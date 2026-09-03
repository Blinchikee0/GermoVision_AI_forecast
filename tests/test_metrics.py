"""Тесты метрик."""

from __future__ import annotations

import numpy as np
import pytest

from germovision.core.metrics import (
    IsotonicCalibrator,
    bootstrap_ci,
    brier_score,
    confusion_counts,
    evaluate_binary,
    evaluate_calibration,
    evaluate_coverage,
    expected_calibration_error,
    interval_coverage,
    multinomial_log_score,
    persistence_forecast,
    pr_auc,
    precision_at_k,
    ranked_probability_score,
    roc_auc,
    sensitivity,
    specificity,
)

# --------------------------------------------------------------------------
# Классификация
# --------------------------------------------------------------------------


def test_confusion_counts_basic():
    c = confusion_counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert (c.tp, c.fn, c.fp, c.tn) == (1, 1, 1, 1)


def test_sensitivity_specificity_perfect():
    y = [1, 1, 0, 0]
    assert sensitivity(y, y) == 1.0
    assert specificity(y, y) == 1.0


def test_sensitivity_is_nan_without_positives():
    """Без положительных объектов чувствительность не определена, а не равна нулю."""
    assert np.isnan(sensitivity([0, 0, 0], [0, 0, 1]))


def test_specificity_is_nan_without_negatives():
    assert np.isnan(specificity([1, 1, 1], [1, 0, 1]))


def test_roc_auc_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert roc_auc(y, y.astype(float)) == 1.0
    rng = np.random.default_rng(0)
    assert 0.35 < roc_auc(y, rng.random(100)) < 0.65


def test_pr_auc_sensitive_to_imbalance():
    """PR-AUC падает при дисбалансе сильнее, чем ROC-AUC — в этом её смысл."""
    rng = np.random.default_rng(1)
    y = np.zeros(1000, dtype=int)
    y[:20] = 1
    score = rng.random(1000)
    score[:20] += 0.45  # слабый, но реальный сигнал
    assert roc_auc(y, score) > pr_auc(y, score)


def test_precision_at_k():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, score, 2) == 1.0
    assert precision_at_k(y, score, 4) == 0.5


def test_precision_at_k_clips_to_size():
    y = np.array([1, 0])
    assert precision_at_k(y, np.array([0.9, 0.1]), 100) == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 200)
    score = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)])
    ci = bootstrap_ci(roc_auc, y, score, n_boot=200, seed=0)
    assert ci.lo <= ci.value <= ci.hi
    assert ci.hi - ci.lo < 0.2
    assert "[" in str(ci)


def test_bootstrap_ci_degenerate_returns_nan_bounds():
    ci = bootstrap_ci(roc_auc, np.zeros(10, dtype=int), np.random.random(10), n_boot=10)
    assert np.isnan(ci.lo) and np.isnan(ci.hi)


def test_evaluate_binary_full_report():
    rng = np.random.default_rng(2)
    y = np.array([0] * 180 + [1] * 20)
    prob = np.clip(np.where(y == 1, rng.normal(0.8, 0.1, 200), rng.normal(0.2, 0.1, 200)), 0, 1)
    rep = evaluate_binary(y, prob, label="Рифампицин", threshold=0.5, n_boot=100)

    assert rep.label == "Рифампицин"
    assert rep.n == 200 and rep.n_positive == 20
    assert rep.sensitivity.value > 0.8
    assert rep.specificity.value > 0.8
    assert "Рифампицин" in rep.to_row()
    assert rep.meets(min_sens=0.5, min_spec=0.5)


def test_evaluate_binary_reports_abstention():
    """Отказ от ответа исключается из метрик, но его доля отчитывается."""
    y = np.array([0, 1] * 50)
    prob = np.where(y == 1, 0.9, 0.1)
    abstained = np.zeros(100, dtype=bool)
    abstained[:20] = True

    rep = evaluate_binary(y, prob, abstained=abstained, n_boot=50)
    assert rep.n == 80
    assert rep.abstention_rate == pytest.approx(0.2)


def test_evaluate_binary_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="формы не совпадают"):
        evaluate_binary([0, 1], [0.5, 0.5, 0.5])


# --------------------------------------------------------------------------
# Калибровка
# --------------------------------------------------------------------------


def test_brier_score_bounds():
    assert brier_score([1, 1], [1.0, 1.0]) == 0.0
    assert brier_score([1, 0], [0.5, 0.5]) == 0.25


def test_ece_near_zero_for_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 5000)
    y = rng.binomial(1, p)  # по построению идеально калибровано
    assert expected_calibration_error(y, p, n_bins=10) < 0.05


def test_ece_large_for_overconfident_model():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.2, 0.8, 3000)
    y = rng.binomial(1, p_true)
    p_over = np.clip((p_true - 0.5) * 3 + 0.5, 0.01, 0.99)  # раздутая уверенность
    assert expected_calibration_error(y, p_over, n_bins=10) > 0.1


def test_isotonic_calibrator_reduces_ece():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.1, 0.9, 4000)
    y = rng.binomial(1, p_true)
    p_bad = np.clip(p_true ** 2, 0.01, 0.99)  # систематическое смещение вниз

    cal = IsotonicCalibrator().fit(p_bad[:2000], y[:2000])
    p_fixed = cal.transform(p_bad[2000:])

    ece_before = expected_calibration_error(y[2000:], p_bad[2000:])
    ece_after = expected_calibration_error(y[2000:], p_fixed)
    assert ece_after < ece_before


def test_isotonic_calibrator_preserves_ranking_approximately():
    """Калибровка неубывающая: порядок объектов сохраняется.

    Преобразование кусочно-постоянно, поэтому часть объектов получает
    одинаковую вероятность, и ROC-AUC сдвигается на доли процента.
    Требовать точного совпадения было бы неверно.
    """
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)
    cal = IsotonicCalibrator().fit(p, y)
    calibrated = cal.transform(p)

    assert roc_auc(y, calibrated) == pytest.approx(roc_auc(y, p), abs=0.02)
    # Строгая проверка монотонности: порядок нигде не инвертирован.
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(calibrated[order]) >= -1e-12)


def test_isotonic_calibrator_requires_fit_first():
    with pytest.raises(RuntimeError, match="не обучен"):
        IsotonicCalibrator().transform([0.5])


def test_isotonic_calibrator_rejects_tiny_sample():
    with pytest.raises(ValueError, match="не менее 10"):
        IsotonicCalibrator().fit([0.1, 0.9], [0, 1])


def test_calibration_report_verdict():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 3000)
    y = rng.binomial(1, p)
    rep = evaluate_calibration(y, p)
    assert rep.is_acceptable
    assert "Brier" in rep.to_markdown()


# --------------------------------------------------------------------------
# Прогноз долей вариантов
# --------------------------------------------------------------------------


def test_multinomial_log_score_rewards_truth():
    counts = np.array([[80, 20], [50, 50]], dtype=float)
    good = np.array([[0.8, 0.2], [0.5, 0.5]])
    bad = np.array([[0.2, 0.8], [0.5, 0.5]])
    assert multinomial_log_score(counts, good) < multinomial_log_score(counts, bad)


def test_multinomial_log_score_weights_by_sample_size():
    """Строка на 1000 наблюдений влияет сильнее строки на 10 — это и нужно."""
    big_wrong = np.array([[1000, 0], [5, 5]], dtype=float)
    probs = np.array([[0.1, 0.9], [0.5, 0.5]])
    small_wrong = np.array([[10, 0], [500, 500]], dtype=float)
    assert multinomial_log_score(big_wrong, probs) > multinomial_log_score(small_wrong, probs)


def test_multinomial_log_score_normalizes_probs():
    counts = np.array([[1, 1]], dtype=float)
    assert multinomial_log_score(counts, np.array([[2.0, 2.0]])) == pytest.approx(
        multinomial_log_score(counts, np.array([[0.5, 0.5]]))
    )


def test_ranked_probability_score_penalizes_distance():
    """Промах на одну градацию штрафуется слабее, чем на три."""
    probs_near = np.array([[0.0, 1.0, 0.0, 0.0]])
    probs_far = np.array([[0.0, 0.0, 0.0, 1.0]])
    obs = np.array([0])
    assert ranked_probability_score(obs, probs_near) < ranked_probability_score(obs, probs_far)


def test_ranked_probability_score_perfect_is_zero():
    assert ranked_probability_score([1], np.array([[0.0, 1.0, 0.0]])) == pytest.approx(0.0)


def test_interval_coverage_counts_hits():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    assert interval_coverage(truth, truth - 0.5, truth + 0.5) == 1.0
    assert interval_coverage(truth, [0.5, 0.5, 1.5, 2.5], [1.5, 1.5, 2.5, 3.5]) == 0.75


def test_coverage_report_flags_overconfident_intervals():
    """Слишком узкие интервалы вокруг прогноза — ложная уверенность."""
    rng = np.random.default_rng(0)
    prediction = rng.normal(size=1000)
    observed = prediction + rng.normal(scale=1.0, size=1000)
    rep = evaluate_coverage(observed, prediction - 0.01, prediction + 0.01, nominal=0.95)
    assert rep.empirical < 0.1
    assert "ЗАНИЖЕНО" in rep.verdict


def test_coverage_report_flags_overly_wide_intervals():
    rng = np.random.default_rng(1)
    prediction = rng.normal(size=1000)
    observed = prediction + rng.normal(scale=1.0, size=1000)
    rep = evaluate_coverage(observed, prediction - 10, prediction + 10, nominal=0.95)
    assert rep.empirical == 1.0
    assert "завышено" in rep.verdict


def test_coverage_report_detects_correct_intervals():
    rng = np.random.default_rng(5)
    mu = rng.normal(size=4000)
    obs = mu + rng.normal(size=4000)
    rep = evaluate_coverage(obs, mu - 1.96, mu + 1.96, nominal=0.95)
    assert rep.verdict == "корректное"


def test_interval_coverage_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="ниже нижней"):
        interval_coverage([1.0], [2.0], [0.0])


def test_persistence_forecast_repeats_last_row():
    hist = np.array([[0.5, 0.5], [0.7, 0.3]])
    fc = persistence_forecast(hist, horizon=3)
    assert fc.shape == (3, 2)
    assert np.allclose(fc[0], [0.7, 0.3])
    assert np.allclose(fc.sum(axis=1), 1.0)


def test_persistence_forecast_rejects_empty_history():
    with pytest.raises(ValueError, match="история пуста"):
        persistence_forecast(np.empty((0, 2)), horizon=1)
