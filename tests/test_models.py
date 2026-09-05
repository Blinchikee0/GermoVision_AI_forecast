"""Тесты моделей GV-Resist и GV-Growth."""

from __future__ import annotations

import numpy as np
import pytest

from germovision.core.splitting import LeakageGuard, temporal_cluster_split
from germovision.data import SyntheticConfig, generate_isolates
from germovision.data.synthetic import generate_lineage_counts
from germovision.models import CatalogueBaseline, Decision, GVGrowth, GVResist, PrevalenceBaseline


@pytest.fixture(scope="module")
def dataset():
    return generate_isolates(SyntheticConfig(n_isolates=1800, seed=11))


@pytest.fixture(scope="module")
def split(dataset):
    sp = temporal_cluster_split(dataset.submission_dates, dataset.clusters)
    # Разделение обязано пройти защиту, иначе тестировать нечего.
    LeakageGuard(dates=dataset.submission_dates, clusters=dataset.clusters).run(sp)
    return sp


@pytest.fixture(scope="module")
def rif(dataset, split):
    return GVResist("RIF", random_state=0).fit(dataset, split)


# --------------------------------------------------------------------------
# GV-Resist: обучение и предсказание
# --------------------------------------------------------------------------


def test_model_fits_and_calibrates(rif):
    assert rif.model_ is not None
    assert rif.calibrator_ is not None, "калибратор не обучен на выделенной части"
    assert rif.conformal_q_ is not None
    assert 0.02 <= rif.threshold_ <= 0.90
    assert rif.operating_point_["tuned"] is True
    assert 0.0 < rif.prevalence_ < 1.0


def test_probabilities_are_in_range(rif, dataset, split):
    p = rif.predict_proba(dataset, split.test)
    assert p.shape == split.test.shape
    assert ((p >= 0) & (p <= 1)).all()


def test_predictions_carry_explanation(rif, dataset, split):
    preds = rif.predict(dataset, split.test[:60])
    assert len(preds) == 60
    for pr in preds:
        assert pr.decision in (Decision.RESISTANT, Decision.SUSCEPTIBLE, Decision.NO_CALL)
        assert pr.explain(), "заключение без обоснования выдаваться не должно"


def test_catalogue_marker_forces_resistant_decision(rif, dataset, split):
    """Маркер группы 1 — референсный стандарт, он имеет приоритет."""
    markers = rif.catalogue.resistance_markers("RIF")
    hits = [i for i in split.test if dataset.mutations[i] & markers]
    if not hits:
        pytest.skip("в тестовой части нет изолятов с маркером из каталога")
    preds = rif.predict(dataset, np.array(hits[:20]))
    assert all(p.decision == Decision.RESISTANT for p in preds)
    assert all(p.source == "catalogue" for p in preds)
    assert all(p.evidence for p in preds)


def test_catalogue_decision_does_not_overwrite_probability(rif, dataset, split):
    """Решение берётся из каталога, вероятность остаётся калиброванной.

    Подмена вероятности постоянным значением разрушила бы калибровку:
    пенетрантность маркеров неполна.
    """
    markers = rif.catalogue.resistance_markers("RIF")
    hits = [i for i in split.test if dataset.mutations[i] & markers][:40]
    if len(hits) < 5:
        pytest.skip("мало изолятов с маркером")
    preds = rif.predict(dataset, np.array(hits))
    probs = {round(p.probability, 4) for p in preds}
    assert len(probs) > 1, "вероятность подменена константой"


def test_local_contributions_are_real(rif, dataset, split):
    """Объяснение получено исключением признаков, а не глобальной важностью."""
    preds = rif.predict(dataset, split.test[:120], explain=True)
    model_based = [p for p in preds if p.source == "model" and p.contributions]
    if not model_based:
        pytest.skip("нет предсказаний с ненулевым вкладом признаков")
    name, delta = model_based[0].contributions[0]
    assert name in rif.feature_names_
    assert abs(delta) > 1e-4


def test_explain_skipped_when_not_requested(rif, dataset, split):
    preds = rif.predict(dataset, split.test[:40], explain=False)
    assert all(not p.contributions for p in preds)


# --------------------------------------------------------------------------
# GV-Resist: отказ от ответа
# --------------------------------------------------------------------------


def test_stricter_alpha_produces_more_abstentions(dataset, split):
    """Требование покрытия выше точности модели вынуждает её молчать."""
    lenient = GVResist("RIF", alpha=0.30, random_state=0).fit(dataset, split)
    strict = GVResist("RIF", alpha=0.01, random_state=0).fit(dataset, split)

    def rate(m):
        preds = m.predict(dataset, split.test, explain=False)
        return sum(p.decision == Decision.NO_CALL for p in preds) / len(preds)

    assert rate(strict) >= rate(lenient)


def test_coverage_tradeoff_is_monotone_in_alpha(rif, dataset, split):
    rows = rif.coverage_tradeoff(dataset, split.test, alphas=(0.02, 0.10, 0.30))
    answered = [r["answer_rate"] for r in rows]
    assert answered[0] <= answered[1]
    assert all(0.0 <= a <= 1.0 for a in answered)


def test_evaluation_reports_both_views(rif, dataset, split):
    ev = rif.evaluate(dataset, split.test, n_boot=60)
    assert 0.0 <= ev.decision_sensitivity.value <= 1.0
    assert 0.0 <= ev.decision_specificity.value <= 1.0
    assert ev.ranking.n > 0
    assert abs(ev.answer_rate + ev.abstention_rate - 1.0) < 1e-9


def test_model_beats_catalogue_on_sensitivity(dataset, split):
    """Смысл ML-уровня — находить устойчивость, которую правила пропускают."""
    model = GVResist("RIF", random_state=0).fit(dataset, split)
    ev = model.evaluate(dataset, split.test, n_boot=60)
    base = CatalogueBaseline("RIF").evaluate(dataset, split.test, n_boot=60)
    assert ev.decision_sensitivity.value > base.sensitivity.value


def test_calibration_requires_separate_part(dataset, split):
    """Без выделенной части калибровка не подменяется обучающей выборкой."""
    from germovision.core.types import Split

    no_calib = Split(train=split.train, test=split.test, strategy="no_calib")
    model = GVResist("RIF", random_state=0).fit(dataset, no_calib)
    assert model.calibrator_ is None
    assert model.conformal_q_ is None
    preds = model.predict(dataset, split.test[:20], explain=False)
    assert all(p.decision != Decision.NO_CALL for p in preds)


def test_fit_rejects_single_class(dataset, split):
    from germovision.core.types import Split

    y = dataset.phenotypes["RIF"]
    only_neg = np.array([i for i in split.train if y[i] == 0.0][:200])
    bad = Split(train=only_neg, test=split.test, strategy="degenerate")
    with pytest.raises(ValueError, match="only one class"):
        GVResist("RIF").fit(dataset, bad)


def test_predict_requires_fit(dataset):
    with pytest.raises(RuntimeError, match="not fitted"):
        GVResist("RIF").predict(dataset)


def test_alpha_must_be_valid():
    with pytest.raises(ValueError, match="alpha"):
        GVResist("RIF", alpha=0.9)


# --------------------------------------------------------------------------
# Базовые модели
# --------------------------------------------------------------------------


def test_catalogue_baseline_is_deterministic(dataset, split):
    b = CatalogueBaseline("RIF")
    p1 = b.predict_proba(dataset, split.test)
    p2 = b.predict_proba(dataset, split.test)
    np.testing.assert_array_equal(p1, p2)
    assert set(np.unique(p1)) <= {0.0, 1.0}


def test_prevalence_baseline_reflects_training_rate(dataset, split):
    b = PrevalenceBaseline("RIF").fit(dataset, split)
    y = dataset.phenotypes["RIF"][split.train]
    expected = float(y[~np.isnan(y)].mean())
    assert b.rate_ == pytest.approx(expected)


# --------------------------------------------------------------------------
# GV-Growth
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def growth():
    counts, times, regions, lineages, truth = generate_lineage_counts(n_weeks=26, seed=5)
    model = GVGrowth(n_bootstrap=40, seed=0).fit(counts, times, regions, lineages)
    return model, truth, times


def test_growth_recovers_true_coefficients(growth):
    """Данные порождены той же моделью — оценка обязана совпасть с истиной."""
    model, truth, _ = growth
    target = "L2_Beijing_MDR"
    j = model.lineages_.index(target)
    estimates = [float(f.slopes[j]) for f in model.fits_.values()]
    assert abs(np.mean(estimates) - truth[target]) < 0.04


def test_growth_distinguishes_rising_from_falling(growth):
    model, truth, _ = growth
    rising = model.lineages_.index("L2_Beijing_MDR")
    falling = model.lineages_.index("L3_CAS")
    for fit in model.fits_.values():
        assert fit.slopes[rising] > fit.slopes[falling]


def test_growth_reference_lineage_is_pinned(growth):
    """Первая линия закреплена нулём — иначе модель неидентифицируема."""
    model, _, _ = growth
    for fit in model.fits_.values():
        assert fit.slopes[0] == 0.0
        assert fit.intercepts[0] == 0.0


def test_growth_probabilities_sum_to_one(growth):
    model, _, times = growth
    region = next(iter(model.fits_))
    p = model.fits_[region].probabilities([0, 5, 10])
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_forecast_intervals_are_valid_and_ordered(growth):
    model, _, times = growth
    region = next(iter(model.fits_))
    point, lo, hi = model.forecast(region, [0, 4, 8], last_time=float(times.max()))
    assert (lo >= 0).all() and (hi <= 1).all()
    assert (lo <= point + 1e-9).all() and (point <= hi + 1e-9).all()


def test_forecast_uncertainty_grows_with_horizon(growth):
    model, _, times = growth
    region = next(iter(model.fits_))
    _, lo, hi = model.forecast(region, [0, 8], last_time=float(times.max()))
    j = model.lineages_.index("L2_Beijing_MDR")
    assert (hi[1][j] - lo[1][j]) > (hi[0][j] - lo[0][j])


def test_small_region_gets_wider_intervals(growth):
    """Меньше секвенирования — шире интервал. Это и есть смысл модели счётчиков."""
    model, _, times = growth
    sizes = {r: f.n_samples for r, f in model.fits_.items()}
    small = min(sizes, key=sizes.get)
    large = max(sizes, key=sizes.get)
    j = model.lineages_.index("L2_Beijing_MDR")
    se_small = model.fits_[small].slope_se[j]
    se_large = model.fits_[large].slope_se[j]
    assert se_small > se_large


def test_growth_table_flags_significance(growth):
    model, _, _ = growth
    rows = model.growth_table()
    rising = [r for r in rows if r["lineage"] == "L2_Beijing_MDR"]
    assert rising and any(r["significant"] for r in rising)
    assert all(r["weekly_pct"] == pytest.approx(np.expm1(r["beta"]) * 100) for r in rows)


def test_forecast_rejects_unknown_region(growth):
    model, _, _ = growth
    with pytest.raises(KeyError, match="not fitted"):
        model.forecast("Атлантида", [4])


def test_growth_rejects_single_lineage():
    with pytest.raises(ValueError, match="at least two lineages"):
        GVGrowth().fit(np.ones((5, 1)), np.arange(5), np.array(["A"] * 5), ["only"])
