"""Тесты быстрых ядер метрик, сохранения моделей и режима предсказания."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from germovision.core.metrics import bootstrap_metrics, evaluate_binary
from germovision.core.metrics._kernels import (
    fast_average_precision,
    fast_roc_auc,
    fast_sens_spec,
)
from germovision.core.splitting import temporal_cluster_split
from germovision.data import SyntheticConfig, generate_isolates
from germovision.models import GVResist
from germovision.persistence import BUNDLE_FORMAT, ModelBundle, load_bundle, save_bundle
from germovision.predict import format_report, load_isolates, predict_isolates

# --------------------------------------------------------------------------
# Быстрые ядра метрик
# --------------------------------------------------------------------------


def test_kernels_match_sklearn():
    """Ядра заменяют sklearn во внутреннем цикле — расхождений быть не должно."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(40, 600))
        y = rng.binomial(1, float(rng.uniform(0.03, 0.6)), n)
        if y.sum() in (0, n):
            continue
        # Округление создаёт совпадающие баллы: обработка связок — самая
        # тонкая часть обеих метрик.
        s = np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        assert fast_roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-10)
        assert fast_average_precision(y, s) == pytest.approx(
            average_precision_score(y, s), abs=1e-10
        )


def test_kernels_handle_degenerate_classes():
    y = np.zeros(20, dtype=int)
    s = np.random.default_rng(0).random(20)
    assert np.isnan(fast_roc_auc(y, s))
    assert np.isnan(fast_average_precision(y, s))


def test_fast_sens_spec_matches_definition():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    pred = np.array([True, True, False, True, False, False, False])
    sens, spec = fast_sens_spec(y, pred)
    assert sens == pytest.approx(2 / 3)
    assert spec == pytest.approx(3 / 4)


def test_bootstrap_metrics_shares_resamples():
    """Общие выборки делают интервалы сопоставимыми между метриками."""
    rng = np.random.default_rng(1)
    y = np.array([0] * 300 + [1] * 100)
    s = np.concatenate([rng.normal(0, 1, 300), rng.normal(1.4, 1, 100)])

    cis = bootstrap_metrics(
        y, s, {"roc": fast_roc_auc, "pr": fast_average_precision}, n_boot=120, seed=3
    )
    assert set(cis) == {"roc", "pr"}
    for ci in cis.values():
        assert ci.lo <= ci.value <= ci.hi
        assert ci.n == 400


def test_bootstrap_metrics_degenerate_returns_nan_bounds():
    y = np.zeros(30, dtype=int)
    s = np.random.default_rng(0).random(30)
    cis = bootstrap_metrics(y, s, {"roc": fast_roc_auc}, n_boot=20)
    assert np.isnan(cis["roc"].lo)


def test_evaluate_binary_still_consistent():
    """Переход на ядра не должен менять результат evaluate_binary."""
    rng = np.random.default_rng(5)
    y = np.array([0] * 200 + [1] * 40)
    p = np.clip(np.where(y == 1, rng.normal(0.75, 0.15, 240), rng.normal(0.25, 0.15, 240)), 0, 1)
    rep = evaluate_binary(y, p, label="test", threshold=0.5, n_boot=200)
    assert rep.roc_auc.value == pytest.approx(roc_auc_score(y, p), abs=1e-10)
    assert rep.pr_auc.value == pytest.approx(average_precision_score(y, p), abs=1e-10)


# --------------------------------------------------------------------------
# Сохранение и загрузка
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    ds = generate_isolates(SyntheticConfig(n_isolates=1500, seed=21))
    sp = temporal_cluster_split(ds.submission_dates, ds.clusters)
    models = {d: GVResist(d, random_state=0).fit(ds, sp) for d in ("RIF", "INH")}
    return ds, sp, models


def test_bundle_roundtrip(tmp_path, trained):
    ds, sp, models = trained
    bundle = ModelBundle(models=models, manifest={"source": "synthetic", "synthetic": True})
    save_bundle(bundle, tmp_path / "m")
    loaded = load_bundle(tmp_path / "m")

    assert loaded.drugs == ["INH", "RIF"]
    assert loaded.manifest["format"] == BUNDLE_FORMAT
    assert "trained_at" in loaded.manifest

    # Предсказания после загрузки совпадают с исходными до последнего знака.
    before = models["RIF"].predict_proba(ds, sp.test)
    after = loaded.models["RIF"].predict_proba(ds, sp.test)
    np.testing.assert_allclose(before, after)


def test_manifest_readable_without_loading_models(tmp_path, trained):
    """Манифест — обычный JSON рядом с моделями.

    Загрузка joblib исполняет код, поэтому решение о доверии к файлу
    должно приниматься до загрузки.
    """
    _, _, models = trained
    save_bundle(ModelBundle(models=models, manifest={"source": "x"}), tmp_path / "m")
    data = json.loads((tmp_path / "m" / "manifest.json").read_text(encoding="utf-8"))
    assert data["drugs"] == ["INH", "RIF"]


def test_load_rejects_incompatible_format(tmp_path, trained):
    _, _, models = trained
    save_bundle(ModelBundle(models=models, manifest={"format": 999}), tmp_path / "m")
    with pytest.raises(ValueError, match="retrain"):
        load_bundle(tmp_path / "m")


def test_load_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No saved models"):
        load_bundle(tmp_path / "нет")


def test_describe_flags_synthetic_and_confirmation(tmp_path, trained):
    _, _, models = trained
    bundle = ModelBundle(
        models=models,
        manifest={
            "source": "synthetic",
            "synthetic": True,
            "quality": {"RIF": {"requires_confirmation": True}},
        },
    )
    text = bundle.describe()
    assert "SYNTHETIC" in text
    assert "RIF" in text


# --------------------------------------------------------------------------
# Режим предсказания
# --------------------------------------------------------------------------


def _write_mutations(path):
    path.write_text(
        "id,gene,mutation\n"
        "A1,rpoB,S450L\n"
        "A1,katG,S315T\n"
        "A2,rpoB,S488A\n",
        encoding="utf-8",
    )
    return path


def test_load_isolates_parses_csv(tmp_path):
    ds = load_isolates(_write_mutations(tmp_path / "m.csv"))
    assert len(ds) == 2
    assert ds.mutations[0] == {"rpoB_S450L", "katG_S315T"}
    assert ds.meta["mode"] == "prediction"
    # Фенотипов нет — это режим применения, а не оценки.
    assert ds.phenotypes == {}


def test_load_isolates_requires_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,ген,замена\nA1,rpoB,S450L\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_isolates(p)


def test_load_isolates_rejects_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("id,gene,mutation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no isolates found"):
        load_isolates(p)


def test_load_isolates_reads_sample_metadata(tmp_path):
    _write_mutations(tmp_path / "m.csv")
    (tmp_path / "s.csv").write_text(
        "id,lineage,country\nA1,L2_Beijing,KZ\n", encoding="utf-8"
    )
    ds = load_isolates(tmp_path / "m.csv", tmp_path / "s.csv")
    assert ds.lineages[0] == "L2_Beijing"
    assert ds.countries[1] == "unknown"


def test_predict_produces_full_report(tmp_path, trained):
    _, _, models = trained
    bundle = ModelBundle(models=models, manifest={"source": "synthetic", "synthetic": True})
    ds = load_isolates(_write_mutations(tmp_path / "m.csv"))
    reports = predict_isolates(bundle, ds)

    assert len(reports) == 2
    assert {d["drug"] for d in reports[0]["drugs"]} == {"RIF", "INH"}
    for d in reports[0]["drugs"]:
        assert d["explanation"]

    text = format_report(reports, bundle)
    assert "A1" in text
    assert "does not replace" in text


def test_catalogue_marker_drives_decision_in_prediction(tmp_path, trained):
    """Изолят с rpoB S450L обязан получить вывод об устойчивости к RIF."""
    _, _, models = trained
    bundle = ModelBundle(models=models, manifest={})
    ds = load_isolates(_write_mutations(tmp_path / "m.csv"))
    reports = predict_isolates(bundle, ds)

    rif = next(d for d in reports[0]["drugs"] if d["drug"] == "RIF")
    assert rif["decision"] == "resistant"
    assert rif["source"] == "catalogue"
    assert "S450L" in rif["explanation"]


# --------------------------------------------------------------------------
# Рабочая точка
# --------------------------------------------------------------------------


def test_operating_point_is_recorded(trained):
    _, _, models = trained
    op = models["RIF"].operating_point_
    assert op["tuned"] is True
    assert 0.01 <= op["threshold"] <= 0.99
    assert "meets_clinical_limit" in op


def test_confirmation_flag_matches_limit(trained):
    """Пометка о фенотипе выставляется ровно тогда, когда лимит не выдержан."""
    _, _, models = trained
    for m in models.values():
        op = m.operating_point_
        assert m.requires_confirmation_ == (not op["meets_clinical_limit"])


def test_evaluation_exposes_operational_metrics(trained):
    ds, sp, models = trained
    ev = models["RIF"].evaluate(ds, sp.test, n_boot=50)
    assert 0.0 <= ev.correctly_closed <= 1.0
    assert 0.0 <= ev.missed_resistance <= 1.0
    assert "correctly closed" in ev.summary_line()
