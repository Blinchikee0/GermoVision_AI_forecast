"""Пайплайн обучения GermoVision.

Запуск:
    python -m germovision.train                       # синтетика, полный прогон
    python -m germovision.train --source data/cryptic # реальные данные
    python -m germovision.train --quick               # быстрый прогон

Порядок шагов зафиксирован протоколом валидации (§ 5.7) и не может быть
изменён без нарушения корректности результатов:

    1. Загрузка данных
    2. Разделение выборки       ← ДО любой обработки признаков
    3. Проверка на утечки       ← падает, если протокол нарушен
    4. Обучение (словарь признаков строится только по train)
    5. Оценка на нетронутом тесте
    6. Внешняя валидация по стране, отсутствующей в обучении
    7. Абляции: что именно даёт прирост
    8. Отчёт

Шаг 3 не является формальностью: при нарушении порядка он останавливает
прогон, а не печатает предупреждение. Метрики, полученные с утечкой,
не должны попадать в отчёт даже помеченными.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core.metrics.calibration import evaluate_calibration
from .core.splitting import LeakageGuard, holdout_group, temporal_cluster_split
from .core.types import LeakageError, Split
from .data import (
    DRUG_NAMES,
    DRUGS,
    IsolateDataset,
    MutationCatalogue,
    SyntheticConfig,
    generate_isolates,
    load_cryptic,
)
from .data.synthetic import generate_lineage_counts
from .models import CatalogueBaseline, GVGrowth, GVResist
from .models.resist import Decision
from .persistence import ModelBundle, save_bundle

#: Целевые пороги гипотезы H1 (§ 2.3 мастер-документа), зафиксированные
#: до эксперимента. Проверяются на внешней валидации, а не на внутреннем
#: тесте: цель — работа в стране, данных из которой в обучении не было.
H1_TARGETS = {"RIF": (0.90, 0.95), "INH": (0.90, 0.95)}


def _ci(m) -> list:
    """Метрика с интервалом в сериализуемом виде."""
    return [m.value, m.lo, m.hi]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _header(title: str) -> None:
    _log()
    _log("=" * 76)
    _log(title)
    _log("=" * 76)


def load_dataset(source: str, n_isolates: int, seed: int) -> IsolateDataset:
    """Загрузить набор данных из указанного источника."""
    if source == "synthetic":
        return generate_isolates(SyntheticConfig(n_isolates=n_isolates, seed=seed))
    return load_cryptic(source)


def build_split(ds: IsolateDataset) -> Split:
    """Разделить выборку по времени, не разрывая кластеры родства."""
    return temporal_cluster_split(ds.submission_dates, ds.clusters, test_size=0.2, calib_size=0.15)


def verify_split(ds: IsolateDataset, split: Split) -> str:
    """Прогнать защиту от утечек. Падает при нарушении протокола."""
    guard = LeakageGuard(dates=ds.submission_dates, clusters=ds.clusters)
    return str(guard.run(split))


def train_drug(
    ds: IsolateDataset,
    split: Split,
    drug: str,
    n_boot: int,
    catalogue: MutationCatalogue,
) -> dict | None:
    """Обучить и оценить одну пару «препарат — модель» вместе с базой."""
    try:
        model = GVResist(drug, catalogue=catalogue).fit(ds, split)
        ev = model.evaluate(ds, split.test, n_boot=n_boot)
        baseline = CatalogueBaseline(drug, catalogue).evaluate(ds, split.test, n_boot=n_boot)
    except ValueError as exc:
        # Препарат с почти отсутствующей устойчивостью в выборке пропускается,
        # но прогон продолжается: остальные двенадцать от этого не зависят.
        return {"drug": drug, "skipped": str(exc)}

    # Калибровка оценивается на тесте: смысл в том, соответствуют ли
    # вероятности реальности на данных, которых модель не видела.
    eval_idx = model._labelled(ds, split.test)
    y_test = ds.phenotypes[drug][eval_idx].astype(int)
    p_test = model.predict_proba(ds, eval_idx)
    calib = evaluate_calibration(y_test, p_test)

    tradeoff = model.coverage_tradeoff(ds, split.test)

    r = ev.ranking
    return {
        "drug": drug,
        "drug_name": DRUG_NAMES.get(drug, drug),
        "n_train": model.n_train_,
        "n_test": r.n,
        "n_positive": r.n_positive,
        "prevalence": round(model.prevalence_, 4),
        "decision": {
            "sensitivity": _ci(ev.decision_sensitivity),
            "specificity": _ci(ev.decision_specificity),
        },
        "ranking": {"pr_auc": _ci(r.pr_auc), "roc_auc": _ci(r.roc_auc)},
        "baseline_catalogue": {
            "sensitivity": _ci(baseline.sensitivity),
            "specificity": _ci(baseline.specificity),
            "pr_auc": _ci(baseline.pr_auc),
        },
        "calibration": {"brier": calib.brier, "ece": calib.ece, "acceptable": calib.is_acceptable},
        "routing": {
            "by_catalogue": ev.n_by_catalogue,
            "by_model": ev.n_evaluated + ev.n_abstained - ev.n_by_catalogue,
            "no_call": ev.n_abstained,
            "no_call_rate": round(ev.abstention_rate, 4),
        },
        "answer_rate": round(ev.answer_rate, 4),
        "correctly_closed": round(ev.correctly_closed, 4),
        "missed_resistance": round(ev.missed_resistance, 4),
        "requires_confirmation": bool(ev.requires_confirmation),
        "operating_point": model.operating_point_,
        "coverage_tradeoff": tradeoff,
        "delta_sensitivity_vs_catalogue": round(
            ev.decision_sensitivity.value - baseline.sensitivity.value, 4
        ),
    }


def external_validation(
    ds: IsolateDataset, drug: str, country: str, n_boot: int, catalogue: MutationCatalogue
) -> dict | None:
    """Обучение без данных страны, проверка на ней.

    Это единственная оценка, отражающая реальный сценарий развёртывания:
    в Казахстане система встретит изоляты, которых в обучающей выборке не
    было. Метрика здесь всегда ниже внутренней, и разрыв между ними — и
    есть честная мера обобщающей способности.
    """
    if country not in set(np.asarray(ds.countries).tolist()):
        return None
    try:
        split = holdout_group(ds.countries, country)
        # Внутри обучающей части выделяем калибровочную по времени.
        train_ds_dates = ds.submission_dates[split.train]
        cutoff = np.quantile(train_ds_dates.astype("datetime64[D]").astype(int), 0.85)
        calib_mask = train_ds_dates.astype("datetime64[D]").astype(int) > cutoff
        split = Split(
            train=split.train[~calib_mask],
            calib=split.train[calib_mask],
            test=split.test,
            strategy="leave_country_out+calib",
            meta={**split.meta, "country": country},
        )
        model = GVResist(drug, catalogue=catalogue).fit(ds, split)
        ev = model.evaluate(ds, split.test, n_boot=n_boot)
        baseline = CatalogueBaseline(drug, catalogue).evaluate(ds, split.test, n_boot=n_boot)
    except ValueError as exc:
        return {"drug": drug, "country": country, "skipped": str(exc)}

    if ev.ranking.n < 20 or ev.ranking.n_positive < 5:
        # Оценка по десятку изолятов не несёт информации: интервал шире
        # самой метрики. Честнее не отчитываться вовсе.
        return {
            "drug": drug,
            "country": country,
            "skipped": f"too little data: n={ev.ranking.n}, resistant {ev.ranking.n_positive}",
        }

    target = H1_TARGETS.get(drug)
    return {
        "drug": drug,
        "drug_name": DRUG_NAMES.get(drug, drug),
        "country": country,
        "n_test": ev.ranking.n,
        "n_positive": ev.ranking.n_positive,
        "sensitivity": _ci(ev.decision_sensitivity),
        "specificity": _ci(ev.decision_specificity),
        "pr_auc": _ci(ev.ranking.pr_auc),
        "abstention_rate": round(ev.abstention_rate, 4),
        "correctly_closed": round(ev.correctly_closed, 4),
        "missed_resistance": round(ev.missed_resistance, 4),
        "baseline_sensitivity": baseline.sensitivity.value,
        "baseline_specificity": baseline.specificity.value,
        "h1_target": list(target) if target else None,
        "h1_met": bool(ev.meets_h1(*target)) if target else None,
    }


def run_ablations(
    ds: IsolateDataset, split: Split, drug: str, n_boot: int, catalogue: MutationCatalogue
) -> list[dict]:
    """Измерить вклад каждой группы признаков (§ 5.10).

    Без этой таблицы нельзя утверждать, что сложные компоненты нужны:
    возможно, всё качество даёт один каталог, а остальное — украшение.
    """
    configs = [
        ("WHO catalogue only (baseline)", None),
        (
            "mutations, no catalogue features",
            dict(
                use_catalogue_features=False, use_burden=False,
                use_discovery=False, use_catalogue_tier=False,
            ),
        ),
        (
            "+ catalogue features",
            dict(use_discovery=False, use_burden=False, use_catalogue_tier=False),
        ),
        ("+ per-gene burden", dict(use_discovery=False, use_catalogue_tier=False)),
        ("+ variants outside target genes", dict(use_catalogue_tier=False)),
        ("full model (+ rule tier)", {}),
        ("... plus lineage context (hurts)", dict(use_context=True)),
    ]

    rows: list[dict] = []
    for name, kwargs in configs:
        try:
            if kwargs is None:
                rep = CatalogueBaseline(drug, catalogue).evaluate(ds, split.test, n_boot=n_boot)
                sens, spec = rep.sensitivity.value, rep.specificity.value
                pr, abst = rep.pr_auc.value, 0.0
            else:
                model = GVResist(drug, catalogue=catalogue, **kwargs).fit(ds, split)
                ev = model.evaluate(ds, split.test, n_boot=n_boot)
                sens = ev.decision_sensitivity.value
                spec = ev.decision_specificity.value
                pr, abst = ev.ranking.pr_auc.value, ev.abstention_rate
        except ValueError as exc:
            rows.append({"config": name, "skipped": str(exc)})
            continue
        rows.append({
            "config": name,
            "sensitivity": sens,
            "specificity": spec,
            "pr_auc": pr,
            "abstention_rate": abst,
        })
    return rows


def fit_growth() -> dict:
    """Подогнать модель роста линий и проверить восстановление истины.

    Синтетические счётчики порождены той же мультиномиальной моделью с
    известными коэффициентами. Сравнение оценки с заложенной истиной —
    проверка того, что модель измеряет то, что заявляет, а не рисует
    правдоподобную кривую.
    """
    counts, times, regions, lineages, truth = generate_lineage_counts()
    model = GVGrowth(n_bootstrap=60).fit(counts, times, regions, lineages)

    recovery = []
    for lin in lineages:
        j = lineages.index(lin)
        estimates = [float(f.slopes[j]) for f in model.fits_.values()]
        recovery.append({
            "lineage": lin,
            "true_beta": truth[lin],
            "mean_estimated_beta": float(np.mean(estimates)),
            "sd_across_regions": float(np.std(estimates)),
        })

    forecasts = {}
    for region in list(model.fits_)[:8]:
        point, lo, hi = model.forecast(region, horizons=[0, 2, 4, 8], last_time=float(times.max()))
        forecasts[region] = {
            "horizons": [0, 2, 4, 8],
            "point": point.tolist(),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "n_samples": model.fits_[region].n_samples,
        }

    observed = []
    for week in sorted(set(times.tolist())):
        mask = times == week
        tot = counts[mask].sum(axis=0)
        observed.append({"week": int(week), "counts": tot.tolist(), "total": float(tot.sum())})

    return {
        "lineages": lineages,
        "tau": model.tau_,
        "recovery": recovery,
        "growth_table": model.growth_table(),
        "forecasts": forecasts,
        "observed_national": observed,
    }


def build_example_reports(
    ds: IsolateDataset,
    split: Split,
    catalogue: MutationCatalogue,
    n_examples: int = 3,
) -> tuple[list[dict], dict[str, GVResist]]:
    """Собрать полные заключения по нескольким изолятам из тестовой части.

    Это то, что реально видит врач: таблица по всем препаратам с решением,
    вероятностью и обоснованием. Выгружаются настоящие выходы обученных
    моделей, а не составленный вручную пример, — иначе интерфейс
    показывал бы то, чего система не делает.

    Отбираются намеренно разные случаи: изолят с множественной
    устойчивостью, чувствительный и такой, по которому система хотя бы
    раз отказалась от ответа. Последний важнее прочих: отказ — штатный
    и правильный исход, и он должен быть виден в интерфейсе.
    """
    models: dict[str, GVResist] = {}
    for drug in DRUGS:
        if drug not in ds.phenotypes:
            continue
        try:
            models[drug] = GVResist(drug, catalogue=catalogue).fit(ds, split)
        except ValueError:
            continue
    if not models:
        return [], {}

    test_idx = np.asarray(split.test, dtype=int)
    all_preds: dict[int, dict[str, object]] = {i: {} for i in test_idx}
    for drug, model in models.items():
        for i, pred in zip(test_idx, model.predict(ds, test_idx, explain=True), strict=True):
            all_preds[i][drug] = pred

    def n_resistant(i: int) -> int:
        return sum(
            1 for pr in all_preds[i].values() if pr.decision == Decision.RESISTANT
        )

    def n_nocall(i: int) -> int:
        return sum(1 for pr in all_preds[i].values() if pr.decision == Decision.NO_CALL)

    ranked_mdr = sorted(test_idx, key=lambda i: -n_resistant(i))
    ranked_clean = sorted(test_idx, key=lambda i: (n_resistant(i), -len(ds.mutations[i])))
    ranked_nocall = sorted(test_idx, key=lambda i: -n_nocall(i))

    chosen: list[int] = []
    for pool in (ranked_mdr, ranked_nocall, ranked_clean):
        for i in pool:
            if i not in chosen:
                chosen.append(int(i))
                break
        if len(chosen) >= n_examples:
            break

    reports: list[dict] = []
    for i in chosen:
        rows = []
        for drug in DRUGS:
            pr = all_preds[i].get(drug)
            if pr is None:
                continue
            truth = ds.phenotypes[drug][i]
            rows.append({
                "drug": drug,
                "drug_name": DRUG_NAMES.get(drug, drug),
                "decision": pr.decision,
                "probability": round(float(pr.probability), 3),
                "source": pr.source,
                "explanation": pr.explain(),
                "ood": bool(pr.ood),
                "phenotype": None if np.isnan(truth) else int(truth),
            })
        reports.append({
            "isolate_id": str(ds.isolate_ids[i]),
            "lineage": str(ds.lineages[i]),
            "country": str(ds.countries[i]),
            "collection_date": str(ds.collection_dates[i]),
            "submission_date": str(ds.submission_dates[i]),
            "mutations": sorted(ds.mutations[i]),
            "n_resistant": n_resistant(i),
            "n_no_call": n_nocall(i),
            "drugs": rows,
        })
    return reports, models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train GermoVision models")
    parser.add_argument(
        "--source", default="synthetic", help="'synthetic' or a directory of CSV files"
    )
    parser.add_argument("--out", default="reports", help="output directory for reports")
    parser.add_argument("--n-isolates", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--external-country", default="KZ")
    parser.add_argument("--catalogue-tsv", default=None, help="full WHO catalogue as TSV")
    parser.add_argument(
        "--quick", action="store_true", help="less data and fewer bootstrap samples"
    )
    parser.add_argument("--no-growth", action="store_true")
    parser.add_argument(
        "--save-models",
        default=None,
        help="directory to save trained models (see germovision.predict)",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    n_boot = 100 if args.quick else 500
    n_isolates = 1500 if args.quick else args.n_isolates
    started = time.time()

    catalogue = (
        MutationCatalogue.from_who_tsv(args.catalogue_tsv)
        if args.catalogue_tsv
        else MutationCatalogue()
    )

    # --- 1. Данные -------------------------------------------------------
    _header("1. DATA")
    ds = load_dataset(args.source, n_isolates, args.seed)
    _log(ds.summary())
    if ds.meta.get("synthetic"):
        _log()
        _log("  ⚠ " + ds.meta["warning"])
    _log(f"\nMutation catalogue: {len(catalogue)} entries")

    # --- 2. Разделение ---------------------------------------------------
    _header("2. SPLIT")
    split = build_split(ds)
    _log(f"{split}")
    for k, v in split.meta.items():
        _log(f"  {k}: {v}")

    # --- 3. Защита -------------------------------------------------------
    _header("3. LEAKAGE GUARD")
    try:
        _log(verify_split(ds, split))
    except LeakageError as exc:
        _log(f"HALTED: {exc}")
        return 1

    # --- 4–5. Обучение и оценка ------------------------------------------
    _header("4. TRAINING AND INTERNAL TEST")
    _log("Closed  — share of isolates given a correct answer within 1-2 days.")
    _log("Missed  — share of resistant isolates confidently called susceptible.")
    _log("Both are computed over all isolates, not only the answered ones.")
    _log("")
    _log(
        f"{'Drug':<16}{'Sensitivity (model)':<22}{'Catalogue':>10}"
        f"{'Closed':>9}{'Missed':>9}{'Answered':>10}"
    )
    _log("-" * 76)

    per_drug: list[dict] = []
    for drug in DRUGS:
        if drug not in ds.phenotypes:
            continue
        res = train_drug(ds, split, drug, n_boot, catalogue)
        if res is None or "skipped" in res:
            _log(f"{drug:<16}skipped: {res.get('skipped', '')[:50] if res else ''}")
            continue
        per_drug.append(res)
        m = res["decision"]["sensitivity"]
        b = res["baseline_catalogue"]["sensitivity"]
        _log(
            f"{res['drug_name']:<16}"
            f"{f'{m[0]:.3f} [{m[1]:.3f}–{m[2]:.3f}]':<22}"
            f"{b[0]:>9.3f}"
            f"{res['correctly_closed']:>10.1%}"
            f"{res['missed_resistance']:>10.1%}"
            f"{res['answer_rate']:>9.1%}"
            f"{'  lab needed' if res['requires_confirmation'] else ''}"
        )

    # --- 6. Внешняя валидация --------------------------------------------
    _header(f"5. EXTERNAL VALIDATION: trained without {args.external_country}, tested on it")
    external: list[dict] = []
    for drug in DRUGS:
        if drug not in ds.phenotypes:
            continue
        res = external_validation(ds, drug, args.external_country, n_boot, catalogue)
        if res is None or "skipped" in res:
            continue
        external.append(res)
        mark = ""
        if res["h1_met"] is True:
            mark = "  [+] H1 met"
        elif res["h1_met"] is False:
            mark = "  [-] H1 NOT met"
        _log(
            f"{res['drug_name']:<16}"
            f"sens {res['sensitivity'][0]:.3f}  "
            f"spec {res['specificity'][0]:.3f}  "
            f"(n={res['n_test']}, resistant {res['n_positive']}){mark}"
        )

    # --- 7. Абляции ------------------------------------------------------
    _header("6. ABLATIONS: what actually helps (rifampicin)")
    ablations = run_ablations(ds, split, "RIF", n_boot, catalogue)
    _log(f"{'Configuration':<40}{'Sens':>8}{'Spec':>8}{'PR-AUC':>9}{'Abstain':>9}")
    _log("-" * 76)
    for row in ablations:
        if "skipped" in row:
            _log(f"{row['config']:<40}skipped")
            continue
        _log(
            f"{row['config']:<40}{row['sensitivity']:>8.3f}"
            f"{row['specificity']:>8.3f}{row['pr_auc']:>9.3f}"
            f"{row['abstention_rate']:>9.1%}"
        )

    # --- 7-бис. Компромисс покрытия --------------------------------------
    _header("7. ANSWER RATE VS ACCURACY (rifampicin)")
    _log("Abstaining sends the sample for a phenotypic test: speed traded for")
    _log("certainty. Where to draw the line is the organisation's call.")
    _log()
    rif = next((r for r in per_drug if r["drug"] == "RIF"), None)
    if rif:
        _log(f"{'alpha':>7}{'Answered':>12}{'Accuracy':>12}{'Sens':>12}{'Spec':>12}")
        _log("-" * 76)
        for row in rif["coverage_tradeoff"]:
            if row.get("answer_rate", 0) == 0:
                _log(f"{row['alpha']:>7.2f}{'0.0%':>12}{'—':>12}{'—':>12}{'—':>12}")
                continue
            _log(
                f"{row['alpha']:>7.2f}{row['answer_rate']:>12.1%}"
                f"{row['accuracy']:>12.3f}{row['sensitivity']:>12.3f}"
                f"{row['specificity']:>12.3f}"
            )

    # --- 8. Модель роста -------------------------------------------------
    growth = None
    if not args.no_growth:
        _header("8. GV-GROWTH: recovering lineage growth coefficients")
        growth = fit_growth()
        _log(f"{'Lineage':<24}{'True beta':>12}{'Estimate':>12}{'Spread':>12}")
        _log("-" * 76)
        for row in growth["recovery"]:
            _log(
                f"{row['lineage']:<24}{row['true_beta']:>12.4f}"
                f"{row['mean_estimated_beta']:>12.4f}{row['sd_across_regions']:>12.4f}"
            )
        _log(f"\nEstimated shrinkage tau = {growth['tau']:.4f}")

    # --- 9. Примеры заключений -------------------------------------------
    _header("9. EXAMPLE ISOLATE REPORTS")
    examples, trained = build_example_reports(ds, split, catalogue)
    for ex in examples:
        _log(
            f"{ex['isolate_id']}  lineage {ex['lineage']}  ({ex['country']}): "
            f"resistant to {ex['n_resistant']} drugs, "
            f"no call on {ex['n_no_call']}"
        )

    # --- 9-бис. Сохранение моделей ---------------------------------------
    if args.save_models and trained:
        quality = {
            r["drug"]: {
                "correctly_closed": r["correctly_closed"],
                "missed_resistance": r["missed_resistance"],
                "requires_confirmation": r["requires_confirmation"],
                "sensitivity": r["decision"]["sensitivity"][0],
                "specificity": r["decision"]["specificity"][0],
            }
            for r in per_drug
        }
        bundle = ModelBundle(
            models=trained,
            manifest={
                "source": args.source,
                "synthetic": bool(ds.meta.get("synthetic")),
                "warning": ds.meta.get("warning", ""),
                "n_train": int(split.train.size),
                "split_strategy": split.strategy,
                "split_meta": split.meta,
                "catalogue_size": len(catalogue),
                "quality": quality,
            },
        )
        saved = save_bundle(bundle, args.save_models)
        _log(f"\nModels saved to: {saved}")
        _log("Use them:  python -m germovision.predict --models "
             f"{saved} --mutations samples.csv")

    # --- 10. Отчёт -------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 1),
        "source": args.source,
        "synthetic": bool(ds.meta.get("synthetic")),
        "warning": ds.meta.get("warning", ""),
        "dataset": {
            "n_isolates": len(ds),
            "n_clusters": int(np.unique(ds.clusters).size),
            "n_countries": int(np.unique(ds.countries).size),
            "n_lineages": int(np.unique(ds.lineages).size),
            "n_variants": len(ds.all_mutation_keys()),
            "date_min": str(ds.submission_dates.min()),
            "date_max": str(ds.submission_dates.max()),
            "countries": sorted(set(np.asarray(ds.countries).tolist())),
            "resistance_rates": {d: ds.resistance_rate(d) for d in ds.drugs},
            "by_country": _country_table(ds),
        },
        "split": {"sizes": split.sizes, "strategy": split.strategy, "meta": split.meta},
        "per_drug": per_drug,
        "external_validation": external,
        "ablations": ablations,
        "growth": growth,
        "example_reports": examples,
        "catalogue_size": len(catalogue),
    }

    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    report_path = out / "report.md"
    report_path.write_text(_markdown_report(payload), encoding="utf-8")

    _header("DONE")
    _log(f"Metrics:  {metrics_path}")
    _log(f"Report:   {report_path}")
    _log(f"Elapsed:  {payload['elapsed_sec']}s")
    return 0


def _country_table(ds: IsolateDataset) -> list[dict]:
    """Доля устойчивости по странам — основа карты в интерфейсе."""
    rows = []
    for country in sorted(set(np.asarray(ds.countries).tolist())):
        mask = np.asarray(ds.countries) == country
        entry = {"country": country, "n": int(mask.sum())}
        for drug in ("RIF", "INH"):
            if drug not in ds.phenotypes:
                continue
            y = ds.phenotypes[drug][mask]
            y = y[~np.isnan(y)]
            entry[f"{drug}_rate"] = float(y.mean()) if y.size else None
            entry[f"{drug}_n"] = int(y.size)
        # МЛУ: устойчивость одновременно к рифампицину и изониазиду.
        if "RIF" in ds.phenotypes and "INH" in ds.phenotypes:
            r, i = ds.phenotypes["RIF"][mask], ds.phenotypes["INH"][mask]
            both = ~np.isnan(r) & ~np.isnan(i)
            entry["mdr_rate"] = (
                float(((r[both] == 1) & (i[both] == 1)).mean()) if both.any() else None
            )
            entry["mdr_n"] = int(both.sum())
        rows.append(entry)
    return rows


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.datetime64):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return str(obj)


def _fmt(triple) -> str:
    if not triple or triple[0] is None or (isinstance(triple[0], float) and np.isnan(triple[0])):
        return "n/a"
    return f"{triple[0]:.3f} [{triple[1]:.3f}–{triple[2]:.3f}]"


def _markdown_report(p: dict) -> str:
    lines = [
        "# GermoVision training report",
        "",
        f"Generated {p['generated_at']} · source `{p['source']}` · "
        f"run time {p['elapsed_sec']}s",
        "",
    ]
    if p["synthetic"]:
        lines += [
            "> **Note.** " + p["warning"],
            "",
        ]

    d = p["dataset"]
    lines += [
        "## Data",
        "",
        f"- Isolates: **{d['n_isolates']}**, relatedness clusters: {d['n_clusters']}",
        f"- Countries: {d['n_countries']}, lineages: {d['n_lineages']}, "
        f"variants: {d['n_variants']}",
        f"- Period: {d['date_min']} — {d['date_max']}",
        f"- Split: {p['split']['strategy']}, {p['split']['sizes']}",
        "",
        "## Internal test",
        "",
        "Closed — share of isolates given a correct answer in 1-2 days instead of",
        "~60. Missed — share of resistant isolates confidently called susceptible:",
        "the only genuinely dangerous outcome. Both are computed over all isolates,",
        "not only the answered ones.",
        "",
        "The Lab column marks drugs where the missed-resistance limit is not met:",
        "genomic prediction alone is insufficient and laboratory confirmation is",
        "required.",
        "",
        "| Drug | N | Resistant | Closed | Missed | Lab | Sensitivity | "
        "Specificity | PR-AUC | Catalogue | Answered |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["per_drug"]:
        lines.append(
            f"| {r['drug_name']} | {r['n_test']} | {r['n_positive']} | "
            f"**{r['correctly_closed']:.1%}** | {r['missed_resistance']:.1%} | "
            f"{'required' if r['requires_confirmation'] else '—'} | "
            f"{_fmt(r['decision']['sensitivity'])} | {_fmt(r['decision']['specificity'])} | "
            f"{_fmt(r['ranking']['pr_auc'])} | {_fmt(r['baseline_catalogue']['sensitivity'])} | "
            f"{r['answer_rate']:.1%} |"
        )

    if p["external_validation"]:
        country = p["external_validation"][0]["country"]
        lines += [
            "",
            f"## External validation (trained without {country})",
            "",
            "The only estimate that reflects a real deployment.",
            "",
            "| Drug | N | Sensitivity | Specificity | H1 target | Verdict |",
            "|---|---|---|---|---|---|",
        ]
        for r in p["external_validation"]:
            target = (
                f"≥{r['h1_target'][0]:.2f} / ≥{r['h1_target'][1]:.2f}" if r["h1_target"] else "—"
            )
            verdict = {True: "met", False: "not met", None: "—"}[r["h1_met"]]
            lines.append(
                f"| {r['drug_name']} | {r['n_test']} | {_fmt(r['sensitivity'])} | "
                f"{_fmt(r['specificity'])} | {target} | {verdict} |"
            )

    lines += [
        "",
        "## Ablations (rifampicin)",
        "",
        "Sensitivity and specificity are computed on the decisions actually issued;",
        "PR-AUC on the calibrated probability.",
        "",
        "| Configuration | Sens | Spec | PR-AUC | Abstained |",
        "|---|---|---|---|---|",
    ]
    for row in p["ablations"]:
        if "skipped" in row:
            continue
        lines.append(
            f"| {row['config']} | {row['sensitivity']:.3f} | "
            f"{row['specificity']:.3f} | {row['pr_auc']:.3f} | "
            f"{row['abstention_rate']:.1%} |"
        )

    rif = next((r for r in p["per_drug"] if r["drug"] == "RIF"), None)
    if rif and rif.get("coverage_tradeoff"):
        lines += [
            "",
            "## Answer rate versus accuracy (rifampicin)",
            "",
            "Abstaining sends the sample for phenotypic testing: the system trades",
            "speed for certainty. Which point on this curve to pick is the",
            "organisation's decision, not the developer's.",
            "",
            "| alpha | Answered | Accuracy | Sens | Spec |",
            "|---|---|---|---|---|",
        ]
        for row in rif["coverage_tradeoff"]:
            if row.get("answer_rate", 0) == 0:
                continue
            lines.append(
                f"| {row['alpha']:.2f} | {row['answer_rate']:.1%} | "
                f"{row['accuracy']:.3f} | {row['sensitivity']:.3f} | "
                f"{row['specificity']:.3f} |"
            )

    if p.get("growth"):
        lines += [
            "",
            "## GV-Growth: recovering growth coefficients",
            "",
            f"Estimated shrinkage tau = {p['growth']['tau']:.4f}",
            "",
            "| Lineage | True beta | Estimated beta | Spread across regions |",
            "|---|---|---|---|",
        ]
        for row in p["growth"]["recovery"]:
            lines.append(
                f"| {row['lineage']} | {row['true_beta']:+.4f} | "
                f"{row['mean_estimated_beta']:+.4f} | {row['sd_across_regions']:.4f} |"
            )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
