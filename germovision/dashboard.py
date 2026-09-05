"""Сборка панели геномного надзора из результатов обучения.

Панель — не отдельный макет с придуманными цифрами, а представление
файла `reports/metrics.json`. Любой прогон обучения порождает свою
панель, и показанное на экране всегда соответствует тому, что модель
реально выдала на удержанной выборке.

Запуск:
    python -m germovision.dashboard
    python -m germovision.dashboard --metrics reports/metrics.json --out dashboard/index.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["build_payload", "render"]

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "template.html"
PLACEHOLDER = "__GV_DATA__"

#: Русские названия линий и препаратов для подписей на графиках.
LINEAGE_LABELS = {
    "L4_Euro_American": "L4 Euro-American",
    "L2_Beijing": "L2 Beijing",
    "L2_Beijing_MDR": "L2 Beijing (МЛУ)",
    "L3_CAS": "L3 CAS",
    "L1_Indo_Oceanic": "L1 Indo-Oceanic",
    "L5_West_African": "L5 West African",
}

COUNTRY_LABELS = {
    "KZ": "Казахстан",
    "RU": "Россия",
    "UZ": "Узбекистан",
    "IN": "Индия",
    "ZA": "ЮАР",
    "CN": "Китай",
    "BR": "Бразилия",
    "GB": "Великобритания",
    "DE": "Германия",
    "PE": "Перу",
}


def _round_list(values, digits: int = 4):
    return [None if v is None else round(float(v), digits) for v in values]


def build_payload(metrics: dict) -> dict:
    """Преобразовать результаты обучения в данные для панели."""
    ds = metrics["dataset"]
    growth = metrics.get("growth") or {}

    external = [r for r in metrics.get("external_validation", []) if "skipped" not in r]
    internal = [r for r in metrics.get("per_drug", []) if "skipped" not in r]

    kz = next((c for c in ds.get("by_country", []) if c["country"] == "KZ"), None)
    rif_int = next((r for r in internal if r["drug"] == "RIF"), None)

    # Средняя доля закрытых без фенотипического теста по основным препаратам.
    core = [r for r in internal if r["drug"] in ("RIF", "INH", "EMB", "LEV", "MXF")]
    answer_rate = sum(r["answer_rate"] for r in core) / len(core) if core else None
    closed = (
        sum(r["correctly_closed"] for r in internal) / len(internal) if internal else None
    )
    n_needs = sum(1 for r in internal if r["requires_confirmation"])

    # Наибольшее преимущество роста среди регионов — главный сигнал панели.
    sig_growth = [
        g for g in growth.get("growth_table", []) if g.get("significant") and g["beta"] > 0
    ]
    top_growth = max(sig_growth, key=lambda g: g["beta"]) if sig_growth else None

    lineages = growth.get("lineages", [])
    observed = []
    for row in growth.get("observed_national", []):
        total = row["total"] or 1.0
        observed.append({
            "week": row["week"],
            "fracs": _round_list([c / total for c in row["counts"]]),
            "total": int(row["total"]),
        })

    forecasts = {}
    for region, f in (growth.get("forecasts") or {}).items():
        forecasts[region] = {
            "horizons": f["horizons"],
            "point": [_round_list(r) for r in f["point"]],
            "lo": [_round_list(r) for r in f["lo"]],
            "hi": [_round_list(r) for r in f["hi"]],
            "n_samples": f["n_samples"],
        }

    return {
        "meta": {
            "generated_at": metrics.get("generated_at"),
            "elapsed_sec": metrics.get("elapsed_sec"),
            "source": metrics.get("source"),
            "synthetic": bool(metrics.get("synthetic")),
            "warning": metrics.get("warning", ""),
            "catalogue_size": metrics.get("catalogue_size"),
            "n_isolates": ds["n_isolates"],
            "n_clusters": ds["n_clusters"],
            "n_countries": ds["n_countries"],
            "n_variants": ds["n_variants"],
            "date_min": ds["date_min"],
            "date_max": ds["date_max"],
            "split": metrics["split"]["sizes"],
            "split_meta": metrics["split"]["meta"],
        },
        "kpi": {
            "mdr_kz": kz["mdr_rate"] if kz else None,
            "mdr_kz_n": kz["mdr_n"] if kz else None,
            "rif_sens_external": external[0]["sensitivity"][0] if external else None,
            "answer_rate": answer_rate,
            "closed": closed,
            "n_needs_confirmation": n_needs,
            "n_drugs": len(internal),
            "top_growth": top_growth,
            "ece_rif": rif_int["calibration"]["ece"] if rif_int else None,
        },
        "external": [
            {
                "drug": r["drug"],
                "name": r["drug_name"],
                "n": r["n_test"],
                "pos": r["n_positive"],
                "sens": _round_list(r["sensitivity"], 3),
                "spec": _round_list(r["specificity"], 3),
                "base_sens": round(r["baseline_sensitivity"], 3),
                "h1": r.get("h1_met"),
            }
            for r in external
        ],
        "internal": [
            {
                "drug": r["drug"],
                "name": r["drug_name"],
                "n": r["n_test"],
                "pos": r["n_positive"],
                "sens": _round_list(r["decision"]["sensitivity"], 3),
                "spec": _round_list(r["decision"]["specificity"], 3),
                "pr_auc": round(r["ranking"]["pr_auc"][0], 3),
                "base_sens": round(r["baseline_catalogue"]["sensitivity"][0], 3),
                "answer_rate": round(r["answer_rate"], 3),
                "closed": round(r["correctly_closed"], 4),
                "missed": round(r["missed_resistance"], 4),
                "needs_confirmation": bool(r["requires_confirmation"]),
                "ece": round(r["calibration"]["ece"], 3),
                "by_catalogue": r["routing"]["by_catalogue"],
                "by_model": r["routing"]["by_model"],
                "no_call": r["routing"]["no_call"],
            }
            for r in internal
        ],
        "tradeoff": rif_int["coverage_tradeoff"] if rif_int else [],
        "ablations": [r for r in metrics.get("ablations", []) if "skipped" not in r],
        "countries": [
            {**c, "label": COUNTRY_LABELS.get(c["country"], c["country"])}
            for c in sorted(
                ds.get("by_country", []),
                key=lambda c: -(c.get("mdr_rate") or 0),
            )
        ],
        "lineages": [{"key": k, "label": LINEAGE_LABELS.get(k, k)} for k in lineages],
        "observed": observed,
        "forecasts": forecasts,
        "growth_table": growth.get("growth_table", []),
        "recovery": growth.get("recovery", []),
        "tau": growth.get("tau"),
        "examples": metrics.get("example_reports", []),
    }


def render(metrics_path: Path, template_path: Path, out_path: Path) -> Path:
    """Собрать HTML-панель, подставив данные в шаблон."""
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"нет файла {metrics_path}. Сначала выполните: python -m germovision.train"
        )
    if not template_path.exists():
        raise FileNotFoundError(f"нет шаблона {template_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = build_payload(metrics)

    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"в шаблоне нет метки {PLACEHOLDER}")

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # </script> внутри данных разорвал бы тег — экранируем на всякий случай.
    data = data.replace("</", "<\\/")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template.replace(PLACEHOLDER, data), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка панели геномного надзора")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--template", default=str(TEMPLATE_PATH))
    parser.add_argument("--out", default="dashboard/index.html")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = render(Path(args.metrics), Path(args.template), Path(args.out))
    size_kb = path.stat().st_size / 1024
    print(f"Панель собрана: {path} ({size_kb:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
