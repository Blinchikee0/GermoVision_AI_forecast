"""Заключение по одному или нескольким изолятам обученными моделями.

Это точка, ради которой всё остальное существует: лаборатория получила
результат секвенирования и хочет знать, что назначать, не дожидаясь
шестидесяти дней культурального теста.

    python -m germovision.train --save-models models/
    python -m germovision.predict --models models/ --mutations sample.csv

Формат входного файла — тот же, что у `mutations.csv` в загрузчике
CRyPTIC, поэтому выгрузка из биоинформатического пайплайна годится без
переделки:

    id,gene,mutation
    TB-2026-0417,rpoB,S450L
    TB-2026-0417,katG,S315T

Необязательный `--samples` добавляет линию и страну; без него они
считаются неизвестными, и признаки, зависящие от контекста, не
используются — по умолчанию их и нет.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data.catalogue import DRUG_NAMES
from .data.schema import IsolateDataset
from .models.resist import Decision
from .persistence import load_bundle

__all__ = ["load_isolates", "predict_isolates", "format_report"]


def load_isolates(
    mutations_path: str | Path, samples_path: str | Path | None = None
) -> IsolateDataset:
    """Собрать набор изолятов из CSV с вариантами.

    Фенотипы отсутствуют — они и не нужны: это режим применения, а не
    оценки. Пустой словарь фенотипов допустим, поскольку `predict` его
    не читает.

    Raises:
        FileNotFoundError: файла нет.
        ValueError: нет обязательных столбцов или ни одного изолята.
    """
    mpath = Path(mutations_path)
    if not mpath.exists():
        raise FileNotFoundError(f"file not found: {mpath}")

    by_id: dict[str, set[str]] = defaultdict(set)
    order: list[str] = []
    with mpath.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        missing = {"id", "gene", "mutation"} - cols.keys()
        if missing:
            raise ValueError(
                f"{mpath.name} is missing columns {sorted(missing)}; "
                f"found {sorted(cols)}"
            )
        for row in reader:
            sid = str(row[cols["id"]]).strip()
            gene = str(row[cols["gene"]]).strip()
            mut = str(row[cols["mutation"]]).strip()
            if not sid:
                continue
            if sid not in by_id:
                order.append(sid)
            if gene and mut:
                by_id[sid].add(f"{gene}_{mut}")

    if not order:
        raise ValueError(f"no isolates found in {mpath.name}")

    meta: dict[str, dict[str, str]] = {}
    if samples_path:
        spath = Path(samples_path)
        if not spath.exists():
            raise FileNotFoundError(f"file not found: {spath}")
        with spath.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
            if "id" not in cols:
                raise ValueError(f"{spath.name} has no id column")
            for row in reader:
                meta[str(row[cols["id"]]).strip()] = {
                    "lineage": str(row.get(cols.get("lineage", ""), "")).strip() or "unknown",
                    "country": str(row.get(cols.get("country", ""), "")).strip() or "unknown",
                }

    today = np.datetime64("today", "D")
    n = len(order)
    return IsolateDataset(
        isolate_ids=np.array(order),
        mutations=[by_id[i] for i in order],
        phenotypes={},
        lineages=np.array([meta.get(i, {}).get("lineage", "unknown") for i in order]),
        countries=np.array([meta.get(i, {}).get("country", "unknown") for i in order]),
        collection_dates=np.array([today] * n),
        submission_dates=np.array([today] * n),
        clusters=np.arange(n),
        meta={"source": str(mpath), "synthetic": False, "mode": "prediction"},
    )


def predict_isolates(bundle, ds: IsolateDataset) -> list[dict]:
    """Получить заключения по всем изолятам и всем препаратам."""
    rows: list[dict] = []
    per_drug = {}
    for drug, model in bundle.models.items():
        per_drug[drug] = model.predict(ds, explain=True)

    for i, isolate_id in enumerate(ds.isolate_ids):
        drugs = []
        for drug in sorted(per_drug, key=lambda d: list(DRUG_NAMES).index(d)):
            pr = per_drug[drug][i]
            drugs.append({
                "drug": drug,
                "drug_name": DRUG_NAMES.get(drug, drug),
                "decision": pr.decision,
                "probability": round(float(pr.probability), 3),
                "source": pr.source,
                "explanation": pr.explain(),
                "needs_confirmation": bool(pr.needs_confirmation),
                "ood": bool(pr.ood),
            })
        rows.append({
            "isolate_id": str(isolate_id),
            "lineage": str(ds.lineages[i]),
            "mutations": sorted(ds.mutations[i]),
            "drugs": drugs,
        })
    return rows


def format_report(reports: list[dict], bundle) -> str:
    """Оформить заключения в читаемый врачом вид."""
    out: list[str] = [
        bundle.describe(),
        "",
        '"standard" in the probability column means the call came from the WHO'
        "\nmutation catalogue rather than the model: it is the reference standard "
        "and takes precedence.",
        "",
    ]

    for rep in reports:
        out.append("=" * 78)
        out.append(f"Isolate {rep['isolate_id']}   lineage: {rep['lineage']}")
        muts = ", ".join(m.replace("_", " ") for m in rep["mutations"])
        out.append(f"Variants found: {len(rep['mutations'])}" + (f" — {muts}" if muts else ""))
        out.append("=" * 78)
        out.append(f"{'Drug':<16}{'Call':<16}{'Prob.':>9}  Basis")
        out.append("-" * 78)

        resistant = []
        for d in rep["drugs"]:
            label = {
                Decision.RESISTANT: "RESISTANT",
                Decision.SUSCEPTIBLE: "susceptible",
                Decision.NO_CALL: "no call",
            }[d["decision"]]
            if d["decision"] == Decision.RESISTANT:
                resistant.append(d["drug_name"])
            # Когда решение принято по каталогу ВОЗ, вероятность модели
            # к нему отношения не имеет и в клиническом заключении не
            # показывается: «УСТОЙЧИВ при вероятности 0,04» выглядит как
            # противоречие, хотя противоречия нет — это две независимые
            # оценки. Вероятность остаётся в JSON для анализа.
            prob = "  standard" if d["source"] == "catalogue" else f"{d['probability']:>9.2f}"
            out.append(f"{d['drug_name']:<16}{label:<16}{prob:>9}  {d['explanation']}")

        out.append("-" * 78)
        if resistant:
            out.append("Predicted resistance: " + ", ".join(resistant))
        else:
            out.append("No resistance predicted.")
        needs = [d["drug_name"] for d in rep["drugs"] if d["needs_confirmation"]]
        if len(needs) == len(rep["drugs"]):
            out.append(
                "Lab confirmation is required for every drug: on the held-out set "
                "the\nshare of missed resistance is above the clinical limit."
            )
        elif needs:
            out.append("Lab confirmation required for: " + ", ".join(needs))
        out.append(
            "\nThis report is advisory and does not replace a clinician's decision."
            "\nThe system is not a certified medical device."
        )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drug-resistance report from an isolate genome"
    )
    parser.add_argument("--models", default="models", help="directory with trained models")
    parser.add_argument("--mutations", required=True, help="CSV: id, gene, mutation")
    parser.add_argument("--samples", default=None, help="CSV with lineage and country")
    parser.add_argument("--json", default=None, help="write reports to JSON")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bundle = load_bundle(args.models)
    ds = load_isolates(args.mutations, args.samples)
    reports = predict_isolates(bundle, ds)

    print(format_report(reports, bundle))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {"manifest": bundle.manifest, "reports": reports},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"JSON written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
