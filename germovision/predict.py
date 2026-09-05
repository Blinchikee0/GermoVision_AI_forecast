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

from .data.catalogue import DRUG_NAMES_RU
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
        raise FileNotFoundError(f"не найден файл {mpath}")

    by_id: dict[str, set[str]] = defaultdict(set)
    order: list[str] = []
    with mpath.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        missing = {"id", "gene", "mutation"} - cols.keys()
        if missing:
            raise ValueError(
                f"в {mpath.name} нет столбцов {sorted(missing)}; "
                f"найдены {sorted(cols)}"
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
        raise ValueError(f"в {mpath.name} не найдено ни одного изолята")

    meta: dict[str, dict[str, str]] = {}
    if samples_path:
        spath = Path(samples_path)
        if not spath.exists():
            raise FileNotFoundError(f"не найден файл {spath}")
        with spath.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
            if "id" not in cols:
                raise ValueError(f"в {spath.name} нет столбца id")
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
        for drug in sorted(per_drug, key=lambda d: list(DRUG_NAMES_RU).index(d)):
            pr = per_drug[drug][i]
            drugs.append({
                "drug": drug,
                "drug_name": DRUG_NAMES_RU.get(drug, drug),
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
        "«стандарт» в столбце вероятности означает, что решение принято по "
        "каталогу\nмутаций ВОЗ, а не моделью: это референсный стандарт, и он "
        "имеет приоритет.",
        "",
    ]

    for rep in reports:
        out.append("=" * 78)
        out.append(f"Изолят {rep['isolate_id']}   линия: {rep['lineage']}")
        muts = ", ".join(m.replace("_", " ") for m in rep["mutations"])
        out.append(f"Найдено вариантов: {len(rep['mutations'])}" + (f" — {muts}" if muts else ""))
        out.append("=" * 78)
        out.append(f"{'Препарат':<16}{'Заключение':<16}{'Вероятн.':>9}  Обоснование")
        out.append("-" * 78)

        resistant = []
        for d in rep["drugs"]:
            label = {
                Decision.RESISTANT: "УСТОЙЧИВ",
                Decision.SUSCEPTIBLE: "чувствителен",
                Decision.NO_CALL: "нет заключения",
            }[d["decision"]]
            if d["decision"] == Decision.RESISTANT:
                resistant.append(d["drug_name"])
            # Когда решение принято по каталогу ВОЗ, вероятность модели
            # к нему отношения не имеет и в клиническом заключении не
            # показывается: «УСТОЙЧИВ при вероятности 0,04» выглядит как
            # противоречие, хотя противоречия нет — это две независимые
            # оценки. Вероятность остаётся в JSON для анализа.
            prob = "  стандарт" if d["source"] == "catalogue" else f"{d['probability']:>9.2f}"
            out.append(f"{d['drug_name']:<16}{label:<16}{prob:>9}  {d['explanation']}")

        out.append("-" * 78)
        if resistant:
            out.append("Прогноз устойчивости: " + ", ".join(resistant))
        else:
            out.append("Прогноз устойчивости не выявлен.")
        needs = [d["drug_name"] for d in rep["drugs"] if d["needs_confirmation"]]
        if len(needs) == len(rep["drugs"]):
            out.append(
                "Фенотипическое подтверждение требуется по всем препаратам: на "
                "удержанной\nвыборке доля пропущенной устойчивости выше "
                "клинического лимита."
            )
        elif needs:
            out.append("Требуют фенотипического подтверждения: " + ", ".join(needs))
        out.append(
            "\nЗаключение носит вспомогательный характер и не заменяет решение "
            "врача.\nСистема не является сертифицированным медицинским изделием."
        )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Заключение о лекарственной устойчивости по геному изолята"
    )
    parser.add_argument("--models", default="models", help="каталог с обученными моделями")
    parser.add_argument("--mutations", required=True, help="CSV: id, gene, mutation")
    parser.add_argument("--samples", default=None, help="CSV с линией и страной")
    parser.add_argument("--json", default=None, help="сохранить заключения в JSON")
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
        print(f"JSON сохранён: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
