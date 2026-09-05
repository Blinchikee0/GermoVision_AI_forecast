"""Маршрутизация разобранных данных к нужной модели.

Один слой между «что за файл принесли» и «что с ним делать». Каждый вид
данных ведёт к своей модели, и результат приводится к общему виду —
набору таблиц, которые можно скачать и подшить к анализу.

Разделение обязанностей: `formats` знает, как читать файлы, `models`
знают, как считать, а этот модуль — что чему соответствует. Добавление
нового формата не требует правок в моделях, и наоборот.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data.catalogue import DRUG_NAMES
from .data.schema import IsolateDataset
from .formats import InputKind, ParsedInput
from .models.escape import GVEscape
from .models.growth import GVGrowth
from .models.resist import Decision

__all__ = ["AnalysisResult", "Table", "AnalysisError", "analyze"]


class AnalysisError(RuntimeError):
    """Данные разобраны, но проанализировать их нельзя."""


@dataclass
class Table:
    """Таблица результата, пригодная для выгрузки в CSV."""

    name: str
    title: str
    columns: list[str]
    rows: list[list[Any]]
    note: str = ""

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buf.getvalue()


@dataclass
class AnalysisResult:
    """Итог анализа одного файла."""

    kind: str
    model: str
    title: str
    summary: str
    tables: list[Table] = field(default_factory=list)
    highlights: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "model": self.model,
            "title": self.title,
            "summary": self.summary,
            "highlights": self.highlights,
            "notes": self.notes,
            "tables": [
                {
                    "name": t.name,
                    "title": t.title,
                    "columns": t.columns,
                    "rows": t.rows,
                    "note": t.note,
                }
                for t in self.tables
            ],
            "payload": self.payload,
        }


# --------------------------------------------------------------------------
# Устойчивость: мутации, VCF, TB-Profiler
# --------------------------------------------------------------------------

_RESISTANCE_KINDS = {InputKind.MUTATIONS, InputKind.VCF, InputKind.TBPROFILER}

_DECISION_LABEL = {
    Decision.RESISTANT: "resistant",
    Decision.SUSCEPTIBLE: "susceptible",
    Decision.NO_CALL: "no call",
}


def _dataset_from_rows(rows: list[dict]) -> IsolateDataset:
    today = np.datetime64("today", "D")
    n = len(rows)
    return IsolateDataset(
        isolate_ids=np.array([r["id"] for r in rows]),
        mutations=[set(r["mutations"]) for r in rows],
        phenotypes={},
        lineages=np.array(["unknown"] * n),
        countries=np.array(["unknown"] * n),
        collection_dates=np.array([today] * n),
        submission_dates=np.array([today] * n),
        clusters=np.arange(n),
        meta={"source": "upload", "synthetic": False, "mode": "prediction"},
    )


def _analyze_resistance(parsed: ParsedInput, bundle) -> AnalysisResult:
    if bundle is None or not getattr(bundle, "models", None):
        raise AnalysisError(
            "Resistance models are not loaded. Train and save them with:\n"
            "    python -m germovision.train --save-models models"
        )

    ds = _dataset_from_rows(parsed.payload)
    order = list(DRUG_NAMES)
    drugs = sorted(bundle.models, key=lambda d: order.index(d) if d in order else 99)

    rows: list[list[Any]] = []
    per_isolate: dict[str, dict[str, str]] = {}
    n_resistant = 0

    for drug in drugs:
        preds = bundle.models[drug].predict(ds, explain=True)
        for i, pr in enumerate(preds):
            sid = str(ds.isolate_ids[i])
            per_isolate.setdefault(sid, {})[drug] = pr.decision
            if pr.decision == Decision.RESISTANT:
                n_resistant += 1
            rows.append([
                sid,
                drug,
                DRUG_NAMES.get(drug, drug),
                _DECISION_LABEL[pr.decision],
                "" if pr.source == "catalogue" else round(float(pr.probability), 3),
                "WHO catalogue" if pr.source == "catalogue" else "model",
                "yes" if pr.needs_confirmation else "no",
                pr.explain(),
            ])

    detail = Table(
        name="resistance",
        title="Drug resistance prediction",
        columns=[
            "Isolate", "Code", "Drug", "Call", "Probability",
            "Source", "Lab confirmation", "Basis",
        ],
        rows=rows,
        note=(
            "Probability is omitted where the WHO mutation catalogue made the call: "
            "it is the reference standard and takes precedence over the model."
        ),
    )

    summary_rows = []
    for sid, decisions in per_isolate.items():
        res = [DRUG_NAMES.get(d, d) for d, v in decisions.items() if v == Decision.RESISTANT]
        nc = [DRUG_NAMES.get(d, d) for d, v in decisions.items() if v == Decision.NO_CALL]
        resistant_set = {d for d, v in decisions.items() if v == Decision.RESISTANT}
        mdr = "yes" if {"RIF", "INH"} <= resistant_set else "no"
        summary_rows.append([
            sid, len(res), mdr, "; ".join(res) or "—", "; ".join(nc) or "—",
        ])

    summary_table = Table(
        name="isolates",
        title="Isolate summary",
        columns=["Isolate", "Resistant to (count)", "MDR", "Resistant to", "No call"],
        rows=summary_rows,
        note=(
            "MDR — multidrug resistance: resistant to both rifampicin and isoniazid."
        ),
    )

    n_mdr = sum(1 for r in summary_rows if r[2] == "yes")
    highlights = [
        {"label": "Isolates", "value": str(len(per_isolate))},
        {"label": "Drugs checked", "value": str(len(drugs))},
        {"label": "Resistance calls", "value": str(n_resistant)},
        {"label": "MDR isolates", "value": str(n_mdr), "tone": "crit" if n_mdr else "good"},
    ]

    notes = list(parsed.notes)
    if bundle.manifest.get("synthetic"):
        notes.append(
            "Models were trained on synthetic data: this shows the pipeline works, "
            "not clinical quality."
        )
    needs = [
        DRUG_NAMES.get(d, d)
        for d, q in (bundle.manifest.get("quality") or {}).items()
        if q.get("requires_confirmation")
    ]
    if needs:
        notes.append(
            "Lab confirmation required for: " + ", ".join(sorted(needs))
        )

    return AnalysisResult(
        kind=parsed.kind,
        model="GV-Resist",
        title="Drug resistance",
        summary=f"{parsed.summary}. {len(drugs)} drugs checked.",
        tables=[summary_table, detail],
        highlights=highlights,
        notes=notes,
        payload={"drugs": drugs, "n_isolates": len(per_isolate)},
    )


# --------------------------------------------------------------------------
# Мутации: белковые и генные последовательности
# --------------------------------------------------------------------------


def _analyze_escape(parsed: ParsedInput, top_candidates: int = 100) -> AnalysisResult:
    model = GVEscape()
    try:
        model.fit(parsed.payload, nucleotide=(parsed.kind == InputKind.GENE_FASTA))
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc

    report = model.analyze(top_candidates=top_candidates)

    observed = Table(
        name="observed_mutations",
        title="Observed substitutions, by descending risk",
        columns=[
            "Mutation", "Position", "Wild type", "Mutant", "Risk",
            "Tolerance", "Salience", "Novelty", "Conservation",
            "Count", "Frequency", "Growth per week",
        ],
        rows=[
            [
                r.label, r.position, r.wildtype, r.mutant, round(r.risk, 4),
                round(r.tolerance, 4), round(r.salience, 4), round(r.novelty, 4),
                round(r.conservation, 4), r.count, round(r.frequency, 6),
                "" if r.trend is None else round(r.trend, 5),
            ]
            for r in report.observed
        ],
        note=(
            "Risk is the geometric mean of three factors: how far the profile "
            "tolerates the substitution, its physicochemical salience, and its "
            "novelty. Growth per week appears where the headers carried dates."
        ),
    )

    candidates = Table(
        name="candidate_mutations",
        title="Candidates: substitutions not yet seen in the data",
        columns=[
            "Mutation", "Position", "Wild type", "Mutant", "Risk",
            "Tolerance", "Salience", "Conservation",
        ],
        rows=[
            [
                r.label, r.position, r.wildtype, r.mutant, round(r.risk, 4),
                round(r.tolerance, 4), round(r.salience, 4), round(r.conservation, 4),
            ]
            for r in report.candidates
        ],
        note=(
            "This list is the early-warning mechanism: a substitution can be scored "
            "before it is ever seen. The estimate comes from a profile model and does "
            "not account for epistasis — how other substitutions change the effect."
        ),
    )

    hotspots = Table(
        name="hotspots",
        title="Hotspots: positions with the most substitution diversity",
        columns=[
            "Position", "Wild type", "Distinct substitutions", "Total count",
            "Max risk", "Conservation", "Substitutions",
        ],
        rows=[
            [
                h["position"], h["wildtype"], h["n_variants"], h["total_count"],
                h["max_risk"], h["conservation"], ", ".join(h["mutations"]),
            ]
            for h in report.hotspots
        ],
        note=(
            "Repeated independent substitutions at one position are a classic "
            "signature of positive selection: evolution keeps revisiting that site."
        ),
    )

    top = report.observed[0] if report.observed else None
    rising = [r for r in report.observed if r.trend and r.trend > 0.05]
    highlights = [
        {"label": "Sequences used", "value": str(report.n_used)},
        {"label": "Observed substitutions", "value": str(len(report.observed))},
        {"label": "Hotspots", "value": str(len(report.hotspots))},
        {
            "label": "Highest risk",
            "value": f"{top.label} · {top.risk:.2f}" if top else "—",
            "tone": "warn" if top and top.risk > 0.5 else "",
        },
    ]
    if rising:
        highlights.append({
            "label": "Rising substitutions",
            "value": str(len(rising)),
            "tone": "crit",
        })

    return AnalysisResult(
        kind=parsed.kind,
        model="GV-Escape",
        title="Evolutionary risk of substitutions",
        summary=report.summary().replace("\n", " · "),
        tables=[observed, candidates, hotspots],
        highlights=highlights,
        notes=list(parsed.notes) + report.notes,
        payload={
            "reference_id": report.reference_id,
            "reference_length": report.reference_length,
            "date_range": report.date_range,
        },
    )


# --------------------------------------------------------------------------
# Динамика линий
# --------------------------------------------------------------------------


def _analyze_growth(parsed: ParsedInput) -> AnalysisResult:
    records = parsed.payload
    regions = sorted({r["region"] for r in records})
    lineages = sorted({r["lineage"] for r in records})
    weeks = sorted({r["week"] for r in records})

    if len(lineages) < 2:
        raise AnalysisError(
            f"the data contain only one lineage ({lineages[0]}). The model describes "
            "how lineages displace one another, so at least two are required"
        )
    if len(weeks) < 4:
        raise AnalysisError(
            f"the data contain {len(weeks)} time points. Estimating a growth rate "
            "needs at least four"
        )

    index = {(r["region"], r["week"]): i for i, r in enumerate(
        [{"region": rg, "week": w} for rg in regions for w in weeks]
    )}
    counts = np.zeros((len(index), len(lineages)), dtype=float)
    for r in records:
        i = index[(r["region"], r["week"])]
        counts[i, lineages.index(r["lineage"])] += r["count"]

    # Референсной берётся самая многочисленная линия: относительно неё
    # знаки коэффициентов совпадают с интуитивным «растёт / убывает».
    ref_idx = GVGrowth.choose_reference(counts, lineages)
    order_idx = [ref_idx] + [j for j in range(len(lineages)) if j != ref_idx]
    counts = counts[:, order_idx]
    lineages = [lineages[j] for j in order_idx]

    keys = list(index)
    keep = counts.sum(axis=1) > 0
    counts = counts[keep]
    kept = [k for k, ok in zip(keys, keep, strict=True) if ok]
    times = np.array([k[1] for k in kept], dtype=float)
    region_labels = np.array([k[0] for k in kept])

    try:
        model = GVGrowth(n_bootstrap=120).fit(counts, times, region_labels, lineages)
    except ValueError as exc:
        raise AnalysisError(f"could not fit the growth model: {exc}") from exc

    growth_rows = [
        [
            g["region"], g["lineage"], round(g["beta"], 5),
            "" if not np.isfinite(g["se"]) else round(g["se"], 5),
            round(g["weekly_pct"], 2), g["n_samples"],
            "yes" if g["significant"] else "no",
        ]
        for g in sorted(model.growth_table(), key=lambda g: -g["beta"])
    ]
    growth = Table(
        name="growth",
        title="Lineage growth advantage by region",
        columns=[
            "Region", "Lineage", "beta", "SE", "Growth per week, %", "Samples", "Significant",
        ],
        rows=growth_rows,
        note=(
            f"Beta is the log growth advantage per week relative to the reference "
            f"lineage \"{lineages[0]}\" (the most abundant one), whose beta is 0 by "
            "construction. \"Significant\" means the confidence interval excludes zero."
        ),
    )

    horizons = [0, 2, 4, 8]
    fc_rows: list[list[Any]] = []
    last = float(times.max())
    for region in model.fits_:
        point, lo, hi = model.forecast(region, horizons, last_time=last)
        for k, h in enumerate(horizons):
            for j, lin in enumerate(lineages):
                fc_rows.append([
                    region, lin, h,
                    round(float(point[k, j]), 4),
                    round(float(lo[k, j]), 4),
                    round(float(hi[k, j]), 4),
                ])
    forecast = Table(
        name="forecast",
        title="Lineage share forecast",
        columns=["Region", "Lineage", "Weeks ahead", "Share", "Lower bound", "Upper bound"],
        rows=fc_rows,
        note="95% interval. It widens where fewer samples were sequenced in the region.",
    )

    rising = [g for g in model.growth_table() if g["significant"] and g["beta"] > 0]
    top = max(rising, key=lambda g: g["beta"]) if rising else None
    highlights = [
        {"label": "Regions", "value": str(len(model.fits_))},
        {"label": "Lineages", "value": str(len(lineages))},
        {"label": "Samples", "value": str(int(counts.sum()))},
        {
            "label": "Significant growth",
            "value": (
                f"{top['lineage']} · +{top['weekly_pct']:.1f}%/wk" if top else "none detected"
            ),
            "tone": "warn" if top else "good",
        },
    ]

    return AnalysisResult(
        kind=parsed.kind,
        model="GV-Growth",
        title="Lineage dynamics",
        summary=f"{parsed.summary}. Shrinkage tau = {model.tau_:.4f}.",
        tables=[growth, forecast],
        highlights=highlights,
        notes=list(parsed.notes),
        payload={"regions": list(model.fits_), "lineages": lineages, "tau": model.tau_},
    )


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


def analyze(parsed: ParsedInput, bundle=None, **kwargs) -> AnalysisResult:
    """Проанализировать разобранные данные подходящей моделью.

    Args:
        parsed: результат `formats.detect_and_parse`.
        bundle: загруженные модели устойчивости. Нужны только для входов,
            ведущих к GV-Resist; GV-Escape и GV-Growth обучаются на самих
            присланных данных и предобученных весов не требуют.

    Raises:
        AnalysisError: данные разобраны, но для анализа непригодны.
    """
    if parsed.kind in _RESISTANCE_KINDS:
        return _analyze_resistance(parsed, bundle)
    if parsed.kind in (InputKind.PROTEIN_FASTA, InputKind.GENE_FASTA):
        return _analyze_escape(parsed, **kwargs)
    if parsed.kind == InputKind.LINEAGE_COUNTS:
        return _analyze_growth(parsed)
    if parsed.kind == InputKind.GENOME_FASTA:
        raise AnalysisError(
            "This is a whole genome. Calling variants from it requires alignment to "
            "a reference and quality control — a bioinformatics pipeline, not a web "
            "form. A result produced in haste would look finished without being "
            "reliable.\n\n"
            "Process the genome with a standard tool and upload its output:\n"
            "  - TB-Profiler produces JSON this system reads directly;\n"
            "  - snpEff on top of bcftools produces an annotated VCF;\n"
            "  - any pipeline: a table of id, gene, mutation will do.\n\n"
            "Individual genes (under 15 000 bp) are accepted and scored for mutation "
            "risk without a pipeline."
        )
    raise AnalysisError(f"unknown input kind: {parsed.kind}")
