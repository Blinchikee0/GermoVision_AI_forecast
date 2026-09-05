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

from .data.catalogue import DRUG_NAMES_RU
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

_DECISION_RU = {
    Decision.RESISTANT: "устойчив",
    Decision.SUSCEPTIBLE: "чувствителен",
    Decision.NO_CALL: "нет заключения",
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
            "модели устойчивости не загружены. Обучите и сохраните их командой:\n"
            "    python -m germovision.train --save-models models"
        )

    ds = _dataset_from_rows(parsed.payload)
    order = list(DRUG_NAMES_RU)
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
                DRUG_NAMES_RU.get(drug, drug),
                _DECISION_RU[pr.decision],
                "" if pr.source == "catalogue" else round(float(pr.probability), 3),
                "каталог ВОЗ" if pr.source == "catalogue" else "модель",
                "да" if pr.needs_confirmation else "нет",
                pr.explain(),
            ])

    detail = Table(
        name="resistance",
        title="Прогноз лекарственной устойчивости",
        columns=[
            "Изолят", "Код", "Препарат", "Заключение", "Вероятность",
            "Источник", "Нужен фенотип", "Обоснование",
        ],
        rows=rows,
        note=(
            "Вероятность не показывается там, где решение принято по каталогу "
            "мутаций ВОЗ: это референсный стандарт, и он имеет приоритет над "
            "оценкой модели."
        ),
    )

    summary_rows = []
    for sid, decisions in per_isolate.items():
        res = [DRUG_NAMES_RU.get(d, d) for d, v in decisions.items() if v == Decision.RESISTANT]
        nc = [DRUG_NAMES_RU.get(d, d) for d, v in decisions.items() if v == Decision.NO_CALL]
        resistant_set = {d for d, v in decisions.items() if v == Decision.RESISTANT}
        mdr = "да" if {"RIF", "INH"} <= resistant_set else "нет"
        summary_rows.append([
            sid, len(res), mdr, "; ".join(res) or "—", "; ".join(nc) or "—",
        ])

    summary_table = Table(
        name="isolates",
        title="Сводка по изолятам",
        columns=["Изолят", "Устойчив к (число)", "МЛУ", "Устойчив к", "Без заключения"],
        rows=summary_rows,
        note=(
            "МЛУ — множественная лекарственная устойчивость: одновременно "
            "к рифампицину и изониазиду."
        ),
    )

    n_mdr = sum(1 for r in summary_rows if r[2] == "да")
    highlights = [
        {"label": "Изолятов", "value": str(len(per_isolate))},
        {"label": "Препаратов проверено", "value": str(len(drugs))},
        {"label": "Прогнозов устойчивости", "value": str(n_resistant)},
        {"label": "Из них МЛУ", "value": f"{n_mdr} изол.", "tone": "crit" if n_mdr else "good"},
    ]

    notes = list(parsed.notes)
    if bundle.manifest.get("synthetic"):
        notes.append(
            "Модели обучены на синтетических данных: результат демонстрирует "
            "работу пайплайна, а не клиническое качество."
        )
    needs = [
        DRUG_NAMES_RU.get(d, d)
        for d, q in (bundle.manifest.get("quality") or {}).items()
        if q.get("requires_confirmation")
    ]
    if needs:
        notes.append(
            "Требуют лабораторного подтверждения: " + ", ".join(sorted(needs))
        )

    return AnalysisResult(
        kind=parsed.kind,
        model="GV-Resist",
        title="Лекарственная устойчивость",
        summary=f"{parsed.summary}. Проверено {len(drugs)} препаратов.",
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
        title="Наблюдённые замены, по убыванию риска",
        columns=[
            "Мутация", "Позиция", "Исходный", "Замена", "Риск",
            "Допустимость", "Заметность", "Новизна", "Консервативность",
            "Наблюдений", "Частота", "Рост в неделю",
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
            "Риск — среднее геометрическое трёх множителей: допустимости замены "
            "по профилю, физико-химической заметности и новизны. Рост в неделю "
            "показан там, где в заголовках нашлись даты."
        ),
    )

    candidates = Table(
        name="candidate_mutations",
        title="Кандидаты: замены, ещё не встречавшиеся в данных",
        columns=[
            "Мутация", "Позиция", "Исходный", "Замена", "Риск",
            "Допустимость", "Заметность", "Консервативность",
        ],
        rows=[
            [
                r.label, r.position, r.wildtype, r.mutant, round(r.risk, 4),
                round(r.tolerance, 4), round(r.salience, 4), round(r.conservation, 4),
            ]
            for r in report.candidates
        ],
        note=(
            "Именно этот список и есть механизм раннего предупреждения: замену "
            "можно оценить до того, как она встретится. Оценка получена профильной "
            "моделью и не учитывает эпистаз — зависимость эффекта от других замен."
        ),
    )

    hotspots = Table(
        name="hotspots",
        title="Горячие точки: позиции с наибольшим разнообразием замен",
        columns=[
            "Позиция", "Исходный", "Разных замен", "Всего наблюдений",
            "Макс. риск", "Консервативность", "Замены",
        ],
        rows=[
            [
                h["position"], h["wildtype"], h["n_variants"], h["total_count"],
                h["max_risk"], h["conservation"], ", ".join(h["mutations"]),
            ]
            for h in report.hotspots
        ],
        note=(
            "Повторные независимые замены в одной позиции — классический признак "
            "положительного отбора: эволюция «пробует» эту позицию снова и снова."
        ),
    )

    top = report.observed[0] if report.observed else None
    rising = [r for r in report.observed if r.trend and r.trend > 0.05]
    highlights = [
        {"label": "Последовательностей", "value": str(report.n_used)},
        {"label": "Наблюдённых замен", "value": str(len(report.observed))},
        {"label": "Горячих точек", "value": str(len(report.hotspots))},
        {
            "label": "Максимальный риск",
            "value": f"{top.label} · {top.risk:.2f}" if top else "—",
            "tone": "warn" if top and top.risk > 0.5 else "",
        },
    ]
    if rising:
        highlights.append({
            "label": "Растущих замен",
            "value": str(len(rising)),
            "tone": "crit",
        })

    return AnalysisResult(
        kind=parsed.kind,
        model="GV-Escape",
        title="Эволюционный риск мутаций",
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
            f"в данных лишь одна линия ({lineages[0]}). Модель описывает, как линии "
            "вытесняют друг друга, поэтому нужно минимум две"
        )
    if len(weeks) < 4:
        raise AnalysisError(
            f"в данных {len(weeks)} моментов времени. Для оценки скорости роста "
            "нужно минимум четыре"
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
        raise AnalysisError(f"не удалось подогнать модель роста: {exc}") from exc

    growth_rows = [
        [
            g["region"], g["lineage"], round(g["beta"], 5),
            "" if not np.isfinite(g["se"]) else round(g["se"], 5),
            round(g["weekly_pct"], 2), g["n_samples"],
            "да" if g["significant"] else "нет",
        ]
        for g in sorted(model.growth_table(), key=lambda g: -g["beta"])
    ]
    growth = Table(
        name="growth",
        title="Преимущество роста линий по регионам",
        columns=[
            "Регион", "Линия", "β", "SE", "Рост в неделю, %", "Образцов", "Значимо",
        ],
        rows=growth_rows,
        note=(
            f"β — логарифмическое преимущество роста за неделю относительно "
            f"референсной линии «{lineages[0]}» (самой многочисленной), у которой "
            "β = 0 по построению. «Значимо» означает, что доверительный интервал "
            "не пересекает ноль."
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
        title="Прогноз долей линий",
        columns=["Регион", "Линия", "Недель вперёд", "Доля", "Нижняя граница", "Верхняя граница"],
        rows=fc_rows,
        note="Интервал 95 %. Он тем шире, чем меньше образцов просеквенировано в регионе.",
    )

    rising = [g for g in model.growth_table() if g["significant"] and g["beta"] > 0]
    top = max(rising, key=lambda g: g["beta"]) if rising else None
    highlights = [
        {"label": "Регионов", "value": str(len(model.fits_))},
        {"label": "Линий", "value": str(len(lineages))},
        {"label": "Образцов", "value": str(int(counts.sum()))},
        {
            "label": "Растёт значимо",
            "value": (
                f"{top['lineage']} · +{top['weekly_pct']:.1f} %/нед" if top else "не выявлено"
            ),
            "tone": "warn" if top else "good",
        },
    ]

    return AnalysisResult(
        kind=parsed.kind,
        model="GV-Growth",
        title="Динамика линий в популяции",
        summary=f"{parsed.summary}. Стягивание τ = {model.tau_:.4f}.",
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
            "Это геном целиком. Вызов вариантов по нему требует выравнивания на "
            "референс и контроля качества — то есть биоинформатического "
            "пайплайна, а не веб-формы: результат, полученный наспех, выглядел "
            "бы готовым, не будучи надёжным.\n\n"
            "Обработайте геном одним из стандартных средств и загрузите его вывод:\n"
            "  • TB-Profiler — даст JSON, который система принимает напрямую;\n"
            "  • snpEff поверх bcftools — даст аннотированный VCF;\n"
            "  • любой пайплайн — подойдёт таблица id, gene, mutation.\n\n"
            "Отдельные гены (короче 15 000 п. н.) принимаются и анализируются "
            "на риск мутаций без пайплайна."
        )
    raise AnalysisError(f"неизвестный вид данных: {parsed.kind}")
