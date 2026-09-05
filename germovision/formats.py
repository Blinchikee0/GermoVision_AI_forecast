"""Распознавание и разбор входных файлов.

Данные о патогене приходят в разном виде: из биоинформатического
пайплайна — VCF, из репозитория последовательностей — FASTA, из системы
надзора — таблица счётчиков, из TB-Profiler — JSON. Заставлять
пользователя приводить их к одному формату вручную значит перекладывать
на него работу, которую программа делает надёжнее.

Формат определяется **по содержимому, а не по расширению**: файл с
вариантами часто называют `.txt`, а выгрузку из таблицы — `.csv`
независимо от того, что внутри. Расширение используется только как
подсказка при неоднозначности.

Каждый формат ведёт к своей модели:

    мутации (CSV) ─────────┐
    VCF ───────────────────┼──> GV-Resist  (лекарственная устойчивость)
    TB-Profiler (JSON) ────┘
    белковые FASTA ───────────> GV-Escape  (риск мутаций)
    счётчики линий (CSV) ─────> GV-Growth  (динамика в популяции)

Нуклеотидный геном целиком осознанно не принимается: вызов вариантов
требует выравнивания на референс и контроля качества прочтений, то есть
полноценного пайплайна. Делать вид, что это можно сделать в веб-форме,
значило бы выдавать ненадёжный результат за готовый. Для такого файла
возвращается объяснение с указанием, чем его обработать.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "InputKind",
    "ParsedInput",
    "FormatError",
    "detect_and_parse",
    "parse_fasta",
    "SUPPORTED",
]


class FormatError(ValueError):
    """Файл не удалось распознать или он повреждён."""


class InputKind:
    """Распознаваемые виды входных данных."""

    MUTATIONS = "mutations"
    VCF = "vcf"
    PROTEIN_FASTA = "protein_fasta"
    GENE_FASTA = "gene_fasta"
    GENOME_FASTA = "genome_fasta"
    LINEAGE_COUNTS = "lineage_counts"
    TBPROFILER = "tbprofiler"


#: Что система умеет принимать — для подсказки в интерфейсе.
SUPPORTED: list[dict[str, str]] = [
    {
        "kind": InputKind.MUTATIONS,
        "title": "Таблица мутаций",
        "ext": ".csv, .tsv, .txt",
        "shape": "столбцы id, gene, mutation",
        "model": "GV-Resist",
        "result": "прогноз лекарственной устойчивости по 13 препаратам",
    },
    {
        "kind": InputKind.VCF,
        "title": "Вызванные варианты",
        "ext": ".vcf, .vcf.gz (распакованный)",
        "shape": "стандартный VCF, желательно с аннотацией snpEff (ANN=)",
        "model": "GV-Resist",
        "result": "прогноз лекарственной устойчивости по 13 препаратам",
    },
    {
        "kind": InputKind.TBPROFILER,
        "title": "Отчёт TB-Profiler",
        "ext": ".json",
        "shape": "поле dr_variants или variants",
        "model": "GV-Resist",
        "result": "прогноз лекарственной устойчивости по 13 препаратам",
    },
    {
        "kind": InputKind.PROTEIN_FASTA,
        "title": "Белковые последовательности",
        "ext": ".fasta, .fa, .faa",
        "shape": "несколько последовательностей одного белка",
        "model": "GV-Escape",
        "result": "ранжирование мутаций по эволюционному риску",
    },
    {
        "kind": InputKind.GENE_FASTA,
        "title": "Нуклеотидные последовательности гена",
        "ext": ".fasta, .fa, .fna",
        "shape": "кодирующая последовательность, длина кратна 3",
        "model": "GV-Escape",
        "result": "трансляция и ранжирование мутаций по риску",
    },
    {
        "kind": InputKind.LINEAGE_COUNTS,
        "title": "Счётчики линий надзора",
        "ext": ".csv, .tsv",
        "shape": "столбцы region, week, lineage, count",
        "model": "GV-Growth",
        "result": "коэффициенты роста линий и прогноз долей",
    },
]

_NUCLEOTIDES = set("ACGTUNRYKMSWBDHVacgtunrykmswbdhv-.")
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYXBZJUO*acdefghiklmnpqrstvwyxbzjuo-.")

#: Порог, выше которого нуклеотидная последовательность считается геномом,
#: а не отдельным геном. Самый длинный ген M. tuberculosis короче 12 т. п. н.
_GENOME_LENGTH = 15000

#: Нулевой байт — надёжный признак двоичного файла.
NUL = bytes([0])


@dataclass
class ParsedInput:
    """Результат разбора файла.

    Args:
        kind: распознанный вид данных.
        payload: разобранное содержимое, форма зависит от вида.
        n_records: число записей — изолятов, последовательностей, строк.
        summary: краткое человекочитаемое описание.
        notes: замечания разбора: пропущенные строки, догадки, ограничения.
    """

    kind: str
    payload: Any
    n_records: int
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def model(self) -> str:
        """Модель, которой адресован этот вид данных."""
        for row in SUPPORTED:
            if row["kind"] == self.kind:
                return row["model"]
        return "—"


# --------------------------------------------------------------------------
# FASTA
# --------------------------------------------------------------------------


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Разобрать FASTA в список пар «заголовок, последовательность».

    Raises:
        FormatError: нет ни одной последовательности.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip() or f"seq_{len(records) + 1}"
            chunks = []
        elif header is not None:
            chunks.append(line)

    if header is not None:
        records.append((header, "".join(chunks)))

    records = [(h, s.upper()) for h, s in records if s]
    if not records:
        raise FormatError("в файле FASTA нет ни одной последовательности")
    return records


def _alphabet_kind(sequences: list[str]) -> str:
    """Определить, нуклеотидная последовательность или белковая.

    Решение по доле символов ACGTUN: у нуклеотидных она близка к единице,
    у белковых заметно ниже, поскольку в них встречаются буквы, которых
    в нуклеотидном алфавите нет.
    """
    sample = "".join(sequences[:20])[:20000]
    if not sample:
        raise FormatError("последовательности пусты")

    letters = [c for c in sample if c.isalpha()]
    if not letters:
        raise FormatError("в последовательностях нет букв")

    nucleotide_share = sum(1 for c in letters if c in "ACGTUN") / len(letters)
    return "nucleotide" if nucleotide_share > 0.9 else "protein"


def _parse_fasta_input(text: str) -> ParsedInput:
    records = parse_fasta(text)
    seqs = [s for _, s in records]
    kind_alpha = _alphabet_kind(seqs)
    notes: list[str] = []

    bad = {c for s in seqs[:20] for c in s} - (
        _NUCLEOTIDES if kind_alpha == "nucleotide" else _AMINO_ACIDS
    )
    if bad:
        notes.append(
            "необычные символы в последовательностях: "
            + ", ".join(sorted(bad)[:8])
            + " — строки с ними будут обработаны как есть"
        )

    lengths = [len(s) for s in seqs]
    span = f"{min(lengths)}–{max(lengths)}" if min(lengths) != max(lengths) else str(lengths[0])

    if kind_alpha == "protein":
        return ParsedInput(
            kind=InputKind.PROTEIN_FASTA,
            payload=records,
            n_records=len(records),
            summary=f"{len(records)} белковых последовательностей, длина {span} а. о.",
            notes=notes,
        )

    if max(lengths) > _GENOME_LENGTH:
        return ParsedInput(
            kind=InputKind.GENOME_FASTA,
            payload=records,
            n_records=len(records),
            summary=f"{len(records)} нуклеотидных последовательностей, длина {span} п. н.",
            notes=notes,
        )

    if max(lengths) % 3 != 0:
        notes.append(
            f"длина {max(lengths)} не кратна трём — при трансляции последние "
            "нуклеотиды будут отброшены"
        )
    return ParsedInput(
        kind=InputKind.GENE_FASTA,
        payload=records,
        n_records=len(records),
        summary=f"{len(records)} нуклеотидных последовательностей гена, длина {span} п. н.",
        notes=notes,
    )


# --------------------------------------------------------------------------
# VCF
# --------------------------------------------------------------------------

_ANN_GENE = re.compile(r"ANN=[^;]*")
_HGVS_P = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})")

_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def _mutation_from_ann(info: str) -> tuple[str, str] | None:
    """Извлечь ген и замену из аннотации snpEff (поле ANN).

    Формат ANN: Allele|Effect|Impact|Gene_Name|Gene_ID|...|HGVS.c|HGVS.p|...
    Берётся первая аннотация — snpEff сортирует их по убыванию значимости.
    """
    match = _ANN_GENE.search(info)
    if not match:
        return None
    first = match.group(0)[4:].split(",")[0]
    parts = first.split("|")
    if len(parts) < 4:
        return None

    gene = parts[3].strip()
    if not gene:
        return None

    for part in parts:
        hit = _HGVS_P.search(part)
        if hit:
            wt = _THREE_TO_ONE.get(hit.group(1).capitalize())
            mut = _THREE_TO_ONE.get(hit.group(3).capitalize())
            if wt and mut:
                return gene, f"{wt}{hit.group(2)}{mut}"
    return None


def _parse_vcf(text: str) -> ParsedInput:
    by_sample: dict[str, set[str]] = {}
    samples: list[str] = []
    notes: list[str] = []
    n_lines = n_used = 0

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            cols = line.split("\t")
            samples = [c.strip() for c in cols[9:]] if len(cols) > 9 else []
            continue
        if not line.strip():
            continue

        fields = line.split("\t")
        if len(fields) < 8:
            continue
        n_lines += 1

        chrom, pos, _id, ref, alt, _qual, flt, info = fields[:8]
        if flt not in ("", ".", "PASS"):
            continue

        parsed = _mutation_from_ann(info)
        if parsed is None:
            # Без аннотации вариант не привязать к гену, а значит и к
            # препарату. Такие строки считаются, но не используются.
            continue
        gene, mutation = parsed
        n_used += 1

        holders = samples or ["sample"]
        if len(fields) > 9 and samples:
            holders = [
                s
                for s, gt in zip(samples, fields[9:], strict=False)
                if gt.split(":")[0] not in ("0/0", "0|0", "./.", ".|.")
            ] or []
        for s in holders:
            by_sample.setdefault(s, set()).add(f"{gene}_{mutation}")

    if not by_sample:
        raise FormatError(
            f"в VCF не нашлось вариантов с аннотацией гена (обработано строк: {n_lines}). "
            "Нужна аннотация snpEff — поле ANN= в столбце INFO. "
            "Аннотировать можно так: snpEff -v <база> input.vcf > annotated.vcf"
        )

    if n_used < n_lines:
        notes.append(
            f"{n_lines - n_used} из {n_lines} вариантов пропущено: нет аннотации "
            "гена или не пройден фильтр"
        )

    rows = [
        {"id": sid, "mutations": sorted(muts)} for sid, muts in sorted(by_sample.items())
    ]
    total = sum(len(r["mutations"]) for r in rows)
    return ParsedInput(
        kind=InputKind.VCF,
        payload=rows,
        n_records=len(rows),
        summary=f"{len(rows)} образцов, {total} аннотированных вариантов",
        notes=notes,
    )


# --------------------------------------------------------------------------
# TB-Profiler
# --------------------------------------------------------------------------


def _parse_tbprofiler(data: Any) -> ParsedInput:
    """Разобрать отчёт TB-Profiler (или список отчётов)."""
    reports = data if isinstance(data, list) else [data]
    rows: list[dict] = []
    notes: list[str] = []
    variant_keys = ("dr_variants", "other_variants", "variants")

    # Признак отчёта TB-Profiler — наличие поля с вариантами. Без этой
    # проверки любой словарь принимался бы за отчёт без единой мутации,
    # и пользователь получал бы пустой результат вместо объяснения.
    looks_like_report = any(
        isinstance(rep, dict) and any(k in rep for k in variant_keys)
        for rep in reports
    )
    if not looks_like_report:
        raise FormatError(
            "в JSON нет отчётов TB-Profiler: ожидается объект или список объектов "
            "с полем dr_variants либо variants"
        )

    for i, rep in enumerate(reports):
        if not isinstance(rep, dict):
            continue
        variants = []
        for key in variant_keys:
            variants.extend(rep.get(key) or [])

        muts: set[str] = set()
        for v in variants:
            if not isinstance(v, dict):
                continue
            gene = v.get("gene") or v.get("gene_name") or v.get("locus_tag")
            change = (
                v.get("change")
                or v.get("protein_change")
                or v.get("nucleotide_change")
                or v.get("hgvs_p")
            )
            if not gene or not change:
                continue
            change = str(change).replace("p.", "").strip()
            for three, one in _THREE_TO_ONE.items():
                change = change.replace(three, one)
            muts.add(f"{gene}_{change}")

        sid = str(rep.get("id") or rep.get("sample_name") or f"sample_{i + 1}")
        rows.append({"id": sid, "mutations": sorted(muts)})
        if not muts:
            notes.append(f"у образца {sid} не найдено вариантов")

    if not rows:
        raise FormatError(
            "в JSON нет отчётов TB-Profiler: ожидается объект или список объектов "
            "с полем dr_variants либо variants"
        )

    total = sum(len(r["mutations"]) for r in rows)
    return ParsedInput(
        kind=InputKind.TBPROFILER,
        payload=rows,
        n_records=len(rows),
        summary=f"{len(rows)} образцов TB-Profiler, {total} вариантов",
        notes=notes,
    )


# --------------------------------------------------------------------------
# Табличные форматы
# --------------------------------------------------------------------------


def _sniff_table(text: str) -> tuple[list[dict], dict[str, str]]:
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    cols = {c.strip().lower(): c for c in (reader.fieldnames or []) if c}
    return list(reader), cols


def _parse_mutations_table(rows: list[dict], cols: dict[str, str]) -> ParsedInput:
    by_id: dict[str, set[str]] = {}
    order: list[str] = []
    skipped = 0

    for row in rows:
        sid = str(row.get(cols["id"], "")).strip()
        if not sid:
            skipped += 1
            continue
        if sid not in by_id:
            by_id[sid] = set()
            order.append(sid)
        gene = str(row.get(cols["gene"], "")).strip()
        mut = str(row.get(cols["mutation"], "")).strip()
        if gene and mut:
            by_id[sid].add(f"{gene}_{mut}")
        else:
            skipped += 1

    if not order:
        raise FormatError("в таблице мутаций нет ни одной строки с идентификатором")

    notes = [f"{skipped} строк пропущено: пустой id, ген или мутация"] if skipped else []
    total = sum(len(v) for v in by_id.values())
    return ParsedInput(
        kind=InputKind.MUTATIONS,
        payload=[{"id": sid, "mutations": sorted(by_id[sid])} for sid in order],
        n_records=len(order),
        summary=f"{len(order)} изолятов, {total} вариантов",
        notes=notes,
    )


def _parse_lineage_counts(rows: list[dict], cols: dict[str, str]) -> ParsedInput:
    records: list[dict] = []
    skipped = 0
    for row in rows:
        try:
            week = float(str(row[cols["week"]]).strip())
            count = float(str(row[cols["count"]]).strip())
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        lineage = str(row.get(cols["lineage"], "")).strip()
        if not lineage or count < 0:
            skipped += 1
            continue
        records.append({
            "region": str(row.get(cols.get("region", ""), "") or "все регионы").strip(),
            "week": week,
            "lineage": lineage,
            "count": count,
        })

    if not records:
        raise FormatError(
            "не удалось прочитать ни одной строки счётчиков: нужны числовые "
            "столбцы week и count и непустой lineage"
        )

    notes = []
    if skipped:
        notes.append(
            f"{skipped} строк пропущено: нечисловые week/count или пустая линия"
        )
    regions = {r["region"] for r in records}
    lineages = {r["lineage"] for r in records}
    return ParsedInput(
        kind=InputKind.LINEAGE_COUNTS,
        payload=records,
        n_records=len(records),
        summary=(
            f"{len(records)} наблюдений, {len(regions)} регионов, "
            f"{len(lineages)} линий, {int(sum(r['count'] for r in records))} образцов"
        ),
        notes=notes,
    )


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


def detect_and_parse(filename: str, content: bytes | str) -> ParsedInput:
    """Определить формат файла и разобрать его.

    Args:
        filename: имя файла — используется только как подсказка.
        content: содержимое.

    Returns:
        ParsedInput.

    Raises:
        FormatError: формат не распознан или файл повреждён.
    """
    if isinstance(content, bytes):
        # Нулевой байт в первых килобайтах — надёжный признак двоичного
        # файла. Без этой проверки .bam или .xlsx успешно «декодируется»
        # в cp1251, превращается в мусорные строки и доходит до разбора
        # таблицы, где выдаёт невнятную ошибку про столбцы.
        if NUL in content[:8192]:
            raise FormatError(
                "файл двоичный, а не текстовый. Распакуйте архив (.gz, .zip) "
                "или экспортируйте данные в CSV. Форматы .bam, .xlsx, .h5 "
                "система не читает — сначала обработайте их своим пайплайном"
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("cp1251")
            except UnicodeDecodeError as exc:
                raise FormatError(
                    "файл не является текстовым. Архивы и двоичные форматы "
                    "(.gz, .bam, .xlsx) нужно распаковать или экспортировать в CSV"
                ) from exc
    else:
        text = content

    if not text.strip():
        raise FormatError("файл пуст")

    name = (filename or "").lower()
    stripped = text.lstrip()

    # 1. FASTA и VCF опознаются по первому значащему символу однозначно.
    if stripped.startswith(">"):
        return _parse_fasta_input(text)
    if stripped.startswith("##fileformat=VCF") or "\n#CHROM" in text[:20000]:
        return _parse_vcf(text)

    # 2. JSON.
    if stripped[0] in "{[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FormatError(f"файл похож на JSON, но не разбирается: {exc}") from exc
        return _parse_tbprofiler(data)

    # 3. Таблицы — по составу столбцов, а не по расширению.
    rows, cols = _sniff_table(text)
    if not cols:
        raise FormatError(
            "формат не распознан. Поддерживаются: таблица мутаций (id, gene, "
            "mutation), счётчики линий (region, week, lineage, count), VCF, "
            "FASTA, отчёт TB-Profiler в JSON"
        )

    if {"id", "gene", "mutation"} <= cols.keys():
        return _parse_mutations_table(rows, cols)
    if {"week", "lineage", "count"} <= cols.keys():
        return _parse_lineage_counts(rows, cols)

    found = ", ".join(sorted(cols)) or "нет заголовка"
    hint = ""
    if name.endswith((".xlsx", ".xls")):
        hint = " Файлы Excel нужно сохранить как CSV."
    raise FormatError(
        f"столбцы таблицы не опознаны (найдены: {found}). Ожидается либо "
        "id, gene, mutation — для прогноза устойчивости, либо "
        "region, week, lineage, count — для динамики линий." + hint
    )
