"""Тесты распознавания и разбора входных форматов."""

from __future__ import annotations

import json

import pytest

from germovision.formats import (
    SUPPORTED,
    FormatError,
    InputKind,
    detect_and_parse,
    parse_fasta,
)

# --------------------------------------------------------------------------
# FASTA
# --------------------------------------------------------------------------

PROTEIN = ">a|2024-01-05\nMKTAYIAKQR\n>b|2024-02-10\nMKTAYIAKQW\n>c|2024-03-01\nMKTAYIAKER\n"
NUCL_GENE = ">g1\n" + "ATGAAAACCGCT" * 4 + "\n>g2\n" + "ATGAAAACCGCT" * 4 + "\n"


def test_parse_fasta_basic():
    recs = parse_fasta(PROTEIN)
    assert len(recs) == 3
    assert recs[0][0].startswith("a|")
    assert recs[0][1] == "MKTAYIAKQR"


def test_parse_fasta_joins_wrapped_lines():
    recs = parse_fasta(">x\nMKT\nAYI\nAKQ\n")
    assert recs[0][1] == "MKTAYIAKQ"


def test_parse_fasta_rejects_empty():
    with pytest.raises(FormatError, match="no sequences"):
        parse_fasta(">только заголовок\n")


def test_detects_protein_fasta():
    p = detect_and_parse("seqs.fasta", PROTEIN)
    assert p.kind == InputKind.PROTEIN_FASTA
    assert p.n_records == 3
    assert p.model == "GV-Escape"


def test_detects_gene_fasta_by_alphabet():
    """Различение белка и нуклеотидов идёт по алфавиту, а не по расширению."""
    p = detect_and_parse("anything.txt", NUCL_GENE)
    assert p.kind == InputKind.GENE_FASTA


def test_detects_genome_by_length():
    genome = ">chr\n" + "ACGT" * 5000 + "\n"
    p = detect_and_parse("genome.fa", genome)
    assert p.kind == InputKind.GENOME_FASTA


def test_warns_when_length_not_multiple_of_three():
    p = detect_and_parse("g.fa", ">x\n" + "ACGTACGTA" * 3 + "AC\n>y\n" + "ACGTACGTA" * 3 + "AC\n")
    assert p.kind == InputKind.GENE_FASTA
    assert any("divisible by three" in n for n in p.notes)


# --------------------------------------------------------------------------
# VCF
# --------------------------------------------------------------------------

VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
NC_000962.3\t761155\t.\tC\tT\t900\tPASS\tANN=T|missense_variant|MODERATE|rpoB|Rv0667|transcript|c.1349C>T|p.Ser450Leu\tGT\t1/1\t0/0
NC_000962.3\t2155168\t.\tC\tG\t900\tPASS\tANN=G|missense_variant|MODERATE|katG|Rv1908c|transcript|c.944G>C|p.Ser315Thr\tGT\t1/1\t1/1
"""


def test_parses_vcf_with_annotation():
    p = detect_and_parse("v.vcf", VCF)
    assert p.kind == InputKind.VCF
    assert p.n_records == 2
    by_id = {r["id"]: r["mutations"] for r in p.payload}
    assert by_id["S1"] == ["katG_S315T", "rpoB_S450L"]
    assert by_id["S2"] == ["katG_S315T"]


def test_vcf_translates_three_letter_code():
    """p.Ser450Leu должно стать S450L — иначе каталог мутаций не найдёт вариант."""
    p = detect_and_parse("v.vcf", VCF)
    assert "rpoB_S450L" in p.payload[0]["mutations"]


def test_vcf_skips_filtered_variants():
    text = VCF.replace("\t900\tPASS\tANN=T|missense", "\t900\tLowQual\tANN=T|missense")
    p = detect_and_parse("v.vcf", text)
    muts = {m for r in p.payload for m in r["mutations"]}
    assert "rpoB_S450L" not in muts


def test_vcf_without_annotation_explains_what_to_do():
    text = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "NC_000962.3\t761155\t.\tC\tT\t900\tPASS\tDP=50\n"
    )
    with pytest.raises(FormatError, match="snpEff"):
        detect_and_parse("v.vcf", text)


# --------------------------------------------------------------------------
# TB-Profiler
# --------------------------------------------------------------------------


def test_parses_tbprofiler_json():
    data = {
        "id": "TB-1",
        "dr_variants": [
            {"gene": "rpoB", "change": "p.Ser450Leu"},
            {"gene": "katG", "change": "S315T"},
        ],
    }
    p = detect_and_parse("r.json", json.dumps(data))
    assert p.kind == InputKind.TBPROFILER
    assert p.payload[0]["id"] == "TB-1"
    assert set(p.payload[0]["mutations"]) == {"rpoB_S450L", "katG_S315T"}


def test_parses_list_of_tbprofiler_reports():
    data = [
        {"id": "A", "dr_variants": [{"gene": "rpoB", "change": "S450L"}]},
        {"id": "B", "variants": [{"gene": "katG", "change": "S315T"}]},
    ]
    p = detect_and_parse("r.json", json.dumps(data))
    assert p.n_records == 2


def test_rejects_unrelated_json():
    with pytest.raises(FormatError, match="TB-Profiler"):
        detect_and_parse("x.json", json.dumps({"foo": "bar"}))


def test_rejects_broken_json():
    with pytest.raises(FormatError, match="does not parse"):
        detect_and_parse("x.json", "{ незакрытая скобка")


# --------------------------------------------------------------------------
# Таблицы
# --------------------------------------------------------------------------


def test_parses_mutations_table():
    p = detect_and_parse("m.csv", "id,gene,mutation\nA,rpoB,S450L\nA,katG,S315T\nB,gyrA,D94G\n")
    assert p.kind == InputKind.MUTATIONS
    assert p.n_records == 2
    assert p.model == "GV-Resist"


def test_parses_tab_separated_table():
    p = detect_and_parse("m.tsv", "id\tgene\tmutation\nA\trpoB\tS450L\n")
    assert p.kind == InputKind.MUTATIONS


def test_parses_lineage_counts():
    rows = ["region,week,lineage,count"]
    for w in range(5):
        rows += [f"KZ,{w},L2,{10 + w}", f"KZ,{w},L4,{20 - w}"]
    p = detect_and_parse("c.csv", "\n".join(rows))
    assert p.kind == InputKind.LINEAGE_COUNTS
    assert p.model == "GV-Growth"
    assert p.n_records == 10


def test_lineage_counts_skip_nonnumeric_rows():
    text = "region,week,lineage,count\nKZ,1,L2,5\nKZ,неделя,L2,x\nKZ,2,L4,7\n"
    p = detect_and_parse("c.csv", text)
    assert p.n_records == 2
    assert any("skipped" in n for n in p.notes)


def test_unknown_columns_explain_expected_shape():
    with pytest.raises(FormatError, match="id, gene, mutation"):
        detect_and_parse("x.csv", "колонка1,колонка2\n1,2\n")


def test_excel_gets_specific_hint():
    with pytest.raises(FormatError, match="Excel"):
        detect_and_parse("данные.xlsx", "колонка1,колонка2\n1,2\n")


# --------------------------------------------------------------------------
# Общие случаи
# --------------------------------------------------------------------------


def test_rejects_binary_content():
    """Двоичный файл отсекается до разбора, а не через невнятную ошибку столбцов."""
    with pytest.raises(FormatError, match="binary"):
        detect_and_parse("a.bam", b"BAM\x01" + bytes([0]) * 8 + b"payload")


def test_rejects_empty_file():
    with pytest.raises(FormatError, match="empty"):
        detect_and_parse("e.csv", "   \n  \n")


def test_accepts_cp1251_encoding():
    """Выгрузки из старых систем нередко приходят в cp1251."""
    text = "id,gene,mutation\nобразец-1,rpoB,S450L\n"
    p = detect_and_parse("m.csv", text.encode("cp1251"))
    assert p.payload[0]["id"] == "образец-1"


def test_extension_does_not_override_content():
    """Файл с вариантами, названный .csv, всё равно распознаётся как VCF."""
    p = detect_and_parse("варианты.csv", VCF)
    assert p.kind == InputKind.VCF


def test_supported_catalogue_is_complete():
    """Каждый вид входа описан в справочнике для интерфейса."""
    described = {row["kind"] for row in SUPPORTED}
    for kind in (
        InputKind.MUTATIONS, InputKind.VCF, InputKind.TBPROFILER,
        InputKind.PROTEIN_FASTA, InputKind.GENE_FASTA, InputKind.LINEAGE_COUNTS,
    ):
        assert kind in described
