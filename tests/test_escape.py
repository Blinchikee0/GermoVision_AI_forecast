"""Тесты GV-Escape и маршрутизации анализа."""

from __future__ import annotations

import numpy as np
import pytest

from germovision.analysis import AnalysisError, analyze
from germovision.formats import detect_and_parse
from germovision.models.escape import (
    AMINO_ACIDS,
    GVEscape,
    _physicochemical_distance,
    translate,
)

# --------------------------------------------------------------------------
# Вспомогательные данные
# --------------------------------------------------------------------------


def make_records(n=180, length=120, seed=5, rising_pos=60, with_dates=True):
    """Набор последовательностей с растущей заменой в заданной позиции."""
    rng = np.random.default_rng(seed)
    base = "".join(rng.choice(list(AMINO_ACIDS), length))
    if base[rising_pos] == "K":
        base = base[:rising_pos] + "A" + base[rising_pos + 1 :]

    records = []
    for i in range(n):
        s = list(base)
        week = i // 5
        if rng.random() < min(0.02 + week * 0.05, 0.9):
            s[rising_pos] = "K"
        for _ in range(int(rng.poisson(0.8))):
            p = int(rng.integers(0, length))
            s[p] = str(rng.choice(list(AMINO_ACIDS)))
        header = f"seq_{i}"
        if with_dates:
            d = np.datetime64("2024-01-01") + np.timedelta64(week * 7, "D")
            header += f"|{d}"
        records.append((header, "".join(s)))
    return base, records


# --------------------------------------------------------------------------
# Трансляция и физико-химия
# --------------------------------------------------------------------------


def test_translate_standard_codons():
    assert translate("ATGAAATTTTAA") == "MKF"      # стоп-кодон обрывает
    assert translate("ATGGCC") == "MA"
    assert translate("ATGGC") == "M"               # неполный кодон отброшен


def test_translate_handles_ambiguity():
    assert "X" in translate("ATGNNNTTT")


def test_physicochemical_distance_reflects_charge():
    """Смена заряда должна весить больше консервативной замены."""
    charge_flip = _physicochemical_distance("E", "K")
    conservative = _physicochemical_distance("L", "I")
    assert charge_flip > conservative
    assert 0.0 <= conservative < charge_flip <= 1.0


def test_identical_residue_has_zero_distance():
    assert _physicochemical_distance("A", "A") == 0.0


# --------------------------------------------------------------------------
# Профиль и риск
# --------------------------------------------------------------------------


def test_fit_builds_profile():
    _, records = make_records()
    m = GVEscape().fit(records)
    assert m.profile_ is not None
    assert m.profile_.shape == (120, 20)
    np.testing.assert_allclose(m.profile_.sum(axis=1), 1.0, atol=1e-9)
    assert m.n_used_ == len(records)


def test_conservation_is_bounded():
    _, records = make_records()
    m = GVEscape().fit(records)
    assert m.conservation_ is not None
    assert (m.conservation_ >= 0).all() and (m.conservation_ <= 1).all()


def test_fit_rejects_tiny_sample():
    _, records = make_records(n=3)
    with pytest.raises(ValueError, match="at least"):
        GVEscape().fit(records)


def test_analyze_requires_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        GVEscape().analyze()


def test_observed_mutations_are_found():
    base, records = make_records()
    report = GVEscape().fit(records).analyze()
    labels = {r.label for r in report.observed}
    # Позиция 60 (индекс) — 61-я в биологической нумерации.
    assert f"{base[60]}61K" in labels


def test_candidates_are_unobserved():
    _, records = make_records()
    report = GVEscape().fit(records).analyze(top_candidates=30)
    seen = {(r.position, r.mutant) for r in report.observed}
    assert len(report.candidates) == 30
    for c in report.candidates:
        assert (c.position, c.mutant) not in seen
        assert c.observed is False


def test_risk_components_are_in_range():
    _, records = make_records()
    report = GVEscape().fit(records).analyze(top_candidates=20)
    for r in report.observed + report.candidates:
        assert 0.0 <= r.risk <= 1.0
        assert 0.0 <= r.tolerance <= 1.0
        assert 0.0 <= r.salience <= 1.0
        assert 0.0 <= r.novelty <= 1.0


def test_observed_sorted_by_risk():
    _, records = make_records()
    report = GVEscape().fit(records).analyze()
    risks = [r.risk for r in report.observed]
    assert risks == sorted(risks, reverse=True)


def test_rising_mutation_gets_positive_trend():
    """Замена, доля которой растёт, должна получить положительный тренд."""
    base, records = make_records(n=260)
    report = GVEscape().fit(records).analyze()
    target = next(r for r in report.observed if r.label == f"{base[60]}61K")
    assert target.trend is not None and target.trend > 0


def test_no_trend_without_dates():
    _, records = make_records(with_dates=False)
    report = GVEscape().fit(records).analyze()
    assert all(r.trend is None for r in report.observed)
    assert any("no dates" in n for n in report.notes)


def test_random_mutations_rarely_get_trend():
    """Тренд возвращается только при значимости — иначе список забьёт шум."""
    _, records = make_records(n=200, rising_pos=60)
    report = GVEscape().fit(records).analyze()
    with_trend = [r for r in report.observed if r.trend is not None]
    # Растущая замена одна; допускаем немного ложных, но не десятки.
    assert 1 <= len(with_trend) <= 5


def test_hotspots_rank_by_variant_count():
    _, records = make_records()
    report = GVEscape().fit(records).analyze()
    counts = [h["n_variants"] for h in report.hotspots]
    assert counts == sorted(counts, reverse=True)


def test_handles_unequal_lengths():
    """Последовательности с делецией сопоставляются, а не отбрасываются."""
    _, records = make_records(n=60)
    trimmed = [(h, s[:-3] if i % 5 == 0 else s) for i, (h, s) in enumerate(records)]
    m = GVEscape().fit(trimmed)
    assert m.n_used_ >= 55
    assert any("aligned" in n for n in m.notes_)


def test_nucleotide_input_is_translated():
    codons = ["ATG", "AAA", "TTT", "GGC", "TGC", "CAT"] * 5
    records = [(f"s{i}", "".join(codons)) for i in range(12)]
    m = GVEscape().fit(records, nucleotide=True)
    assert m.reference_ == translate("".join(codons))


# --------------------------------------------------------------------------
# Маршрутизация анализа
# --------------------------------------------------------------------------


def _fasta_text(records):
    return "\n".join(f">{h}\n{s}" for h, s in records) + "\n"


def test_analyze_routes_protein_to_escape():
    _, records = make_records()
    parsed = detect_and_parse("p.fasta", _fasta_text(records))
    result = analyze(parsed, top_candidates=25)
    assert result.model == "GV-Escape"
    assert {t.name for t in result.tables} == {
        "observed_mutations", "candidate_mutations", "hotspots"
    }
    assert result.highlights


def test_tables_export_to_csv():
    _, records = make_records()
    result = analyze(detect_and_parse("p.fasta", _fasta_text(records)), top_candidates=5)
    csv_text = result.tables[0].to_csv()
    lines = csv_text.strip().split("\n")
    assert lines[0].startswith("Mutation,")
    assert len(lines) == len(result.tables[0].rows) + 1


def test_result_serialises_to_dict():
    _, records = make_records()
    result = analyze(detect_and_parse("p.fasta", _fasta_text(records)), top_candidates=5)
    d = result.to_dict()
    assert set(d) >= {"kind", "model", "title", "summary", "tables", "highlights", "notes"}
    assert isinstance(d["tables"][0]["rows"], list)


def test_genome_gets_actionable_explanation():
    """Отказ по геному должен объяснять, чем его обработать."""
    parsed = detect_and_parse("genome.fa", ">chr\n" + "ACGT" * 5000 + "\n")
    with pytest.raises(AnalysisError, match="TB-Profiler"):
        analyze(parsed)


def test_resistance_without_models_explains_how_to_train():
    parsed = detect_and_parse("m.csv", "id,gene,mutation\nA,rpoB,S450L\n")
    with pytest.raises(AnalysisError, match="save-models"):
        analyze(parsed, bundle=None)


def test_growth_needs_two_lineages():
    rows = ["region,week,lineage,count"] + [f"KZ,{w},L2,{10 + w}" for w in range(6)]
    with pytest.raises(AnalysisError, match="only one lineage"):
        analyze(detect_and_parse("c.csv", "\n".join(rows)))


def test_growth_needs_enough_timepoints():
    rows = ["region,week,lineage,count"]
    for w in range(2):
        rows += [f"KZ,{w},L2,{10}", f"KZ,{w},L4,{20}"]
    with pytest.raises(AnalysisError, match="time points"):
        analyze(detect_and_parse("c.csv", "\n".join(rows)))


def test_growth_picks_most_abundant_reference():
    """Референс — самая многочисленная линия, иначе знаки β читаются наоборот."""
    rows = ["region,week,lineage,count"]
    for w in range(12):
        rows.append(f"KZ,{w},L4_major,{max(1, 60 - w * 2)}")
        rows.append(f"KZ,{w},L2_rising,{max(1, 3 + w * 3)}")
    result = analyze(detect_and_parse("c.csv", "\n".join(rows)))
    assert "L4_major" in result.tables[0].note
    rising = [r for r in result.tables[0].rows if r[1] == "L2_rising"]
    assert rising and rising[0][2] > 0  # β положительна у растущей линии
