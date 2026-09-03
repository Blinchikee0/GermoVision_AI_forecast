"""Тесты главной метрики проекта — упреждения (Lead Time)."""

from __future__ import annotations

import numpy as np
import pytest

from germovision.core.metrics import (
    OfficialEvent,
    Signal,
    evaluate_lead_time,
    lead_time_at_budget,
)


@pytest.fixture
def omicron_like():
    """Сценарий по мотивам Омикрона: сигнал за 21 день до объявления ВОЗ."""
    signals = [
        Signal("BA.1", "2021-11-05", 0.91, "ZA"),
        Signal("BA.1", "2021-11-12", 0.95, "ZA"),
        Signal("BA.1", "2021-11-20", 0.97, "ZA"),
    ]
    events = [OfficialEvent("BA.1", "2021-11-26", "ZA")]
    return signals, events


def test_lead_time_uses_first_crossing(omicron_like):
    """Упреждение считается по первому пересечению порога, не по последнему."""
    signals, events = omicron_like
    rep = evaluate_lead_time(signals, events, threshold=0.9, observation_days=90, n_regions=1)
    assert rep.outcomes[0].lead_time_days == 21
    assert rep.outcomes[0].detected_early


def test_higher_threshold_shrinks_lead_time(omicron_like):
    """Более строгий порог даёт меньшее упреждение — базовый компромисс системы."""
    signals, events = omicron_like
    lenient = evaluate_lead_time(signals, events, 0.90, observation_days=90, n_regions=1)
    strict = evaluate_lead_time(signals, events, 0.96, observation_days=90, n_regions=1)
    assert lenient.median_lead_time > strict.median_lead_time


def test_missed_event_is_not_counted_as_zero(omicron_like):
    """Пропуск не входит в медиану, но снижает полноту обнаружения."""
    signals, events = omicron_like
    rep = evaluate_lead_time(signals, events, threshold=0.99, observation_days=90, n_regions=1)
    assert rep.detection_rate == 0.0
    assert np.isnan(rep.median_lead_time)
    assert not rep.outcomes[0].detected


def test_signal_after_official_date_is_not_early():
    signals = [Signal("X", "2021-12-10", 0.9)]
    events = [OfficialEvent("X", "2021-11-26")]
    rep = evaluate_lead_time(signals, events, 0.5, observation_days=90, n_regions=1)
    assert not rep.outcomes[0].detected  # only_before_official=True по умолчанию


def test_late_signal_counted_with_negative_lead_time():
    """С only_before_official=False запаздывание измеряется как отрицательное LT."""
    signals = [Signal("X", "2021-12-10", 0.9)]
    events = [OfficialEvent("X", "2021-11-26")]
    rep = evaluate_lead_time(
        signals, events, 0.5, observation_days=90, n_regions=1, only_before_official=False
    )
    assert rep.outcomes[0].lead_time_days == -14
    assert rep.outcomes[0].detected
    assert not rep.outcomes[0].detected_early


def test_false_alarms_counted_per_object():
    signals = [
        Signal("REAL", "2021-11-05", 0.9, "ZA"),
        Signal("NOISE1", "2021-11-06", 0.9, "ZA"),
        Signal("NOISE2", "2021-11-07", 0.9, "ZA"),
        Signal("NOISE2", "2021-11-08", 0.95, "ZA"),  # тот же объект — одна тревога
    ]
    events = [OfficialEvent("REAL", "2021-11-26", "ZA")]
    rep = evaluate_lead_time(signals, events, 0.5, observation_days=91, n_regions=1)
    assert rep.false_alarms == 2
    assert rep.false_alarms_per_region_quarter == pytest.approx(2.0, abs=0.05)


def test_regions_are_tracked_separately():
    """Один и тот же вариант в разных регионах — разные объекты наблюдения."""
    signals = [
        Signal("BA.1", "2021-11-05", 0.9, "ZA"),
        Signal("BA.1", "2021-11-20", 0.9, "KZ"),
    ]
    events = [
        OfficialEvent("BA.1", "2021-11-26", "ZA"),
        OfficialEvent("BA.1", "2021-12-10", "KZ"),
    ]
    rep = evaluate_lead_time(signals, events, 0.5, observation_days=90, n_regions=2)
    lts = sorted(o.lead_time_days for o in rep.outcomes)
    assert lts == [20.0, 21.0]
    assert rep.false_alarms == 0


def test_evaluate_lead_time_rejects_empty_events():
    with pytest.raises(ValueError, match="список событий пуст"):
        evaluate_lead_time([Signal("X", "2021-01-01", 0.5)], [], 0.5)


# --------------------------------------------------------------------------
# Выбор порога по бюджету ложных тревог
# --------------------------------------------------------------------------


def _budget_scenario():
    """Настоящие события дают высокие баллы, шум — умеренные."""
    signals = []
    for i, obj in enumerate(["E1", "E2", "E3"]):
        signals.append(Signal(obj, f"2021-11-0{i + 1}", 0.95))
    for i in range(10):
        signals.append(Signal(f"N{i}", "2021-11-15", 0.60))
    events = [
        OfficialEvent("E1", "2021-11-26"),
        OfficialEvent("E2", "2021-11-27"),
        OfficialEvent("E3", "2021-11-28"),
    ]
    return signals, events


def test_lead_time_at_budget_picks_threshold_excluding_noise():
    signals, events = _budget_scenario()
    thr, rep = lead_time_at_budget(
        signals, events, max_false_alarms_per_region_quarter=1.0,
        observation_days=91, n_regions=1,
    )
    assert thr > 0.60  # шум отсечён
    assert rep.false_alarms == 0
    assert rep.detection_rate == 1.0
    assert rep.median_lead_time > 0


def test_lead_time_at_budget_respects_generous_budget():
    """При щедром бюджете допускается низкий порог и больше ложных тревог."""
    signals, events = _budget_scenario()
    _, rep = lead_time_at_budget(
        signals, events, max_false_alarms_per_region_quarter=50.0,
        observation_days=91, n_regions=1,
    )
    assert rep.detection_rate == 1.0


def test_lead_time_at_budget_fails_when_budget_impossible():
    signals = [
        Signal("REAL", "2021-11-05", 0.5),
        *[Signal(f"N{i}", "2021-11-06", 0.9) for i in range(20)],
    ]
    events = [OfficialEvent("REAL", "2021-11-26")]
    with pytest.raises(ValueError, match="ни один порог не укладывается в бюджет"):
        lead_time_at_budget(
            signals, events, max_false_alarms_per_region_quarter=0.1,
            observation_days=91, n_regions=1,
        )


def test_h4_requires_both_conditions():
    """H4 не подтверждается выполнением только одного из двух условий."""
    signals, events = _budget_scenario()
    _, rep = lead_time_at_budget(
        signals, events, 1.0, observation_days=91, n_regions=1
    )
    assert rep.meets_h4(max_false_alarms_per_region_quarter=1.0)
    assert not rep.meets_h4(max_false_alarms_per_region_quarter=0.0) or rep.false_alarms == 0


def test_report_renders_markdown(omicron_like):
    signals, events = omicron_like
    rep = evaluate_lead_time(signals, events, 0.9, observation_days=90, n_regions=1)
    md = rep.to_markdown()
    assert "BA.1" in md and "Медианное упреждение" in md
    assert "медиана" in str(rep)
