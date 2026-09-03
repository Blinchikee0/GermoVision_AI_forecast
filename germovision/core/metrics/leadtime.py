"""Lead Time — упреждение системы. Главная метрика проекта.

Формулировка «система выявляет мутации раньше» бессмысленна без точки
отсчёта. Здесь она задана строго:

    LT = D_official − D_system

где `D_official` — дата официального признания угрозы (ВОЗ объявила
вариант значимым, национальный центр выпустил предупреждение), а
`D_system` — дата первого сигнала GermoVision при работе исключительно
на данных, доступных на тот момент.

Ключевое требование к корректности измерения обеспечивается не этим
модулем, а протоколом: сигналы должны быть получены на замороженных
срезах данных (`temporal.forward_chaining`), иначе измеряется не
упреждение, а знание будущего.

Второе требование — Lead Time никогда не отчитывается в одиночку.
Система, поднимающая тревогу ежедневно, формально обладает огромным
упреждением и при этом бесполезна: её перестают слушать через неделю.
Поэтому основной результат — `lead_time_at_budget`: максимальное
упреждение при заданном бюджете ложных тревог.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Signal",
    "OfficialEvent",
    "EventOutcome",
    "LeadTimeReport",
    "evaluate_lead_time",
    "lead_time_at_budget",
]


def _d(value) -> np.datetime64:
    return np.datetime64(value, "D")


@dataclass(frozen=True)
class Signal:
    """Сигнал системы по объекту наблюдения.

    Args:
        object_id: идентификатор объекта (линия, вариант, комбинация мутаций).
        date: дата сигнала.
        score: балл риска, выданный слоем объединения (§ 5.5).
        region: регион; объекты в разных регионах отслеживаются раздельно.
    """

    object_id: str
    date: np.datetime64
    score: float
    region: str = "global"

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", _d(self.date))


@dataclass(frozen=True)
class OfficialEvent:
    """Момент официального признания угрозы — точка отсчёта упреждения."""

    object_id: str
    official_date: np.datetime64
    region: str = "global"

    def __post_init__(self) -> None:
        object.__setattr__(self, "official_date", _d(self.official_date))


@dataclass(frozen=True)
class EventOutcome:
    """Результат по одному событию."""

    object_id: str
    region: str
    official_date: np.datetime64
    system_date: np.datetime64 | None
    lead_time_days: float  # NaN, если событие не обнаружено вовсе

    @property
    def detected(self) -> bool:
        return self.system_date is not None

    @property
    def detected_early(self) -> bool:
        """Обнаружено строго до официального признания."""
        return self.detected and self.lead_time_days > 0

    def __str__(self) -> str:
        if not self.detected:
            return f"{self.object_id} ({self.region}): ПРОПУЩЕНО"
        sign = "+" if self.lead_time_days > 0 else ""
        return (
            f"{self.object_id} ({self.region}): {sign}{self.lead_time_days:.0f} дн. "
            f"(сигнал {self.system_date}, официально {self.official_date})"
        )


@dataclass
class LeadTimeReport:
    """Сводный отчёт: упреждение вместе с ценой, которой оно достигнуто."""

    threshold: float
    outcomes: list[EventOutcome] = field(default_factory=list)
    false_alarms: int = 0
    observation_days: int = 0
    n_regions: int = 1

    @property
    def detection_rate(self) -> float:
        """Доля событий, по которым система вообще подала сигнал."""
        if not self.outcomes:
            return float("nan")
        return sum(o.detected for o in self.outcomes) / len(self.outcomes)

    @property
    def early_detection_rate(self) -> float:
        """Доля событий, обнаруженных до официального признания."""
        if not self.outcomes:
            return float("nan")
        return sum(o.detected_early for o in self.outcomes) / len(self.outcomes)

    @property
    def median_lead_time(self) -> float:
        """Медиана упреждения по обнаруженным событиям.

        Пропущенные события в медиану не входят — иначе метрика зависела
        бы от произвольного значения, приписанного пропуску. Поэтому
        медиана всегда читается вместе с `detection_rate`: высокая
        медиана при низкой полноте означает, что система ловит только
        самые очевидные случаи.
        """
        vals = [o.lead_time_days for o in self.outcomes if o.detected]
        return float(np.median(vals)) if vals else float("nan")

    @property
    def iqr_lead_time(self) -> tuple[float, float]:
        vals = [o.lead_time_days for o in self.outcomes if o.detected]
        if not vals:
            return (float("nan"), float("nan"))
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        return (float(q1), float(q3))

    @property
    def false_alarms_per_region_quarter(self) -> float:
        """Ложные тревоги в пересчёте на один регион за квартал."""
        if self.observation_days <= 0 or self.n_regions <= 0:
            return float("nan")
        region_quarters = self.n_regions * (self.observation_days / 91.31)
        return self.false_alarms / region_quarters if region_quarters > 0 else float("nan")

    def meets_h4(self, max_false_alarms_per_region_quarter: float) -> bool:
        """Проверка центральной гипотезы проекта H4.

        Упреждение положительно И бюджет ложных тревог соблюдён.
        Выполнение только одного из условий гипотезу не подтверждает.
        """
        return (
            self.median_lead_time > 0
            and self.false_alarms_per_region_quarter <= max_false_alarms_per_region_quarter
        )

    def to_markdown(self) -> str:
        q1, q3 = self.iqr_lead_time
        lines = [
            f"**Порог:** {self.threshold:.3f}",
            "",
            f"- Медианное упреждение: **{self.median_lead_time:.0f} дн.** "
            f"(IQR {q1:.0f}–{q3:.0f})",
            f"- Обнаружено событий: {self.detection_rate:.0%} "
            f"({sum(o.detected for o in self.outcomes)} из {len(self.outcomes)})",
            f"- Обнаружено с положительным упреждением: {self.early_detection_rate:.0%}",
            f"- Ложных тревог: {self.false_alarms} "
            f"({self.false_alarms_per_region_quarter:.2f} на регион в квартал)",
            "",
            "| Событие | Регион | Упреждение | Сигнал | Официально |",
            "|---|---|---|---|---|",
        ]
        ordered = sorted(
            self.outcomes,
            key=lambda o: (not o.detected, -o.lead_time_days if o.detected else 0.0),
        )
        for o in ordered:
            lt = f"{o.lead_time_days:+.0f} дн." if o.detected else "пропуск"
            sd = str(o.system_date) if o.system_date is not None else "—"
            lines.append(f"| {o.object_id} | {o.region} | {lt} | {sd} | {o.official_date} |")
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"LeadTime(порог={self.threshold:.3f}): "
            f"медиана {self.median_lead_time:.0f} дн., "
            f"полнота {self.detection_rate:.0%}, "
            f"ложных тревог {self.false_alarms_per_region_quarter:.2f}/регион·квартал"
        )


def evaluate_lead_time(
    signals: list[Signal],
    events: list[OfficialEvent],
    threshold: float,
    observation_days: int | None = None,
    n_regions: int | None = None,
    only_before_official: bool = True,
) -> LeadTimeReport:
    """Посчитать упреждение и число ложных тревог при заданном пороге.

    Args:
        signals: все сигналы системы за период наблюдения.
        events: события, впоследствии признанные значимыми.
        threshold: порог балла, выше которого сигнал считается тревогой.
        observation_days: длительность периода наблюдения. По умолчанию
            вычисляется по диапазону дат сигналов.
        n_regions: число регионов под наблюдением. По умолчанию — число
            различных регионов среди сигналов.
        only_before_official: учитывать только сигналы, поданные до
            официального признания. False позволяет измерить и
            запаздывание — иногда это тоже нужно знать.

    Returns:
        LeadTimeReport.

    Raises:
        ValueError: если список событий пуст.
    """
    if not events:
        raise ValueError("список событий пуст: упреждение измерить не от чего")

    fired = [s for s in signals if s.score >= threshold]
    dates = [s.date for s in signals]

    if observation_days is None:
        observation_days = (
            int((max(dates) - min(dates)).astype(int)) if len(dates) > 1 else 0
        )
    if n_regions is None:
        n_regions = len({s.region for s in signals}) or 1

    outcomes: list[EventOutcome] = []
    for ev in events:
        relevant = [
            s
            for s in fired
            if s.object_id == ev.object_id
            and s.region == ev.region
            and (not only_before_official or s.date < ev.official_date)
        ]
        if relevant:
            first = min(relevant, key=lambda s: s.date)
            lt = float((ev.official_date - first.date).astype(int))
            outcomes.append(
                EventOutcome(ev.object_id, ev.region, ev.official_date, first.date, lt)
            )
        else:
            outcomes.append(
                EventOutcome(ev.object_id, ev.region, ev.official_date, None, float("nan"))
            )

    known = {(e.object_id, e.region) for e in events}
    false_alarms = len({(s.object_id, s.region) for s in fired} - known)

    return LeadTimeReport(
        threshold=threshold,
        outcomes=outcomes,
        false_alarms=false_alarms,
        observation_days=observation_days,
        n_regions=n_regions,
    )


def lead_time_at_budget(
    signals: list[Signal],
    events: list[OfficialEvent],
    max_false_alarms_per_region_quarter: float,
    thresholds=None,
    observation_days: int | None = None,
    n_regions: int | None = None,
) -> tuple[float, LeadTimeReport]:
    """Найти порог, дающий наибольшее упреждение в рамках бюджета тревог.

    Это основной операционный результат системы. Порог выбирается не по
    F1 и не по точке Юдена, а по критерию, который формулирует
    эпидемиолог: «сколько ложных тревог в квартал я готов разобрать».
    Система лишь показывает, какое упреждение достижимо при таком
    бюджете.

    Args:
        signals: сигналы системы.
        events: официально признанные события.
        max_false_alarms_per_region_quarter: бюджет ложных тревог.
        thresholds: перебираемые пороги. По умолчанию — все наблюдённые
            значения баллов, что гарантирует нахождение оптимума.
        observation_days: длительность наблюдения.
        n_regions: число регионов.

    Returns:
        Пара (лучший порог, отчёт при нём).

    Raises:
        ValueError: если ни один порог не укладывается в бюджет.
    """
    if thresholds is None:
        thresholds = np.unique([s.score for s in signals])
    thresholds = np.sort(np.atleast_1d(np.asarray(thresholds, dtype=float)))

    best: tuple[float, LeadTimeReport] | None = None
    for thr in thresholds:
        rep = evaluate_lead_time(
            signals, events, float(thr), observation_days, n_regions
        )
        if rep.false_alarms_per_region_quarter > max_false_alarms_per_region_quarter:
            continue
        if np.isnan(rep.median_lead_time):
            continue
        # Первичный критерий — полнота обнаружения, вторичный — упреждение.
        # Система, ловящая одно событие с огромным запасом и пропускающая
        # остальные, хуже системы, ловящей все с умеренным упреждением.
        key = (rep.detection_rate, rep.median_lead_time)
        if best is None or key > (best[1].detection_rate, best[1].median_lead_time):
            best = (float(thr), rep)

    if best is None:
        raise ValueError(
            f"ни один порог не укладывается в бюджет "
            f"{max_false_alarms_per_region_quarter} ложных тревог на регион в квартал. "
            "Либо бюджет слишком строгий, либо модель недостаточно специфична"
        )
    return best
