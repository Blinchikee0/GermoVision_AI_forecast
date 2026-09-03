"""Метрики вероятностного прогноза долей вариантов.

Данные геномного надзора — мультиномиальные счётчики: «из 120
просеквенированных образцов 34 относятся к линии X». Отсюда следуют
требования к метрикам, которые версия 1.0 не выполняла:

* Доли зависимы — они в сумме дают единицу; оценивать их независимой
  регрессией некорректно.
* Неопределённость зависит от объёма выборки — доля 30 % при 10 и при
  1000 образцов означает разную уверенность. Метрика обязана учитывать
  знаменатель.
* Прогноз вероятностный, поэтому оценивается правильной функцией потерь
  (proper scoring rule), а не средней ошибкой точечной оценки.

Отдельно проверяется покрытие интервалов: модель, дающая формально
точный прогноз при систематически заниженной неопределённости, опасна —
она создаёт ложную уверенность.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "multinomial_log_score",
    "ranked_probability_score",
    "interval_coverage",
    "CoverageReport",
    "evaluate_coverage",
    "persistence_forecast",
]

_EPS = 1e-12


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, None)
    return p / p.sum(axis=-1, keepdims=True)


def multinomial_log_score(counts, probs) -> float:
    """Средняя отрицательная логарифмическая правдоподобность на наблюдение.

    Правильная функция потерь: минимизируется истинным распределением,
    поэтому её нельзя «обмануть» осторожным прогнозом. Меньше — лучше.

    Args:
        counts: массив (T × V) наблюдённых счётчиков по вариантам.
        probs: массив (T × V) предсказанных вероятностей.

    Returns:
        Средний по наблюдениям отрицательный логарифм правдоподобия.
        Взвешивание по числу образцов происходит естественно: строка,
        основанная на 1000 наблюдений, влияет на итог в 100 раз сильнее
        строки из 10 наблюдений.
    """
    c = np.asarray(counts, dtype=float)
    p = _normalize(np.asarray(probs, dtype=float))
    if c.shape != p.shape:
        raise ValueError(f"формы не совпадают: {c.shape} и {p.shape}")

    total = c.sum()
    if total == 0:
        return float("nan")
    return float(-np.sum(c * np.log(p)) / total)


def ranked_probability_score(observed_bin, probs) -> float:
    """Ранжированная вероятностная оценка для упорядоченных категорий.

    Применима там, где категории имеют естественный порядок — например,
    прогноз числа случаев по градациям. В отличие от логарифмической
    потери, штрафует за расстояние: прогноз, промахнувшийся на одну
    градацию, наказывается слабее, чем промахнувшийся на пять.

    Args:
        observed_bin: индекс наблюдённой категории для каждого наблюдения.
        probs: массив (T × K) предсказанных вероятностей категорий.

    Returns:
        Средний RPS; 0 — идеально.
    """
    obs = np.asarray(observed_bin, dtype=int)
    p = _normalize(np.asarray(probs, dtype=float))
    if p.ndim != 2:
        raise ValueError("probs должен быть двумерным (T × K)")
    if obs.size != p.shape[0]:
        raise ValueError("число наблюдений не совпадает с числом строк probs")

    n_cat = p.shape[1]
    onehot = np.zeros_like(p)
    onehot[np.arange(obs.size), np.clip(obs, 0, n_cat - 1)] = 1.0

    cum_pred = np.cumsum(p, axis=1)
    cum_obs = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_pred - cum_obs) ** 2, axis=1) / (n_cat - 1)))


def interval_coverage(y_true, lower, upper) -> float:
    """Доля истинных значений, попавших в предсказанный интервал."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("формы y_true, lower и upper должны совпадать")
    if np.any(hi < lo):
        raise ValueError("верхняя граница интервала ниже нижней")
    return float(np.mean((yt >= lo) & (yt <= hi)))


@dataclass
class CoverageReport:
    """Отчёт о качестве интервалов неопределённости."""

    nominal: float
    empirical: float
    n: int
    mean_width: float

    @property
    def verdict(self) -> str:
        """Словесная оценка соответствия покрытия заявленному уровню.

        Допуск ±3 п.п. — компромисс между строгостью и разумной
        погрешностью на выборках порядка сотен наблюдений.
        """
        diff = self.empirical - self.nominal
        if abs(diff) <= 0.03:
            return "корректное"
        if diff < 0:
            return "ЗАНИЖЕНО — интервалы слишком узкие, ложная уверенность"
        return "завышено — интервалы избыточно широкие, прогноз малополезен"

    def __str__(self) -> str:
        return (
            f"покрытие {self.empirical:.1%} при заявленных {self.nominal:.0%} "
            f"(n={self.n}, средняя ширина {self.mean_width:.3f}) — {self.verdict}"
        )


def evaluate_coverage(y_true, lower, upper, nominal: float = 0.95) -> CoverageReport:
    """Проверить, соответствует ли фактическое покрытие заявленному."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return CoverageReport(
        nominal=nominal,
        empirical=interval_coverage(y_true, lo, hi),
        n=int(np.asarray(y_true).size),
        mean_width=float(np.mean(hi - lo)),
    )


def persistence_forecast(history, horizon: int = 1) -> np.ndarray:
    """Базовый прогноз «дальше будет как сейчас».

    Обязательная точка отсчёта (§ 5.7, правило: нет baseline — нет
    интерпретируемой метрики). Прогноз, не превосходящий persistence,
    не имеет практической ценности независимо от абсолютных значений
    своих метрик.

    Args:
        history: массив (T × V) наблюдённых долей по времени.
        horizon: длина прогноза в шагах.

    Returns:
        Массив (horizon × V) — последнее наблюдение, повторённое horizon раз.
    """
    h = np.asarray(history, dtype=float)
    if h.ndim != 2:
        raise ValueError("history должен быть двумерным (T × V)")
    if h.shape[0] == 0:
        raise ValueError("история пуста")
    if horizon <= 0:
        raise ValueError("horizon должен быть положительным")
    return np.tile(_normalize(h[-1]), (horizon, 1))
