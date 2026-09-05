"""Метрики классификации для задач с редким положительным классом.

Исправление дефекта D10: версия 1.0 оценивала обнаружение опасных
мутаций метриками регрессии (MAPE, MAE, MSE). Это неверно дважды —
задача классификационная, а MAPE к тому же теряет смысл при значениях,
близких к нулю, каковыми и являются частоты редких мутаций.

Здесь собраны метрики, соответствующие природе задачи, и — обязательно —
их доверительные интервалы. Значение «AUC 0,82», приведённое без
интервала на выборке в несколько сотен объектов, неинформативно.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ._kernels import fast_average_precision, fast_roc_auc, fast_sens_spec

__all__ = [
    "MetricCI",
    "bootstrap_metrics",
    "ConfusionCounts",
    "confusion_counts",
    "sensitivity",
    "specificity",
    "precision_at_k",
    "pr_auc",
    "roc_auc",
    "bootstrap_ci",
    "ClassificationReport",
    "evaluate_binary",
]


@dataclass(frozen=True)
class MetricCI:
    """Значение метрики с бутстрэп-интервалом."""

    value: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        if np.isnan(self.value):
            return "н/д"
        return f"{self.value:.3f} [{self.lo:.3f}–{self.hi:.3f}]"


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n_pos(self) -> int:
        return self.tp + self.fn

    @property
    def n_neg(self) -> int:
        return self.tn + self.fp


def confusion_counts(y_true, y_pred) -> ConfusionCounts:
    """Подсчитать элементы матрицы ошибок для бинарной задачи."""
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)
    if yt.shape != yp.shape:
        raise ValueError(f"формы не совпадают: {yt.shape} и {yp.shape}")
    return ConfusionCounts(
        tp=int(np.sum(yt & yp)),
        fp=int(np.sum(~yt & yp)),
        tn=int(np.sum(~yt & ~yp)),
        fn=int(np.sum(yt & ~yp)),
    )


def sensitivity(y_true, y_pred) -> float:
    """Чувствительность (recall, доля выявленных устойчивых изолятов).

    Ключевая метрика GV-Resist: пропуск устойчивости означает назначение
    неработающей схемы — самая дорогая ошибка системы.
    """
    c = confusion_counts(y_true, y_pred)
    return float("nan") if c.n_pos == 0 else c.tp / c.n_pos


def specificity(y_true, y_pred) -> float:
    """Специфичность (доля верно определённых чувствительных изолятов).

    Ложная тревога означает отказ от работающего препарата и переход на
    более токсичную схему.
    """
    c = confusion_counts(y_true, y_pred)
    return float("nan") if c.n_neg == 0 else c.tn / c.n_neg


def precision_at_k(y_true, y_score, k: int) -> float:
    """Точность среди k объектов с наибольшим баллом.

    Операционная метрика GV-Escape: если эпидемиолог готов рассмотреть
    сто наиболее подозрительных замен, сколько из них окажутся реальными.
    """
    yt = np.asarray(y_true).astype(bool)
    ys = np.asarray(y_score, dtype=float)
    if k <= 0:
        raise ValueError("k должно быть положительным")
    k = min(k, ys.size)
    top = np.argsort(-ys, kind="stable")[:k]
    return float(yt[top].mean())


def pr_auc(y_true, y_score) -> float:
    """Площадь под кривой «точность — полнота».

    Предпочтительна перед ROC-AUC при сильном дисбалансе: ROC-AUC остаётся
    высоким даже когда среди предсказанных положительных подавляющее
    большинство ложных, поскольку учитывает долю от большого
    отрицательного класса.
    """
    yt = np.asarray(y_true).astype(int)
    if yt.sum() == 0 or yt.sum() == yt.size:
        return float("nan")
    return float(average_precision_score(yt, np.asarray(y_score, dtype=float)))


def roc_auc(y_true, y_score) -> float:
    """Площадь под ROC-кривой."""
    yt = np.asarray(y_true).astype(int)
    if yt.sum() == 0 or yt.sum() == yt.size:
        return float("nan")
    return float(roc_auc_score(yt, np.asarray(y_score, dtype=float)))


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> MetricCI:
    """Доверительный интервал метрики методом бутстрэпа.

    Требование § 5.8: каждая метрика отчитывается с интервалом.
    Стратифицированная передискретизация сохраняет число положительных
    объектов — иначе при редком классе часть выборок окажется вырожденной.

    Args:
        metric_fn: функция (y_true, y_score) -> float.
        y_true: истинные метки.
        y_score: баллы или предсказания.
        n_boot: число бутстрэп-выборок.
        alpha: уровень значимости (0,05 даёт 95 %-й интервал).
        seed: сид генератора.

    Returns:
        MetricCI с точечной оценкой и границами перцентильного интервала.
    """
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    point = float(metric_fn(yt, ys))

    pos_idx = np.flatnonzero(yt.astype(bool))
    neg_idx = np.flatnonzero(~yt.astype(bool))
    if pos_idx.size == 0 or neg_idx.size == 0:
        return MetricCI(point, float("nan"), float("nan"), int(yt.size))

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        vals[b] = metric_fn(yt[idx], ys[idx])

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return MetricCI(point, float("nan"), float("nan"), int(yt.size))
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return MetricCI(point, float(lo), float(hi), int(yt.size))


def bootstrap_metrics(
    y_true,
    y_score,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, MetricCI]:
    """Интервалы сразу для набора метрик на общих бутстрэп-выборках.

    Отличие от четырёх независимых вызовов `bootstrap_ci` не только в
    скорости. Общие выборки делают интервалы **совместимыми между собой**:
    чувствительность и специфичность в одной строке отчёта посчитаны на
    одних и тех же передискретизациях, поэтому их разброс сопоставим.
    При независимом ресемплинге каждая метрика видела бы свою реальность.

    Побочный эффект — вчетверо меньше операций передискретизации, что и
    было основным источником времени прогона.

    Args:
        y_true: истинные метки.
        y_score: баллы или предсказания.
        metrics: словарь «имя → функция (y_true, y_score) -> float».
        n_boot: число бутстрэп-выборок.
        alpha: уровень значимости.
        seed: сид.

    Returns:
        Словарь «имя → MetricCI».
    """
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    point = {name: float(fn(yt, ys)) for name, fn in metrics.items()}

    pos_idx = np.flatnonzero(yt.astype(bool))
    neg_idx = np.flatnonzero(~yt.astype(bool))
    n = int(yt.size)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {k: MetricCI(v, float("nan"), float("nan"), n) for k, v in point.items()}

    rng = np.random.default_rng(seed)
    # Стратифицированная передискретизация: число положительных объектов
    # сохраняется, иначе при редком классе часть выборок вырождается.
    draws_pos = rng.choice(pos_idx, size=(n_boot, pos_idx.size), replace=True)
    draws_neg = rng.choice(neg_idx, size=(n_boot, neg_idx.size), replace=True)

    values = {name: np.empty(n_boot, dtype=float) for name in metrics}
    for b in range(n_boot):
        idx = np.concatenate([draws_pos[b], draws_neg[b]])
        yb, sb = yt[idx], ys[idx]
        for name, fn in metrics.items():
            values[name][b] = fn(yb, sb)

    out: dict[str, MetricCI] = {}
    for name, vals in values.items():
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            out[name] = MetricCI(point[name], float("nan"), float("nan"), n)
            continue
        lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
        out[name] = MetricCI(point[name], float(lo), float(hi), n)
    return out


@dataclass
class ClassificationReport:
    """Отчёт по одной бинарной задаче (одному препарату)."""

    label: str
    n: int
    n_positive: int
    threshold: float
    sensitivity: MetricCI
    specificity: MetricCI
    pr_auc: MetricCI
    roc_auc: MetricCI
    abstention_rate: float = 0.0

    def meets(self, min_sens: float, min_spec: float) -> bool:
        """Проверить соответствие целевым порогам гипотезы H1."""
        return self.sensitivity.value >= min_sens and self.specificity.value >= min_spec

    def to_row(self) -> str:
        return (
            f"| {self.label} | {self.n} | {self.n_positive} | "
            f"{self.sensitivity} | {self.specificity} | "
            f"{self.pr_auc} | {self.roc_auc} | {self.abstention_rate:.1%} |"
        )

    @staticmethod
    def header() -> str:
        return (
            "| Препарат | N | Устойчивых | Чувствительность | Специфичность | "
            "PR-AUC | ROC-AUC | Отказов |\n"
            "|---|---|---|---|---|---|---|---|"
        )


def evaluate_binary(
    y_true,
    y_score,
    label: str = "",
    threshold: float = 0.5,
    abstained: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> ClassificationReport:
    """Полная оценка бинарной задачи с интервалами.

    Args:
        y_true: истинные метки (0/1).
        y_score: предсказанные вероятности.
        label: имя задачи (например, название препарата).
        threshold: порог перевода вероятности в решение.
        abstained: булев массив «модель отказалась от ответа». Такие
            объекты исключаются из расчёта метрик, но их доля
            отчитывается: метрика, посчитанная по лёгким объектам после
            отказа от трудных, без указания доли отказов вводит в
            заблуждение.
        n_boot: число бутстрэп-выборок.
        seed: сид.

    Returns:
        ClassificationReport.
    """
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    if yt.shape != ys.shape:
        raise ValueError(f"формы не совпадают: {yt.shape} и {ys.shape}")

    abstention_rate = 0.0
    if abstained is not None:
        mask = ~np.asarray(abstained).astype(bool)
        abstention_rate = float(1.0 - mask.mean())
        yt, ys = yt[mask], ys[mask]

    if yt.size == 0:
        raise ValueError("после исключения отказов не осталось объектов")

    cis = bootstrap_metrics(
        yt,
        ys,
        {
            "sensitivity": lambda a, b: fast_sens_spec(a, b >= threshold)[0],
            "specificity": lambda a, b: fast_sens_spec(a, b >= threshold)[1],
            "pr_auc": fast_average_precision,
            "roc_auc": fast_roc_auc,
        },
        n_boot=n_boot,
        seed=seed,
    )

    return ClassificationReport(
        label=label,
        n=int(yt.size),
        n_positive=int(yt.sum()),
        threshold=threshold,
        sensitivity=cis["sensitivity"],
        specificity=cis["specificity"],
        pr_auc=cis["pr_auc"],
        roc_auc=cis["roc_auc"],
        abstention_rate=abstention_rate,
    )
