"""Быстрые ядра метрик для внутреннего цикла бутстрэпа.

Профилирование показало, что 63 % времени прогона уходило на бутстрэп, а
внутри него больше половины — на проверку входных данных внутри sklearn:
`type_of_target` и декораторы валидации параметров вызывались десятки
тысяч раз подряд на одних и тех же по форме массивах.

Проверять форму имеет смысл один раз, а не на каждой из 500 выборок.
Здесь метрики посчитаны напрямую в numpy без валидации; корректность
относительно sklearn закреплена тестами `test_kernels_match_sklearn`.

Публичные функции модуля `classification` продолжают вызывать sklearn:
на одиночном вызове его накладные расходы несущественны, а поведение
на краевых случаях лучше проверено.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fast_roc_auc", "fast_average_precision", "fast_sens_spec"]


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC через статистику Манна — Уитни.

    Площадь под ROC-кривой равна вероятности того, что случайно выбранный
    положительный объект получит балл выше случайного отрицательного.
    Это в точности нормированная U-статистика, поэтому достаточно одной
    сортировки. Совпадения баллов обрабатываются средними рангами — так
    же, как это делает sklearn.

    Args:
        y_true: метки 0/1.
        y_score: баллы.

    Returns:
        Значение в [0, 1] либо NaN, если один из классов пуст.
    """
    n = y_true.size
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    s_sorted = y_score[order]

    # Средние ранги для групп одинаковых баллов, без цикла по группам:
    # накопленная сумма флагов «начался новый блок» даёт номер блока,
    # а границы блоков задают средний ранг внутри каждого.
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(s_sorted[1:], s_sorted[:-1], out=is_new[1:])

    group_start = np.flatnonzero(is_new)
    group_end = np.append(group_start[1:], n)
    group_rank = 0.5 * (group_start + group_end + 1)  # нумерация рангов с 1
    ranks = group_rank[np.cumsum(is_new) - 1]

    rank_sum = float(ranks[y_true[order] == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fast_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Средняя точность — площадь под кривой «точность — полнота».

    Считается тем же способом, что и `sklearn.metrics.average_precision_score`:
    суммой приращений полноты, взвешенных точностью в соответствующей точке.
    Интерполяция не применяется — это важно, поскольку интерполированная
    версия систематически завышает оценку при редком классе.
    """
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == y_true.size:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order].astype(np.float64)
    s = y_score[order]

    # Пороги ставятся только там, где балл меняется: объекты с одинаковым
    # баллом неразличимы и должны учитываться одной точкой кривой.
    distinct = np.flatnonzero(np.diff(s))
    idx = np.concatenate([distinct, [y.size - 1]])

    tps = np.cumsum(y)[idx]
    fps = 1.0 + idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def fast_sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Чувствительность и специфичность одним проходом.

    Args:
        y_true: метки 0/1.
        y_pred: булевы решения.

    Returns:
        Пара (чувствительность, специфичность); NaN, если класс пуст.
    """
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = y_true.size - n_pos
    sens = float(y_pred[pos].sum()) / n_pos if n_pos else float("nan")
    spec = float((~y_pred[~pos]).sum()) / n_neg if n_neg else float("nan")
    return sens, spec
