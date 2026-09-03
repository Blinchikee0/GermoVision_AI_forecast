"""Внешняя валидация: обучение на одних группах, проверка на другой.

Реализует правило 3 протокола валидации: модель обучается на изолятах
из 22 стран и проверяется на 23-й, отсутствовавшей в обучении. Это
воспроизводит реальный сценарий развёртывания в Казахстане, данных из
которого в обучающей выборке нет.

Метрика на внешней валидации почти всегда ниже внутренней. Разрыв между
ними — и есть честная оценка того, насколько модель обобщается.
"""

from __future__ import annotations

import numpy as np

from ..types import Split

__all__ = ["leave_one_group_out", "holdout_group"]


def holdout_group(groups, held_out, val_groups=None) -> Split:
    """Отложить одну группу (например, страну) целиком в тест.

    Args:
        groups: метка группы для каждого объекта.
        held_out: группа (или список групп), уходящая в тест.
        val_groups: группы, уходящие в валидацию. Не должны пересекаться
            с held_out.

    Returns:
        Split со стратегией "leave_group_out".

    Raises:
        ValueError: если группа не найдена или пересекается с валидацией.
    """
    g = np.asarray(groups)
    test_groups = np.atleast_1d(np.asarray(held_out))

    missing = set(test_groups.tolist()) - set(np.unique(g).tolist())
    if missing:
        raise ValueError(f"группы отсутствуют в данных: {sorted(missing)}")

    val_idx = None
    val_mask = np.zeros(g.size, dtype=bool)
    if val_groups is not None:
        vg = np.atleast_1d(np.asarray(val_groups))
        clash = set(vg.tolist()) & set(test_groups.tolist())
        if clash:
            raise ValueError(f"группы одновременно в тесте и валидации: {sorted(clash)}")
        val_mask = np.isin(g, vg)
        val_idx = np.flatnonzero(val_mask)

    test_mask = np.isin(g, test_groups)
    train_idx = np.flatnonzero(~test_mask & ~val_mask)
    test_idx = np.flatnonzero(test_mask)

    return Split(
        train=train_idx,
        test=test_idx,
        val=val_idx,
        strategy="leave_group_out",
        meta={
            "held_out": test_groups.tolist(),
            "n_train_groups": int(np.unique(g[train_idx]).size),
        },
    )


def leave_one_group_out(groups, min_size: int = 1) -> list[Split]:
    """Породить по одному разделению на каждую группу.

    Args:
        groups: метка группы для каждого объекта.
        min_size: группы меньше этого размера пропускаются — оценка
            метрики по трём объектам не несёт информации.

    Returns:
        Список Split, по одному на подходящую группу.

    Raises:
        ValueError: если групп меньше двух.
    """
    g = np.asarray(groups)
    uniq, counts = np.unique(g, return_counts=True)
    if uniq.size < 2:
        raise ValueError("нужно минимум две группы для внешней валидации")

    splits: list[Split] = []
    for name, size in zip(uniq, counts, strict=True):
        if size < min_size:
            continue
        splits.append(holdout_group(g, name))

    if not splits:
        raise ValueError(f"ни одна группа не достигает min_size={min_size}")
    return splits
