"""Разделение выборки с учётом родства объектов.

Исправление дефекта версии 1.0, который сам по себе завышает метрики
сильнее большинства прочих: изоляты из одной вспышки почти идентичны и
не являются независимыми наблюдениями. При случайном разделении близкие
родственники попадают одновременно в train и test, и модель на тесте
фактически узнаёт объекты, которые уже видела.

Правило проекта: все члены одного филогенетического кластера попадают
целиком либо в обучающую, либо в тестовую часть.
"""

from __future__ import annotations

import numpy as np

from ..types import Split

__all__ = ["cluster_by_distance", "cluster_split"]


def cluster_by_distance(distances: np.ndarray, threshold: float) -> np.ndarray:
    """Одиночная связь (single linkage) по матрице попарных расстояний.

    Два изолята считаются связанными, если расстояние между ними не
    превышает порог; кластер — связная компонента такого графа. Для
    геномных данных расстояние обычно измеряется в числе различающихся
    позиций (SNP), а порог задаётся эпидемиологически: например, ≤ 5 SNP
    для *M. tuberculosis* трактуется как вероятная недавняя передача.

    Args:
        distances: квадратная симметричная матрица расстояний (n × n).
        threshold: порог связи (включительно).

    Returns:
        Массив длины n с номерами кластеров, пронумерованными с нуля
        в порядке первого появления.

    Raises:
        ValueError: если матрица не квадратная.
    """
    dist = np.asarray(distances, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distances должна быть квадратной матрицей")

    n = dist.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # сжатие пути
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in np.flatnonzero(dist[i, i + 1 :] <= threshold) + i + 1:
            union(i, int(j))

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def cluster_split(
    clusters,
    test_size: float = 0.2,
    val_size: float = 0.0,
    seed: int = 0,
) -> Split:
    """Разделить выборку так, чтобы кластер не пересекал границу частей.

    Целевые доли выдерживаются приближённо: кластеры неделимы, поэтому
    точное попадание в долю невозможно. Используется жадная укладка —
    кластеры перебираются от крупных к мелким, каждый отправляется в ту
    часть, которая сильнее всего недобрала до своей цели.

    Args:
        clusters: метка кластера для каждого объекта.
        test_size: целевая доля тестовой части.
        val_size: целевая доля валидационной части.
        seed: сид для перемешивания кластеров одинакового размера.

    Returns:
        Split со стратегией "cluster".

    Raises:
        ValueError: если доли заданы некорректно или кластеров слишком
            мало, чтобы получить непустые части.
    """
    labels = np.asarray(clusters)
    if labels.ndim != 1:
        raise ValueError("clusters должен быть одномерным массивом")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size должен лежать в (0, 1)")
    if not 0.0 <= val_size < 1.0 or test_size + val_size >= 1.0:
        raise ValueError("некорректные доли: test_size + val_size должно быть < 1")

    uniq, sizes = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        raise ValueError(
            f"кластеров {uniq.size}: разделить без утечки невозможно. "
            "Проверьте порог кластеризации"
        )

    n_total = labels.size
    targets = {
        "train": (1.0 - test_size - val_size) * n_total,
        "test": test_size * n_total,
    }
    if val_size > 0:
        targets["val"] = val_size * n_total

    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(uniq.size), -sizes))

    assigned: dict[str, list[int]] = {k: [] for k in targets}
    filled = dict.fromkeys(targets, 0.0)

    for pos in order:
        cl, size = uniq[pos], float(sizes[pos])
        # Часть с наибольшим относительным дефицитом получает кластер.
        deficit = {k: (targets[k] - filled[k]) / targets[k] for k in targets}
        best = max(deficit, key=lambda k: deficit[k])
        assigned[best].append(cl)
        filled[best] += size

    idx = {k: np.flatnonzero(np.isin(labels, v)) for k, v in assigned.items()}
    for part, arr in idx.items():
        if arr.size == 0:
            raise ValueError(
                f"часть '{part}' пуста: кластеров слишком мало для заданных долей"
            )

    return Split(
        train=idx["train"],
        test=idx["test"],
        val=idx.get("val"),
        strategy="cluster",
        meta={
            "n_clusters": int(uniq.size),
            "target_test_size": test_size,
            "actual_test_size": round(idx["test"].size / n_total, 4),
            "largest_cluster": int(sizes.max()),
        },
    )
