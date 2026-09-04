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

__all__ = ["cluster_by_distance", "cluster_split", "temporal_cluster_split"]


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


def temporal_cluster_split(
    dates,
    clusters,
    test_size: float = 0.2,
    calib_size: float = 0.15,
) -> Split:
    """Временнóе разделение, не разрывающее кластеры родства.

    Правила 1 и 2 протокола валидации конфликтуют напрямую: отсечка по
    календарной дате рассекает кластер, часть членов которого
    депонирована до неё, а часть — после. Соблюсти оба правила
    одновременно можно, только сделав неделимой единицей **кластер, а не
    изолят**.

    Кластеры упорядочиваются по дате первого появления и нарезаются по
    накопленной доле изолятов. Обучающая часть оказывается строго раньше
    калибровочной, а та — раньше тестовой, при этом ни один кластер не
    пересекает границу.

    Побочный эффект, о котором нужно знать: границы частей смещаются
    относительно ровных календарных дат — ровно на длину кластеров,
    пересекавших отсечку. Фактические границы возвращаются в `meta`.

    Args:
        dates: даты доступности записей (`submission_date`).
        clusters: метки кластеров родства.
        test_size: целевая доля тестовой части.
        calib_size: целевая доля калибровочной части. Ноль отключает её,
            но тогда становятся недоступны калибровка вероятностей и
            конформный отказ от ответа.

    Returns:
        Split со стратегией "temporal_cluster".

    Raises:
        ValueError: если доли некорректны или кластеров слишком мало.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    labels = np.asarray(clusters)
    if d.size != labels.size:
        raise ValueError("длины dates и clusters должны совпадать")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size должен лежать в (0, 1)")
    if not 0.0 <= calib_size < 1.0 or test_size + calib_size >= 1.0:
        raise ValueError("test_size + calib_size должно быть < 1")

    uniq = np.unique(labels)
    if uniq.size < 3:
        raise ValueError(
            f"кластеров {uniq.size}: разделить на три части без утечки невозможно"
        )

    # Кластер датируется по своему самому раннему изоляту: именно тогда
    # он впервые стал наблюдаемым.
    first_seen = np.array([d[labels == c].min() for c in uniq])
    order = np.argsort(first_seen, kind="stable")

    sizes = np.array([int((labels == c).sum()) for c in uniq])
    cum = np.cumsum(sizes[order]) / labels.size

    train_end_frac = 1.0 - test_size - calib_size
    calib_end_frac = 1.0 - test_size

    train_clusters = uniq[order][cum <= train_end_frac]
    calib_clusters = uniq[order][(cum > train_end_frac) & (cum <= calib_end_frac)]
    test_clusters = uniq[order][cum > calib_end_frac]

    if train_clusters.size == 0 or test_clusters.size == 0:
        raise ValueError("после нарезки по кластерам одна из частей пуста")

    train_idx = np.flatnonzero(np.isin(labels, train_clusters))
    calib_idx = np.flatnonzero(np.isin(labels, calib_clusters)) if calib_clusters.size else None
    test_idx = np.flatnonzero(np.isin(labels, test_clusters))

    return Split(
        train=train_idx,
        calib=calib_idx,
        test=test_idx,
        strategy="temporal_cluster",
        meta={
            "n_clusters": int(uniq.size),
            "train_last_date": str(d[train_idx].max()),
            "test_first_date": str(d[test_idx].min()),
            "actual_test_size": round(test_idx.size / labels.size, 4),
            "date_semantics": "submission_date",
            "note": (
                "границы смещены относительно календарных дат, поскольку "
                "неделимой единицей является кластер родства"
            ),
        },
    )
