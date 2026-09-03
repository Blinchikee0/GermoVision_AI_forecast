"""Тесты разделения выборки."""

from __future__ import annotations

import numpy as np
import pytest

from germovision.core.splitting import (
    cluster_by_distance,
    cluster_split,
    forward_chaining,
    holdout_group,
    leave_one_group_out,
    temporal_split,
)
from germovision.core.types import Split, SplitContractError

# --------------------------------------------------------------------------
# Контракт Split
# --------------------------------------------------------------------------


def test_split_rejects_overlap():
    with pytest.raises(SplitContractError, match="пересекаются"):
        Split(train=np.array([0, 1, 2]), test=np.array([2, 3]))


def test_split_rejects_duplicates_inside_part():
    with pytest.raises(SplitContractError, match="дубликаты"):
        Split(train=np.array([0, 1, 1]), test=np.array([2, 3]))


def test_split_rejects_empty_test():
    with pytest.raises(SplitContractError, match="тестовая часть пуста"):
        Split(train=np.array([0, 1]), test=np.array([], dtype=int))


def test_split_indices_are_immutable():
    """Индексы нельзя менять после создания — молчаливая правка = утечка."""
    split = Split(train=np.array([0, 1]), test=np.array([2, 3]))
    with pytest.raises(ValueError):
        split.train[0] = 99


# --------------------------------------------------------------------------
# Временнóе разделение
# --------------------------------------------------------------------------


@pytest.fixture
def dates_90():
    return np.array(
        [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(90)]
    )


def test_temporal_split_respects_boundary(dates_90):
    split = temporal_split(dates_90, train_end="2021-02-01")
    assert dates_90[split.train].max() <= np.datetime64("2021-02-01")
    assert dates_90[split.test].min() > np.datetime64("2021-02-01")
    assert split.n_total == 90


def test_temporal_split_with_validation(dates_90):
    split = temporal_split(dates_90, train_end="2021-02-01", val_end="2021-02-15")
    assert dates_90[split.train].max() <= np.datetime64("2021-02-01")
    assert dates_90[split.val].min() > np.datetime64("2021-02-01")
    assert dates_90[split.val].max() <= np.datetime64("2021-02-15")
    assert dates_90[split.test].min() > np.datetime64("2021-02-15")


def test_temporal_split_calibration_part_is_disjoint(dates_90):
    """Калибровочная часть отделена от обучения (§ 5.9)."""
    split = temporal_split(dates_90, train_end="2021-02-20", calib_fraction=0.2)
    assert split.calib is not None
    assert np.intersect1d(split.train, split.calib).size == 0
    # Калибровка берётся из хвоста — ближе по времени к тесту.
    assert dates_90[split.calib].min() >= dates_90[split.train].max() - np.timedelta64(1, "D")


def test_temporal_split_rejects_bad_boundaries(dates_90):
    with pytest.raises(ValueError, match="строго позже"):
        temporal_split(dates_90, train_end="2021-02-01", val_end="2021-01-15")


def test_forward_chaining_produces_ordered_folds(dates_90):
    splits = forward_chaining(
        dates_90, initial_train_end="2021-01-20", horizon_days=7, step_days=7, n_folds=5
    )
    assert len(splits) == 5
    prev_train = 0
    for s in splits:
        assert dates_90[s.train].max() < dates_90[s.test].min()
        assert s.train.size > prev_train  # обучающее окно растёт
        prev_train = s.train.size


def test_forward_chaining_embargo_creates_gap(dates_90):
    splits = forward_chaining(
        dates_90,
        initial_train_end="2021-01-20",
        horizon_days=7,
        step_days=7,
        n_folds=3,
        embargo_days=5,
    )
    for s in splits:
        gap = (dates_90[s.test].min() - dates_90[s.train].max()).astype(int)
        assert gap > 5


# --------------------------------------------------------------------------
# Разделение по кластерам родства
# --------------------------------------------------------------------------


def test_cluster_by_distance_finds_components():
    """Три изолята: два близких (3 SNP) и один далёкий (47 SNP)."""
    dist = np.array(
        [
            [0, 3, 47],
            [3, 0, 45],
            [47, 45, 0],
        ],
        dtype=float,
    )
    labels = cluster_by_distance(dist, threshold=5)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_cluster_by_distance_single_linkage_chains():
    """Одиночная связь объединяет цепочку A-B-C даже если A и C далеки."""
    dist = np.array(
        [
            [0, 4, 8],
            [4, 0, 4],
            [8, 4, 0],
        ],
        dtype=float,
    )
    assert len(np.unique(cluster_by_distance(dist, threshold=5))) == 1


def test_cluster_split_keeps_clusters_intact():
    clusters = np.repeat(np.arange(20), 5)  # 20 кластеров по 5 объектов
    split = cluster_split(clusters, test_size=0.3, seed=1)
    train_cl = set(clusters[split.train].tolist())
    test_cl = set(clusters[split.test].tolist())
    assert train_cl & test_cl == set()
    assert split.n_total == clusters.size


def test_cluster_split_approximates_target_proportion():
    clusters = np.repeat(np.arange(50), 4)
    split = cluster_split(clusters, test_size=0.25, seed=0)
    actual = split.test.size / clusters.size
    assert abs(actual - 0.25) < 0.1


def test_cluster_split_rejects_single_cluster():
    with pytest.raises(ValueError, match="разделить без утечки невозможно"):
        cluster_split(np.zeros(100, dtype=int), test_size=0.2)


# --------------------------------------------------------------------------
# Внешняя валидация по группам
# --------------------------------------------------------------------------


def test_holdout_group_excludes_country_entirely():
    groups = np.array(["KZ"] * 10 + ["UK"] * 20 + ["IN"] * 15)
    split = holdout_group(groups, "KZ")
    assert set(groups[split.test].tolist()) == {"KZ"}
    assert "KZ" not in set(groups[split.train].tolist())
    assert split.test.size == 10


def test_holdout_group_rejects_unknown_group():
    with pytest.raises(ValueError, match="отсутствуют в данных"):
        holdout_group(np.array(["UK", "IN"]), "KZ")


def test_holdout_group_rejects_test_val_clash():
    groups = np.array(["KZ"] * 5 + ["UK"] * 5 + ["IN"] * 5)
    with pytest.raises(ValueError, match="одновременно в тесте и валидации"):
        holdout_group(groups, "KZ", val_groups=["KZ", "UK"])


def test_leave_one_group_out_covers_every_group():
    groups = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    splits = leave_one_group_out(groups)
    assert len(splits) == 3
    held = {s.meta["held_out"][0] for s in splits}
    assert held == {"A", "B", "C"}


def test_leave_one_group_out_skips_tiny_groups():
    groups = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 2)
    splits = leave_one_group_out(groups, min_size=5)
    assert len(splits) == 2
