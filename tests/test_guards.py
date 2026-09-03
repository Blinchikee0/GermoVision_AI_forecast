"""Тесты защиты от утечки данных.

Центральный тест файла — `test_v1_augmentation_leak_is_caught`: он
воспроизводит ошибку версии 1.0 (аугментация до разделения выборки) и
проверяет, что защита её обнаруживает. Если этот тест начнёт проходить
без срабатывания защиты, значит защита сломана, и всем метрикам проекта
больше нельзя верить.
"""

from __future__ import annotations

import numpy as np
import pytest

from germovision.core.splitting import (
    LeakageGuard,
    augment_train_only,
    check_cluster_integrity,
    check_no_exact_duplicates,
    check_no_near_duplicates,
    check_temporal_order,
    cluster_split,
    temporal_split,
)
from germovision.core.types import LeakageError, Split


@pytest.fixture
def dates_100():
    return np.array(
        [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(100)]
    )


# --------------------------------------------------------------------------
# Временнáя проверка
# --------------------------------------------------------------------------


def test_temporal_order_passes_on_clean_split(dates_100):
    split = temporal_split(dates_100, train_end="2021-02-15")
    assert "зазор" in check_temporal_order(split, dates_100)


def test_temporal_order_catches_random_split(dates_100):
    """Случайное перемешивание — дефект D11 — обнаруживается."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(100)
    bad = Split(train=idx[:70], test=idx[70:], strategy="random")
    with pytest.raises(LeakageError, match="утечка из будущего"):
        check_temporal_order(bad, dates_100)


def test_temporal_order_enforces_embargo(dates_100):
    split = temporal_split(dates_100, train_end="2021-02-15")
    with pytest.raises(LeakageError, match="меньше требуемого embargo"):
        check_temporal_order(split, dates_100, embargo_days=30)


# --------------------------------------------------------------------------
# Проверка кластеров родства
# --------------------------------------------------------------------------


def test_cluster_integrity_passes_on_cluster_split():
    clusters = np.repeat(np.arange(20), 5)
    split = cluster_split(clusters, test_size=0.3, seed=0)
    assert "пересечений нет" in check_cluster_integrity(split, clusters)


def test_cluster_integrity_catches_split_family():
    """Разрез посреди кластера родства обнаруживается.

    Кластеры идут по пять объектов подряд, поэтому граница на индексе 72
    рассекает кластер 14: изоляты 70–71 попадают в обучение, 72–74 — в тест.
    """
    clusters = np.repeat(np.arange(20), 5)
    bad = Split(train=np.arange(72), test=np.arange(72, 100), strategy="random")
    with pytest.raises(LeakageError, match="кластеров родства пересекают границу"):
        check_cluster_integrity(bad, clusters)


# --------------------------------------------------------------------------
# ГЛАВНЫЙ ТЕСТ: утечка через аугментацию (дефект D9)
# --------------------------------------------------------------------------


def _make_dataset(n: int = 300, d: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = (x[:, 0] + 0.5 * rng.normal(size=n) > 1.2).astype(int)  # ~10 % положительных
    return x, y


def test_v1_augmentation_leak_is_caught():
    """Воспроизведение ошибки версии 1.0 и проверка, что она обнаружена.

    Версия 1.0: «увеличил долю целевых данных до 20 % и добавил
    контролируемый гауссовский шум», после чего «данные делятся на
    выборки». При таком порядке зашумлённые копии одной записи попадают
    и в train, и в test.
    """
    x, y = _make_dataset()
    rng = np.random.default_rng(42)

    # --- НЕПРАВИЛЬНО: сначала аугментация, потом разделение ---------------
    pos = np.flatnonzero(y == 1)
    dup = rng.choice(pos, size=200, replace=True)
    x_aug = np.concatenate([x, x[dup] + rng.normal(scale=0.01, size=(200, x.shape[1]))])

    order = rng.permutation(x_aug.shape[0])
    leaky = Split(train=order[:400], test=order[400:], strategy="v1_broken")

    with pytest.raises(LeakageError, match="аугментация выполнена до разделения"):
        check_no_near_duplicates(leaky, x_aug)


def test_correct_order_passes():
    """Тот же объём аугментации, но выполненный после разделения, проходит."""
    x, y = _make_dataset()
    clusters = np.arange(x.shape[0])  # каждый объект независим
    split = cluster_split(clusters, test_size=0.3, seed=0)

    def augment(x_tr, y_tr, rng):
        pos = np.flatnonzero(y_tr == 1)
        if pos.size == 0:
            return np.empty((0, x_tr.shape[1])), np.empty(0, dtype=int)
        dup = rng.choice(pos, size=200, replace=True)
        noise = rng.normal(scale=0.01, size=(200, x_tr.shape[1]))
        return x_tr[dup] + noise, y_tr[dup]

    x_tr, y_tr = augment_train_only(split, x, y, augment, seed=42)

    assert x_tr.shape[0] == split.train.size + 200
    assert y_tr.shape[0] == x_tr.shape[0]
    # Тестовая часть не тронута и остаётся чистой.
    check_no_near_duplicates(split, x)


def test_augment_train_only_never_touches_test():
    """API структурно не позволяет аугментировать тест."""
    x, y = _make_dataset(n=100)
    split = cluster_split(np.arange(100), test_size=0.3, seed=0)

    def augment(x_tr, y_tr, rng):
        return x_tr[:10] + 0.001, y_tr[:10]

    x_tr, _ = augment_train_only(split, x, y, augment)
    # Возвращается только обучающая часть; тестовую функция не отдаёт вовсе.
    assert x_tr.shape[0] == split.train.size + 10


def test_augment_train_only_validates_shapes():
    x, y = _make_dataset(n=60)
    split = cluster_split(np.arange(60), test_size=0.3, seed=0)

    def bad_augment(x_tr, y_tr, rng):
        return x_tr[:5], y_tr[:3]  # рассогласование

    with pytest.raises(ValueError, match="объектов и"):
        augment_train_only(split, x, y, bad_augment)


# --------------------------------------------------------------------------
# Точные дубликаты
# --------------------------------------------------------------------------


def test_exact_duplicates_caught():
    x = np.arange(40, dtype=float).reshape(20, 2)
    x[15] = x[3]  # запись продублирована
    bad = Split(train=np.arange(10), test=np.arange(10, 20))
    with pytest.raises(LeakageError, match="побитово совпадают"):
        check_no_exact_duplicates(bad, x)


def test_exact_duplicates_pass_when_clean():
    x = np.arange(40, dtype=float).reshape(20, 2)
    split = Split(train=np.arange(10), test=np.arange(10, 20))
    assert "совпадений нет" in check_no_exact_duplicates(split, x)


# --------------------------------------------------------------------------
# Сводная проверка
# --------------------------------------------------------------------------


def test_guard_runs_all_checks(dates_100):
    x, y = _make_dataset(n=100)
    clusters = np.arange(100)
    split = temporal_split(dates_100, train_end="2021-02-20")

    guard = LeakageGuard(dates=dates_100, clusters=clusters, x=x)
    report = guard.run(split)

    assert report.passed
    assert "temporal_order" in report.checks
    assert "cluster_integrity" in report.checks
    assert "exact_duplicates" in report.checks
    assert "near_duplicates" in report.checks


def test_guard_records_skipped_checks(dates_100):
    """Пропуск проверки фиксируется — «не проверяли» ≠ «всё хорошо»."""
    split = temporal_split(dates_100, train_end="2021-02-20")
    report = LeakageGuard(dates=dates_100).run(split)
    assert "ПРОПУЩЕНО" in report.checks
    assert "cluster_integrity" in report.checks["ПРОПУЩЕНО"]


def test_guard_collects_all_failures_without_raising(dates_100):
    rng = np.random.default_rng(0)
    idx = rng.permutation(100)
    bad = Split(train=idx[:70], test=idx[70:], strategy="random")
    clusters = np.repeat(np.arange(20), 5)

    report = LeakageGuard(dates=dates_100, clusters=clusters).run(
        bad, raise_on_fail=False
    )
    assert not report.passed
    assert "temporal_order" in report.failures
    assert "cluster_integrity" in report.failures
    assert "ОБНАРУЖЕНА УТЕЧКА" in str(report)
