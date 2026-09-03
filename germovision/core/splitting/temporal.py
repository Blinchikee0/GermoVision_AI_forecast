"""Временнóе разделение выборки.

Исправление дефекта D11 версии 1.0: случайное перемешивание выборки
означает, что модель при обучении видит будущее, и метрики завышаются.
Единственный корректный протокол для эпидемиологических данных —
обучение на данных до даты T, проверка на данных после T.

Важно, какую дату использовать. Для измерения упреждения (Lead Time)
берётся `submission_date` — дата, когда запись стала доступна, а не
`collection_date` — дата взятия образца. Образец, собранный 1 ноября,
но депонированный 20 ноября, физически не был доступен системе
5 ноября; использование даты сбора создало бы скрытую утечку.
"""

from __future__ import annotations

import numpy as np

from ..types import Split

__all__ = ["temporal_split", "forward_chaining"]


def _as_datetime(dates) -> np.ndarray:
    arr = np.asarray(dates)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]")
    return arr.astype("datetime64[D]")


def temporal_split(
    dates,
    train_end,
    test_start=None,
    val_end=None,
    calib_fraction: float = 0.0,
    seed: int = 0,
) -> Split:
    """Разделить выборку по времени одной отсечкой.

    Args:
        dates: даты доступности записей (`submission_date`).
        train_end: последняя дата, включаемая в обучение (включительно).
        test_start: первая дата теста. По умолчанию — сразу после
            `val_end` либо `train_end`. Явное указание позволяет
            оставить зазор (embargo) между обучением и тестом.
        val_end: если задано, интервал (train_end, val_end] становится
            валидационной частью.
        calib_fraction: доля обучающей части, отводимая под калибровку
            вероятностей. Калибровка на обучающих данных даёт
            оптимистичный результат, поэтому нужна отдельная часть.
        seed: сид для выбора калибровочной подвыборки.

    Returns:
        Split с частями train / val / calib / test.

    Raises:
        ValueError: если границы заданы в неверном порядке.
    """
    d = _as_datetime(dates)
    t_end = np.datetime64(train_end, "D")
    v_end = np.datetime64(val_end, "D") if val_end is not None else None

    if v_end is not None and v_end <= t_end:
        raise ValueError("val_end должен быть строго позже train_end")

    boundary = v_end if v_end is not None else t_end
    t_start = np.datetime64(test_start, "D") if test_start is not None else boundary
    if t_start < boundary:
        raise ValueError("test_start не может быть раньше конца обучения/валидации")

    train_idx = np.flatnonzero(d <= t_end)
    val_idx = (
        np.flatnonzero((d > t_end) & (d <= v_end)) if v_end is not None else None
    )
    test_idx = np.flatnonzero(d > (t_start if test_start is not None else boundary))

    calib_idx = None
    if calib_fraction > 0.0:
        if not 0.0 < calib_fraction < 1.0:
            raise ValueError("calib_fraction должен лежать в (0, 1)")
        rng = np.random.default_rng(seed)
        n_calib = max(1, int(round(train_idx.size * calib_fraction)))
        if n_calib >= train_idx.size:
            raise ValueError("calib_fraction забирает всю обучающую часть")
        # Калибровочная часть берётся из хвоста обучающего периода:
        # она должна быть ближе по времени к тесту, чем основное обучение.
        tail = train_idx[np.argsort(d[train_idx], kind="stable")][-n_calib:]
        calib_idx = np.sort(tail)
        train_idx = np.setdiff1d(train_idx, calib_idx)
        del rng

    return Split(
        train=train_idx,
        val=val_idx,
        calib=calib_idx,
        test=test_idx,
        strategy="temporal",
        meta={
            "train_end": str(t_end),
            "val_end": str(v_end) if v_end is not None else None,
            "test_start": str(t_start),
            "date_semantics": "submission_date",
        },
    )


def forward_chaining(
    dates,
    initial_train_end,
    horizon_days: int,
    step_days: int,
    n_folds: int,
    embargo_days: int = 0,
) -> list[Split]:
    """Скользящая проверка вперёд (blocked forward chaining).

    Обучение на данных до недели k, прогноз на неделю k+h, сдвиг окна,
    повтор. Метрика усредняется по всем сдвигам — одна отсечка даёт
    оценку, зависящую от случайности выбора даты.

    Args:
        dates: даты доступности записей.
        initial_train_end: конец обучения на первом шаге.
        horizon_days: длина тестового окна в днях.
        step_days: сдвиг окна между шагами.
        n_folds: число шагов.
        embargo_days: зазор между обучением и тестом. Нужен, когда
            запись может обновиться задним числом.

    Returns:
        Список Split; шаги, где обучающая или тестовая часть оказалась
        пустой, пропускаются.
    """
    if horizon_days <= 0 or step_days <= 0 or n_folds <= 0:
        raise ValueError("horizon_days, step_days и n_folds должны быть > 0")

    d = _as_datetime(dates)
    base = np.datetime64(initial_train_end, "D")
    splits: list[Split] = []

    for k in range(n_folds):
        train_end = base + np.timedelta64(k * step_days, "D")
        test_start = train_end + np.timedelta64(embargo_days, "D")
        test_end = test_start + np.timedelta64(horizon_days, "D")

        train_idx = np.flatnonzero(d <= train_end)
        test_idx = np.flatnonzero((d > test_start) & (d <= test_end))
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        splits.append(
            Split(
                train=train_idx,
                test=test_idx,
                strategy="forward_chaining",
                meta={
                    "fold": k,
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "embargo_days": embargo_days,
                    "date_semantics": "submission_date",
                },
            )
        )

    if not splits:
        raise ValueError(
            "ни один шаг не дал непустых частей — проверьте initial_train_end и горизонт"
        )
    return splits
