"""Базовые типы и контракты ядра GermoVision.

Здесь определён `Split` — единственный способ, которым в проекте
разрешено разделять выборку. Любая модель принимает `Split`, а не
самостоятельно нарезанные массивы: это гарантирует, что разделение
прошло через проверки `guards.LeakageGuard`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "Fold",
    "Split",
    "LeakageError",
    "SplitContractError",
    "AlertLevel",
    "freeze",
]


class Fold(str, Enum):
    """Часть выборки."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class LeakageError(AssertionError):
    """Обнаружена утечка данных между частями выборки.

    Наследуется от AssertionError намеренно: утечка — это нарушение
    инварианта проекта, а не восстановимая ошибка. Её нельзя
    перехватить и продолжить обучение.
    """


class SplitContractError(ValueError):
    """Нарушен контракт объекта Split (пересечение, пустая часть, дубли)."""


class AlertLevel(str, Enum):
    """Уровни оповещения (§ 5.6 мастер-документа)."""

    WATCH = "watch"          # зелёный: слабый сигнал, действий не требует
    ATTENTION = "attention"  # жёлтый: два и более независимых сигнала
    ALARM = "alarm"          # красный: высокий риск, требуется реакция


def freeze(arr: np.ndarray) -> np.ndarray:
    """Вернуть копию массива, запрещённую к изменению.

    Индексы разделения не должны меняться после создания `Split`:
    молчаливая правка индексов — самый незаметный способ получить утечку.
    """
    out = np.asarray(arr, dtype=np.int64).copy()
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class Split:
    """Разделение выборки на части.

    Хранит позиционные индексы (не метки), поэтому одинаково работает
    с numpy-массивами, pandas-таблицами и списками.

    Args:
        train: индексы обучающей части.
        test: индексы тестовой части.
        val: индексы валидационной части (может отсутствовать).
        calib: индексы калибровочной части. Отдельная часть нужна
            потому, что калибровать вероятности на обучающей выборке
            нельзя — калибровка окажется оптимистичной (§ 5.9).
        strategy: имя стратегии, породившей разделение.
        meta: произвольные метаданные (границы дат, имя страны и т. п.).
    """

    train: np.ndarray
    test: np.ndarray
    val: np.ndarray | None = None
    calib: np.ndarray | None = None
    strategy: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", freeze(self.train))
        object.__setattr__(self, "test", freeze(self.test))
        if self.val is not None:
            object.__setattr__(self, "val", freeze(self.val))
        if self.calib is not None:
            object.__setattr__(self, "calib", freeze(self.calib))
        self._validate()

    def _validate(self) -> None:
        if self.train.size == 0:
            raise SplitContractError("обучающая часть пуста")
        if self.test.size == 0:
            raise SplitContractError("тестовая часть пуста")

        for name, idx in self.parts().items():
            if idx.size != np.unique(idx).size:
                raise SplitContractError(f"дубликаты индексов в части '{name}'")

        names = list(self.parts())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = np.intersect1d(self.parts()[a], self.parts()[b])
                if overlap.size:
                    raise SplitContractError(
                        f"части '{a}' и '{b}' пересекаются "
                        f"({overlap.size} общих индексов, например {overlap[:5].tolist()})"
                    )

    def parts(self) -> dict[str, np.ndarray]:
        """Непустые части разделения в виде словаря."""
        out: dict[str, np.ndarray] = {"train": self.train, "test": self.test}
        if self.val is not None and self.val.size:
            out["val"] = self.val
        if self.calib is not None and self.calib.size:
            out["calib"] = self.calib
        return out

    @property
    def sizes(self) -> dict[str, int]:
        return {k: int(v.size) for k, v in self.parts().items()}

    @property
    def n_total(self) -> int:
        return sum(self.sizes.values())

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}={v}" for k, v in self.sizes.items())
        return f"Split(strategy={self.strategy!r}, {sizes})"
