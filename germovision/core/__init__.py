"""Ядро GermoVision — Часть 1.

Здесь живёт то, что делает результаты проекта научно корректными:
разделение выборки, защита от утечек и метрики. Модели (Части 3–6)
обязаны пользоваться этими средствами и не имеют права нарезать
выборку самостоятельно.
"""

from . import metrics, splitting
from .types import AlertLevel, Fold, LeakageError, Split, SplitContractError

__all__ = [
    "Split",
    "Fold",
    "AlertLevel",
    "LeakageError",
    "SplitContractError",
    "splitting",
    "metrics",
]
