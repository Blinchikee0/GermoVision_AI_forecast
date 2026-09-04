"""Базовые модели для сравнения.

Исправление дефекта D12: версия 1.0 не приводила ни одной модели
сравнения. Без базы абсолютное значение метрики не интерпретируемо —
если простое правило даёт AUC 0,93, а сложная модель 0,94, то сложность
не оправдана.

Главная база проекта — не «случайное угадывание», а **действующий
стандарт**: применение каталога мутаций ВОЗ. Утверждение «наша модель
полезна» имеет смысл только если она превосходит то, чем пользуются уже
сегодня.
"""

from __future__ import annotations

import numpy as np

from ..core.metrics.classification import ClassificationReport, evaluate_binary
from ..data.catalogue import DRUG_NAMES_RU, MutationCatalogue
from ..data.schema import IsolateDataset

__all__ = ["CatalogueBaseline", "PrevalenceBaseline"]


class CatalogueBaseline:
    """Действующий стандарт: правила каталога мутаций ВОЗ.

    Обучения не требует — каталог задан извне. Метод `fit` присутствует
    только для единообразия интерфейса.

    Выдаёт 1 при наличии маркера группы 1–2 и 0 иначе. Именно так
    работают применяемые сегодня инструменты интерпретации (TB-Profiler
    и аналоги), поэтому это честная точка отсчёта.
    """

    def __init__(self, drug: str, catalogue: MutationCatalogue | None = None) -> None:
        self.drug = drug
        self.catalogue = catalogue or MutationCatalogue()

    def fit(self, ds: IsolateDataset, split=None) -> CatalogueBaseline:  # noqa: ARG002
        return self

    def predict_proba(self, ds: IsolateDataset, idx: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(len(ds)) if idx is None else np.asarray(idx, dtype=int)
        markers = self.catalogue.resistance_markers(self.drug)
        return np.array([1.0 if ds.mutations[i] & markers else 0.0 for i in rows])

    def evaluate(
        self, ds: IsolateDataset, idx: np.ndarray, n_boot: int = 500
    ) -> ClassificationReport:
        y_all = ds.phenotypes[self.drug]
        eval_idx = np.array([i for i in idx if not np.isnan(y_all[i])], dtype=int)
        if eval_idx.size == 0:
            raise ValueError(f"{self.drug}: нет измеренных фенотипов в выборке")
        return evaluate_binary(
            y_all[eval_idx].astype(int),
            self.predict_proba(ds, eval_idx),
            label=f"{DRUG_NAMES_RU.get(self.drug, self.drug)} (каталог ВОЗ)",
            threshold=0.5,
            n_boot=n_boot,
        )


class PrevalenceBaseline:
    """Постоянный ответ, равный доле устойчивых в обучении.

    Нужна как проверка на вырожденность: если основная модель не
    превосходит эту базу, значит она не выучила ничего, кроме частоты
    класса. Показывает также, почему accuracy непригодна как метрика —
    при доле устойчивых 5 % эта модель даёт 95 % точности и нулевую пользу.
    """

    def __init__(self, drug: str) -> None:
        self.drug = drug
        self.rate_: float = float("nan")

    def fit(self, ds: IsolateDataset, split) -> PrevalenceBaseline:
        y = ds.phenotypes[self.drug][split.train]
        y = y[~np.isnan(y)]
        if y.size == 0:
            raise ValueError(f"{self.drug}: в обучении нет измеренных фенотипов")
        self.rate_ = float(y.mean())
        return self

    def predict_proba(self, ds: IsolateDataset, idx: np.ndarray | None = None) -> np.ndarray:
        n = len(ds) if idx is None else len(np.asarray(idx))
        return np.full(n, self.rate_, dtype=float)
