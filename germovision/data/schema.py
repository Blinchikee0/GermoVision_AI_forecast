"""Модель данных: набор клинических изолятов с генотипом и фенотипом.

Единая структура, к которой приводятся все источники — CRyPTIC, локальные
данные лаборатории, синтетический генератор. Модели работают только с ней
и ничего не знают о происхождении данных.

Поля разделения дат (`collection_date` и `submission_date`) хранятся
раздельно намеренно: для измерения упреждения нужна дата, когда запись
стала доступна системе, а не дата взятия образца (см. § 5.7, правило 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .catalogue import DRUGS

__all__ = ["IsolateDataset"]


@dataclass
class IsolateDataset:
    """Набор изолятов.

    Args:
        isolate_ids: идентификаторы (n,).
        mutations: для каждого изолята — множество ключей вариантов вида
            `rpoB_S450L`.
        phenotypes: словарь «препарат → массив (n,)» со значениями 1
            (устойчив), 0 (чувствителен) и NaN (не тестировался). Пропуски
            реальны: в CRyPTIC не для каждого изолята измерены все 13 МИК.
        lineages: линия *M. tuberculosis* (L1–L4, Beijing и т. д.).
        countries: страна происхождения — используется для внешней
            валидации leave-one-country-out.
        collection_dates: даты взятия образца.
        submission_dates: даты появления записи в базе.
        clusters: идентификаторы кластеров родства. Все члены кластера
            обязаны попадать в одну часть выборки.
        meta: происхождение и параметры источника.
    """

    isolate_ids: np.ndarray
    mutations: list[set[str]]
    phenotypes: dict[str, np.ndarray]
    lineages: np.ndarray
    countries: np.ndarray
    collection_dates: np.ndarray
    submission_dates: np.ndarray
    clusters: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.isolate_ids)
        lengths = {
            "mutations": len(self.mutations),
            "lineages": len(self.lineages),
            "countries": len(self.countries),
            "collection_dates": len(self.collection_dates),
            "submission_dates": len(self.submission_dates),
            "clusters": len(self.clusters),
        }
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise ValueError(f"field lengths do not match the isolate count ({n}): {bad}")

        for drug, arr in self.phenotypes.items():
            if len(arr) != n:
                raise ValueError(f"phenotype '{drug}' has length {len(arr)}, expected {n}")

        late = np.asarray(self.submission_dates) < np.asarray(self.collection_dates)
        if late.any():
            raise ValueError(
                f"{int(late.sum())} records were deposited before the collection date — "
                "check the source; this invalidates any lead-time measurement"
            )

    def __len__(self) -> int:
        return len(self.isolate_ids)

    @property
    def drugs(self) -> tuple[str, ...]:
        """Препараты, для которых есть хотя бы одно измерение."""
        return tuple(d for d in DRUGS if d in self.phenotypes)

    def all_mutation_keys(self) -> list[str]:
        """Отсортированный список всех встреченных вариантов."""
        return sorted({m for muts in self.mutations for m in muts})

    def labelled_mask(self, drug: str) -> np.ndarray:
        """Маска изолятов, у которых измерен фенотип к препарату."""
        if drug not in self.phenotypes:
            raise KeyError(f"no phenotype for drug '{drug}'")
        return ~np.isnan(self.phenotypes[drug])

    def resistance_rate(self, drug: str) -> float:
        """Доля устойчивых среди протестированных."""
        mask = self.labelled_mask(drug)
        return float(np.nan) if not mask.any() else float(self.phenotypes[drug][mask].mean())

    def subset(self, idx: np.ndarray) -> IsolateDataset:
        """Взять подмножество изолятов по позиционным индексам."""
        idx = np.asarray(idx, dtype=int)
        return IsolateDataset(
            isolate_ids=self.isolate_ids[idx],
            mutations=[self.mutations[i] for i in idx],
            phenotypes={d: a[idx] for d, a in self.phenotypes.items()},
            lineages=self.lineages[idx],
            countries=self.countries[idx],
            collection_dates=self.collection_dates[idx],
            submission_dates=self.submission_dates[idx],
            clusters=self.clusters[idx],
            meta={**self.meta, "subset_of": self.meta.get("source", "unknown")},
        )

    def summary(self) -> str:
        """Сводка для Data Card и логов обучения."""
        lines = [
            f"Isolates:   {len(self)}",
            f"Countries:  {len(np.unique(self.countries))}",
            f"Lineages:   {len(np.unique(self.lineages))}",
            f"Clusters:   {len(np.unique(self.clusters))}",
            f"Variants:   {len(self.all_mutation_keys())}",
            f"Period:     {self.submission_dates.min()} — {self.submission_dates.max()}",
            "",
            "| Drug | Tested | Resistant | Share |",
            "|---|---|---|---|",
        ]
        for drug in self.drugs:
            mask = self.labelled_mask(drug)
            n_tested = int(mask.sum())
            n_res = int(np.nansum(self.phenotypes[drug][mask]))
            rate = n_res / n_tested if n_tested else float("nan")
            lines.append(f"| {drug} | {n_tested} | {n_res} | {rate:.1%} |")
        return "\n".join(lines)
