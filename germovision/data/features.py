"""Построение матрицы признаков из генотипа изолята.

Тонкий, но существенный момент: **словарь признаков строится только по
обучающей части**. Если собрать список мутаций по всему набору, включая
тест, произойдёт утечка — редкая мутация, встречающаяся лишь в тестовых
изолятах, получит собственный столбец, и модель узнает о существовании
объектов, которых видеть не должна. Утечка тихая: она не ломает метрики
целиком, но систематически их завышает.

Поэтому `FeatureBuilder` устроен как обычный трансформер: `fit` на
обучающей части, `transform` на любой другой. Мутации, не встреченные
при обучении, попадают в агрегированные признаки (число неизвестных
вариантов в гене), но не получают собственных столбцов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalogue import DRUG_GENES, MutationCatalogue
from .schema import IsolateDataset

__all__ = ["FeatureMatrix", "FeatureBuilder"]


@dataclass
class FeatureMatrix:
    """Матрица признаков с именами столбцов и разметкой групп."""

    x: np.ndarray
    names: list[str]
    groups: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.x.shape[1] != len(self.names):
            raise ValueError(
                f"столбцов {self.x.shape[1]}, имён {len(self.names)}"
            )
        if not self.groups:
            self.groups = ["other"] * len(self.names)

    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    def group_indices(self, group: str) -> np.ndarray:
        """Индексы столбцов заданной группы — нужно для абляций (§ 5.10)."""
        return np.array([i for i, g in enumerate(self.groups) if g == group], dtype=int)


class FeatureBuilder:
    """Преобразователь «генотип → числовые признаки» для одного препарата.

    Признаки строятся раздельно по препаратам: для рифампицина осмысленны
    варианты в rpoB, для фторхинолонов — в gyrA/gyrB. Общая матрица по
    всем генам сразу дала бы модели тысячи заведомо нерелевантных
    столбцов и ухудшила бы обобщение на редких мутациях.

    Группы признаков:
        `mutation`  — индикатор конкретного варианта (частого в обучении);
        `catalogue` — сводка по каталогу ВОЗ: есть ли маркер группы 1, 2,
                      минимальный уровень доказательности среди найденных;
        `burden`    — число вариантов в каждом релевантном гене, включая
                      не встречавшиеся при обучении;
        `context`   — линия возбудителя.

    Example:
        >>> fb = FeatureBuilder("RIF").fit(train_ds)
        >>> fm = fb.transform(test_ds)
    """

    def __init__(
        self,
        drug: str,
        catalogue: MutationCatalogue | None = None,
        min_count: int = 3,
        use_catalogue: bool = True,
        use_burden: bool = True,
        use_context: bool = True,
    ) -> None:
        self.drug = drug
        self.catalogue = catalogue or MutationCatalogue()
        self.min_count = min_count
        self.use_catalogue = use_catalogue
        self.use_burden = use_burden
        self.use_context = use_context

        self.genes: tuple[str, ...] = DRUG_GENES.get(drug, ())
        self.vocabulary_: list[str] = []
        self.lineages_: list[str] = []
        self.names_: list[str] = []
        self.groups_: list[str] = []
        self._fitted = False

    def _relevant(self, muts: set[str]) -> set[str]:
        """Оставить только варианты в генах, связанных с препаратом."""
        return {m for m in muts if m.split("_", 1)[0] in self.genes}

    def fit(self, ds: IsolateDataset, idx: np.ndarray | None = None) -> FeatureBuilder:
        """Построить словарь признаков по обучающей части.

        Args:
            ds: полный набор данных.
            idx: индексы обучающей части. **Обязателен** при работе с
                разделённой выборкой: без него словарь соберётся по всем
                данным, включая тест, что является утечкой.
        """
        rows = range(len(ds)) if idx is None else np.asarray(idx, dtype=int)

        counts: dict[str, int] = {}
        for i in rows:
            for m in self._relevant(ds.mutations[i]):
                counts[m] = counts.get(m, 0) + 1

        self.vocabulary_ = sorted(k for k, c in counts.items() if c >= self.min_count)
        self.lineages_ = (
            sorted(set(np.asarray(ds.lineages)[list(rows)].tolist())) if self.use_context else []
        )

        names: list[str] = list(self.vocabulary_)
        groups: list[str] = ["mutation"] * len(self.vocabulary_)

        if self.use_catalogue:
            names += ["cat_group1", "cat_group2", "cat_min_group", "cat_n_markers"]
            groups += ["catalogue"] * 4
        if self.use_burden:
            names += [f"burden_{g}" for g in self.genes] + ["burden_unknown"]
            groups += ["burden"] * (len(self.genes) + 1)
        if self.use_context:
            names += [f"lineage_{lin}" for lin in self.lineages_]
            groups += ["context"] * len(self.lineages_)

        self.names_, self.groups_ = names, groups
        self._fitted = True
        return self

    def transform(self, ds: IsolateDataset, idx: np.ndarray | None = None) -> FeatureMatrix:
        """Преобразовать изоляты в матрицу признаков.

        Raises:
            RuntimeError: если `fit` не вызывался.
        """
        if not self._fitted:
            raise RuntimeError("FeatureBuilder не обучен: сначала вызовите fit()")

        rows = np.arange(len(ds)) if idx is None else np.asarray(idx, dtype=int)
        vocab_index = {m: j for j, m in enumerate(self.vocabulary_)}
        markers_g1 = {
            e.key for e in self.catalogue.entries if e.drug == self.drug and e.group == 1
        }
        markers_g2 = {
            e.key for e in self.catalogue.entries if e.drug == self.drug and e.group == 2
        }

        x = np.zeros((rows.size, len(self.names_)), dtype=np.float32)
        offset = len(self.vocabulary_)

        for r, i in enumerate(rows):
            muts = self._relevant(ds.mutations[i])
            for m in muts:
                j = vocab_index.get(m)
                if j is not None:
                    x[r, j] = 1.0

            col = offset
            if self.use_catalogue:
                hits = [e for m in muts for e in self.catalogue.lookup(m, self.drug)]
                x[r, col] = float(bool(muts & markers_g1))
                x[r, col + 1] = float(bool(muts & markers_g2))
                # Минимальный уровень доказательности среди найденных строк.
                # 6 означает «в каталоге ничего не найдено» — отдельное
                # значение, а не пропуск: отсутствие записи само по себе
                # информативно.
                x[r, col + 2] = float(min((e.group for e in hits), default=6))
                x[r, col + 3] = float(len(hits))
                col += 4

            if self.use_burden:
                for g in self.genes:
                    x[r, col] = float(sum(1 for m in muts if m.startswith(f"{g}_")))
                    col += 1
                x[r, col] = float(sum(1 for m in muts if m not in vocab_index))
                col += 1

            if self.use_context:
                lin = str(ds.lineages[i])
                if lin in self.lineages_:
                    x[r, col + self.lineages_.index(lin)] = 1.0

        return FeatureMatrix(x=x, names=list(self.names_), groups=list(self.groups_))

    def fit_transform(self, ds: IsolateDataset, idx: np.ndarray | None = None) -> FeatureMatrix:
        return self.fit(ds, idx).transform(ds, idx)
