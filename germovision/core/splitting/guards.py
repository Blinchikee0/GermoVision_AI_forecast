"""Защита от утечки данных.

Это главный модуль ядра. Он существует потому, что версия 1.0 проекта
содержала утечку, делавшую все её метрики недействительными: аугментация
(добавление гауссовского шума и дублирование редкого класса) выполнялась
**до** разделения выборки. В результате зашумлённые копии одной и той же
записи оказывались одновременно в обучающей и тестовой части, и модель
на тесте «узнавала» примеры, которые уже видела.

Такую ошибку невозможно заметить, глядя на метрики: они просто хорошие.
Единственная защита — автоматическая проверка, запускаемая в CI.

Модуль решает задачу двумя способами:

1. `LeakageGuard` — набор проверок, падающих при обнаружении утечки.
2. `augment_train_only` — API, в котором неправильный порядок операций
   невыразим: функция принимает уже готовый `Split`, поэтому
   аугментировать до разделения технически нельзя.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..types import LeakageError, Split

# (x_train, y_train, rng) -> дополнительные (x, y)
AugmentFn = Callable[
    [np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]
]

__all__ = [
    "GuardReport",
    "LeakageGuard",
    "check_temporal_order",
    "check_cluster_integrity",
    "check_no_exact_duplicates",
    "check_no_near_duplicates",
    "augment_train_only",
]


@dataclass
class GuardReport:
    """Результат работы проверок."""

    passed: bool
    checks: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = "ПРОЙДЕНО" if self.passed else "ОБНАРУЖЕНА УТЕЧКА"
        lines = [f"LeakageGuard: {head}"]
        for name, detail in self.checks.items():
            mark = "  ok " if name not in self.failures else "  ХХ "
            lines.append(f"{mark}{name}: {detail}")
        return "\n".join(lines)


def _row_hashes(x: np.ndarray) -> np.ndarray:
    """Устойчивые хеши строк матрицы признаков."""
    arr = np.ascontiguousarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.array(
        [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in arr],
        dtype=object,
    )


def check_temporal_order(split: Split, dates, embargo_days: int = 0) -> str:
    """Проверить, что ни одна обучающая запись не позже тестовой.

    Args:
        split: проверяемое разделение.
        dates: даты доступности записей (`submission_date`).
        embargo_days: требуемый минимальный зазор в днях.

    Returns:
        Строка с описанием результата.

    Raises:
        LeakageError: если обучение содержит записи из тестового периода.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    train_max = d[split.train].max()
    test_min = d[split.test].min()
    gap = (test_min - train_max).astype(int)

    if gap <= 0:
        n_after = int((d[split.train] >= test_min).sum())
        raise LeakageError(
            f"утечка из будущего: {n_after} обучающих записей датированы не раньше "
            f"начала теста (train_max={train_max}, test_min={test_min}). "
            "Разделение должно быть строго временным (§ 5.7, правило 1)"
        )
    if gap < embargo_days:
        raise LeakageError(
            f"зазор {gap} дн. меньше требуемого embargo={embargo_days} дн."
        )
    return f"train_max={train_max}, test_min={test_min}, зазор={gap} дн."


def check_cluster_integrity(split: Split, clusters) -> str:
    """Проверить, что ни один кластер родства не пересекает границу частей.

    Raises:
        LeakageError: если кластер представлен более чем в одной части.
    """
    labels = np.asarray(clusters)
    parts = split.parts()
    seen: dict[object, str] = {}
    shared: list[tuple[object, str, str]] = []

    for name, idx in parts.items():
        for cl in np.unique(labels[idx]):
            key = cl.item() if hasattr(cl, "item") else cl
            if key in seen and seen[key] != name:
                shared.append((key, seen[key], name))
            else:
                seen.setdefault(key, name)

    if shared:
        preview = "; ".join(f"кластер {c} в '{a}' и '{b}'" for c, a, b in shared[:5])
        raise LeakageError(
            f"{len(shared)} кластеров родства пересекают границу частей: {preview}. "
            "Близкородственные изоляты не являются независимыми наблюдениями "
            "(§ 5.7, правило 2)"
        )
    return f"{len(seen)} кластеров, пересечений нет"


def check_no_exact_duplicates(split: Split, x) -> str:
    """Проверить отсутствие побитово одинаковых строк в разных частях.

    Ловит дублирование записей — вторую половину аугментации версии 1.0
    («увеличил долю целевых данных до 20 %»).

    Raises:
        LeakageError: если одинаковая строка признаков есть и в train, и в test.
    """
    h = _row_hashes(np.asarray(x))
    train_h = set(h[split.train].tolist())
    dup = [i for i in split.test if h[i] in train_h]

    if dup:
        raise LeakageError(
            f"{len(dup)} тестовых строк побитово совпадают с обучающими "
            f"(например, индексы {dup[:5]}). Вероятная причина — дублирование "
            "записей до разделения выборки (§ 5.7, правило 4)"
        )
    return f"проверено {split.test.size} тестовых строк, совпадений нет"


def check_no_near_duplicates(
    split: Split,
    x,
    ratio: float = 0.1,
    max_suspicious_fraction: float = 0.02,
) -> str:
    """Проверить отсутствие почти-дубликатов между обучением и тестом.

    Это главная проверка модуля. Аугментация гауссовским шумом не создаёт
    побитовых копий — она создаёт близкие точки, и проверка на точное
    совпадение их пропускает. Нужен статистический критерий.

    Идея: для каждой тестовой точки берётся расстояние до ближайшей
    обучающей и сравнивается с характерным масштабом расстояний в данных.
    Зашумлённая копия отстоит от оригинала на величину порядка амплитуды
    шума — то есть на порядки меньше типичного расстояния между разными
    объектами. Такой разрыв в масштабах и есть подпись утечки.

    Масштаб оценивается по расстоянию до k-го соседа, а не до первого.
    Это принципиально: при массовой утечке обучающая выборка сама набита
    почти-дубликатами, поэтому расстояние до первого соседа внутри
    обучения тоже оказывается крошечным, и порог, построенный на нём,
    перестаёт что-либо ловить. Расстояние до k-го соседа отражает
    реальную плотность данных и дублированием не искажается.

    Args:
        split: проверяемое разделение.
        x: матрица признаков (n × d).
        ratio: доля характерного масштаба, ниже которой близость считается
            подозрительной.
        max_suspicious_fraction: допустимая доля подозрительных тестовых
            точек.

    Returns:
        Строка с описанием результата.

    Raises:
        LeakageError: если доля подозрительно близких точек превышает порог.
    """
    from sklearn.neighbors import NearestNeighbors

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    train_x, test_x = arr[split.train], arr[split.test]
    n_train = train_x.shape[0]
    if n_train < 5 or test_x.shape[0] < 1:
        return "недостаточно точек для статистической проверки — пропущено"

    k = int(min(10, max(2, n_train // 20)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(train_x)

    # Масштаб данных: медиана расстояния до k-го соседа внутри обучения.
    within, _ = nn.kneighbors(train_x, n_neighbors=k + 1)
    scale = float(np.median(within[:, k]))
    if scale <= 0.0:
        return "вырожденные данные (нулевой масштаб) — проверка неприменима"

    across, _ = nn.kneighbors(test_x, n_neighbors=1)
    across_nn = across[:, 0]

    threshold = ratio * scale
    suspicious = int((across_nn <= threshold).sum())
    fraction = suspicious / test_x.shape[0]

    if fraction > max_suspicious_fraction:
        raise LeakageError(
            f"{suspicious} из {test_x.shape[0]} тестовых точек ({fraction:.1%}) лежат "
            f"к обучающим ближе {threshold:.4g} при характерном масштабе данных "
            f"{scale:.4g} — разрыв на порядки. Признак того, что "
            "аугментация выполнена до разделения выборки: в тесте находятся "
            "зашумлённые копии обучающих записей (дефект D9, § 5.7, правило 4)"
        )
    return (
        f"подозрительных {suspicious}/{test_x.shape[0]} ({fraction:.1%}) "
        f"при допуске {max_suspicious_fraction:.0%}, масштаб {scale:.3g}"
    )


class LeakageGuard:
    """Набор проверок разделения выборки.

    Запускается перед каждым обучением и в CI. Проверки, для которых не
    переданы нужные данные, пропускаются — но факт пропуска фиксируется
    в отчёте, чтобы «не проверяли» нельзя было спутать с «проверили и
    всё хорошо».

    Example:
        >>> guard = LeakageGuard(dates=dates, clusters=clusters, x=X)
        >>> report = guard.run(split)     # бросит LeakageError при утечке
        >>> print(report)
    """

    def __init__(
        self,
        dates=None,
        clusters=None,
        x=None,
        embargo_days: int = 0,
        near_duplicate_check: bool = True,
    ) -> None:
        self.dates = dates
        self.clusters = clusters
        self.x = x
        self.embargo_days = embargo_days
        self.near_duplicate_check = near_duplicate_check

    def run(self, split: Split, raise_on_fail: bool = True) -> GuardReport:
        """Выполнить все применимые проверки.

        Args:
            split: проверяемое разделение.
            raise_on_fail: бросать исключение при первой ошибке. False
                позволяет собрать полный отчёт обо всех нарушениях.

        Returns:
            GuardReport с результатами.

        Raises:
            LeakageError: при обнаружении утечки, если raise_on_fail=True.
        """
        report = GuardReport(passed=True)
        checks: list[tuple[str, Callable[[], str]]] = [
            ("disjoint", lambda: f"части не пересекаются, всего {split.n_total} записей")
        ]
        if self.dates is not None:
            checks.append((
                "temporal_order",
                lambda: check_temporal_order(split, self.dates, self.embargo_days),
            ))
        if self.clusters is not None:
            checks.append((
                "cluster_integrity",
                lambda: check_cluster_integrity(split, self.clusters),
            ))
        if self.x is not None:
            checks.append((
                "exact_duplicates",
                lambda: check_no_exact_duplicates(split, self.x),
            ))
            if self.near_duplicate_check:
                checks.append(("near_duplicates", lambda: check_no_near_duplicates(split, self.x)))

        for name, fn in checks:
            try:
                report.checks[name] = fn()
            except LeakageError as exc:
                report.passed = False
                report.failures.append(name)
                report.checks[name] = str(exc)
                if raise_on_fail:
                    raise

        skipped = []
        if self.dates is None:
            skipped.append("temporal_order")
        if self.clusters is None:
            skipped.append("cluster_integrity")
        if self.x is None:
            skipped.extend(["exact_duplicates", "near_duplicates"])
        if skipped:
            report.checks["ПРОПУЩЕНО"] = ", ".join(skipped) + " (не переданы данные)"

        return report


def augment_train_only(
    split: Split,
    x,
    y,
    augment_fn: AugmentFn,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Применить аугментацию исключительно к обучающей части.

    Функция принимает готовый `Split`, поэтому вызвать её до разделения
    выборки невозможно — неправильный порядок операций здесь просто
    невыразим. Это структурная защита: она надёжнее инструкции в
    документации, которую можно не прочитать.

    Args:
        split: разделение, прошедшее проверки LeakageGuard.
        x: полная матрица признаков.
        y: полный вектор меток.
        augment_fn: функция (x_train, y_train, rng) -> (x_aug, y_aug),
            возвращающая **дополнительные** объекты (не включая исходные).
        seed: сид генератора случайных чисел.

    Returns:
        Пара (x_train_augmented, y_train_augmented) — только обучающая
        часть. Тестовая часть не возвращается намеренно: её нельзя
        аугментировать ни при каких условиях.

    Raises:
        ValueError: если augment_fn вернула несогласованные размеры.
    """
    arr_x, arr_y = np.asarray(x), np.asarray(y)
    x_train, y_train = arr_x[split.train], arr_y[split.train]

    rng = np.random.default_rng(seed)
    x_extra, y_extra = augment_fn(x_train, y_train, rng)
    x_extra, y_extra = np.asarray(x_extra), np.asarray(y_extra)

    if x_extra.shape[0] != y_extra.shape[0]:
        raise ValueError(
            f"augment_fn вернула {x_extra.shape[0]} объектов и {y_extra.shape[0]} меток"
        )
    if x_extra.size and x_extra.shape[1:] != x_train.shape[1:]:
        raise ValueError(
            f"форма аугментированных признаков {x_extra.shape[1:]} "
            f"не совпадает с исходной {x_train.shape[1:]}"
        )

    if x_extra.size == 0:
        return x_train, y_train
    return np.concatenate([x_train, x_extra]), np.concatenate([y_train, y_extra])
