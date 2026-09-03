"""Калибровка вероятностей.

Врачу нужна не «оценка модели», а вероятность: значение 0,85 должно
означать, что в 85 случаях из 100 изолят действительно окажется
устойчивым. Модель может хорошо ранжировать объекты и при этом выдавать
систематически смещённые вероятности — ранжирующие метрики (ROC-AUC,
PR-AUC) этого не показывают.

Отдельное правило проекта: калибровать нужно на выделенной части
выборки, не пересекающейся ни с обучением, ни с тестом. Калибровка на
обучающих данных выглядит идеальной и на новых данных не работает.
Поэтому `Split` содержит отдельное поле `calib`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
    "CalibrationReport",
    "evaluate_calibration",
    "IsotonicCalibrator",
]


def brier_score(y_true, y_prob) -> float:
    """Средний квадрат отклонения вероятности от исхода.

    Оценивает одновременно и различающую способность, и калибровку.
    Меньше — лучше; 0,25 соответствует постоянному ответу 0,5.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"формы не совпадают: {yt.shape} и {yp.shape}")
    return float(np.mean((yp - yt) ** 2))


def reliability_curve(
    y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Кривая надёжности: предсказанная вероятность против наблюдаемой частоты.

    Args:
        y_true: истинные метки (0/1).
        y_prob: предсказанные вероятности.
        n_bins: число интервалов.
        strategy: "quantile" — равное число объектов в интервале
            (устойчивее при скошенном распределении вероятностей);
            "uniform" — равная ширина интервалов.

    Returns:
        (средняя предсказанная вероятность, наблюдаемая частота,
        число объектов) по непустым интервалам.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(yp, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([yp.min(), yp.max() + 1e-9])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy должна быть 'quantile' или 'uniform'")

    idx = np.clip(np.digitize(yp, edges[1:-1], right=True), 0, edges.size - 2)
    pred, obs, counts = [], [], []
    for b in range(edges.size - 1):
        mask = idx == b
        if not mask.any():
            continue
        pred.append(yp[mask].mean())
        obs.append(yt[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(pred), np.array(obs), np.array(counts)


def expected_calibration_error(
    y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Ожидаемая ошибка калибровки (ECE).

    Средневзвешенное по интервалам расхождение между предсказанной
    вероятностью и наблюдаемой частотой. Ноль — идеальная калибровка.
    Ориентир для медицинского применения: ECE не выше 0,05.
    """
    pred, obs, counts = reliability_curve(y_true, y_prob, n_bins, strategy)
    if counts.sum() == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(pred - obs)) / counts.sum())


@dataclass
class CalibrationReport:
    brier: float
    ece: float
    n_bins: int
    bin_predicted: np.ndarray
    bin_observed: np.ndarray
    bin_counts: np.ndarray

    @property
    def is_acceptable(self) -> bool:
        """Соответствие ориентиру ECE ≤ 0,05 для медицинского применения."""
        return self.ece <= 0.05

    def to_markdown(self) -> str:
        lines = [
            f"Brier = {self.brier:.4f}, ECE = {self.ece:.4f} "
            f"({'приемлемо' if self.is_acceptable else 'ТРЕБУЕТСЯ КАЛИБРОВКА'})",
            "",
            "| Предсказано | Наблюдалось | N |",
            "|---|---|---|",
        ]
        for p, o, c in zip(
            self.bin_predicted, self.bin_observed, self.bin_counts, strict=True
        ):
            lines.append(f"| {p:.3f} | {o:.3f} | {c} |")
        return "\n".join(lines)


def evaluate_calibration(y_true, y_prob, n_bins: int = 10) -> CalibrationReport:
    """Собрать полный отчёт о калибровке."""
    pred, obs, counts = reliability_curve(y_true, y_prob, n_bins)
    return CalibrationReport(
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins),
        n_bins=n_bins,
        bin_predicted=pred,
        bin_observed=obs,
        bin_counts=counts,
    )


class IsotonicCalibrator:
    """Изотоническая калибровка вероятностей.

    Монотонно преобразует выход модели так, чтобы предсказанные
    вероятности соответствовали наблюдаемым частотам. Меняется
    осмысленность абсолютных значений, а не порядок объектов.

    Преобразование неубывающее, но кусочно-постоянное: часть объектов
    получает одинаковую вероятность. Поэтому ранжирующие метрики
    сохраняются лишь приближённо — совпадения засчитываются как половина
    правильного упорядочивания, и ROC-AUC может измениться на доли
    процента в любую сторону.

    Обучается на калибровочной части `Split.calib`. Попытка обучить её
    на обучающей части — методологическая ошибка, поэтому класс требует
    явно передать данные и не имеет доступа к обучению модели.

    Example:
        >>> cal = IsotonicCalibrator().fit(p_calib, y_calib)
        >>> p_test_calibrated = cal.transform(p_test)
    """

    def __init__(self) -> None:
        self._model = None

    def fit(self, y_prob, y_true) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        yp = np.asarray(y_prob, dtype=float)
        yt = np.asarray(y_true, dtype=float)
        if yp.size < 10:
            raise ValueError(
                f"для калибровки нужно не менее 10 объектов, передано {yp.size}"
            )
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(yp, yt)
        return self

    def transform(self, y_prob) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("калибратор не обучен: сначала вызовите fit()")
        return np.asarray(self._model.predict(np.asarray(y_prob, dtype=float)))
