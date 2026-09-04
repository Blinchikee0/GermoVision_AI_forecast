"""GV-Growth — прогноз динамики долей линий возбудителя.

Обоснование выбора модели
=========================

Данные геномного надзора имеют строго определённую природу: в регионе за
неделю просеквенировано N образцов, из них n₁ линии A, n₂ линии B и так
далее. Это **мультиномиальные счётные данные**, а не непрерывный
временной ряд. Отсюда три следствия, которые версия 1.0 нарушала,
применяя 1D-CNN:

1. **Доли зависимы** — они в сумме дают единицу. Предсказывать их
   независимой регрессией некорректно.
2. **Неопределённость зависит от объёма выборки.** Доля 30 % при 10 и при
   1000 образцов означает совершенно разную уверенность. Регрессия на
   долях эту информацию теряет; мультиномиальная модель учитывает её
   естественным образом, потому что знаменатель входит в правдоподобие.
3. **Логистический рост — не эмпирика, а следствие теории.** Если у
   варианта есть постоянное селективное преимущество, отношение его
   частоты к частоте базового варианта растёт экспоненциально, а сама
   доля — по логистической кривой. Это прямое следствие уравнений
   динамики отбора, а не подобранная форма кривой.

Именно поэтому в реальной практике надзора применяется мультиномиальная
логистическая регрессия — она используется, в частности, в прогнозах
Nextstrain по SARS-CoV-2.

**Зачем иерархичность.** В большинстве регионов Казахстана за неделю
секвенируется несколько десятков образцов. Независимая оценка по такому
объёму даёт бессмысленно широкие интервалы. Иерархическая модель
стягивает региональные коэффициенты роста к общему среднему: регион с
малой выборкой заимствует силу у остальных, а регион с большой —
сохраняет собственную оценку. Величина стягивания оценивается из данных
(эмпирический байес), а не назначается вручную.

**Что получается на выходе.** Коэффициент β — логарифмическое
преимущество роста в единицу времени, величина с прямым биологическим
смыслом, которую можно обсуждать с эпидемиологом и сравнивать между
регионами. Ни один выход свёрточной сети таким свойством не обладает.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = ["GrowthFit", "GVGrowth"]

_EPS = 1e-12


@dataclass
class GrowthFit:
    """Результат подгонки по одному региону."""

    region: str
    lineages: list[str]
    intercepts: np.ndarray
    slopes: np.ndarray
    n_observations: int
    n_samples: int
    t0: float = 0.0
    slope_se: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def probabilities(self, t) -> np.ndarray:
        """Предсказанные доли линий в момент(ы) времени t."""
        tt = np.atleast_1d(np.asarray(t, dtype=float)) - self.t0
        eta = self.intercepts[None, :] + self.slopes[None, :] * tt[:, None]
        return np.exp(eta - logsumexp(eta, axis=1, keepdims=True))

    def growth_advantage(self, lineage: str) -> float:
        """Логарифмическое преимущество роста линии за единицу времени."""
        return float(self.slopes[self.lineages.index(lineage)])

    def weekly_advantage_pct(self, lineage: str) -> float:
        """Преимущество роста в процентах за неделю — для отчёта людям."""
        return float(np.expm1(self.growth_advantage(lineage)) * 100.0)


class GVGrowth:
    """Иерархическая мультиномиальная логистическая модель роста линий.

    Example:
        >>> model = GVGrowth().fit(counts, times, regions, lineages)
        >>> forecast, lo, hi = model.forecast("KZ-Aktobe", horizons=[2, 4, 8])
    """

    def __init__(
        self,
        shrinkage: float | None = None,
        intercept_prior_sd: float = 5.0,
        max_iter: int = 500,
        n_bootstrap: int = 200,
        seed: int = 0,
    ) -> None:
        """
        Args:
            shrinkage: сила стягивания региональных наклонов к общему
                среднему (τ). None — оценивать из данных.
            intercept_prior_sd: слабый прайор на свободные члены, нужен
                только для численной устойчивости при полном разделении.
            max_iter: предел итераций оптимизатора.
            n_bootstrap: число параметрических бутстрэп-выборок для
                интервалов. Пересемплирование выполняется из
                мультиномиального распределения с подогнанными долями,
                что корректно учитывает зависимость неопределённости от
                объёма секвенирования.
            seed: сид.
        """
        self.shrinkage = shrinkage
        self.intercept_prior_sd = intercept_prior_sd
        self.max_iter = max_iter
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        self.lineages_: list[str] = []
        self.regions_: list[str] = []
        self.global_slopes_: np.ndarray | None = None
        self.tau_: float = float("nan")
        self.fits_: dict[str, GrowthFit] = {}

    # -- внутреннее ------------------------------------------------------

    @staticmethod
    def _nll(params, counts, times, prior_slopes, tau, intercept_sd):
        """Отрицательный логарифм апостериорной плотности (с точностью до константы)."""
        v = counts.shape[1]
        a = np.concatenate([[0.0], params[: v - 1]])   # референсная линия закреплена
        b = np.concatenate([[0.0], params[v - 1 :]])   # для идентифицируемости

        eta = a[None, :] + b[None, :] * times[:, None]
        log_p = eta - logsumexp(eta, axis=1, keepdims=True)
        nll = -np.sum(counts * log_p)

        nll += 0.5 * np.sum((a[1:] / intercept_sd) ** 2)
        if prior_slopes is not None and np.isfinite(tau) and tau > 0:
            nll += 0.5 * np.sum(((b[1:] - prior_slopes[1:]) / tau) ** 2)
        return nll

    def _fit_one(self, counts, times, prior_slopes, tau) -> tuple[np.ndarray, np.ndarray]:
        v = counts.shape[1]
        x0 = np.zeros(2 * (v - 1))
        if prior_slopes is not None:
            x0[v - 1 :] = prior_slopes[1:]

        res = minimize(
            self._nll,
            x0,
            args=(counts, times, prior_slopes, tau, self.intercept_prior_sd),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        a = np.concatenate([[0.0], res.x[: v - 1]])
        b = np.concatenate([[0.0], res.x[v - 1 :]])
        return a, b

    # -- обучение --------------------------------------------------------

    def fit(
        self,
        counts: np.ndarray,
        times: np.ndarray,
        regions: np.ndarray,
        lineages: list[str],
    ) -> GVGrowth:
        """Подогнать модель.

        Args:
            counts: массив (n_obs × n_lineages) счётчиков.
            times: моменты наблюдений (обычно номер недели).
            regions: регион каждого наблюдения.
            lineages: названия линий; первая считается референсной.

        Raises:
            ValueError: несогласованные размеры или менее двух линий.
        """
        counts = np.asarray(counts, dtype=float)
        times = np.asarray(times, dtype=float)
        regions = np.asarray(regions)

        if counts.ndim != 2:
            raise ValueError("counts должен быть двумерным (наблюдения × линии)")
        if counts.shape[1] < 2:
            raise ValueError("нужно минимум две линии")
        if not (counts.shape[0] == times.size == regions.size):
            raise ValueError("длины counts, times и regions должны совпадать")
        if len(lineages) != counts.shape[1]:
            raise ValueError("число названий линий не совпадает с числом столбцов")

        self.lineages_ = list(lineages)
        self.regions_ = sorted(set(regions.tolist()))
        t0 = float(times.min())
        t_centered = times - t0

        # Шаг 1. Общая модель по объединённым данным — оценка среднего
        # наклона по каждой линии.
        _, global_b = self._fit_one(counts, t_centered, None, np.inf)
        self.global_slopes_ = global_b

        # Шаг 2. Оценка разброса наклонов между регионами (эмпирический байес).
        raw: dict[str, np.ndarray] = {}
        for reg in self.regions_:
            mask = regions == reg
            if counts[mask].sum() < 10:
                continue
            _, b = self._fit_one(counts[mask], t_centered[mask], None, np.inf)
            raw[reg] = b

        if self.shrinkage is not None:
            tau = float(self.shrinkage)
        elif len(raw) >= 3:
            spread = np.std(np.array([b[1:] for b in raw.values()]), axis=0)
            # Нижняя граница не даёт модели выродиться в полное объединение
            # регионов, когда наблюдаемый разброс случайно оказался нулевым.
            tau = float(max(np.median(spread), 1e-3))
        else:
            tau = 0.05
        self.tau_ = tau

        # Шаг 3. Финальная подгонка по регионам со стягиванием к общему среднему.
        rng = np.random.default_rng(self.seed)
        self.fits_ = {}
        for reg in self.regions_:
            mask = regions == reg
            c_reg, t_reg = counts[mask], t_centered[mask]
            if c_reg.sum() == 0:
                continue

            a, b = self._fit_one(c_reg, t_reg, self.global_slopes_, tau)
            se = self._bootstrap_se(c_reg, t_reg, a, b, tau, rng)

            self.fits_[str(reg)] = GrowthFit(
                region=str(reg),
                lineages=self.lineages_,
                intercepts=a,
                slopes=b,
                n_observations=int(mask.sum()),
                n_samples=int(c_reg.sum()),
                t0=t0,
                slope_se=se,
                meta={"tau": tau, "shrunk_toward_global": True},
            )

        if not self.fits_:
            raise ValueError("ни один регион не содержит достаточно данных")
        return self

    def _bootstrap_se(self, counts, times, a, b, tau, rng) -> np.ndarray:
        """Стандартные ошибки наклонов параметрическим бутстрэпом.

        Счётчики пересемплируются из мультиномиального распределения с
        подогнанными долями и **исходным объёмом выборки** каждой недели.
        Так неопределённость автоматически оказывается шире там, где
        секвенировано мало образцов, — что и требуется.
        """
        n_totals = counts.sum(axis=1).astype(int)
        eta = a[None, :] + b[None, :] * times[:, None]
        probs = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))

        slopes = []
        for _ in range(self.n_bootstrap):
            resampled = np.array(
                [rng.multinomial(n, p) for n, p in zip(n_totals, probs, strict=True)],
                dtype=float,
            )
            if resampled.sum() == 0:
                continue
            try:
                _, b_star = self._fit_one(resampled, times, self.global_slopes_, tau)
                slopes.append(b_star)
            except (ValueError, FloatingPointError):
                continue

        if len(slopes) < 10:
            return np.full_like(b, np.nan)
        return np.std(np.array(slopes), axis=0)

    # -- прогноз ---------------------------------------------------------

    def forecast(
        self, region: str, horizons: list[int], last_time: float | None = None, z: float = 1.96
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Спрогнозировать доли линий на заданные горизонты.

        Args:
            region: регион.
            horizons: горизонты в единицах времени (обычно недели).
            last_time: момент, от которого отсчитывается горизонт.
            z: множитель для интервала; 1,96 даёт примерно 95 %.

        Returns:
            Тройка (доли, нижняя граница, верхняя граница), каждая формы
            (len(horizons) × n_lineages).

        Raises:
            KeyError: регион не подгонялся.
        """
        if region not in self.fits_:
            raise KeyError(f"регион '{region}' отсутствует; есть: {sorted(self.fits_)}")

        fit = self.fits_[region]
        base = fit.t0 if last_time is None else last_time
        times = np.array([base + h for h in horizons], dtype=float)

        point = fit.probabilities(times)
        se = fit.slope_se
        if se is None or np.all(np.isnan(se)):
            return point, point, point

        # Интервал строится в пространстве логитов и переводится обратно —
        # так границы гарантированно остаются в [0, 1].
        #
        # Неопределённость каждой линии оценивается по отдельности, при
        # остальных линиях, зафиксированных в точечной оценке. Сдвигать
        # все линии одновременно и в противоположные стороны неверно:
        # доли связаны условием нормировки, и такой «интервал» всегда
        # упирался бы в 0 и 1, то есть не нёс бы информации.
        dt = times - fit.t0
        lo = np.empty_like(point)
        hi = np.empty_like(point)
        se_safe = np.nan_to_num(se)

        for k, d in enumerate(dt):
            delta = z * se_safe * abs(d)
            eta = fit.intercepts + fit.slopes * d
            for j in range(eta.size):
                for bound, sign in ((lo, -1.0), (hi, +1.0)):
                    shifted = eta.copy()
                    shifted[j] += sign * delta[j]
                    bound[k, j] = float(np.exp(shifted[j] - logsumexp(shifted)))

        return point, np.clip(lo, 0, 1), np.clip(hi, 0, 1)

    def growth_table(self) -> list[dict]:
        """Сводка коэффициентов роста по регионам и линиям — для панели."""
        rows: list[dict] = []
        for region, fit in self.fits_.items():
            for j, lin in enumerate(fit.lineages):
                se = float(fit.slope_se[j]) if fit.slope_se is not None else float("nan")
                beta = float(fit.slopes[j])
                rows.append({
                    "region": region,
                    "lineage": lin,
                    "beta": beta,
                    "se": se,
                    "weekly_pct": float(np.expm1(beta) * 100.0),
                    "n_samples": fit.n_samples,
                    "significant": bool(np.isfinite(se) and se > 0 and abs(beta) > 1.96 * se),
                })
        return rows
