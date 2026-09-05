"""GV-Resist — предсказание лекарственной устойчивости по геному изолята.

Задача: дан набор вариантов, найденных у клинического изолята. Оценить
вероятность устойчивости к каждому препарату, выдать решение с указанием
уровня достоверности и обоснованием — либо честно отказаться от ответа.

Практический смысл: фенотипический тест лекарственной чувствительности
при туберкулёзе занимает порядка 60 дней, поскольку *M. tuberculosis*
растёт крайне медленно. Геномное предсказание даёт результат за 1–2 дня.
Для Казахстана, где около 26 % новых случаев — с множественной
лекарственной устойчивостью против 3,2 % в мире, это означает, что
примерно каждый четвёртый пациент перестаёт получать заведомо
неработающую схему на протяжении полутора месяцев.

Обоснование архитектуры
=======================

**Почему два уровня, а не одна модель.** Каталог мутаций ВОЗ — это
официальный референсный стандарт, построенный на более чем 52 000
изолятов. Игнорировать его в пользу «своей нейросети» было бы неверно и
клинически, и методологически: врач должен видеть, что вывод соответствует
стандарту, и иметь возможность его проверить. Поэтому первый уровень —
правила каталога, второй — обучаемая модель, работающая там, где каталог
молчит: редкие варианты, их сочетания, недавно внедрённые препараты.

**Почему градиентный бустинг, а не нейронная сеть.** Данные табличные и
разреженные, объём — порядка 10⁴ объектов. В этом режиме бустинг не
уступает нейросетям по качеству и превосходит их по трём параметрам,
которые для медицинского применения важнее долей процента AUC:
устойчивость к переобучению на малых выборках, скорость (минуты, а не
часы) и интерпретируемость. Нейросеть на таком объёме — это дефект D13
версии 1.0 в новой обёртке.

**Почему обязательна калибровка.** Врачу нужна вероятность, а не «оценка
модели»: 0,85 должно означать, что в 85 случаях из 100 изолят
действительно устойчив. Ранжирующие метрики этого не гарантируют.

**Почему модель обязана уметь молчать.** Модель, которая всегда отвечает,
в медицине опаснее модели, которая иногда отказывается. Явный отказ
отправляет образец на фенотипическое тестирование — безопасный исход.
Уверенная ошибка приводит к неверной терапии. Отказ реализован через
конформное предсказание, дающее формальную гарантию покрытия.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.metrics.calibration import IsotonicCalibrator
from ..core.metrics.classification import (
    ClassificationReport,
    MetricCI,
    bootstrap_ci,
    evaluate_binary,
    sensitivity,
    specificity,
)
from ..core.types import Split
from ..data.catalogue import DRUG_NAMES_RU, CatalogueEntry, MutationCatalogue
from ..data.features import FeatureBuilder
from ..data.schema import IsolateDataset

__all__ = ["Decision", "ResistancePrediction", "DrugEvaluation", "GVResist"]


class Decision:
    """Возможные решения системы."""

    RESISTANT = "resistant"
    SUSCEPTIBLE = "susceptible"
    NO_CALL = "no_call"


@dataclass
class ResistancePrediction:
    """Заключение по одному изоляту и одному препарату."""

    isolate_id: str
    drug: str
    decision: str
    probability: float
    source: str  # "catalogue" | "model"
    evidence: list[CatalogueEntry] = field(default_factory=list)
    contributions: list[tuple[str, float]] = field(default_factory=list)
    ood: bool = False
    reason: str = ""
    needs_confirmation: bool = False

    @property
    def drug_name(self) -> str:
        return DRUG_NAMES_RU.get(self.drug, self.drug)

    def explain(self) -> str:
        """Обоснование для врача. Заключение без обоснования не выдаётся."""
        if self.decision == Decision.NO_CALL:
            return f"no call — {self.reason}"
        # Пометка о необходимости подтверждения выводится один раз в
        # сводке заключения, а не в каждой строке: продублированная
        # тринадцать раз, она перестаёт читаться и теряет смысл.
        note = ""
        if self.source == "catalogue" and self.evidence:
            top = self.evidence[0]
            return f"{top.key} — WHO catalogue, group {top.group}: {top.note}{note}"
        if self.contributions:
            parts = [f"{name} ({delta:+.2f})" for name, delta in self.contributions[:3]]
            return "model, feature contributions: " + ", ".join(parts) + note
        return "model: no known marker found" + note


@dataclass
class DrugEvaluation:
    """Оценка качества по одному препарату.

    Содержит два взгляда на одни и те же предсказания, и они не
    взаимозаменяемы:

    * `ranking` — метрики по калиброванной вероятности. Отвечают на
      вопрос «насколько хорошо модель различает устойчивые и
      чувствительные изоляты» и позволяют сравнивать конфигурации.
    * `decision_*` — метрики по фактически выдаваемому решению, включая
      влияние правил каталога и отказов от ответа. Именно с этим
      сталкивается врач.

    Отчитываться только первыми — значит скрыть эффект правил и отказов;
    только вторыми — потерять сравнимость между конфигурациями.
    """

    drug: str
    ranking: ClassificationReport
    decision_sensitivity: MetricCI
    decision_specificity: MetricCI
    n_evaluated: int
    n_abstained: int
    n_by_catalogue: int

    @property
    def abstention_rate(self) -> float:
        total = self.n_evaluated + self.n_abstained
        return self.n_abstained / total if total else 0.0

    correctly_closed: float = float("nan")
    missed_resistance: float = float("nan")
    requires_confirmation: bool = False

    @property
    def answer_rate(self) -> float:
        """Доля изолятов, закрытых за 1–2 дня без фенотипического теста.

        Практически именно эта величина определяет пользу системы:
        отвечая по половине образцов за сутки вместо шестидесяти дней,
        она сокращает нагрузку на лабораторию вдвое, а не «ошибается
        в половине случаев».
        """
        return 1.0 - self.abstention_rate

    def summary_line(self) -> str:
        """Одна строка, отвечающая на вопрос «что это даёт»."""
        flag = " ⚠ lab confirmation required" if self.requires_confirmation else ""
        return (
            f"{self.drug}: correctly closed {self.correctly_closed:.1%}, "
            f"resistance missed {self.missed_resistance:.1%}, "
            f"answered {self.answer_rate:.1%}{flag}"
        )

    def meets_h1(self, min_sens: float = 0.90, min_spec: float = 0.95) -> bool:
        """Проверка целевых порогов гипотезы H1 по фактическим решениям."""
        return (
            self.decision_sensitivity.value >= min_sens
            and self.decision_specificity.value >= min_spec
        )


class GVResist:
    """Двухуровневая модель устойчивости для одного препарата.

    Example:
        >>> model = GVResist("RIF").fit(dataset, split)
        >>> preds = model.predict(dataset, split.test)
        >>> preds[0].explain()
    """

    def __init__(
        self,
        drug: str,
        catalogue: MutationCatalogue | None = None,
        alpha: float = 0.10,
        max_missed_resistance: float = 0.10,
        max_utility_sacrifice: float = 0.15,
        min_count: int = 3,
        use_catalogue_tier: bool = True,
        use_catalogue_features: bool = True,
        use_burden: bool = True,
        use_context: bool = False,
        use_discovery: bool = True,
        ood_threshold: float = 0.5,
        random_state: int = 0,
    ) -> None:
        """
        Args:
            drug: код препарата.
            catalogue: каталог мутаций; по умолчанию встроенное ядро.
            alpha: уровень конформного предсказания. Задаёт, когда система
                отказывается отвечать: при 0,10 гарантируется, что истинная
                метка попадает в выдаваемое множество не реже чем в 90 %
                случаев.
            max_missed_resistance: доля устойчивых изолятов, которым
                допустимо выдать уверенное «чувствителен». Это единственный
                по-настоящему опасный исход системы, и он ограничивается
                явно.

                Значение 0,10 соответствует клинической планке порядка 90 %
                чувствительности, принятой для молекулярной диагностики
                туберкулёза. Более строгие значения обходятся дорого и
                нелинейно и зависит от разделяющей способности модели.
                Достижимость лимита проверяется: если он не выдержан,
                препарат помечается как требующий фенотипического
                подтверждения.
            max_utility_sacrifice: сколько закрытых случаев допустимо
                отдать ради соблюдения лимита пропусков. При превышении
                этой цены лимит не навязывается — вместо этого выставляется
                пометка о необходимости подтверждения.

                Величина задаёт компромисс, а не «точность». Требование
                покрытия выше, чем собственная точность модели, математически
                вынуждает её отказываться от ответа: получить 95 % покрытия
                одиночными ответами при точности 85 % невозможно. Поэтому
                чем строже alpha, тем больше отказов. Значение 0,10 выбрано
                как рабочее; окончательный выбор — за организацией, исходя
                из того, сколько образцов она готова отправлять на
                фенотипическое подтверждение. Кривая компромисса строится
                методом `coverage_tradeoff`.
            min_count: минимальная частота варианта в обучении, при которой
                он получает собственный столбец признаков.
            use_catalogue_tier: включать ли уровень правил. Отключение —
                режим абляции для оценки вклада каталога.
            ood_threshold: доля неизвестных вариантов, при которой изолят
                считается вышедшим за пределы обучающего распределения.
            random_state: сид.
        """
        if not 0.0 < alpha < 0.5:
            raise ValueError("alpha must lie in (0, 0.5)")
        self.drug = drug
        self.catalogue = catalogue or MutationCatalogue()
        self.alpha = alpha
        self.max_missed_resistance = max_missed_resistance
        self.max_utility_sacrifice = max_utility_sacrifice
        self.ood_threshold = ood_threshold
        self.use_catalogue_tier = use_catalogue_tier
        self.random_state = random_state

        self.features = FeatureBuilder(
            drug,
            catalogue=self.catalogue,
            min_count=min_count,
            use_catalogue=use_catalogue_features,
            use_burden=use_burden,
            use_context=use_context,
            use_discovery=use_discovery,
        )
        self.model_ = None
        self.calibrator_: IsotonicCalibrator | None = None
        self.conformal_q_: float | None = None
        self.threshold_: float = 0.5
        self.requires_confirmation_: bool = False
        self.operating_point_: dict = {"tuned": False}
        self.feature_names_: list[str] = []
        self.n_train_: int = 0
        self.prevalence_: float = float("nan")
        self._calib_split: Split | None = None

    # -- обучение ---------------------------------------------------------

    def _labelled(self, ds: IsolateDataset, idx: np.ndarray) -> np.ndarray:
        """Оставить из индексов только изоляты с измеренным фенотипом."""
        y = ds.phenotypes[self.drug]
        return np.asarray([i for i in idx if not np.isnan(y[i])], dtype=int)

    def fit(self, ds: IsolateDataset, split: Split) -> GVResist:
        """Обучить модель на обучающей части разделения.

        Словарь признаков строится строго по `split.train`; калибратор и
        конформный порог — по `split.calib`, если он задан, иначе по
        валидационной части. Тестовая часть не используется ни на одном
        шаге.

        Raises:
            ValueError: если в обучающей части нет обоих классов.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        train_idx = self._labelled(ds, split.train)
        if train_idx.size < 20:
            raise ValueError(
                f"{self.drug}: only {train_idx.size} training isolates have a measured phenotype"
            )

        y_train = ds.phenotypes[self.drug][train_idx].astype(int)
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"{self.drug}: the training part contains only one class — cannot fit"
            )

        self.features.fit(ds, train_idx)
        self.feature_names_ = list(self.features.names_)
        x_train = self.features.transform(ds, train_idx).x
        self.n_train_ = int(train_idx.size)
        self.prevalence_ = float(y_train.mean())

        # Веса классов: устойчивых изолятов меньше, а пропуск устойчивости —
        # самая дорогая ошибка системы, поэтому редкий класс усиливается.
        weights = np.where(y_train == 1, 1.0 / max(self.prevalence_, 1e-3), 1.0)
        weights *= len(weights) / weights.sum()

        self.model_ = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.random_state,
        )
        self.model_.fit(x_train, y_train, sample_weight=weights)

        self._calib_split = split
        self._fit_calibration(ds, split)
        return self

    def _fit_calibration(self, ds: IsolateDataset, split: Split) -> None:
        """Откалибровать вероятности, порог решения и конформный порог.

        Три величины оцениваются на выделенной части выборки, не
        пересекающейся ни с обучением, ни с тестом. Если такой части нет,
        калибровка пропускается целиком — а не подменяется обучающей
        выборкой: на обучении она выглядит идеальной и на новых данных не
        работает, так что молчаливая подмена хуже отсутствия калибровки.
        """
        source = split.calib if split.calib is not None else split.val
        if source is None:
            return

        calib_idx = self._labelled(ds, source)
        if calib_idx.size < 30:
            return

        y_cal = ds.phenotypes[self.drug][calib_idx].astype(int)
        if len(np.unique(y_cal)) < 2:
            return

        p_raw = self._raw_proba(ds, calib_idx)
        self.calibrator_ = IsotonicCalibrator().fit(p_raw, y_cal)
        p_cal = np.clip(self.calibrator_.transform(p_raw), 1e-6, 1 - 1e-6)

        # Отказ от ответа: маргинальное конформное предсказание.
        # Нонконформность — 1 минус вероятность истинной метки.
        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        self.conformal_q_ = self._quantile(scores, self.alpha)

        # Клиническая асимметрия: порог решения.
        self.threshold_ = self._tune_threshold(p_cal, y_cal)

    @staticmethod
    def _quantile(scores: np.ndarray, alpha: float) -> float | None:
        """Конформный квантиль с поправкой на объём выборки.

        Множитель (n + 1) / n — стандартная поправка split conformal: без
        неё гарантия покрытия нарушается на малых выборках, а именно они
        и встречаются у редких препаратов.
        """
        n = scores.size
        if n < 20:
            return None
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    def _tune_threshold(self, p: np.ndarray, y: np.ndarray) -> float:
        """Выбрать порог решения по пользе, которую система приносит.

        Максимизируется доля изолятов, **закрытых верно** — то есть
        получивших правильный ответ за сутки вместо шестидесяти дней.
        Величина считается от полного числа изолятов, поэтому оптимум не
        вырождается: назвать всех устойчивыми так же плохо, как назвать
        всех чувствительными.

        Почему цель не «выдержать лимит пропусков любой ценой». Такой
        лимит выполним всегда — достаточно понизить порог, — но цена
        нелинейна и зависит от того, насколько хорошо модель вообще
        разделяет классы. На рифампицине переход к лимиту 10 % стоит
        14 процентных пунктов закрытых случаев; на левофлоксацине, где
        разделение слабее, — уже 69, и от системы не остаётся пользы.
        Отказ не бесплатен: образец возвращается к шестидесятидневному
        ожиданию, то есть к тому, от чего система должна избавлять.

        Поэтому порог выбирается по пользе, а достижимость клинической
        планки **проверяется и объявляется**. Если лимит пропусков не
        выдержан, препарат помечается как требующий фенотипического
        подтверждения, и пометка доходит до заключения врача. Система,
        которая делает вид, что справилась, опаснее системы, которая
        сообщает, где ей нельзя доверять одной.
        """
        n_pos = int((y == 1).sum())
        if n_pos < 10 or (y == 0).sum() < 10:
            return 0.5

        candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 60)))
        best_t, best_closed, best_missed = 0.5, -1.0, 1.0
        for t in candidates:
            said_res = p >= t
            closed = float((said_res == (y == 1)).mean())
            missed = float((~said_res & (y == 1)).sum()) / n_pos
            if closed > best_closed:
                best_t, best_closed, best_missed = float(t), closed, missed

        # Проверка клинической планки при выбранном пороге.
        meets = best_missed <= self.max_missed_resistance

        # Если планка не выдержана, смотрим, во что обошлось бы её
        # соблюдение. Когда цена умеренная — платим её: пропуск
        # устойчивости дороже лишнего фенотипического теста.
        if not meets:
            strict_t = float(
                np.quantile(p[y == 1], self.max_missed_resistance, method="lower")
            )
            said_res = p >= strict_t
            strict_closed = float((said_res == (y == 1)).mean())
            if strict_closed >= best_closed - self.max_utility_sacrifice:
                best_t = strict_t
                best_closed = strict_closed
                best_missed = float((~said_res & (y == 1)).sum()) / n_pos
                meets = True

        self.requires_confirmation_ = not meets
        self.operating_point_ = {
            "tuned": True,
            "threshold": round(best_t, 4),
            "correctly_closed_calib": round(best_closed, 4),
            "missed_resistance_calib": round(best_missed, 4),
            "clinical_limit": self.max_missed_resistance,
            "meets_clinical_limit": bool(meets),
            "n_positive_calib": n_pos,
        }
        return float(np.clip(best_t, 0.01, 0.99))

    # -- предсказание -----------------------------------------------------

    def _raw_proba(self, ds: IsolateDataset, idx: np.ndarray) -> np.ndarray:
        x = self.features.transform(ds, idx).x
        return self.model_.predict_proba(x)[:, 1]

    def predict_proba(self, ds: IsolateDataset, idx: np.ndarray | None = None) -> np.ndarray:
        """Калиброванная вероятность устойчивости."""
        if self.model_ is None:
            raise RuntimeError("model is not fitted: call fit() first")
        rows = np.arange(len(ds)) if idx is None else np.asarray(idx, dtype=int)
        p = self._raw_proba(ds, rows)
        if self.calibrator_ is not None:
            p = self.calibrator_.transform(p)
        return np.clip(p, 0.0, 1.0)

    def _local_contributions(
        self, x_row: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Локальное объяснение методом исключения признаков.

        Для каждого активного признака вычисляется, насколько упадёт
        предсказанная вероятность, если этот признак обнулить. В отличие
        от глобальной важности это ответ на вопрос «почему модель так
        решила **для этого изолята**», а активных признаков у изолята
        единицы, поэтому расчёт дёшев.
        """
        active = np.flatnonzero(x_row)
        if active.size == 0:
            return []

        base = float(self.model_.predict_proba(x_row.reshape(1, -1))[0, 1])
        variants = np.repeat(x_row.reshape(1, -1), active.size, axis=0)
        for r, j in enumerate(active):
            variants[r, j] = 0.0
        without = self.model_.predict_proba(variants)[:, 1]

        deltas = [
            (self.feature_names_[j], base - float(p))
            for j, p in zip(active, without, strict=True)
        ]
        deltas.sort(key=lambda t: -abs(t[1]))
        return [d for d in deltas[:top_k] if abs(d[1]) > 1e-4]

    def _ood_fraction(self, ds: IsolateDataset, i: int) -> float:
        """Доля вариантов изолята, не встречавшихся при обучении."""
        relevant = self.features._relevant(ds.mutations[i])
        if not relevant:
            return 0.0
        known = set(self.features.vocabulary_)
        return sum(1 for m in relevant if m not in known) / len(relevant)

    def predict(
        self, ds: IsolateDataset, idx: np.ndarray | None = None, explain: bool = True
    ) -> list[ResistancePrediction]:
        """Выдать заключения по изолятам.

        Args:
            explain: вычислять ли локальный вклад признаков. При массовой
                оценке качества объяснения не нужны, а стоят дорого —
                каждое требует отдельного прохода модели по числу активных
                признаков изолята.

        Порядок принятия решения:
            1. Каталог ВОЗ нашёл маркер группы 1–2 → устойчив, источник
               «каталог». Это референсный стандарт, он имеет приоритет.
            2. Иначе — конформное множество по калиброванной вероятности.
               Множество из двух меток означает неуверенность → отказ.
            3. Изолят с большой долей неизвестных вариантов помечается как
               вышедший за пределы обучающего распределения.
        """
        if self.model_ is None:
            raise RuntimeError("model is not fitted: call fit() first")

        rows = np.arange(len(ds)) if idx is None else np.asarray(idx, dtype=int)
        probs = self.predict_proba(ds, rows)
        fm = self.features.transform(ds, rows)

        out: list[ResistancePrediction] = []
        for r, i in enumerate(rows):
            p = float(probs[r])
            muts = ds.mutations[i]
            ood_frac = self._ood_fraction(ds, i)
            is_ood = ood_frac > self.ood_threshold

            cat_decision, evidence = (
                self.catalogue.predict(muts, self.drug)
                if self.use_catalogue_tier
                else (None, [])
            )

            if cat_decision is True:
                # Каталог задаёт решение, но НЕ подменяет вероятность.
                # Подмена (например, на 0,95) разрушила бы калибровку:
                # пенетрантность маркеров неполна, и часть изолятов с
                # маркером фенотипически чувствительна. Вероятность должна
                # оставаться честной оценкой, решение — соответствовать
                # референсному стандарту.
                out.append(
                    ResistancePrediction(
                        isolate_id=str(ds.isolate_ids[i]),
                        drug=self.drug,
                        decision=Decision.RESISTANT,
                        probability=p,
                        source="catalogue",
                        evidence=evidence,
                        ood=is_ood,
                        needs_confirmation=self.requires_confirmation_,
                    )
                )
                continue

            decision, reason = self._conformal_decision(p)
            if decision == Decision.NO_CALL and is_ood:
                reason = (
                    f"{ood_frac:.0%} of this isolate's variants were unseen in training — "
                    "phenotypic confirmation required"
                )
            elif is_ood and decision == Decision.SUSCEPTIBLE:
                # Заявлять чувствительность по изоляту с незнакомым
                # генотипом опасно: именно так пропускается устойчивость,
                # вызванная неизвестным механизмом.
                decision = Decision.NO_CALL
                reason = (
                    f"{ood_frac:.0%} of this isolate's variants were unseen in training; "
                    "a susceptible call would not be reliable"
                )

            out.append(
                ResistancePrediction(
                    isolate_id=str(ds.isolate_ids[i]),
                    drug=self.drug,
                    decision=decision,
                    probability=p,
                    source="model",
                    contributions=self._local_contributions(fm.x[r]) if explain else [],
                    ood=is_ood,
                    reason=reason,
                    needs_confirmation=self.requires_confirmation_,
                )
            )
        return out

    def _conformal_decision(self, p: float) -> tuple[str, str]:
        """Решение по одному изоляту.

        Разделение обязанностей: конформное предсказание отвечает на
        вопрос «достаточно ли данных, чтобы вообще отвечать», порог — на
        вопрос «что именно ответить».

        Если обе метки совместимы с калибровочными данными, система
        отказывается от ответа: это штатный и безопасный исход, образец
        уходит на фенотипическое подтверждение. Если ни одна не
        совместима, изолят нетипичен — тоже отказ.
        """
        if self.conformal_q_ is None:
            return (
                Decision.RESISTANT if p >= self.threshold_ else Decision.SUSCEPTIBLE,
                "no calibration set available; abstention is disabled",
            )

        q = self.conformal_q_
        plausible_resistant = (1.0 - p) <= q
        plausible_susceptible = p <= q

        if plausible_resistant and plausible_susceptible:
            return Decision.NO_CALL, (
                f"both hypotheses are compatible with the data at the {1 - self.alpha:.0%} "
                f"level (probability {p:.2f})"
            )
        if not plausible_resistant and not plausible_susceptible:
            return Decision.NO_CALL, (
                f"atypical isolate: neither hypothesis agrees with the training data "
                f"(probability {p:.2f})"
            )

        # Ровно одна гипотеза правдоподобна. Метку назначает порог: он
        # несёт клиническую асимметрию, тогда как конформное множество
        # симметрично по построению. В редком случае расхождения выбор
        # делается в сторону «устойчив» — ошибиться в эту сторону дешевле.
        return (
            Decision.RESISTANT
            if (p >= self.threshold_ or not plausible_susceptible)
            else Decision.SUSCEPTIBLE
        ), ""


    def coverage_tradeoff(
        self, ds: IsolateDataset, idx: np.ndarray, alphas=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    ) -> list[dict]:
        """Компромисс «доля ответов — точность ответов».

        Отказ от ответа не бесплатен и не вреден сам по себе: он переводит
        образец на фенотипическое тестирование, то есть меняет скорость на
        достоверность. Решение о том, где провести границу, принимает
        организация, а не разработчик. Задача системы — показать цену
        каждого варианта.

        Перебирается уровень конформного предсказания, то есть готовность
        системы отвечать. Порог решения при этом не трогается: он задан
        клиническим критерием и от готовности отвечать не зависит.

        Returns:
            Список словарей с долей отвеченных изолятов, а также точностью,
            чувствительностью и специфичностью среди них.
        """
        eval_idx = self._labelled(ds, idx)
        if eval_idx.size == 0:
            raise ValueError(f"{self.drug}: no measured phenotypes in this subset")

        y = ds.phenotypes[self.drug][eval_idx].astype(int)
        saved = (self.alpha, self.conformal_q_, self.threshold_)

        rows: list[dict] = []
        try:
            for a in alphas:
                self.alpha = a
                self._fit_calibration(ds, self._calib_split)
                preds = self.predict(ds, eval_idx, explain=False)
                answered = np.array([q.decision != Decision.NO_CALL for q in preds])
                if not answered.any():
                    rows.append({"alpha": a, "answer_rate": 0.0})
                    continue
                decided = np.array(
                    [q.decision == Decision.RESISTANT for q in preds]
                )[answered]
                y_ans = y[answered]
                rows.append({
                    "alpha": a,
                    "answer_rate": float(answered.mean()),
                    "accuracy": float((decided == y_ans.astype(bool)).mean()),
                    "sensitivity": sensitivity(y_ans, decided),
                    "specificity": specificity(y_ans, decided),
                    "n_answered": int(answered.sum()),
                })
        finally:
            self.alpha, self.conformal_q_, self.threshold_ = saved

        return rows

    # -- оценка -----------------------------------------------------------

    def evaluate(
        self, ds: IsolateDataset, idx: np.ndarray, n_boot: int = 500
    ) -> DrugEvaluation:
        """Оценить качество на удержанной части.

        Изоляты, по которым система отказалась от ответа, исключаются из
        расчёта метрик, но их доля отчитывается отдельно: метрика,
        посчитанная после отказа от трудных случаев, без указания доли
        отказов вводит в заблуждение.
        """
        eval_idx = self._labelled(ds, idx)
        if eval_idx.size == 0:
            raise ValueError(f"{self.drug}: no measured phenotypes in the test part")

        y = ds.phenotypes[self.drug][eval_idx].astype(int)
        preds = self.predict(ds, eval_idx, explain=False)
        p = np.array([q.probability for q in preds])
        abstained = np.array([q.decision == Decision.NO_CALL for q in preds])

        if abstained.all():
            raise ValueError(f"{self.drug}: the system abstained on every isolate")

        ranking = evaluate_binary(
            y,
            p,
            label=DRUG_NAMES_RU.get(self.drug, self.drug),
            threshold=self.threshold_,
            abstained=abstained,
            n_boot=n_boot,
            seed=self.random_state,
        )

        answered = ~abstained
        y_ans = y[answered]
        decided = np.array(
            [q.decision == Decision.RESISTANT for q in preds], dtype=float
        )[answered]

        # Операционные метрики считаются от ПОЛНОГО числа изолятов, а не
        # от числа отвеченных: именно так измеряется польза системы.
        said_res = np.array([q.decision == Decision.RESISTANT for q in preds])
        correct = answered & (said_res == (y == 1))
        n_pos_all = int((y == 1).sum())
        missed = int((answered & ~said_res & (y == 1)).sum())

        return DrugEvaluation(
            drug=self.drug,
            ranking=ranking,
            correctly_closed=float(correct.mean()),
            missed_resistance=(missed / n_pos_all) if n_pos_all else float("nan"),
            requires_confirmation=self.requires_confirmation_,
            decision_sensitivity=bootstrap_ci(
                lambda a, b: sensitivity(a, b >= 0.5),
                y_ans, decided, n_boot, seed=self.random_state,
            ),
            decision_specificity=bootstrap_ci(
                lambda a, b: specificity(a, b >= 0.5),
                y_ans, decided, n_boot, seed=self.random_state + 1,
            ),
            n_evaluated=int(answered.sum()),
            n_abstained=int(abstained.sum()),
            n_by_catalogue=sum(1 for q in preds if q.source == "catalogue"),
        )
