"""Генератор синтетического набора изолятов.

Зачем он нужен. Данные CRyPTIC открыты, но занимают десятки гигабайт и
требуют отдельной загрузки; данные пациентов из клиник вообще не могут
попасть в репозиторий. Чтобы пайплайн можно было запустить и проверить
целиком в любой момент — от загрузки до отчёта, — нужен генератор,
воспроизводящий структуру реальных данных.

**Это не замена реальных данных.** Метрики, полученные на синтетике,
характеризуют корректность пайплайна, а не качество модели, и в отчётах
помечаются как синтетические. Для проверки гипотезы H1 нужен CRyPTIC.

Что воспроизводится из реальной структуры данных:

1. **Филогенетическая структура.** Изоляты приходят кластерами передачи:
   члены одного кластера почти идентичны. Без этого разделение по
   кластерам родства нечего было бы проверять.
2. **Неполнота каталога.** Часть устойчивости вызвана вариантами, которых
   в каталоге ВОЗ нет. Именно на них ML-уровень должен давать прирост над
   правилами — если этого не заложить, задача выродится.
3. **Неполная пенетрантность.** Наличие маркера не гарантирует
   фенотипическую устойчивость: типичные значения 0,90–0,98.
4. **Ошибка фенотипического теста.** Культуральный метод сам по себе
   неидеален, порядка 2–4 % расхождений при повторном тестировании.
5. **Необъяснённая устойчивость.** Небольшая доля устойчивых изолятов не
   несёт ни одного известного маркера — механизм неизвестен. Это ставит
   потолок достижимой чувствительности и не даёт получить нереалистичные
   100 %.
6. **Пропуски фенотипов.** Новые препараты тестируются реже старых.
7. **Временнóй дрейф.** Устойчивость к бедаквилину появляется лишь
   в поздние годы — препарат вошёл в широкую практику недавно.
8. **Различия между странами и линиями.** Доля МЛУ в Казахстане около
   26 % против 3,2 % в мире; линия Beijing (L2) ассоциирована с
   повышенной частотой лекарственной устойчивости.
"""

from __future__ import annotations

import numpy as np

from .catalogue import DRUGS, MutationCatalogue
from .schema import IsolateDataset

__all__ = ["SyntheticConfig", "generate_isolates", "NOVEL_MARKERS"]


#: Варианты, отсутствующие в каталоге, но реально повышающие устойчивость.
#: Моделируют неполноту каталога: правила их не видят, ML-уровень —
#: должен обнаружить. Обозначения намеренно правдоподобны (те же гены,
#: соседние кодоны), чтобы задача не решалась тривиально по имени гена.
NOVEL_MARKERS: dict[str, list[tuple[str, float]]] = {
    "RIF": [("rpoB_I491F", 0.85), ("rpoB_V170F", 0.70)],
    "INH": [("katG_W328L", 0.75), ("katG_D419H", 0.60)],
    "EMB": [("embB_D1024N", 0.55), ("embA_c-16g", 0.50)],
    "LEV": [("gyrA_G88C", 0.80)],
    "MXF": [("gyrA_G88C", 0.75)],
    "BDQ": [("Rv0678_R90C", 0.65), ("mmpL5_LoF", 0.45)],
    "CFZ": [("Rv0678_R90C", 0.60)],
    "LZD": [("rplC_T460C", 0.55)],
    "AMI": [("rrs_c1402t", 0.70)],
    "KAN": [("rrs_c1402t", 0.70), ("whiB7_c-73t", 0.50)],
    "ETH": [("ethR_LoF", 0.55)],
    "DLM": [("fbiD_LoF", 0.60)],
    "RFB": [("rpoB_I491F", 0.80)],
}

#: Доля МЛУ и объём выборки по странам. Казахстан выделен как основная
#: цель внедрения: доля МЛУ среди новых случаев там около 26 % против
#: 3,2 % в мире (ВОЗ), и данных из Центральной Азии в мировых наборах мало.
COUNTRY_PROFILE: dict[str, tuple[float, float]] = {
    # страна: (относительный вес выборки, множитель частоты устойчивости)
    "KZ": (0.08, 2.6),
    "RU": (0.10, 2.4),
    "UZ": (0.05, 2.2),
    "IN": (0.18, 1.4),
    "ZA": (0.14, 1.3),
    "CN": (0.12, 1.2),
    "BR": (0.08, 0.9),
    "GB": (0.09, 0.5),
    "DE": (0.06, 0.5),
    "PE": (0.10, 1.1),
}

#: Линии и их относительная склонность к накоплению устойчивости.
LINEAGE_PROFILE: dict[str, tuple[float, float]] = {
    "L2_Beijing": (0.32, 1.8),
    "L4_Euro_American": (0.38, 1.0),
    "L3_CAS": (0.15, 1.1),
    "L1_Indo_Oceanic": (0.11, 0.8),
    "L5_West_African": (0.04, 0.7),
}

#: Базовая частота устойчивости к препарату в популяции надзора и доля
#: изолятов, для которых фенотип вообще измеряется.
DRUG_PROFILE: dict[str, tuple[float, float]] = {
    # препарат: (базовая частота устойчивости, доля протестированных)
    "RIF": (0.16, 0.98),
    "RFB": (0.13, 0.88),
    "INH": (0.20, 0.98),
    "EMB": (0.10, 0.95),
    "LEV": (0.08, 0.92),
    "MXF": (0.08, 0.90),
    "BDQ": (0.02, 0.70),
    "LZD": (0.02, 0.78),
    "CFZ": (0.03, 0.72),
    "DLM": (0.02, 0.60),
    "AMI": (0.05, 0.85),
    "KAN": (0.07, 0.83),
    "ETH": (0.11, 0.86),
}

#: Год, с которого препарат вошёл в широкую практику. До этого момента
#: устойчивость к нему практически не встречается — селективного давления
#: ещё не было.
DRUG_INTRODUCED: dict[str, int] = {"BDQ": 2019, "DLM": 2020, "LZD": 2017, "CFZ": 2018}


class SyntheticConfig:
    """Параметры генератора.

    Args:
        n_isolates: число изолятов.
        mean_cluster_size: средний размер кластера передачи.
        year_start, year_end: период наблюдения.
        penetrance_known: вероятность фенотипической устойчивости при
            наличии маркера из каталога.
        unexplained_fraction: доля устойчивых изолятов, не несущих ни
            одного известного маркера — механизм неизвестен. Задаёт
            потолок достижимой чувствительности любой геномной модели.
        novel_fraction: доля устойчивых, несущих вариант, которого нет в
            каталоге. Именно на них ML-уровень даёт прирост над правилами.
        phenotype_error: вероятность ошибки фенотипического теста.
        seed: сид генератора.
    """

    def __init__(
        self,
        n_isolates: int = 6000,
        mean_cluster_size: float = 3.5,
        year_start: int = 2015,
        year_end: int = 2024,
        penetrance_known: float = 0.94,
        unexplained_fraction: float = 0.10,
        novel_fraction: float = 0.25,
        phenotype_error: float = 0.03,
        seed: int = 20260904,
    ) -> None:
        if n_isolates < 100:
            raise ValueError("at least 100 isolates are required")
        if not 0.5 <= penetrance_known <= 1.0:
            raise ValueError("penetrance_known must lie in [0.5, 1.0]")
        self.n_isolates = n_isolates
        self.mean_cluster_size = mean_cluster_size
        self.year_start = year_start
        self.year_end = year_end
        if not 0.0 <= unexplained_fraction + novel_fraction < 1.0:
            raise ValueError("unexplained_fraction + novel_fraction must be < 1")
        self.penetrance_known = penetrance_known
        self.unexplained_fraction = unexplained_fraction
        self.novel_fraction = novel_fraction
        self.phenotype_error = phenotype_error
        self.seed = seed


def _sample_categorical(rng, profile: dict[str, tuple[float, float]], size: int):
    keys = list(profile)
    weights = np.array([profile[k][0] for k in keys], dtype=float)
    weights /= weights.sum()
    return rng.choice(keys, size=size, p=weights)


def generate_isolates(
    config: SyntheticConfig | None = None,
    catalogue: MutationCatalogue | None = None,
) -> IsolateDataset:
    """Сгенерировать набор изолятов со структурой реальных данных.

    Returns:
        IsolateDataset с пометкой `synthetic=True` в метаданных. Эта
        пометка проходит через весь пайплайн до отчёта, чтобы синтетические
        метрики нельзя было по недосмотру предъявить как реальные.
    """
    cfg = config or SyntheticConfig()
    cat = catalogue or MutationCatalogue()
    rng = np.random.default_rng(cfg.seed)

    # --- Кластеры передачи ------------------------------------------------
    n_clusters = max(2, int(cfg.n_isolates / cfg.mean_cluster_size))
    sizes = 1 + rng.poisson(cfg.mean_cluster_size - 1, size=n_clusters)
    sizes = np.maximum(sizes, 1)
    # Подгоняем суммарный размер под заданное число изолятов.
    while sizes.sum() < cfg.n_isolates:
        sizes[rng.integers(n_clusters)] += 1
    order = rng.permutation(n_clusters)
    cluster_ids: list[int] = []
    for c in order:
        cluster_ids.extend([int(c)] * int(sizes[c]))
        if len(cluster_ids) >= cfg.n_isolates:
            break
    cluster_ids = np.array(cluster_ids[: cfg.n_isolates])
    n = cluster_ids.size

    # --- Признаки уровня кластера (наследуются всеми членами) ------------
    uniq_clusters = np.unique(cluster_ids)
    cl_country = dict(
        zip(
            uniq_clusters,
            _sample_categorical(rng, COUNTRY_PROFILE, uniq_clusters.size),
            strict=True,
        )
    )
    cl_lineage = dict(
        zip(
            uniq_clusters,
            _sample_categorical(rng, LINEAGE_PROFILE, uniq_clusters.size),
            strict=True,
        )
    )
    cl_year = dict(
        zip(
            uniq_clusters,
            rng.integers(cfg.year_start, cfg.year_end + 1, size=uniq_clusters.size),
            strict=True,
        )
    )

    countries = np.array([cl_country[c] for c in cluster_ids])
    lineages = np.array([cl_lineage[c] for c in cluster_ids])
    years = np.array([cl_year[c] for c in cluster_ids])

    # --- Даты -------------------------------------------------------------
    day_of_year = rng.integers(0, 365, size=n)
    collection = np.array([
        np.datetime64(f"{y}-01-01") + np.timedelta64(int(d), "D")
        for y, d in zip(years, day_of_year, strict=True)
    ])
    # Задержка депонирования: от 10 до ~120 дней. Именно она делает
    # submission_date единственно корректной датой для измерения упреждения.
    lag = 10 + rng.gamma(shape=2.0, scale=20.0, size=n).astype(int)
    submission = collection + np.array([np.timedelta64(int(x), "D") for x in lag])

    # --- Устойчивость -----------------------------------------------------
    mutations: list[set[str]] = [set() for _ in range(n)]
    phenotypes: dict[str, np.ndarray] = {}

    # Фон: филогенетические маркеры, не связанные с устойчивостью.
    # Нужны, чтобы модель училась их игнорировать, а не хвататься за
    # любой сигнал, коррелирующий с линией.
    for i in range(n):
        if lineages[i] == "L2_Beijing" and rng.random() < 0.75:
            mutations[i].add("rpoB_S488A")
        if rng.random() < 0.30:
            mutations[i].add("gyrA_S95T")
        if rng.random() < 0.15:
            mutations[i].add("embB_E378A")

    for drug in DRUGS:
        base_rate, tested_frac = DRUG_PROFILE[drug]
        markers = sorted(cat.resistance_markers(drug))
        novel = NOVEL_MARKERS.get(drug, [])

        # Вероятность устойчивости для конкретного изолята.
        p_res = np.full(n, base_rate, dtype=float)
        p_res *= np.array([COUNTRY_PROFILE[c][1] for c in countries])
        p_res *= np.array([LINEAGE_PROFILE[str(lin)][1] for lin in lineages])

        introduced = DRUG_INTRODUCED.get(drug)
        if introduced is not None:
            # До внедрения препарата селективного давления нет.
            p_res *= np.where(years >= introduced, 1.0, 0.05)
        p_res = np.clip(p_res, 0.0, 0.85)

        is_resistant = rng.random(n) < p_res
        phenotype = np.zeros(n, dtype=float)

        # Доли считаются от числа УСТОЙЧИВЫХ изолятов, а не от всей
        # выборки: иначе для редких препаратов (базовая частота 2 %) доля
        # необъяснённой устойчивости превысила бы единицу, и ни один
        # маркер вообще не был бы поставлен.
        thr_unexplained = cfg.unexplained_fraction
        thr_novel = thr_unexplained + cfg.novel_fraction

        for i in np.flatnonzero(is_resistant):
            roll = rng.random()
            if roll < thr_unexplained or not markers:
                # Механизм неизвестен. Это и есть потолок чувствительности
                # любой геномной модели: такую устойчивость не поймать.
                phenotype[i] = 1.0
            elif novel and roll < thr_novel:
                key, penetrance = novel[rng.integers(len(novel))]
                mutations[i].add(key)
                phenotype[i] = 1.0 if rng.random() < penetrance else 0.0
            else:
                key = markers[rng.integers(len(markers))]
                mutations[i].add(key)
                phenotype[i] = 1.0 if rng.random() < cfg.penetrance_known else 0.0

        # Ошибка фенотипического теста — двусторонняя.
        flip = rng.random(n) < cfg.phenotype_error
        phenotype[flip] = 1.0 - phenotype[flip]

        # Пропуски: новые препараты тестируются реже.
        untested = rng.random(n) > tested_frac
        phenotype[untested] = np.nan
        phenotypes[drug] = phenotype

    isolate_ids = np.array([f"SYN-{i:06d}" for i in range(n)])

    return IsolateDataset(
        isolate_ids=isolate_ids,
        mutations=mutations,
        phenotypes=phenotypes,
        lineages=lineages,
        countries=countries,
        collection_dates=collection,
        submission_dates=submission,
        clusters=cluster_ids,
        meta={
            "source": "synthetic",
            "synthetic": True,
            "seed": cfg.seed,
            "n_clusters": int(uniq_clusters.size),
            "penetrance_known": cfg.penetrance_known,
            "unexplained_fraction": cfg.unexplained_fraction,
            "novel_fraction": cfg.novel_fraction,
            "phenotype_error": cfg.phenotype_error,
            "warning": (
                "SYNTHETIC DATA. These metrics show that the pipeline is correct, "
                "not that the model is clinically good. Hypothesis H1 requires the "
                "CRyPTIC dataset."
            ),
        },
    )


#: Регионы Казахстана с оценкой недельного объёма секвенирования.
#: Различие объёмов принципиально: именно оно проверяет, действительно ли
#: иерархическая модель роста помогает регионам с малой выборкой.
KZ_REGIONS: dict[str, int] = {
    "Almaty": 40,
    "Astana": 35,
    "Shymkent": 25,
    "Aktobe": 12,
    "Karaganda": 15,
    "Pavlodar": 8,
    "Atyrau": 7,
    "Kostanay": 9,
}


def generate_lineage_counts(
    lineages: list[str] | None = None,
    regions: dict[str, int] | None = None,
    n_weeks: int = 30,
    true_advantage: dict[str, float] | None = None,
    seed: int = 7,
):
    """Сгенерировать счётчики линий по регионам и неделям.

    Данные порождаются той же мультиномиальной логистической моделью,
    которую подгоняет GV-Growth, — с известными истинными коэффициентами
    роста. Это позволяет проверить, восстанавливает ли модель то, что
    заложено, а не только «красиво ли выглядит график».

    Returns:
        Кортеж (counts, times, region_labels, lineages, true_advantage).
    """
    lins = lineages or ["L4_Euro_American", "L2_Beijing", "L2_Beijing_MDR", "L3_CAS"]
    regs = regions or KZ_REGIONS
    truth = true_advantage or {
        "L4_Euro_American": 0.0,     # референс
        "L2_Beijing": 0.02,
        "L2_Beijing_MDR": 0.11,      # растущая устойчивая линия
        "L3_CAS": -0.03,
    }
    rng = np.random.default_rng(seed)

    counts, times, region_labels = [], [], []
    for region, weekly_n in regs.items():
        # Региональный наклон отклоняется от общего — есть что стягивать.
        offsets = {lin: rng.normal(0.0, 0.02) for lin in lins}
        intercepts = np.array([rng.normal(0.0, 0.4) for _ in lins])
        intercepts[0] = 0.0
        slopes = np.array([truth[lin] + offsets[lin] for lin in lins])
        slopes[0] = 0.0

        for week in range(n_weeks):
            eta = intercepts + slopes * week
            p = np.exp(eta - eta.max())
            p /= p.sum()
            # Объём секвенирования колеблется от недели к неделе.
            n = max(1, int(rng.poisson(weekly_n)))
            counts.append(rng.multinomial(n, p))
            times.append(week)
            region_labels.append(region)

    return (
        np.array(counts, dtype=float),
        np.array(times, dtype=float),
        np.array(region_labels),
        lins,
        truth,
    )
