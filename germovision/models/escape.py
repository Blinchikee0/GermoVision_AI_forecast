"""GV-Escape — оценка эволюционного риска аминокислотных замен.

Задача: по набору последовательностей одного белка сказать, какие замены
опасны — то есть жизнеспособны и при этом заметно меняют свойства белка.
Это прямой ответ на вопрос «какие мутации патогена стоит отслеживать».

Что здесь реализовано, а что нет
================================

Современный подход к этой задаче — белковые языковые модели (ESM-2) и
надстройки над ними вроде EVEscape. Они требуют десятков гигабайт весов
и видеокарты, и подключаются в проекте отдельно. Здесь реализован
**классический профильный метод**, который эти модели вытеснили, но не
обесценили: позиционно-специфичный профиль частот аминокислот.

Честная оценка возможностей: профильная модель считает позиции
независимыми и потому не видит эпистаз — зависимость эффекта замены от
других замен в том же геноме. Языковая модель это частично улавливает.
Зато профиль обучается за секунды на данных самого пользователя, не
требует ускорителя и полностью объясним: вклад каждой компоненты в
итоговый балл виден в выдаче. Для задачи «отранжировать замены и указать
на подозрительные» этого достаточно, а разница с ESM-2 должна быть
измерена, а не предположена, — протокол измерения описан в § 5.2
мастер-документа.

Как считается риск
==================

Риск раскладывается на три множителя — та же схема, что в EVEscape:

    Риск = Допустимость × Заметность × Новизна

**Допустимость** — насколько замена согласуется с тем, что эволюция уже
допускала в этой позиции. Оценивается по профилю частот со сглаживанием.
Замена в позиции, где встречались разные остатки, скорее сохранит
жизнеспособность; замена в строго консервативной позиции чаще летальна,
и такой вариант до популяции не дойдёт.

**Заметность** — насколько замена меняет физико-химические свойства
остатка: заряд, гидрофобность, объём. Замена лейцина на изолейцин
антигенно почти незаметна; замена глутамата на лизин переворачивает
заряд и способна разрушить связывание антитела.

**Новизна** — редкость замены в наблюдённых данных. Уже
распространившийся вариант отслеживать поздно; интерес представляет
редкий, но допустимый и заметный.

Если в заголовках последовательностей есть даты, дополнительно
оценивается **скорость роста** доли замены во времени: растущая доля —
самостоятельный сигнал раннего предупреждения, независимый от свойств
самой замены.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

__all__ = ["AMINO_ACIDS", "MutationRisk", "GVEscape"]

#: Стандартные двадцать аминокислот в алфавитном порядке однобуквенного кода.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

#: Гидрофобность по шкале Кайта — Дулиттла.
_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

#: Объём остатка в кубических ангстремах (значения Замятнина).
_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

#: Заряд при физиологическом pH. Гистидин заряжен частично.
_CHARGE = {a: 0.0 for a in AMINO_ACIDS} | {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1
}

#: Стандартный генетический код для трансляции нуклеотидных последовательностей.
_CODONS = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?")


def translate(sequence: str) -> str:
    """Перевести нуклеотидную последовательность в аминокислотную.

    Неполный последний кодон отбрасывается; кодоны с неоднозначными
    символами дают `X`; трансляция останавливается на первом стоп-кодоне,
    если он не последний.
    """
    seq = sequence.upper().replace("U", "T").replace("-", "")
    out: list[str] = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        aa = _CODONS.get(seq[i : i + 3], "X")
        if aa == "*":
            break
        out.append(aa)
    return "".join(out)


def _physicochemical_distance(wt: str, mut: str) -> float:
    """Нормированное физико-химическое расстояние между остатками в [0, 1]."""
    if wt not in _HYDROPATHY or mut not in _HYDROPATHY:
        return 0.5
    d_hydro = abs(_HYDROPATHY[wt] - _HYDROPATHY[mut]) / 9.0      # диапазон −4,5…4,5
    d_vol = abs(_VOLUME[wt] - _VOLUME[mut]) / 167.7              # диапазон 60,1…227,8
    d_charge = abs(_CHARGE[wt] - _CHARGE[mut]) / 2.0             # диапазон −1…1
    # Заряд весит больше остальных: его смена сильнее всего влияет на
    # связывание антител и на взаимодействие с рецептором.
    return float(np.clip(0.3 * d_hydro + 0.25 * d_vol + 0.45 * d_charge, 0.0, 1.0))


def _extract_date(header: str) -> np.datetime64 | None:
    """Вытащить дату из заголовка FASTA, если она там есть."""
    m = _DATE_RE.search(header)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        return None
    day = int(m.group(3)) if m.group(3) else 1
    day = min(max(day, 1), 28)
    try:
        return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "D")
    except ValueError:
        return None


@dataclass
class MutationRisk:
    """Оценка риска одной замены."""

    position: int
    wildtype: str
    mutant: str
    risk: float
    tolerance: float
    salience: float
    novelty: float
    count: int
    frequency: float
    conservation: float
    trend: float | None = None
    observed: bool = True

    @property
    def label(self) -> str:
        return f"{self.wildtype}{self.position}{self.mutant}"

    def as_dict(self) -> dict:
        return {
            "mutation": self.label,
            "position": self.position,
            "wildtype": self.wildtype,
            "mutant": self.mutant,
            "risk": round(self.risk, 4),
            "tolerance": round(self.tolerance, 4),
            "salience": round(self.salience, 4),
            "novelty": round(self.novelty, 4),
            "conservation": round(self.conservation, 4),
            "count": self.count,
            "frequency": round(self.frequency, 6),
            "trend_per_week": None if self.trend is None else round(self.trend, 5),
            "observed": self.observed,
        }


@dataclass
class EscapeReport:
    """Результат анализа набора последовательностей."""

    reference_id: str
    reference_length: int
    n_sequences: int
    n_used: int
    observed: list[MutationRisk] = field(default_factory=list)
    candidates: list[MutationRisk] = field(default_factory=list)
    hotspots: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    date_range: tuple[str, str] | None = None

    def summary(self) -> str:
        lines = [
            f"Reference: {self.reference_id} ({self.reference_length} aa)",
            f"Sequences: {self.n_sequences}, used {self.n_used}",
            f"Observed substitutions: {len(self.observed)}",
            f"Candidates not yet observed: {len(self.candidates)}",
        ]
        if self.date_range:
            lines.append(f"Period: {self.date_range[0]} — {self.date_range[1]}")
        return "\n".join(lines)


class GVEscape:
    """Профильная модель риска аминокислотных замен.

    Example:
        >>> model = GVEscape().fit(records)   # [(заголовок, последовательность), …]
        >>> report = model.analyze(top_candidates=50)
        >>> report.observed[0].label
    """

    def __init__(self, pseudocount: float = 0.5, min_sequences: int = 5) -> None:
        """
        Args:
            pseudocount: сглаживание профиля. Без него замена, ни разу не
                встреченная в позиции, получила бы нулевую вероятность и
                бесконечно низкую допустимость — а отсутствие в выборке из
                сотни последовательностей ничего не доказывает.
            min_sequences: минимум последовательностей для осмысленного профиля.
        """
        if pseudocount <= 0:
            raise ValueError("pseudocount must be positive")
        self.pseudocount = pseudocount
        self.min_sequences = min_sequences

        self.reference_: str = ""
        self.reference_id_: str = ""
        self.profile_: np.ndarray | None = None
        self.conservation_: np.ndarray | None = None
        self.observed_counts_: dict[tuple[int, str], int] = {}
        self.observed_dates_: dict[tuple[int, str], list[np.datetime64]] = {}
        self.dates_: list[np.datetime64 | None] = []
        self.n_sequences_: int = 0
        self.n_used_: int = 0
        self.notes_: list[str] = []

    # -- обучение ---------------------------------------------------------

    def fit(
        self,
        records: list[tuple[str, str]],
        reference: str | None = None,
        nucleotide: bool = False,
    ) -> GVEscape:
        """Построить профиль по набору последовательностей.

        Args:
            records: пары «заголовок, последовательность».
            reference: референсная последовательность. По умолчанию берётся
                самая частая длина, а из неё — первая последовательность:
                она заведомо сопоставима с большинством остальных.
            nucleotide: последовательности нуклеотидные и требуют трансляции.

        Raises:
            ValueError: последовательностей слишком мало или они несопоставимы.
        """
        if nucleotide:
            records = [(h, translate(s)) for h, s in records]
        records = [(h, s.upper()) for h, s in records if s]

        if len(records) < self.min_sequences:
            raise ValueError(
                f"at least {self.min_sequences} sequences are required, "
                f"{len(records)} given. A profile built on fewer carries no "
                "information about which substitutions evolution tolerates"
            )

        self.n_sequences_ = len(records)
        lengths = [len(s) for _, s in records]
        modal_len = int(np.bincount(np.array(lengths)).argmax())

        if reference is not None:
            self.reference_ = reference.upper()
            self.reference_id_ = "user-supplied"
        else:
            idx = next(i for i, ln in enumerate(lengths) if ln == modal_len)
            self.reference_ = records[idx][1]
            self.reference_id_ = records[idx][0][:80]

        ref_len = len(self.reference_)
        counts = np.full((ref_len, len(AMINO_ACIDS)), self.pseudocount, dtype=float)
        self.observed_counts_ = {}
        self.observed_dates_ = {}
        self.notes_ = []
        self.dates_ = []

        n_used = n_aligned = 0
        for header, seq in records:
            pairs = self._map_positions(seq)
            if pairs is None:
                continue
            if len(seq) != ref_len:
                n_aligned += 1
            n_used += 1
            date = _extract_date(header)
            self.dates_.append(date)

            for pos, aa in pairs:
                j = _AA_INDEX.get(aa)
                if j is None:
                    continue
                counts[pos, j] += 1.0
                wt = self.reference_[pos]
                if aa != wt and wt in _AA_INDEX:
                    key = (pos, aa)
                    self.observed_counts_[key] = self.observed_counts_.get(key, 0) + 1
                    if date is not None:
                        self.observed_dates_.setdefault(key, []).append(date)

        if n_used < self.min_sequences:
            raise ValueError(
                f"only {n_used} sequences could be mapped to the reference. "
                "Check that the file contains a single protein"
            )

        self.n_used_ = n_used
        if n_aligned:
            self.notes_.append(
                f"{n_aligned} sequences differed in length from the reference and were "
                "aligned; insertions and deletions are not included in the profile"
            )

        self.profile_ = counts / counts.sum(axis=1, keepdims=True)
        # Консервативность: единица минус нормированная энтропия Шеннона.
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(self.profile_ * np.log(self.profile_), axis=1)
        self.conservation_ = 1.0 - ent / np.log(len(AMINO_ACIDS))
        return self

    def _map_positions(self, seq: str) -> list[tuple[int, str]] | None:
        """Сопоставить остатки последовательности позициям референса.

        Для последовательностей равной длины сопоставление прямое — так
        устроен выровненный FASTA, самый частый случай. Для остальных
        используются совпадающие блоки: последовательности одного белка
        различаются единичными заменами и короткими вставками, и блочное
        сопоставление здесь и точнее, и на порядки быстрее полного
        динамического выравнивания.
        """
        ref = self.reference_
        if len(seq) == len(ref):
            return [(i, a) for i, a in enumerate(seq) if a.isalpha()]

        if not seq or abs(len(seq) - len(ref)) > 0.25 * len(ref):
            return None

        pairs: list[tuple[int, str]] = []
        matcher = SequenceMatcher(None, ref, seq, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # Равные блоки и замены равной длины дают позиционное
            # соответствие один к одному; всё остальное — вставки и
            # делеции, которые профиль замен не описывает.
            if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
                pairs.extend((i1 + k, seq[j1 + k]) for k in range(i2 - i1))
        return pairs if len(pairs) >= 0.5 * len(ref) else None

    # -- оценка риска -----------------------------------------------------

    def _tolerance(self, pos: int, wt: str, mut: str) -> float:
        """Насколько эволюция допускает такую замену в этой позиции."""
        assert self.profile_ is not None
        j_wt, j_mut = _AA_INDEX.get(wt), _AA_INDEX.get(mut)
        if j_wt is None or j_mut is None:
            return 0.0
        ratio = self.profile_[pos, j_mut] / max(self.profile_[pos, j_wt], 1e-9)
        # Логистическое сжатие: отношение частот меняется на порядки, а
        # балл должен оставаться в [0, 1] и быть сопоставимым между позициями.
        return float(1.0 / (1.0 + np.exp(-np.log(max(ratio, 1e-12)) / 2.0 - 2.0)))

    def _trend(self, key: tuple[int, str]) -> float | None:
        """Скорость роста доли замены во времени, на неделю.

        Доля замены сравнивается между первой и второй половиной периода;
        разность логитов, делённая на расстояние между серединами половин,
        даёт скорость роста.

        Возвращается только **статистически значимый** тренд. Без этой
        проверки редкая замена, случайно попавшая во вторую половину чаще
        первой, объявлялась бы растущей: на выборке из десятка наблюдений
        такое происходит постоянно, и список «растущих» заполнялся бы
        шумом. Сигнал раннего предупреждения, который срабатывает от шума,
        бесполезен — его перестают читать.
        """
        dates = self.observed_dates_.get(key, [])
        all_dates = [d for d in self.dates_ if d is not None]
        if len(dates) < 8 or len(all_dates) < 30:
            return None

        t_all = np.array([d.astype("datetime64[D]").astype(int) for d in all_dates])
        t_mut = np.array([d.astype("datetime64[D]").astype(int) for d in dates])
        span = t_all.max() - t_all.min()
        if span < 28:
            return None

        mid = t_all.min() + span / 2
        n1, n2 = int((t_all <= mid).sum()), int((t_all > mid).sum())
        k1, k2 = int((t_mut <= mid).sum()), int((t_mut > mid).sum())
        if n1 < 10 or n2 < 10:
            return None

        # Поправка Хальдейна — Анскомба: +0,5 к каждой ячейке таблицы.
        # Она же делает определённой стандартную ошибку при нулевых ячейках.
        a, b = k1 + 0.5, n1 - k1 + 0.5
        c, d = k2 + 0.5, n2 - k2 + 0.5
        delta = np.log(c / d) - np.log(a / b)
        se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
        if abs(delta) < 1.96 * se:
            return None

        weeks = span / 14.0  # расстояние между серединами половин, в неделях
        return float(delta / max(weeks, 1e-6))

    def _risk(self, pos: int, wt: str, mut: str, count: int) -> MutationRisk:
        assert self.profile_ is not None and self.conservation_ is not None
        tolerance = self._tolerance(pos, wt, mut)
        salience = _physicochemical_distance(wt, mut)
        freq = count / max(self.n_used_, 1)
        # Новизна: уже распространившийся вариант отслеживать поздно.
        novelty = float(np.exp(-8.0 * freq))
        risk = float(tolerance * salience * novelty) ** (1.0 / 3.0)

        return MutationRisk(
            position=pos + 1,  # нумерация позиций в биологии с единицы
            wildtype=wt,
            mutant=mut,
            risk=risk,
            tolerance=tolerance,
            salience=salience,
            novelty=novelty,
            count=count,
            frequency=freq,
            conservation=float(self.conservation_[pos]),
            trend=self._trend((pos, mut)) if count else None,
            observed=count > 0,
        )

    def analyze(self, top_candidates: int = 100, min_count: int = 1) -> EscapeReport:
        """Оценить наблюдённые замены и предложить кандидатов.

        Args:
            top_candidates: сколько ещё не наблюдавшихся замен вернуть.
                Именно они и есть механизм раннего предупреждения: оценить
                замену можно до того, как она встретится.
            min_count: минимальное число наблюдений замены.

        Raises:
            RuntimeError: модель не обучена.
        """
        if self.profile_ is None:
            raise RuntimeError("model is not fitted: call fit() first")

        observed = [
            self._risk(pos, self.reference_[pos], mut, count)
            for (pos, mut), count in self.observed_counts_.items()
            if count >= min_count and self.reference_[pos] in _AA_INDEX
        ]
        observed.sort(key=lambda r: -r.risk)

        seen = set(self.observed_counts_)
        candidates: list[MutationRisk] = []
        for pos, wt in enumerate(self.reference_):
            if wt not in _AA_INDEX:
                continue
            for mut in AMINO_ACIDS:
                if mut == wt or (pos, mut) in seen:
                    continue
                candidates.append(self._risk(pos, wt, mut, 0))
        candidates.sort(key=lambda r: -r.risk)
        candidates = candidates[:top_candidates]

        # Горячие точки: позиции с наибольшим числом разных наблюдённых замен.
        by_pos: dict[int, list[MutationRisk]] = {}
        for r in observed:
            by_pos.setdefault(r.position, []).append(r)
        hotspots = sorted(
            (
                {
                    "position": pos,
                    "wildtype": rs[0].wildtype,
                    "n_variants": len(rs),
                    "total_count": sum(r.count for r in rs),
                    "max_risk": round(max(r.risk for r in rs), 4),
                    "conservation": round(rs[0].conservation, 4),
                    "mutations": [r.label for r in sorted(rs, key=lambda r: -r.count)[:6]],
                }
                for pos, rs in by_pos.items()
            ),
            key=lambda h: (-h["n_variants"], -h["total_count"]),
        )[:25]

        dates = [d for d in self.dates_ if d is not None]
        date_range = (str(min(dates)), str(max(dates))) if len(dates) >= 2 else None
        notes = list(self.notes_)
        if not dates:
            notes.append(
                "The headers carry no dates, so growth rates cannot be estimated. "
                "Dates are recognised as YYYY-MM-DD anywhere in the header."
            )

        return EscapeReport(
            reference_id=self.reference_id_,
            reference_length=len(self.reference_),
            n_sequences=self.n_sequences_,
            n_used=self.n_used_,
            observed=observed,
            candidates=candidates,
            hotspots=hotspots,
            notes=notes,
            date_range=date_range,
        )
