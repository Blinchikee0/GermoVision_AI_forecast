"""Каталог мутаций лекарственной устойчивости *M. tuberculosis*.

Первый уровень модели GV-Resist — это не машинное обучение, а применение
референсного стандарта. Каталог мутаций ВОЗ (2-е изд., 2023) построен на
анализе более 52 000 изолятов с сопряжёнными данными секвенирования и
фенотипического тестирования из 67 стран и классифицирует свыше 30 000
вариантов по уровню доказательности связи с резистентностью.

Врач должен видеть, что вывод системы соответствует официальному
стандарту, а не «мнению нейросети». Поэтому правила каталога имеют
приоритет над предсказанием ML-модели, и каждое заключение ссылается на
конкретную строку каталога.

Здесь закодировано ядро каталога — наиболее изученные и клинически
значимые варианты. Полный каталог ВОЗ подключается через
`MutationCatalogue.from_who_tsv()`; встроенное ядро используется как
запасной вариант и для воспроизводимых тестов.

Градация уровней доказательности ВОЗ:
    1 — ассоциирован с резистентностью
    2 — ассоциирован с резистентностью (промежуточный)
    3 — неопределённая значимость
    4 — не ассоциирован с резистентностью (промежуточный)
    5 — не ассоциирован с резистентностью
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DRUGS",
    "DRUG_NAMES_RU",
    "DRUG_GENES",
    "CatalogueEntry",
    "MutationCatalogue",
    "CORE_CATALOGUE",
]

#: 13 препаратов планшета UKMYC6, использованного консорциумом CRyPTIC.
#: Пиразинамида в списке нет: его тестирование требует кислой среды и на
#: этом планшете не проводится.
DRUGS: tuple[str, ...] = (
    "RIF",  # рифампицин
    "RFB",  # рифабутин
    "INH",  # изониазид
    "EMB",  # этамбутол
    "LEV",  # левофлоксацин
    "MXF",  # моксифлоксацин
    "BDQ",  # бедаквилин
    "LZD",  # линезолид
    "CFZ",  # клофазимин
    "DLM",  # деламанид
    "AMI",  # амикацин
    "KAN",  # канамицин
    "ETH",  # этионамид
)

DRUG_NAMES_RU: dict[str, str] = {
    "RIF": "Рифампицин",
    "RFB": "Рифабутин",
    "INH": "Изониазид",
    "EMB": "Этамбутол",
    "LEV": "Левофлоксацин",
    "MXF": "Моксифлоксацин",
    "BDQ": "Бедаквилин",
    "LZD": "Линезолид",
    "CFZ": "Клофазимин",
    "DLM": "Деламанид",
    "AMI": "Амикацин",
    "KAN": "Канамицин",
    "ETH": "Этионамид",
}

#: Гены, в которых для каждого препарата известны механизмы устойчивости.
#: Используется для отбора признаков и для OOD-проверки: мутация в гене,
#: не связанном с препаратом, не должна влиять на предсказание.
DRUG_GENES: dict[str, tuple[str, ...]] = {
    "RIF": ("rpoB",),
    "RFB": ("rpoB",),
    "INH": ("katG", "fabG1", "inhA", "ahpC"),
    "EMB": ("embB", "embA", "embC"),
    "LEV": ("gyrA", "gyrB"),
    "MXF": ("gyrA", "gyrB"),
    "BDQ": ("Rv0678", "atpE", "pepQ"),
    "LZD": ("rrl", "rplC"),
    "CFZ": ("Rv0678", "pepQ"),
    "DLM": ("ddn", "fbiA", "fbiB", "fbiC", "fgd1"),
    "AMI": ("rrs",),
    "KAN": ("rrs", "eis"),
    "ETH": ("ethA", "fabG1", "inhA"),
}


@dataclass(frozen=True)
class CatalogueEntry:
    """Строка каталога: связь варианта с устойчивостью к препарату.

    Args:
        gene: имя гена.
        mutation: обозначение варианта — аминокислотная замена (`S450L`),
            промоторная замена (`c-15t`) или обобщённая потеря функции (`LoF`).
        drug: код препарата.
        group: уровень доказательности ВОЗ (1–5).
        note: краткое пояснение для отображения врачу.
    """

    gene: str
    mutation: str
    drug: str
    group: int
    note: str = ""

    @property
    def key(self) -> str:
        """Канонический идентификатор варианта, например `rpoB_S450L`."""
        return f"{self.gene}_{self.mutation}"

    @property
    def confers_resistance(self) -> bool:
        """Уровни 1 и 2 трактуются как основание для вывода об устойчивости."""
        return self.group in (1, 2)


def _entries() -> list[CatalogueEntry]:
    """Ядро каталога: наиболее изученные варианты по 13 препаратам."""
    e: list[CatalogueEntry] = []

    # --- Рифампицин: область RRDR гена rpoB (кодоны 426–452) -------------
    # Замены в этой области дают более 95 % всей устойчивости к рифампицину.
    rif_high = [
        "S450L", "S450W", "H445Y", "H445D", "H445R",
        "D435V", "D435Y", "Q432K", "Q432P",
    ]
    for m in rif_high:
        e.append(
            CatalogueEntry("rpoB", m, "RIF", 1, "RRDR rpoB — устойчивость к рифампицину")
        )
    e.append(CatalogueEntry("rpoB", "L452P", "RIF", 1, "RRDR rpoB"))
    e.append(
        CatalogueEntry("rpoB", "L430P", "RIF", 2, "RRDR rpoB, пограничная устойчивость")
    )
    e.append(CatalogueEntry("rpoB", "D435V", "RIF", 1, "RRDR rpoB"))

    # Рифабутин: перекрёстная устойчивость неполная. Часть замен rpoB даёт
    # устойчивость к рифампицину при сохранении чувствительности к
    # рифабутину — клинически важное различие, которое нельзя терять.
    for m in ["S450L", "S450W", "H445Y", "H445R", "Q432K"]:
        e.append(
            CatalogueEntry("rpoB", m, "RFB", 1, "перекрёстная устойчивость с рифампицином")
        )
    for m in ["D435V", "D435Y", "L452P", "L430P"]:
        e.append(
            CatalogueEntry(
                "rpoB", m, "RFB", 4,
                "устойчивость к RIF при сохранении чувствительности к RFB",
            )
        )

    # --- Изониазид -------------------------------------------------------
    e.append(CatalogueEntry("katG", "S315T", "INH", 1, "katG S315T — высокий уровень устойчивости"))
    e.append(CatalogueEntry("katG", "S315N", "INH", 1, "katG S315N"))
    e.append(CatalogueEntry("katG", "LoF", "INH", 1, "потеря функции katG"))
    e.append(
        CatalogueEntry("fabG1", "c-15t", "INH", 1, "промотор inhA — низкий уровень устойчивости")
    )
    e.append(CatalogueEntry("fabG1", "c-8t", "INH", 2, "промотор inhA"))
    e.append(CatalogueEntry("inhA", "S94A", "INH", 1, "мишень inhA"))
    e.append(CatalogueEntry("inhA", "I194T", "INH", 2, "мишень inhA"))

    # --- Этамбутол -------------------------------------------------------
    embb = [("M306V", 1), ("M306I", 1), ("M306L", 2), ("G406A", 1), ("G406D", 1), ("Q497R", 1)]
    for m, g in embb:
        e.append(CatalogueEntry("embB", m, "EMB", g, "embB — мишень этамбутола"))

    # --- Фторхинолоны: QRDR gyrA (кодоны 88–94) --------------------------
    for m in ["A90V", "D94G", "D94N", "D94Y", "D94H", "D94A", "S91P"]:
        for drug in ("LEV", "MXF"):
            e.append(CatalogueEntry("gyrA", m, drug, 1, "QRDR gyrA — устойчивость к фторхинолонам"))
    for m in ["N538D", "E501D"]:
        for drug in ("LEV", "MXF"):
            e.append(CatalogueEntry("gyrB", m, drug, 2, "gyrB"))

    # --- Бедаквилин и клофазимин: общий механизм через Rv0678 ------------
    # Перекрёстная устойчивость реальна и клинически значима: потеря
    # функции Rv0678 снимает репрессию эффлюксной помпы MmpL5.
    e.append(CatalogueEntry("Rv0678", "LoF", "BDQ", 2, "Rv0678 — эффлюкс MmpL5"))
    e.append(CatalogueEntry("Rv0678", "LoF", "CFZ", 2, "перекрёстная устойчивость BDQ/CFZ"))
    e.append(CatalogueEntry("atpE", "D28N", "BDQ", 1, "atpE — мишень бедаквилина"))
    e.append(CatalogueEntry("atpE", "A63P", "BDQ", 1, "atpE — мишень бедаквилина"))
    e.append(CatalogueEntry("pepQ", "LoF", "BDQ", 3, "pepQ — значимость не установлена"))

    # --- Линезолид -------------------------------------------------------
    e.append(CatalogueEntry("rrl", "g2814t", "LZD", 2, "23S рРНК"))
    e.append(CatalogueEntry("rrl", "g2270c", "LZD", 2, "23S рРНК"))
    e.append(CatalogueEntry("rplC", "C154R", "LZD", 1, "rplC — рибосомный белок L3"))

    # --- Деламанид: активируется ферментами биосинтеза F420 --------------
    for gene in ("ddn", "fbiA", "fbiB", "fbiC", "fgd1"):
        e.append(CatalogueEntry(gene, "LoF", "DLM", 2, "нарушение активации пролекарства"))

    # --- Аминогликозиды --------------------------------------------------
    e.append(
        CatalogueEntry("rrs", "a1401g", "AMI", 1, "16S рРНК — высокий уровень устойчивости")
    )
    e.append(CatalogueEntry("rrs", "a1484g", "AMI", 1, "16S рРНК"))
    e.append(CatalogueEntry("rrs", "a1401g", "KAN", 1, "перекрёстная устойчивость AMI/KAN"))
    e.append(CatalogueEntry("rrs", "a1484g", "KAN", 1, "16S рРНК"))
    for m in ["c-14t", "c-12t", "g-10a", "c-37t"]:
        e.append(
            CatalogueEntry("eis", m, "KAN", 1, "промотор eis — устойчивость только к канамицину")
        )

    # --- Этионамид: общая мишень с изониазидом ---------------------------
    e.append(CatalogueEntry("ethA", "LoF", "ETH", 1, "ethA — активация пролекарства"))
    e.append(CatalogueEntry("fabG1", "c-15t", "ETH", 1, "перекрёстная устойчивость INH/ETH"))
    e.append(CatalogueEntry("inhA", "S94A", "ETH", 1, "общая мишень с изониазидом"))

    # --- Варианты, НЕ связанные с устойчивостью --------------------------
    # Их присутствие в каталоге не менее важно: оно позволяет системе
    # уверенно сказать «эта мутация ни на что не влияет», вместо того
    # чтобы молчать и порождать необоснованную тревогу.
    e.append(CatalogueEntry("rpoB", "S488A", "RIF", 5, "филогенетический маркер, не влияет"))
    e.append(CatalogueEntry("embB", "E378A", "EMB", 5, "филогенетический маркер"))
    e.append(CatalogueEntry("gyrA", "S95T", "LEV", 5, "филогенетический полиморфизм"))
    e.append(CatalogueEntry("gyrA", "S95T", "MXF", 5, "филогенетический полиморфизм"))
    e.append(CatalogueEntry("gyrA", "E21Q", "LEV", 5, "филогенетический полиморфизм"))
    e.append(CatalogueEntry("gyrA", "G668D", "MXF", 5, "филогенетический полиморфизм"))

    return e


CORE_CATALOGUE: list[CatalogueEntry] = _entries()


class MutationCatalogue:
    """Каталог мутаций с быстрым поиском по варианту и по препарату.

    Example:
        >>> cat = MutationCatalogue()
        >>> hits = cat.lookup("rpoB_S450L", "RIF")
        >>> hits[0].group
        1
    """

    def __init__(self, entries: list[CatalogueEntry] | None = None) -> None:
        self.entries = list(entries if entries is not None else CORE_CATALOGUE)
        self._by_key_drug: dict[tuple[str, str], list[CatalogueEntry]] = defaultdict(list)
        self._by_drug: dict[str, list[CatalogueEntry]] = defaultdict(list)
        for entry in self.entries:
            self._by_key_drug[(entry.key, entry.drug)].append(entry)
            self._by_drug[entry.drug].append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, mutation_key: str, drug: str) -> list[CatalogueEntry]:
        """Найти строки каталога для варианта и препарата."""
        return list(self._by_key_drug.get((mutation_key, drug), []))

    def resistance_markers(self, drug: str) -> set[str]:
        """Ключи вариантов, дающих устойчивость к препарату (группы 1–2)."""
        return {e.key for e in self._by_drug.get(drug, []) if e.confers_resistance}

    def known_keys(self) -> set[str]:
        """Все варианты, присутствующие в каталоге."""
        return {e.key for e in self.entries}

    def genes_for(self, drug: str) -> tuple[str, ...]:
        """Гены, связанные с препаратом."""
        return DRUG_GENES.get(drug, ())

    def predict(self, mutations: set[str], drug: str) -> tuple[bool | None, list[CatalogueEntry]]:
        """Правило первого уровня GV-Resist.

        Args:
            mutations: ключи вариантов, найденных у изолята.
            drug: код препарата.

        Returns:
            Пара (решение, обоснование). Решение `True` — устойчив;
            `None` — каталог не даёт ответа, вопрос передаётся ML-уровню.
            Каталог никогда не утверждает чувствительность: отсутствие
            известного маркера не доказывает отсутствие устойчивости,
            поскольку механизмы изучены не полностью.
        """
        hits = [
            entry
            for key in mutations
            for entry in self.lookup(key, drug)
            if entry.confers_resistance
        ]
        if hits:
            hits.sort(key=lambda x: x.group)
            return True, hits
        return None, []

    @classmethod
    def from_who_tsv(cls, path: str | Path) -> MutationCatalogue:
        """Загрузить полный каталог ВОЗ из выгрузки в формате TSV.

        Ожидаются столбцы `gene`, `mutation`, `drug`, `group`; регистр и
        порядок значения не имеют. Дополнительные столбцы игнорируются,
        что позволяет подключать разные выпуски каталога без правки кода.

        Raises:
            FileNotFoundError: файл отсутствует.
            ValueError: отсутствуют обязательные столбцы.
        """
        import csv

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"каталог не найден: {p}")

        entries: list[CatalogueEntry] = []
        with p.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            required = {"gene", "mutation", "drug", "group"}
            missing = required - cols.keys()
            if missing:
                raise ValueError(f"в каталоге нет столбцов: {sorted(missing)}")

            for row in reader:
                try:
                    group = int(str(row[cols["group"]]).strip()[0])
                except (ValueError, IndexError):
                    continue
                entries.append(
                    CatalogueEntry(
                        gene=row[cols["gene"]].strip(),
                        mutation=row[cols["mutation"]].strip(),
                        drug=row[cols["drug"]].strip().upper()[:3],
                        group=group,
                        note="каталог ВОЗ",
                    )
                )
        if not entries:
            raise ValueError(f"каталог пуст после разбора: {p}")
        return cls(entries)
