"""Загрузка реальных данных: набор CRyPTIC и совместимые выгрузки.

Набор CRyPTIC — 12 289 клинических изолятов *M. tuberculosis* из 23 стран
пяти континентов, для каждого измерена МИК к 13 препаратам в едином
формате. Это крупнейший согласованный генотип-фенотипический набор по
туберкулёзу и основной источник обучения GV-Resist.

Данные не хранятся в репозитории: они занимают десятки гигабайт и
распространяются через публичный FTP EMBL-EBI. Загрузчик ожидает, что
пользователь скачал их сам, и работает с обобщённым табличным форматом,
чтобы подходить и для локальных выгрузок лаборатории.

Ожидаемая структура каталога:

    data/cryptic/
        samples.csv     — id, country, lineage, collection_date [, submission_date]
        mutations.csv   — id, gene, mutation
        phenotypes.csv  — id, drug, phenotype  (R/S либо 1/0)

Столбцы сопоставляются без учёта регистра; лишние игнорируются.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .catalogue import DRUGS
from .schema import IsolateDataset

__all__ = ["load_cryptic", "assign_clusters_by_genotype"]

_RESISTANT_TOKENS = {"r", "res", "resistant", "1", "1.0", "true"}
_SUSCEPTIBLE_TOKENS = {"s", "sus", "susceptible", "0", "0.0", "false"}


def _read_csv(path: Path, required: set[str]) -> tuple[list[dict], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"file not found: {path}. Expected the directory layout documented "
            "in germovision/data/cryptic.py"
        )
    with path.open(encoding="utf-8", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        missing = required - cols.keys()
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(cols)}"
            )
        return list(reader), cols


def assign_clusters_by_genotype(mutations: list[set[str]], threshold: int = 5) -> np.ndarray:
    """Приблизительная кластеризация родства по совпадению мутаций.

    Полноценная кластеризация требует попарных расстояний по всему геному
    (см. `core.splitting.cluster_by_distance`). Здесь используется
    приближение: изоляты, различающиеся не более чем `threshold`
    вариантами, объединяются одиночной связью.

    Приближение осознанно консервативно: оно скорее объединит лишнее, чем
    разделит родственников. Для защиты от утечки это правильная сторона
    ошибки — объединив лишнее, мы потеряем немного данных, а разделив
    родственников, получили бы завышенные метрики.

    Args:
        mutations: множества вариантов по изолятам.
        threshold: максимальное число различий внутри кластера.

    Returns:
        Массив меток кластеров.
    """
    n = len(mutations)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Кандидатов ищем через обратный индекс: сравнивать все пары при
    # десятках тысяч изолятов невозможно.
    index: dict[str, list[int]] = defaultdict(list)
    for i, muts in enumerate(mutations):
        for m in muts:
            index[m].append(i)

    for holders in index.values():
        if len(holders) > 500:  # неспецифичный маркер, для родства бесполезен
            continue
        for a_pos, i in enumerate(holders):
            for j in holders[a_pos + 1 :]:
                if find(i) == find(j):
                    continue
                if len(mutations[i] ^ mutations[j]) <= threshold:
                    union(i, j)

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def load_cryptic(
    directory: str | Path,
    cluster_threshold: int = 5,
    default_submission_lag_days: int = 30,
) -> IsolateDataset:
    """Загрузить набор изолятов из каталога с тремя CSV.

    Args:
        directory: каталог с `samples.csv`, `mutations.csv`, `phenotypes.csv`.
        cluster_threshold: порог кластеризации родства.
        default_submission_lag_days: задержка депонирования, применяемая,
            если в данных нет `submission_date`. Оценка упреждения при
            отсутствии реальной даты депонирования становится
            приблизительной, и об этом делается пометка в метаданных.

    Returns:
        IsolateDataset.

    Raises:
        FileNotFoundError: отсутствует один из файлов.
        ValueError: нет обязательных столбцов или ни одного изолята.
    """
    root = Path(directory)

    samples, scols = _read_csv(root / "samples.csv", {"id", "country", "collection_date"})
    muts_rows, mcols = _read_csv(root / "mutations.csv", {"id", "gene", "mutation"})
    phen_rows, pcols = _read_csv(root / "phenotypes.csv", {"id", "drug", "phenotype"})

    ids: list[str] = []
    countries: list[str] = []
    lineages: list[str] = []
    collection: list[np.datetime64] = []
    submission: list[np.datetime64] = []
    has_submission = "submission_date" in scols

    for row in samples:
        sid = str(row[scols["id"]]).strip()
        if not sid:
            continue
        ids.append(sid)
        countries.append(str(row[scols["country"]]).strip() or "UNK")
        lineages.append(
            str(row[scols["lineage"]]).strip() if "lineage" in scols else "unknown"
        )
        coll = np.datetime64(str(row[scols["collection_date"]]).strip()[:10], "D")
        collection.append(coll)
        if has_submission and str(row[scols["submission_date"]]).strip():
            submission.append(np.datetime64(str(row[scols["submission_date"]]).strip()[:10], "D"))
        else:
            submission.append(coll + np.timedelta64(default_submission_lag_days, "D"))

    if not ids:
        raise ValueError(f"no isolates found in {root / 'samples.csv'}")

    pos = {sid: i for i, sid in enumerate(ids)}

    mutations: list[set[str]] = [set() for _ in ids]
    for row in muts_rows:
        i = pos.get(str(row[mcols["id"]]).strip())
        if i is None:
            continue
        gene = str(row[mcols["gene"]]).strip()
        mut = str(row[mcols["mutation"]]).strip()
        if gene and mut:
            mutations[i].add(f"{gene}_{mut}")

    phenotypes = {d: np.full(len(ids), np.nan, dtype=float) for d in DRUGS}
    unknown_drugs: set[str] = set()
    for row in phen_rows:
        i = pos.get(str(row[pcols["id"]]).strip())
        if i is None:
            continue
        drug = str(row[pcols["drug"]]).strip().upper()[:3]
        if drug not in phenotypes:
            unknown_drugs.add(drug)
            continue
        token = str(row[pcols["phenotype"]]).strip().lower()
        if token in _RESISTANT_TOKENS:
            phenotypes[drug][i] = 1.0
        elif token in _SUSCEPTIBLE_TOKENS:
            phenotypes[drug][i] = 0.0
        # Прочее (промежуточная категория, брак теста) остаётся NaN.

    phenotypes = {d: a for d, a in phenotypes.items() if not np.isnan(a).all()}
    if not phenotypes:
        raise ValueError(
            "No recognised phenotype. Values R/S or 1/0 are expected, with drug "
            f"codes from {list(DRUGS)}"
        )

    clusters = assign_clusters_by_genotype(mutations, threshold=cluster_threshold)

    return IsolateDataset(
        isolate_ids=np.array(ids),
        mutations=mutations,
        phenotypes=phenotypes,
        lineages=np.array(lineages),
        countries=np.array(countries),
        collection_dates=np.array(collection),
        submission_dates=np.array(submission),
        clusters=clusters,
        meta={
            "source": str(root),
            "synthetic": False,
            "n_clusters": int(np.unique(clusters).size),
            "cluster_threshold": cluster_threshold,
            "submission_dates_real": has_submission,
            "unknown_drugs_skipped": sorted(unknown_drugs),
            "notes": (
                ""
                if has_submission
                else "submission_date is absent from the source and estimated as "
                f"collection_date + {default_submission_lag_days} days; "
                "lead-time measurement is therefore approximate"
            ),
        },
    )
