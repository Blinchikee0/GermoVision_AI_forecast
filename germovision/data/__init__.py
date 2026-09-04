"""Часть 2 — данные.

Приводит все источники к единой структуре `IsolateDataset`: реальный
набор CRyPTIC, локальные выгрузки лабораторий, синтетический генератор
для проверки пайплайна.
"""

from .catalogue import (
    DRUG_GENES,
    DRUG_NAMES_RU,
    DRUGS,
    CatalogueEntry,
    MutationCatalogue,
)
from .cryptic import assign_clusters_by_genotype, load_cryptic
from .features import FeatureBuilder, FeatureMatrix
from .schema import IsolateDataset
from .synthetic import SyntheticConfig, generate_isolates

__all__ = [
    "DRUGS",
    "DRUG_NAMES_RU",
    "DRUG_GENES",
    "CatalogueEntry",
    "MutationCatalogue",
    "IsolateDataset",
    "FeatureBuilder",
    "FeatureMatrix",
    "load_cryptic",
    "assign_clusters_by_genotype",
    "generate_isolates",
    "SyntheticConfig",
]
