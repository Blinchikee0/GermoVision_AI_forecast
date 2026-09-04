"""Разделение выборки — единственный разрешённый в проекте способ.

Правила (§ 5.7 мастер-документа):
1. Разделение по времени, не случайное.
2. Родственные объекты целиком в одной части.
3. Внешняя валидация по географии.
4. Аугментация только после разделения и только в train.
"""

from .cluster import cluster_by_distance, cluster_split, temporal_cluster_split
from .grouped import holdout_group, leave_one_group_out
from .guards import (
    GuardReport,
    LeakageGuard,
    augment_train_only,
    check_cluster_integrity,
    check_no_exact_duplicates,
    check_no_near_duplicates,
    check_temporal_order,
)
from .temporal import forward_chaining, temporal_split

__all__ = [
    "temporal_split",
    "forward_chaining",
    "cluster_split",
    "temporal_cluster_split",
    "cluster_by_distance",
    "holdout_group",
    "leave_one_group_out",
    "LeakageGuard",
    "GuardReport",
    "augment_train_only",
    "check_temporal_order",
    "check_cluster_integrity",
    "check_no_exact_duplicates",
    "check_no_near_duplicates",
]
