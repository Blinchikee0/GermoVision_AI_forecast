"""Метрики проекта.

Соответствие метрики типу задачи — требование § 5.8. Метрики регрессии
(MAPE, MAE, MSE) в проекте не используются: задачи классификационные
и вероятностные, а MAPE к тому же неустойчив при значениях около нуля.
"""

from .calibration import (
    CalibrationReport,
    IsotonicCalibrator,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    reliability_curve,
)
from .classification import (
    ClassificationReport,
    MetricCI,
    bootstrap_ci,
    bootstrap_metrics,
    confusion_counts,
    evaluate_binary,
    pr_auc,
    precision_at_k,
    roc_auc,
    sensitivity,
    specificity,
)
from .forecasting import (
    CoverageReport,
    evaluate_coverage,
    interval_coverage,
    multinomial_log_score,
    persistence_forecast,
    ranked_probability_score,
)
from .leadtime import (
    EventOutcome,
    LeadTimeReport,
    OfficialEvent,
    Signal,
    evaluate_lead_time,
    lead_time_at_budget,
)

__all__ = [
    "MetricCI",
    "ClassificationReport",
    "confusion_counts",
    "sensitivity",
    "specificity",
    "precision_at_k",
    "pr_auc",
    "roc_auc",
    "bootstrap_ci",
    "bootstrap_metrics",
    "evaluate_binary",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
    "CalibrationReport",
    "evaluate_calibration",
    "IsotonicCalibrator",
    "multinomial_log_score",
    "ranked_probability_score",
    "interval_coverage",
    "CoverageReport",
    "evaluate_coverage",
    "persistence_forecast",
    "Signal",
    "OfficialEvent",
    "EventOutcome",
    "LeadTimeReport",
    "evaluate_lead_time",
    "lead_time_at_budget",
]
