"""drift_detector — ML drift detection for production data pipelines.

Data drift   : PSI, Kolmogorov–Smirnov, chi-square per-feature tests.
Concept drift: DDM, ADWIN-style sliding window, Page–Hinkley error monitors.
Schema drift : column/dtype comparison between reference and current frames.
"""

from drift_detector.data_drift import (
    psi,
    ks_drift,
    chi_square_drift,
    FeatureDriftResult,
    per_feature_drift,
)
from drift_detector.concept_drift import (
    DDMDetector,
    PageHinkleyDetector,
    AdwinDetector,
    monitor_error_stream,
)
from drift_detector.schema_drift import compare_schemas, SchemaDriftResult
from drift_detector.drift_report import (
    DriftConfig,
    DriftReport,
    detect_drift,
    detect_drift_csv,
    report_to_json,
)

__all__ = [
    "psi",
    "ks_drift",
    "chi_square_drift",
    "FeatureDriftResult",
    "per_feature_drift",
    "DDMDetector",
    "PageHinkleyDetector",
    "AdwinDetector",
    "monitor_error_stream",
    "compare_schemas",
    "SchemaDriftResult",
    "DriftConfig",
    "DriftReport",
    "detect_drift",
    "detect_drift_csv",
    "report_to_json",
]

__version__ = "1.0.0"
__author__ = "Anusha Mukka"
