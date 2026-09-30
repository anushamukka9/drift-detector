"""Streaming drift monitoring: score production batches against a frozen reference.

Fit once on reference (training) data, then feed production batches as they
arrive. Each batch is compared against the *frozen* reference profile with the
same PSI / chi-square tests as :func:`drift_detector.per_feature_drift`, so
batches are O(batch size) in memory — the full reference sample is never kept.

Numeric features reuse the reference quantile breakpoints for every batch
(standard PSI practice: buckets must not move with the data being measured).
Categorical features reuse the reference value counts. The KS test is not used
here because it needs the raw reference sample; PSI decides numeric verdicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

from drift_detector.data_drift import (
    FeatureDriftResult,
    _categorical_counts,
    _chi_square_from_counts,
    _is_categorical,
    _psi_from_counts,
    _quantile_breakpoints,
    _severity_for_psi,
    _severity_for_pvalue,
    _to_1d,
)
from drift_detector.drift_report import DriftConfig


@dataclass
class BatchDriftResult:
    """Drift outcome for one production batch."""

    batch_id: str
    batch_rows: int
    features: List[FeatureDriftResult] = field(default_factory=list)
    verdict: str = "NO SIGNIFICANT DRIFT"

    @property
    def drifted_features(self) -> List[FeatureDriftResult]:
        return [f for f in self.features if f.drifted]

    @property
    def alerted(self) -> bool:
        return self.verdict == "DRIFT DETECTED"

    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "batch_rows": self.batch_rows,
            "verdict": self.verdict,
            "alerted": self.alerted,
            "drifted_features": [f.to_dict() for f in self.drifted_features],
            "features": [f.to_dict() for f in self.features],
        }


class StreamingDriftMonitor:
    """Stateful monitor: fit on reference data, then score batches.

    Example:
        monitor = StreamingDriftMonitor(DriftConfig(psi_threshold=0.2))
        monitor.fit(reference_columns)
        for batch_id, batch in production_batches():
            result = monitor.update(batch, batch_id=batch_id)
            if result.alerted:
                page_oncall(result)
        print(monitor.summary())
    """

    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()
        self._fitted = False
        self._numeric_breaks: Dict[str, np.ndarray] = {}
        self._numeric_ref_counts: Dict[str, np.ndarray] = {}
        self._categorical_ref_counts: Dict[str, Dict[str, int]] = {}
        self._features: List[str] = []
        self.history: List[BatchDriftResult] = []

    # -- fitting --------------------------------------------------------

    def fit(self, reference: Dict[str, np.ndarray]) -> "StreamingDriftMonitor":
        """Freeze the reference profile. Returns self for chaining."""
        if not reference:
            raise ValueError("fit requires a non-empty reference dataset")
        self._features = sorted(reference)
        self._numeric_breaks = {}
        self._numeric_ref_counts = {}
        self._categorical_ref_counts = {}
        for name in self._features:
            values = _to_1d(reference[name])
            if _is_categorical(values):
                self._categorical_ref_counts[name] = _categorical_counts(values)
            else:
                numeric = np.asarray(values, dtype=float)
                valid = numeric[~np.isnan(numeric)]  # nulls are schema drift's job
                if valid.size == 0:
                    self._numeric_breaks[name] = np.array([])
                    self._numeric_ref_counts[name] = np.array([], dtype=int)
                    continue
                breaks = _quantile_breakpoints(valid, self.config.psi_buckets)
                self._numeric_breaks[name] = breaks
                counts, _ = np.histogram(valid, bins=breaks)
                self._numeric_ref_counts[name] = counts
        self._fitted = True
        self.history = []
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    # -- scoring ----------------------------------------------------------

    def _score_numeric(self, name: str, values: np.ndarray) -> FeatureDriftResult:
        numeric = _to_1d(np.asarray(values, dtype=float))
        if numeric.size == 0:
            raise ValueError(f"batch feature {name!r} is empty")
        valid = numeric[~np.isnan(numeric)]  # nulls are schema drift's job
        breaks = self._numeric_breaks[name]
        if breaks.size == 0:
            # Reference had no valid values: drifted iff the batch has any.
            drifted = valid.size > 0
            return FeatureDriftResult(
                feature=name, kind="numeric", test="psi",
                score=float("inf") if drifted else 0.0,
                p_value=None, drifted=drifted,
                severity="high" if drifted else "none",
            )
        if breaks.size < 2:
            # Constant reference feature: any deviation in the batch is drift.
            drifted = bool(valid.size and np.any(valid != breaks[0]))
            return FeatureDriftResult(
                feature=name, kind="numeric", test="psi", score=0.0,
                p_value=None, drifted=drifted,
                severity="high" if drifted else "none",
            )
        if valid.size == 0:
            return FeatureDriftResult(
                feature=name, kind="numeric", test="psi",
                score=float("inf"), p_value=None, drifted=True, severity="high",
            )
        batch_counts, _ = np.histogram(valid, bins=breaks)
        score = _psi_from_counts(self._numeric_ref_counts[name], batch_counts)
        severity = _severity_for_psi(score)
        return FeatureDriftResult(
            feature=name, kind="numeric", test="psi", score=score,
            p_value=None, drifted=score >= self.config.psi_threshold,
            severity=severity,
        )

    def _score_categorical(self, name: str, values: np.ndarray) -> FeatureDriftResult:
        counts = _categorical_counts(values)
        if sum(counts.values()) == 0:
            raise ValueError(f"batch feature {name!r} is empty")
        stat, p = _chi_square_from_counts(self._categorical_ref_counts[name], counts)
        severity = _severity_for_pvalue(p, stat > len(counts))
        return FeatureDriftResult(
            feature=name, kind="categorical", test="chi-square",
            score=stat, p_value=p,
            drifted=p < self.config.p_value_threshold, severity=severity,
        )

    def _verdict(self, features: List[FeatureDriftResult]) -> str:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        worst = max((order[f.severity] for f in features), default=0)
        if worst >= order.get(self.config.fail_on, 3):
            return "DRIFT DETECTED"
        if worst >= order["medium"]:
            return "DRIFT SUSPECTED"
        return "NO SIGNIFICANT DRIFT"

    def update(
        self, batch: Dict[str, np.ndarray], batch_id: str | None = None
    ) -> BatchDriftResult:
        """Score one production batch against the frozen reference profile."""
        if not self._fitted:
            raise ValueError("call fit() before update()")
        missing = [n for n in self._features if n not in batch]
        if missing:
            raise ValueError(f"batch is missing reference features: {missing}")
        results = []
        for name in self._features:
            values = batch[name]
            if name in self._categorical_ref_counts:
                results.append(self._score_categorical(name, values))
            else:
                results.append(self._score_numeric(name, values))
        result = BatchDriftResult(
            batch_id=batch_id if batch_id is not None else f"batch-{len(self.history)}",
            batch_rows=int(_to_1d(batch[self._features[0]]).size),
            features=results,
            verdict=self._verdict(results),
        )
        self.history.append(result)
        return result

    # -- reporting --------------------------------------------------------

    def summary(self) -> List[Dict]:
        """One row per scored batch: id, verdict, and drifted feature names."""
        return [
            {
                "batch_id": r.batch_id,
                "verdict": r.verdict,
                "alerted": r.alerted,
                "drifted_features": [f.feature for f in r.drifted_features],
            }
            for r in self.history
        ]

    def alerts(self) -> List[BatchDriftResult]:
        """Batches whose verdict flipped to DRIFT DETECTED."""
        return [r for r in self.history if r.alerted]
