"""Data drift: per-feature statistical tests between reference and current data.

Numeric features    -> Population Stability Index (PSI) + Kolmogorov–Smirnov.
Categorical features -> chi-square test of independence on value counts.

Rule-of-thumb alert bands for PSI (industry convention):
    < 0.10  no significant shift
    0.10–0.25 moderate shift (investigate)
    >= 0.25 significant shift (alert)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def _to_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).ravel()


def _is_categorical(values: np.ndarray) -> bool:
    return values.dtype.kind in ("U", "S", "O")


def psi(expected: Sequence, actual: Sequence, buckets: int = 10,
        epsilon: float = 1e-6) -> float:
    """Population Stability Index between an expected and an actual distribution.

    Buckets are quantile cut-points of the expected distribution. ``epsilon``
    avoids division by zero when a bucket is empty in either sample.
    """
    exp = _to_1d(np.asarray(expected, dtype=float))
    act = _to_1d(np.asarray(actual, dtype=float))
    if exp.size == 0 or act.size == 0:
        raise ValueError("PSI requires non-empty inputs")
    if np.all(exp == exp[0]) and np.all(act == act[0]) and exp[0] == act[0]:
        return 0.0

    quantiles = np.linspace(0, 100, buckets + 1)
    breakpoints = np.unique(np.percentile(exp, quantiles))
    if breakpoints.size < 2:
        return 0.0  # constant feature in the reference sample

    exp_counts, _ = np.histogram(exp, bins=breakpoints)
    act_counts, _ = np.histogram(act, bins=breakpoints)

    exp_perc = exp_counts / exp_counts.sum()
    act_perc = act_counts / act_counts.sum()

    exp_perc = np.clip(exp_perc, epsilon, None)
    act_perc = np.clip(act_perc, epsilon, None)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def ks_drift(reference: Sequence, current: Sequence) -> Tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test; returns (statistic, p-value)."""
    ref = _to_1d(np.asarray(reference, dtype=float))
    cur = _to_1d(np.asarray(current, dtype=float))
    if ref.size == 0 or cur.size == 0:
        raise ValueError("KS test requires non-empty inputs")
    if np.all(ref == ref[0]) and np.all(cur == cur[0]) and ref[0] == cur[0]:
        return 0.0, 1.0
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def chi_square_drift(reference: Sequence, current: Sequence) -> Tuple[float, float]:
    """Chi-square test of independence on categorical value counts.

    Returns (statistic, p-value). Categories seen in either sample are aligned
    into one contingency table; unseen cells get count 0.
    """
    ref = _to_1d(np.asarray(reference, dtype=str))
    cur = _to_1d(np.asarray(current, dtype=str))
    if ref.size == 0 or cur.size == 0:
        raise ValueError("chi-square test requires non-empty inputs")

    categories = sorted(set(ref) | set(cur))
    if len(categories) == 1:
        return 0.0, 1.0

    table = np.array([
        [np.count_nonzero(ref == c) for c in categories],
        [np.count_nonzero(cur == c) for c in categories],
    ], dtype=float)
    statistic, pvalue, _, _ = stats.chi2_contingency(table)
    return float(statistic), float(pvalue)


@dataclass
class FeatureDriftResult:
    feature: str
    kind: str          # "numeric" | "categorical"
    test: str          # "psi" | "ks" | "chi-square"
    score: float       # PSI value, KS statistic, or chi-square statistic
    p_value: float | None
    drifted: bool
    severity: str      # "none" | "low" | "medium" | "high"

    def to_dict(self) -> Dict:
        return asdict(self)


def _severity_for_psi(value: float) -> str:
    if value >= 0.25:
        return "high"
    if value >= 0.10:
        return "medium"
    return "none"


def _severity_for_pvalue(p: float, score_high: bool) -> str:
    if p < 0.001 and score_high:
        return "high"
    if p < 0.05:
        return "medium"
    if p < 0.10:
        return "low"
    return "none"


def per_feature_drift(
    reference: Dict[str, np.ndarray],
    current: Dict[str, np.ndarray],
    psi_threshold: float = 0.25,
    p_value_threshold: float = 0.05,
    buckets: int = 10,
    features: List[str] | None = None,
) -> List[FeatureDriftResult]:
    """Run the appropriate drift test for every shared feature.

    Numeric features get both PSI and the KS test (PSI decides the verdict);
    categorical features get the chi-square test.
    """
    names = features or sorted(set(reference) & set(current))
    results: List[FeatureDriftResult] = []
    for name in names:
        ref = _to_1d(reference[name])
        cur = _to_1d(current[name])
        if _is_categorical(ref) or _is_categorical(cur):
            stat, p = chi_square_drift(ref, cur)
            severity = _severity_for_pvalue(p, stat > len(set(ref) | set(cur)))
            results.append(FeatureDriftResult(
                feature=name, kind="categorical", test="chi-square",
                score=stat, p_value=p,
                drifted=p < p_value_threshold, severity=severity,
            ))
        else:
            score = psi(ref, cur, buckets=buckets)
            ks_stat, ks_p = ks_drift(ref, cur)
            severity = _severity_for_psi(score)
            results.append(FeatureDriftResult(
                feature=name, kind="numeric", test="psi",
                score=score, p_value=ks_p,
                drifted=score >= psi_threshold, severity=severity,
            ))
    return results
