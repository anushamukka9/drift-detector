"""Schema drift: structural comparison between reference and current datasets.

Catches the pipeline-breaking class of drift — columns added or removed,
dtype changes, and null-rate shifts — before any statistical test runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np


def _dtype_name(values: np.ndarray) -> str:
    kind = values.dtype.kind
    if kind in ("U", "S", "O"):
        return "categorical"
    if kind in ("i", "u"):
        return "integer"
    if kind == "f":
        return "float"
    if kind == "b":
        return "boolean"
    return str(values.dtype)


def _null_rate(values: np.ndarray) -> float:
    arr = values.ravel()
    if arr.dtype.kind == "f":
        return float(np.isnan(arr).mean())
    flat = arr.astype(object).ravel()
    return float(sum(v is None or (isinstance(v, float) and np.isnan(v)) for v in flat) / max(len(flat), 1))


@dataclass
class SchemaDriftResult:
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    dtype_changed: Dict[str, Dict[str, str]] = field(default_factory=dict)
    null_rate_shift: Dict[str, Dict[str, float]] = field(default_factory=dict)
    drifted: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def compare_schemas(
    reference: Dict[str, np.ndarray],
    current: Dict[str, np.ndarray],
    null_rate_threshold: float = 0.10,
) -> SchemaDriftResult:
    """Compare column sets, dtypes, and null rates between two datasets."""
    ref_cols = set(reference)
    cur_cols = set(current)

    added = sorted(cur_cols - ref_cols)
    removed = sorted(ref_cols - cur_cols)

    dtype_changed: Dict[str, Dict[str, str]] = {}
    null_rate_shift: Dict[str, Dict[str, float]] = {}

    for col in sorted(ref_cols & cur_cols):
        ref_dt, cur_dt = _dtype_name(reference[col]), _dtype_name(current[col])
        if ref_dt != cur_dt:
            dtype_changed[col] = {"reference": ref_dt, "current": cur_dt}
        ref_null, cur_null = _null_rate(reference[col]), _null_rate(current[col])
        if abs(cur_null - ref_null) > null_rate_threshold:
            null_rate_shift[col] = {
                "reference": round(ref_null, 4),
                "current": round(cur_null, 4),
                "delta": round(cur_null - ref_null, 4),
            }

    drifted = bool(added or removed or dtype_changed or null_rate_shift)
    return SchemaDriftResult(
        columns_added=added,
        columns_removed=removed,
        dtype_changed=dtype_changed,
        null_rate_shift=null_rate_shift,
        drifted=drifted,
    )
