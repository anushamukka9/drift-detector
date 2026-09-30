"""Report assembly: run schema + data drift checks and emit a JSON report."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from drift_detector.data_drift import FeatureDriftResult, per_feature_drift
from drift_detector.schema_drift import SchemaDriftResult, compare_schemas


@dataclass
class DriftConfig:
    psi_threshold: float = 0.25
    p_value_threshold: float = 0.05
    null_rate_threshold: float = 0.10
    psi_buckets: int = 10
    fail_on: str = "high"  # severity level that flips the verdict

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DriftReport:
    schema: SchemaDriftResult
    features: List[FeatureDriftResult]
    config: DriftConfig
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reference_rows: int = 0
    current_rows: int = 0

    @property
    def drifted_features(self) -> List[FeatureDriftResult]:
        return [f for f in self.features if f.drifted]

    @property
    def verdict(self) -> str:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        worst = max((order[f.severity] for f in self.features), default=0)
        if self.schema.drifted or worst >= order.get(self.config.fail_on, 3):
            return "DRIFT DETECTED"
        if worst >= order["medium"]:
            return "DRIFT SUSPECTED"
        return "NO SIGNIFICANT DRIFT"

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "generated_at": self.generated_at,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "config": self.config.to_dict(),
            "schema_drift": self.schema.to_dict(),
            "drifted_features": [f.to_dict() for f in self.drifted_features],
            "features": [f.to_dict() for f in self.features],
        }


def detect_drift(
    reference: Dict[str, np.ndarray],
    current: Dict[str, np.ndarray],
    config: DriftConfig | None = None,
) -> DriftReport:
    """Run the full drift battery over two in-memory column dicts."""
    config = config or DriftConfig()
    schema = compare_schemas(reference, current, null_rate_threshold=config.null_rate_threshold)
    shared = sorted(set(reference) & set(current))
    features = per_feature_drift(
        reference, current,
        psi_threshold=config.psi_threshold,
        p_value_threshold=config.p_value_threshold,
        buckets=config.psi_buckets,
        features=shared,
    )
    return DriftReport(
        schema=schema,
        features=features,
        config=config,
        reference_rows=len(next(iter(reference.values()))) if reference else 0,
        current_rows=len(next(iter(current.values()))) if current else 0,
    )


def load_csv(path: str) -> Dict[str, np.ndarray]:
    """Load a CSV into a column dict; numeric columns become float arrays."""
    columns: Dict[str, List] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row found")
        for name in reader.fieldnames:
            columns[name] = []
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(row[name])
    out: Dict[str, np.ndarray] = {}
    for name, values in columns.items():
        try:
            out[name] = np.array([float(v) for v in values])
        except (TypeError, ValueError):
            out[name] = np.array(values, dtype=object)
    return out


def detect_drift_csv(reference_path: str, current_path: str,
                     config: DriftConfig | None = None) -> DriftReport:
    """Run the full drift battery over two CSV files."""
    return detect_drift(load_csv(reference_path), load_csv(current_path), config)


def report_to_json(report: DriftReport, path: str | None = None) -> str:
    """Serialize a report to JSON; optionally write it to ``path``."""
    text = json.dumps(report.to_dict(), indent=2)
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def report_to_markdown(report: DriftReport, path: str | None = None) -> str:
    """Render a report as human-readable Markdown; optionally write to ``path``."""
    lines = ["# Drift Report", ""]
    lines.append(f"**Verdict: {report.verdict}**")
    lines.append("")
    lines.append(f"Generated: {report.generated_at}")
    lines.append(
        f"Reference rows: {report.reference_rows} · "
        f"Current rows: {report.current_rows}"
    )
    lines.append("")

    lines.append("## Schema drift")
    lines.append("")
    schema = report.schema
    if not schema.drifted:
        lines.append("No schema changes detected.")
    else:
        if schema.columns_added:
            lines.append(f"Columns added: {', '.join(schema.columns_added)}")
        if schema.columns_removed:
            lines.append(f"Columns removed: {', '.join(schema.columns_removed)}")
        for col, change in schema.dtype_changed.items():
            lines.append(
                f"Dtype changed: `{col}` "
                f"{change['reference']} -> {change['current']}"
            )
        for col, shift in schema.null_rate_shift.items():
            lines.append(
                f"Null-rate shift: `{col}` "
                f"{shift['reference']:.3f} -> {shift['current']:.3f} "
                f"(delta {shift['delta']:+.3f})"
            )
    lines.append("")

    lines.append("## Per-feature drift")
    lines.append("")
    if not report.features:
        lines.append("No shared features to compare.")
    else:
        lines.append("| Feature | Kind | Test | Score | p-value | Severity | Drifted |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for f in report.features:
            p = f"{f.p_value:.4f}" if f.p_value is not None else "n/a"
            mark = "yes" if f.drifted else "no"
            lines.append(
                f"| `{f.feature}` | {f.kind} | {f.test} | {f.score:.4f} "
                f"| {p} | {f.severity} | {mark} |"
            )
    lines.append("")

    if report.drifted_features:
        lines.append("## Drifted features")
        lines.append("")
        for f in report.drifted_features:
            lines.append(
                f"- **{f.feature}** ({f.kind}, {f.test}): "
                f"score={f.score:.4f}, severity={f.severity}"
            )
        lines.append("")

    text = "\n".join(lines)
    if path:
        with open(path, "w") as fh:
            fh.write(text)
    return text
