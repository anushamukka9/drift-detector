"""Before/after drift comparison: simulate a production data change end to end.

Generates a reference CSV (training data) and a current CSV (serving data)
where three things went wrong at once:

  1. the `age` distribution shifted older (data drift),
  2. a new `app_version` column appeared (schema drift: column added),
  3. the `income` null rate jumped from ~5% to ~30% (schema drift: nulls).

Then it runs the full drift battery, prints a readable summary, and writes
JSON and Markdown reports next to this script.

Run:  python examples/before_after.py
"""

import csv
import os
import tempfile

import numpy as np

from drift_detector import (
    DriftConfig,
    StreamingDriftMonitor,
    detect_drift_csv,
    report_to_json,
    report_to_markdown,
)
from drift_detector.drift_report import load_csv

rng = np.random.default_rng(2026)


def with_nulls(values, null_rate):
    out = []
    for v in values:
        out.append("NaN" if rng.random() < null_rate else f"{v:.1f}")
    return out


def write_reference(path):
    n = 5000
    ages = rng.normal(35, 5, n).round(1)
    incomes = rng.normal(70_000, 15_000, n).round(0)
    cities = rng.choice(["austin", "seattle", "denver"], n, p=[0.5, 0.3, 0.2])
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["age", "income", "city"])
        for a, inc, c in zip(ages, with_nulls(incomes, 0.05), cities):
            writer.writerow([a, inc, c])


def write_current(path):
    n = 5000
    ages = rng.normal(48, 6, n).round(1)          # drifted older
    incomes = rng.normal(70_000, 15_000, n).round(0)
    cities = rng.choice(["austin", "seattle", "denver"], n, p=[0.5, 0.3, 0.2])
    versions = rng.choice(["2.1", "2.2"], n)      # brand-new column
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["age", "income", "city", "app_version"])
        for a, inc, c, v in zip(ages, with_nulls(incomes, 0.30), cities, versions):
            writer.writerow([a, inc, c, v])


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp()
    ref_path = os.path.join(tmp, "reference.csv")
    cur_path = os.path.join(tmp, "current.csv")
    write_reference(ref_path)
    write_current(cur_path)

    print("== before/after comparison ==")
    report = detect_drift_csv(ref_path, cur_path, DriftConfig())
    print(f"Verdict: {report.verdict}")
    print(f"Schema added columns: {report.schema.columns_added}")
    print(f"Schema null-rate shifts: {sorted(report.schema.null_rate_shift)}")
    for f in report.drifted_features:
        print(f"Drifted: {f.feature} ({f.kind}, {f.test}) "
              f"score={f.score:.4f} severity={f.severity}")
    assert report.verdict == "DRIFT DETECTED"

    json_path = os.path.join(root, "before_after_report.json")
    md_path = os.path.join(root, "before_after_report.md")
    report_to_json(report, json_path)
    report_to_markdown(report, md_path)
    print(f"\nwrote {json_path}\nwrote {md_path}")

    print("\n== streaming monitor: same story, batch by batch ==")
    monitor = StreamingDriftMonitor(DriftConfig()).fit(load_csv(ref_path))
    current = load_csv(cur_path)
    n = len(current["age"])
    batch_size = 1000
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        batch = {name: values[sl] for name, values in current.items()
                 if name in ("age", "income", "city")}
        result = monitor.update(batch, batch_id=f"week-{(i // batch_size) + 1}")
        drifted = [f.feature for f in result.drifted_features]
        print(f"{result.batch_id}: {result.verdict}  drifted={drifted}")
    assert monitor.alerts(), "expected at least one alerting batch"

    print("\nDone — before/after comparison works as expected.")


if __name__ == "__main__":
    main()
