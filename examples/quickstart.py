"""End-to-end drift-detector demo: synthetic data, CSV comparison, concept drift.

Run:  python examples/quickstart.py
"""

import csv
import os
import tempfile

import numpy as np

from drift_detector import (
    DDMDetector,
    DriftConfig,
    compare_schemas,
    detect_drift_csv,
    monitor_error_stream,
    report_to_json,
)

rng = np.random.default_rng(7)


def write_csv(path, ages, cities):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["age", "city"])
        writer.writerows(zip(ages, cities))


def main() -> None:
    tmp = tempfile.mkdtemp()
    ref_path = os.path.join(tmp, "reference.csv")
    cur_path = os.path.join(tmp, "current.csv")

    # Reference: what the model was trained on.
    write_csv(ref_path,
              ages=rng.normal(35, 5, 3000).round(1),
              cities=rng.choice(["austin", "seattle"], 3000, p=[0.6, 0.4]))
    # Current: serving data where the age distribution shifted older.
    write_csv(cur_path,
              ages=rng.normal(48, 5, 3000).round(1),
              cities=rng.choice(["austin", "seattle"], 3000, p=[0.6, 0.4]))

    print("== schema check ==")
    from drift_detector.drift_report import load_csv
    print(compare_schemas(load_csv(ref_path), load_csv(cur_path)))

    print("\n== data-drift report ==")
    report = detect_drift_csv(ref_path, cur_path, DriftConfig())
    print(report_to_json(report))
    print("Verdict:", report.verdict)
    assert report.verdict == "DRIFT DETECTED"

    print("\n== concept drift on an error stream ==")
    errors = [0] * 400 + [0] * 20 + [1] * 10 + [1] * 150  # model degrades
    final, fired = monitor_error_stream(errors, detector="ddm")
    print(f"DDM final state: {final}, fired at {len(fired)} positions")

    detector = DDMDetector()
    for i, e in enumerate(errors):
        if detector.update(e) == "drift":
            print(f"DDM raised drift at sample {i}")
            break

    print("\nDone — drift-detector works as expected.")


if __name__ == "__main__":
    main()
