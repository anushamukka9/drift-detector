"""CLI: compare two CSVs for drift and print/write a JSON report.

Usage:
    drift-detector compare reference.csv current.csv [options]
"""

from __future__ import annotations

import argparse
import sys

from drift_detector import (
    DriftConfig,
    detect_drift_csv,
    report_to_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-detector",
        description="Detect data, concept, and schema drift between datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cmp = sub.add_parser("compare", help="Compare two CSV files for drift")
    cmp.add_argument("reference", help="Path to the reference (baseline) CSV")
    cmp.add_argument("current", help="Path to the current (production) CSV")
    cmp.add_argument("--psi-threshold", type=float, default=0.25,
                     help="PSI value at/above which a numeric feature drifts (default: 0.25)")
    cmp.add_argument("--p-value-threshold", type=float, default=0.05,
                     help="p-value below which a categorical feature drifts (default: 0.05)")
    cmp.add_argument("--null-rate-threshold", type=float, default=0.10,
                     help="Null-rate delta that flags schema drift (default: 0.10)")
    cmp.add_argument("--buckets", type=int, default=10,
                     help="PSI quantile buckets (default: 10)")
    cmp.add_argument("--fail-on", default="high", choices=["low", "medium", "high"],
                     help="Severity that flips the verdict (default: high)")
    cmp.add_argument("-o", "--output", default=None,
                     help="Write the JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        config = DriftConfig(
            psi_threshold=args.psi_threshold,
            p_value_threshold=args.p_value_threshold,
            null_rate_threshold=args.null_rate_threshold,
            psi_buckets=args.buckets,
            fail_on=args.fail_on,
        )
        report = detect_drift_csv(args.reference, args.current, config)
        text = report_to_json(report, args.output)
        print(text)
        print(f"\nVerdict: {report.verdict}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
