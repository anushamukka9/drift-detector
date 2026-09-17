# drift-detector

Detect **data drift**, **concept drift**, and **schema drift** in ML pipelines — with a Python API, a JSON drift report, and a CLI that compares two CSVs.

## Why

Models rarely fail because the code changed. They fail because the *world* changed:
incoming data drifts from the training distribution, prediction error rates creep
up, or an upstream pipeline quietly renames a column. `drift-detector` catches all
three, with one report and configurable alert thresholds.

## Install

```bash
pip install drift-detector        # (once published)
# or from source:
pip install .
```

Requires Python ≥ 3.9, `numpy`, `scipy`.

## Quickstart

```bash
# Compare today's serving data against your training reference:
drift-detector compare data/reference.csv data/current.csv -o report.json
```

```python
from drift_detector import detect_drift_csv, DriftConfig

report = detect_drift_csv("data/reference.csv", "data/current.csv",
                          DriftConfig(psi_threshold=0.25))
print(report.verdict)                      # NO SIGNIFICANT DRIFT | DRIFT SUSPECTED | DRIFT DETECTED
print([f.feature for f in report.drifted_features])
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable end-to-end
demo, and [`docs/usage.md`](docs/usage.md) for the full guide.

## What's inside

| Module | What it detects | Method |
|---|---|---|
| `data_drift` | Feature distribution shift | PSI + KS test (numeric), chi-square (categorical) |
| `concept_drift` | Model error-rate drift | DDM, ADWIN-style, Page–Hinkley online detectors |
| `schema_drift` | Structural pipeline changes | Column add/remove, dtype change, null-rate shift |
| `drift_report` | Unified verdict | `DriftConfig` thresholds + JSON report |

## Thresholds (defaults)

- **PSI**: < 0.10 none · 0.10–0.25 investigate · ≥ 0.25 drifted
- **chi-square**: drifted when p < 0.05
- **Schema**: drifted on any column add/remove, dtype change, or null-rate delta > 0.10
- Verdict flips to `DRIFT DETECTED` on schema drift or any feature at the
  configured `fail_on` severity (default `high`); tune everything via `DriftConfig`.

## Development

```bash
pytest                # run the test suite (PYTHONPATH=src without install)
```

## License

MIT — Copyright 2026 Anusha Mukka.
