# drift-detector

Detect **data drift**, **concept drift**, and **schema drift** in ML pipelines — with a Python API, JSON and Markdown drift reports, a streaming batch monitor, and a CLI that compares two CSVs.

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
from drift_detector import detect_drift_csv, DriftConfig, report_to_markdown

report = detect_drift_csv("data/reference.csv", "data/current.csv",
                          DriftConfig(psi_threshold=0.25))
print(report.verdict)                      # NO SIGNIFICANT DRIFT | DRIFT SUSPECTED | DRIFT DETECTED
print([f.feature for f in report.drifted_features])
print(report_to_markdown(report))          # human-readable summary
```

For production, score serving data batch by batch without keeping the full
reference sample in memory:

```python
from drift_detector import StreamingDriftMonitor, DriftConfig

monitor = StreamingDriftMonitor(DriftConfig()).fit(reference_columns)
for batch_id, batch in serving_batches():      # each batch: {feature: array}
    result = monitor.update(batch, batch_id=batch_id)
    if result.alerted:
        alert(f"drift in batch {batch_id}: "
              f"{[f.feature for f in result.drifted_features]}")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable end-to-end
demo, [`examples/before_after.py`](examples/before_after.py) for a simulated
production incident (distribution shift + new column + null-rate jump), and
[`docs/usage.md`](docs/usage.md) for the full guide.

## What's inside

| Module | What it detects | Method |
|---|---|---|
| `data_drift` | Feature distribution shift | PSI + KS test (numeric), chi-square (categorical) |
| `concept_drift` | Model error-rate drift | DDM, ADWIN-style, Page–Hinkley online detectors |
| `schema_drift` | Structural pipeline changes | Column add/remove, dtype change, null-rate shift |
| `streaming` | Per-batch production drift | `StreamingDriftMonitor`: fit-once, score batches |
| `drift_report` | Unified verdict | `DriftConfig` thresholds + JSON/Markdown reports |

## Thresholds (defaults)

- **PSI**: < 0.10 none · 0.10–0.25 investigate · ≥ 0.25 drifted (infinite when a batch falls entirely outside the reference range)
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
