# drift-detector usage guide

## 1. Data drift — has the input distribution shifted?

Data drift means the features your model sees in production no longer look like
the features it was trained on. Check per feature:

```python
from drift_detector import psi, ks_drift, chi_square_drift, per_feature_drift
import numpy as np

# Numeric features: Population Stability Index + Kolmogorov–Smirnov
psi(reference_age, current_age)          # 0.0 = identical, >= 0.25 = alert
stat, p = ks_drift(reference_age, current_age)

# Categorical features: chi-square test of independence
stat, p = chi_square_drift(reference_city, current_city)   # p < 0.05 -> drifted

# All features at once (auto-picks the right test per dtype)
results = per_feature_drift(reference_cols, current_cols,
                            psi_threshold=0.25, p_value_threshold=0.05)
for r in results:
    if r.drifted:
        print(r.feature, r.test, round(r.score, 3), r.severity)
```

PSI bands follow industry convention: **< 0.10** no shift, **0.10–0.25**
investigate, **≥ 0.25** significant shift. The KS p-value is reported alongside
every numeric result as corroborating evidence.

## 2. Concept drift — is the model still right?

Concept drift means P(y | X) changed: the same inputs now produce different
outcomes, so error rates rise even when the input distribution looks stable.
Feed a stream of per-prediction outcomes (`0` = correct, `1` = error) into one
of the online detectors:

```python
from drift_detector import DDMDetector, PageHinkleyDetector, AdwinDetector

detector = DDMDetector()
for error in error_stream:
    state = detector.update(error)   # "ok" | "warning" | "drift"
    if state == "warning":
        # start buffering fresh training data
    if state == "drift":
        # retrain, then detector.reset()

# Or run a detector over a recorded stream in one call:
from drift_detector import monitor_error_stream
final, fired_indices = monitor_error_stream(errors, detector="adwin")
```

- **DDM** — best for gradual degradation; gives early `warning` before `drift`.
- **ADWIN-style** — good for abrupt shifts; compares a recent window against history.
- **Page–Hinkley** — sensitive to small sustained increases in error mean.

## 3. Schema drift — did the pipeline change shape?

Run first, before any statistics — a renamed column makes the other tests
meaningless:

```python
from drift_detector import compare_schemas
result = compare_schemas(reference_cols, current_cols)
print(result.columns_added, result.columns_removed, result.dtype_changed)
```

## 4. The full report and the CLI

```python
from drift_detector import DriftConfig, detect_drift_csv, report_to_json

config = DriftConfig(psi_threshold=0.25, p_value_threshold=0.05,
                     null_rate_threshold=0.10, fail_on="high")
report = detect_drift_csv("reference.csv", "current.csv", config)
report_to_json(report, "drift-report.json")
print(report.verdict)
```

```bash
drift-detector compare reference.csv current.csv \
    --psi-threshold 0.2 --p-value-threshold 0.01 --fail-on medium \
    -o drift-report.json
```

The JSON report contains the verdict, per-feature scores and severities, the
schema diff, row counts, and the config used — everything an incident review or
a model-monitoring dashboard needs.

## 5. Putting it in a pipeline

A typical nightly job:

1. Export today's serving features to `current.csv`.
2. `drift-detector compare reference.csv current.csv -o report.json`.
3. Alert (PagerDuty/Slack/email) when the verdict is `DRIFT DETECTED`; open a
   review ticket on `DRIFT SUSPECTED`.
4. Feed the model's logged error stream into `DDMDetector` for concept drift.
5. On confirmed drift: retrain, promote `current.csv` to the new reference.
