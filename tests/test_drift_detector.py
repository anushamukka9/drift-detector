"""Test suite for drift_detector."""

import json
import math

import numpy as np
import pytest

from drift_detector import (
    DriftConfig,
    DDMDetector,
    PageHinkleyDetector,
    AdwinDetector,
    chi_square_drift,
    compare_schemas,
    detect_drift,
    detect_drift_csv,
    ks_drift,
    monitor_error_stream,
    per_feature_drift,
    psi,
    report_to_json,
)
from drift_detector.cli import main as cli_main

rng = np.random.default_rng(42)


# ---------------------------------------------------------------- PSI
def test_psi_identical_distributions_is_zero():
    a = rng.normal(0, 1, 2000)
    assert psi(a, rng.normal(0, 1, 2000)) < 0.05


def test_psi_shifted_distribution_is_high():
    ref = rng.normal(0, 1, 3000)
    cur = rng.normal(2.0, 1, 3000)
    assert psi(ref, cur) > 0.25


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        psi([], [1, 2, 3])


# ---------------------------------------------------------------- KS
def test_ks_same_distribution_high_pvalue():
    ref = rng.normal(0, 1, 1500)
    cur = rng.normal(0, 1, 1500)
    stat, p = ks_drift(ref, cur)
    assert stat < 0.08 and p > 0.05


def test_ks_shifted_distribution_low_pvalue():
    ref = rng.normal(0, 1, 1500)
    cur = rng.normal(1.5, 1, 1500)
    stat, p = ks_drift(ref, cur)
    assert p < 0.001 and stat > 0.2


# ---------------------------------------------------------------- chi-square
def test_chi_square_stable_categories():
    ref = rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])
    cur = rng.choice(["a", "b", "c"], size=2000, p=[0.5, 0.3, 0.2])
    _, p = chi_square_drift(ref, cur)
    assert p > 0.05


def test_chi_square_shifted_categories():
    ref = rng.choice(["a", "b"], size=2000, p=[0.9, 0.1])
    cur = rng.choice(["a", "b"], size=2000, p=[0.1, 0.9])
    stat, p = chi_square_drift(ref, cur)
    assert p < 0.001 and stat > 100


# ---------------------------------------------------------------- per-feature
def test_per_feature_drift_flags_shifted_numeric():
    ref = {"age": rng.normal(35, 5, 3000), "city": rng.choice(["x", "y"], 3000)}
    cur = {"age": rng.normal(55, 5, 3000), "city": rng.choice(["x", "y"], 3000)}
    results = per_feature_drift(ref, cur)
    by_name = {r.feature: r for r in results}
    assert by_name["age"].drifted and by_name["age"].kind == "numeric"
    assert not by_name["city"].drifted and by_name["city"].kind == "categorical"


def test_per_feature_drift_flags_shifted_categorical():
    ref = {"tier": rng.choice(["free", "paid"], size=3000, p=[0.8, 0.2])}
    cur = {"tier": rng.choice(["free", "paid"], size=3000, p=[0.2, 0.8])}
    (result,) = per_feature_drift(ref, cur)
    assert result.drifted and result.test == "chi-square"


# ---------------------------------------------------------------- schema
def test_schema_drift_detects_added_removed_and_dtype_change():
    ref = {"a": np.array([1, 2, 3]), "b": np.array(["x", "y", "z"])}
    cur = {"a": np.array([1.5, 2.5, 3.5]), "c": np.array([9, 9, 9])}
    result = compare_schemas(ref, cur)
    assert result.drifted
    assert result.columns_added == ["c"]
    assert result.columns_removed == ["b"]
    assert result.dtype_changed == {"a": {"reference": "integer", "current": "float"}}


def test_schema_drift_clean_when_identical():
    ref = {"a": np.array([1.0, 2.0]), "b": np.array(["x", "y"])}
    cur = {"a": np.array([1.0, 2.0]), "b": np.array(["x", "y"])}
    assert not compare_schemas(ref, cur).drifted


# ---------------------------------------------------------------- concept drift
def test_ddm_fires_on_error_rate_increase():
    stable = [0] * 400 + [1] * 20
    drifted = stable + [1] * 200
    d = DDMDetector()
    states = [d.update(e) for e in drifted]
    assert "drift" in states


def test_ddm_stays_ok_on_stable_errors():
    errors = [1 if i % 20 == 0 else 0 for i in range(500)]
    d = DDMDetector()
    assert all(d.update(e) != "drift" for e in errors)


def test_page_hinkley_fires_on_mean_shift():
    errors = [0] * 200 + [1] * 100
    d = PageHinkleyDetector(threshold=0.05)
    states = [d.update(e) for e in errors]
    assert states[-1] == "drift"


def test_adwin_fires_on_mean_shift():
    errors = [0] * 300 + [1] * 200
    d = AdwinDetector(window_size=100, min_samples=120)
    states = [d.update(e) for e in errors]
    assert "drift" in states


def test_monitor_error_stream_returns_fired_indices():
    errors = [0] * 300 + [1] * 150
    final, fired = monitor_error_stream(errors, detector="ddm")
    assert final == "drift"
    assert fired and all(isinstance(i, int) for i in fired)


def test_monitor_error_stream_rejects_unknown_detector():
    with pytest.raises(ValueError):
        monitor_error_stream([0, 1, 0], detector="nope")


# ---------------------------------------------------------------- report + CLI
def _csvs(tmp_path):
    ref = tmp_path / "ref.csv"
    cur = tmp_path / "cur.csv"
    ref.write_text("age,city\n" + "\n".join(
        f"{a},{c}" for a, c in zip(rng.normal(35, 5, 500), rng.choice(["x", "y"], 500))))
    cur.write_text("age,city,extra\n" + "\n".join(
        f"{a},{c},1" for a, c in zip(rng.normal(60, 5, 500), rng.choice(["x", "y"], 500))))
    return str(ref), str(cur)


def test_detect_drift_csv_verdict_and_schema(tmp_path):
    ref_path, cur_path = _csvs(tmp_path)
    report = detect_drift_csv(ref_path, cur_path, DriftConfig())
    assert report.verdict == "DRIFT DETECTED"
    assert report.schema.columns_added == ["extra"]
    assert any(f.feature == "age" and f.drifted for f in report.features)
    payload = json.loads(report_to_json(report))
    assert payload["verdict"] == "DRIFT DETECTED"


def test_report_no_drift_when_stable(tmp_path):
    ref = tmp_path / "r.csv"
    cur = tmp_path / "c.csv"
    rows = "\n".join(f"{a},{c}" for a, c in zip(rng.normal(35, 5, 800), rng.choice(["x", "y"], 800)))
    ref.write_text("age,city\n" + rows)
    cur.write_text("age,city\n" + rows)
    report = detect_drift_csv(str(ref), str(cur), DriftConfig())
    assert report.verdict == "NO SIGNIFICANT DRIFT"


def test_cli_compare_writes_json(tmp_path, capsys):
    ref_path, cur_path = _csvs(tmp_path)
    out = tmp_path / "report.json"
    rc = cli_main(["compare", ref_path, cur_path, "-o", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "DRIFT DETECTED"
    assert "drifted_features" in payload
