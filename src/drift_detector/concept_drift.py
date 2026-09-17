"""Concept drift: online error-rate monitors for supervised models.

Feed a stream of per-prediction error indicators (0 = correct, 1 = error).
Three classic detectors are included:

* :class:`DDMDetector` — Drift Detection Method (Gama et al. 2004). Tracks the
  error rate p and its std-dev s; warns when p + s > p_min + 2*s_min and
  alarms when p + s > p_min + 3*s_min.
* :class:`AdwinDetector` — simplified ADWIN: compares the mean of a recent
  window against the mean of the older history; alarms when the gap exceeds
  a Hoeffding-style bound.
* :class:`PageHinkleyDetector` — Page–Hinkley test for an increase in the
  mean of the error stream.

Each detector exposes ``update(error) -> str`` returning "ok", "warning"
or "drift", and ``reset()`` after retraining.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Tuple


class DDMDetector:
    """Drift Detection Method on a binary error stream."""

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0,
                 min_samples: int = 30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.errors = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.state = "ok"

    def update(self, error: int | bool) -> str:
        self.n += 1
        self.errors += int(bool(error))
        if self.n < self.min_samples:
            return "ok"
        p = self.errors / self.n
        s = math.sqrt(p * (1 - p) / self.n)
        if p + s < self.p_min + self.s_min:
            self.p_min, self.s_min = p, s
        if p + s > self.p_min + self.drift_level * self.s_min:
            self.state = "drift"
        elif p + s > self.p_min + self.warning_level * self.s_min:
            self.state = "warning"
        else:
            self.state = "ok"
        return self.state

    @property
    def error_rate(self) -> float:
        return self.errors / self.n if self.n else 0.0


class AdwinDetector:
    """ADWIN-style two-window mean-shift detector.

    Keeps the full history in a deque and a sliding recent window; the drift
    bound is a Hoeffding bound scaled by ``delta``. ``reset()`` drops history
    (call after a confirmed drift + retrain).
    """

    def __init__(self, window_size: int = 100, delta: float = 0.05,
                 min_samples: int = 60):
        self.window_size = window_size
        self.delta = delta
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self.recent: deque = deque(maxlen=self.window_size)
        self.history: deque = deque(maxlen=self.window_size * 4)
        self.state = "ok"

    def _bound(self, n0: int, n1: int) -> float:
        m = 1.0 / (1.0 / n0 + 1.0 / n1)  # harmonic mean
        return math.sqrt(math.log(2.0 / self.delta) / (2.0 * m))

    def update(self, error: int | bool) -> str:
        value = float(bool(error))
        self.recent.append(value)
        self.history.append(value)
        if len(self.history) < self.min_samples:
            return "ok"
        old = list(self.history)[: len(self.history) - len(self.recent)]
        if not old or not self.recent:
            return "ok"
        mean_old = sum(old) / len(old)
        mean_new = sum(self.recent) / len(self.recent)
        bound = self._bound(len(old), len(self.recent))
        self.state = "drift" if abs(mean_new - mean_old) > bound else "ok"
        return self.state


class PageHinkleyDetector:
    """Page–Hinkley test for an upward shift in the error stream mean."""

    def __init__(self, threshold: float = 0.05, alpha: float = 0.005,
                 min_samples: int = 30):
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.sum_diff = 0.0
        self.min_sum = 0.0
        self.state = "ok"

    def update(self, error: int | bool) -> str:
        value = float(bool(error))
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.sum_diff += value - self.mean - self.alpha
        self.min_sum = min(self.min_sum, self.sum_diff)
        if self.n >= self.min_samples and self.sum_diff - self.min_sum > self.threshold:
            self.state = "drift"
        else:
            self.state = "ok"
        return self.state


def monitor_error_stream(
    errors: List[int | bool],
    detector: str = "ddm",
    **kwargs,
) -> Tuple[str, List[int]]:
    """Run a named detector ("ddm" | "adwin" | "page-hinkley") over an error list.

    Returns (final_state, indices_where_drift_or_warning_fired).
    """
    detectors = {
        "ddm": DDMDetector,
        "adwin": AdwinDetector,
        "page-hinkley": PageHinkleyDetector,
    }
    if detector not in detectors:
        raise ValueError(f"unknown detector {detector!r}; choose from {sorted(detectors)}")
    d = detectors[detector](**kwargs)
    fired: List[int] = []
    final = "ok"
    for i, e in enumerate(errors):
        state = d.update(e)
        final = state
        if state in ("warning", "drift"):
            fired.append(i)
    return final, fired
