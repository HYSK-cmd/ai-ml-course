"""Data / prediction drift statistics for monitoring (numpy only)."""
from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index.
    Bin edges: the i/bins quantiles of ``expected`` (np.quantile, default linear method) for
    i = 1..bins-1, plus -inf and +inf at the ends. A value equal to an edge goes to the LEFT bin
    (np.searchsorted(inner_edges, x, side="left")). Proportions are clipped to at least eps.
    PSI = sum((a - e) * ln(a / e)). Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 shifted."""
    raise NotImplementedError


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |ECDF_a(x) - ECDF_b(x)| over all x.
    O((n+m) log(n+m)) -- sort, don't loop over pairs."""
    raise NotImplementedError


class DriftMonitor:
    """Keeps a reference sample and a rolling window (last ``window`` values) of live values.
    ``update(values)`` appends; ``report()`` -> {"psi": float, "ks": float, "drifted": bool} where
    drifted = psi > psi_threshold. Before the window has ``min_samples`` values, report() returns
    {"psi": nan, "ks": nan, "drifted": False}."""

    def __init__(self, reference: np.ndarray, window: int = 1000, min_samples: int = 100,
                 psi_threshold: float = 0.25):
        raise NotImplementedError

    def update(self, values) -> None:
        raise NotImplementedError

    def report(self) -> dict:
        raise NotImplementedError
