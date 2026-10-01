"""Shared grader helpers. These are the grader's own tools -- they never call your code's gradcheck."""
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"


def numgrad(f, arrays, eps=1e-6):
    """Central-difference gradient of scalar f(*arrays) w.r.t. each array (float64)."""
    grads = []
    for a in arrays:
        g = np.zeros_like(a, dtype=np.float64)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]
            a[i] = old + eps
            fp = f(*arrays)
            a[i] = old - eps
            fm = f(*arrays)
            a[i] = old
            g[i] = (fp - fm) / (2 * eps)
        grads.append(g)
    return grads


def need_data(*relpaths):
    """Skip the test (with a helpful message) if a dataset file is missing."""
    for rel in relpaths:
        if not (DATA / rel).exists():
            pytest.skip(f"missing data/{rel} -- run: python scripts/get_data.py")
    return [DATA / rel for rel in relpaths]
