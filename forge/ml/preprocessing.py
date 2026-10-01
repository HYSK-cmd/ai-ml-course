from __future__ import annotations

import numpy as np


class StandardScaler:
    """z = (x - mean_) / scale_, with scale_ = population std (ddof=0); constant columns get scale_ 1."""

    def fit(self, X: np.ndarray) -> StandardScaler:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
