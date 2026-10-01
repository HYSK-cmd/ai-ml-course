from __future__ import annotations

import numpy as np


class PCA:
    """PCA via SVD of the centered data (not via eigh of the covariance -- know why).
    Attributes: mean_, components_ (n_components, d) orthonormal rows, explained_variance_ (ddof=1),
    explained_variance_ratio_ (fraction of TOTAL variance)."""

    def __init__(self, n_components: int):
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> PCA:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
