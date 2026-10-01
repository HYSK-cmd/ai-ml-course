from __future__ import annotations

import numpy as np


class GaussianMixture:
    """Full-covariance Gaussian mixture fit with EM.

    Init: run your KMeans (n_init=1, same random_state) and use its clusters for means/covs/weights.
    E-step: responsibilities in LOG space (log-sum-exp!) -- points far from every component must not
    produce nan. M-step: weighted means, covariances (+ reg_covar * I), weights.
    Stop when the mean log-likelihood improves by less than tol, or after max_iter.
    Attributes: weights_ (k,), means_ (k, d), covariances_ (k, d, d), converged_ (bool),
    lower_bound_history_ (mean log-likelihood after every E-step -- EM guarantees it never decreases).
    """

    def __init__(self, n_components: int = 1, max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6,
                 random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> GaussianMixture:
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """log p(x) for each row."""
        raise NotImplementedError

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
