from __future__ import annotations

import numpy as np


def kmeans_plusplus(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: first center uniform at random; each next center is a row of X sampled with
    probability proportional to D(x)^2 (squared distance to the nearest chosen center).
    Returns (n_clusters, n_features), every row is a row of X."""
    raise NotImplementedError


class KMeans:
    """Lloyd's algorithm with k-means++ init, best of n_init runs (lowest inertia).
    Stop a run when the centers move less than tol (sum of squared shifts) or after max_iter.
    If a cluster goes empty, re-seed it with the point farthest from its assigned center.
    Distances must be vectorized: ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2.
    Attributes: cluster_centers_, labels_, inertia_ (sum of squared distances), n_iter_."""

    def __init__(self, n_clusters: int = 8, n_init: int = 10, max_iter: int = 300, tol: float = 1e-4,
                 random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> KMeans:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
