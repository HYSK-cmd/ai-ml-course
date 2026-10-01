"""Approximate nearest-neighbour search by maximum inner product (numpy only).

IVF (inverted file index, the core of FAISS IndexIVFFlat):
  train(vectors): k-means the vectors into n_lists centroids -- use YOUR forge.ml.KMeans.
  add(vectors, ids): assign each vector to its nearest centroid; store each list's vectors
                     CONTIGUOUSLY (one array per list, or one big array sorted by list + offsets).
  search(queries, k): for each query, score all centroids, take the n_probe best lists, score only
                      the vectors in those lists, return the top k.
Assign/probe lists by inner product with the centroids (for normalized vectors this matches L2).
Top-k selection: np.argpartition, then sort only those k.
"""
from __future__ import annotations

import numpy as np


def brute_force_search(vectors: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product. Returns (scores (Q, k), ids (Q, k)), each row sorted descending."""
    raise NotImplementedError


class IVFIndex:
    def __init__(self, n_lists: int = 100, n_probe: int = 8, seed: int = 0):
        raise NotImplementedError

    def train(self, vectors: np.ndarray) -> IVFIndex:
        raise NotImplementedError

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """ids default to 0..n-1."""
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as brute_force_search. If fewer than k candidates were scanned, pad ids with
        -1 and scores with -inf."""
        raise NotImplementedError
