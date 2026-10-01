"""STRETCH -- second stage: re-rank retrieved candidates with YOUR Week 2 GradientBoostingClassifier.

Features for a (user, item) pair -- all computed from TRAIN data only:
  retrieval score (two-tower dot product), item popularity (log count), user activity (log count),
  item mean rating, user mean rating, genre affinity (share of the user's train interactions whose
  genres overlap the item's). Add your own.
Training labels: split the TRAIN interactions once more in time (last 10% per user = "future"),
retrieve candidates for each user from the earlier part, label 1 if the candidate is in the future
part. Never use the test set to train the ranker. And train a SEPARATE two-tower on the earlier part
to produce those candidates and the retrieval-score feature: the full-train model has already seen
the "future" interactions, so its scores would leak the labels.
"""
from __future__ import annotations

import numpy as np


class FeatureBuilder:
    def __init__(self, train, ratings, item_genres: np.ndarray, user_vecs: np.ndarray, item_vecs: np.ndarray):
        """train: implicit train frame with u, i columns; ratings: explicit-rating train frame with u, i, rating."""
        raise NotImplementedError

    def build(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """(n_pairs, n_features) float array for aligned arrays of user and item ids."""
        raise NotImplementedError


class Ranker:
    def __init__(self, n_estimators: int = 100, max_depth: int = 3, learning_rate: float = 0.1):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ranker:
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
