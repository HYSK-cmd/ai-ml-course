"""CART decision trees.

Rules (match sklearn so the tests can compare):
* Candidate thresholds are midpoints between consecutive DISTINCT sorted values of a feature.
  A sample goes LEFT if x[feature] <= threshold.
* Choose the (feature, threshold) with the largest impurity decrease
  N_node*I(node) - N_left*I(left) - N_right*I(right). Split even if the decrease is 0
  (XOR needs that), as long as the node is impure.
* A node becomes a leaf if: it is pure, depth == max_depth, n_samples < min_samples_split,
  or no split leaves >= min_samples_leaf samples on each side.
* The split search must be VECTORIZED per feature (sort once, then prefix sums over the sorted
  labels/targets) -- the speed test fits a depth-8 tree on 16k rows x 8 features.
* Represent the tree however you like (flat arrays are fastest). Leaves get integer ids for apply().

Week 2 boosting needs to overwrite leaf values of a fitted DecisionTreeRegressor (Newton steps).
Design a way to do that (e.g. a method that takes {leaf_id: value}).
"""
from __future__ import annotations

import numpy as np


class DecisionTreeClassifier:
    """criterion: "gini" or "entropy". max_features: None (all) or int -- if int, each node considers
    a random subset of that many features (needed for the random forest stretch goal)."""

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 criterion: str = "gini", max_features: int | None = None, random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, n_classes) class frequencies of the leaf, columns ordered like classes_."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Integer leaf id for each row."""
        raise NotImplementedError

    def get_depth(self) -> int:
        """Root-only tree has depth 0."""
        raise NotImplementedError

    def get_n_leaves(self) -> int:
        raise NotImplementedError


class DecisionTreeRegressor:
    """Squared-error criterion; a leaf predicts the mean target of its training samples."""

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: int | None = None, random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_depth(self) -> int:
        raise NotImplementedError

    def get_n_leaves(self) -> int:
        raise NotImplementedError
