"""Gradient boosting (functional gradient descent) and random forests, built on YOUR trees."""
from __future__ import annotations

import numpy as np


class GradientBoostingRegressor:
    """Squared loss. F_0 = mean(y); each stage fits a DecisionTreeRegressor to the negative gradient
    (the residuals) and adds learning_rate * tree(x).
    subsample < 1 -> each stage trains on a random subset (without replacement) of that fraction.
    Attributes: init_ (float), estimators_ (list of trees), train_loss_ (MSE on the training set
    after each stage, len == n_estimators)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
                 min_samples_leaf: int = 1, subsample: float = 1.0, random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientBoostingClassifier:
    """BINARY classification with log-loss (labels: any two values; classes_ = np.unique(y)).

    F_0 = log-odds of the positive class (classes_[1]). Each stage:
      residual r = y - sigmoid(F)                       (negative gradient)
      fit a DecisionTreeRegressor to r                  (finds the partition)
      set each leaf's value to the NEWTON step  sum(r) / sum(p * (1 - p))  over its samples
      F += learning_rate * tree(x)
    Attributes: init_, estimators_, train_loss_ (mean log-loss after each stage)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
                 min_samples_leaf: int = 1, subsample: float = 1.0, random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
        raise NotImplementedError

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw score F(x) (log-odds)."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, 2) -- columns [P(classes_[0]), P(classes_[1])]."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomForestClassifier:
    """STRETCH. Bagging (bootstrap rows) + random feature subsets (max_features, default sqrt(d))
    over your DecisionTreeClassifier. predict_proba averages the trees' probabilities."""

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, max_features: int | None = None,
                 random_state: int | None = None):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
