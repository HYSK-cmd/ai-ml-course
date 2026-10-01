from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """C[i, j] = number of samples with true label labels[i] predicted as labels[j].
    labels default: sorted union of y_true and y_pred."""
    raise NotImplementedError


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1) -> tuple[float, float, float]:
    """Binary precision, recall, F1 for pos_label. Define 0/0 as 0.0."""
    raise NotImplementedError


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC-AUC (y_true in {0,1}) = P(score of random positive > random negative), ties count 1/2.
    Must be O(n log n) (rank-based), not O(n_pos * n_neg)."""
    raise NotImplementedError


def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Binary log-loss; p = P(y=1), clipped to [eps, 1-eps]."""
    raise NotImplementedError


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError
