from __future__ import annotations

import numpy as np


class LinearRegression:
    """Ridge-regularized least squares, solved in closed form (no gradient descent).

    Minimize ||y - X w - b||^2 + l2 * ||w||^2   (the intercept b is NOT penalized).
    Hint from the math week: center X and y first, then the intercept falls out.
    Must not crash when X has collinear columns and l2 == 0 (use a least-squares solver, not inv()).
    Attributes after fit: coef_ (n_features,), intercept_ (float).
    """

    def __init__(self, l2: float = 0.0, fit_intercept: bool = True):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LogisticRegression:
    """Multinomial (softmax) logistic regression trained with YOUR forge.autograd + forge.optim.

    loss = mean cross-entropy + 0.5 * l2 * ||W||^2   (bias not penalized)
    Full-batch optimization for ``max_iter`` steps with forge.optim.Adam(lr).
    Labels can be any type (ints, strings): classes_ = np.unique(y); predict returns labels from classes_.
    Binary problems are just 2 classes.

    Attributes: classes_, coef_ (n_features, n_classes), intercept_ (n_classes,), loss_history_ (list[float]).
    """

    def __init__(self, lr: float = 0.1, l2: float = 0.0, max_iter: int = 500):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
