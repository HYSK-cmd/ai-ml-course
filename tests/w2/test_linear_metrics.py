"""Week 2 autograder: linear models, preprocessing, metrics, model selection."""
import numpy as np
import pytest
from sklearn import datasets, linear_model
from sklearn import metrics as skm
from sklearn.model_selection import KFold as SkKFold

import forge.autograd
from forge.ml import KFold, LinearRegression, LogisticRegression, StandardScaler, cross_val_score, train_test_split
from forge.ml.metrics import (accuracy_score, confusion_matrix, log_loss, mean_squared_error, precision_recall_f1,
                              r2_score, roc_auc_score)

rng = np.random.default_rng(0)


# ------------------------------------------------------------------------------- preprocessing
def test_standard_scaler():
    X = rng.normal(3, 2, size=(100, 4))
    X[:, 2] = 7.0  # constant column
    s = StandardScaler().fit(X)
    Z = s.transform(X)
    np.testing.assert_allclose(Z[:, [0, 1, 3]].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z[:, [0, 1, 3]].std(0), 1, atol=1e-12)
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(s.inverse_transform(Z), X)
    np.testing.assert_allclose(StandardScaler().fit_transform(X), Z)


# ------------------------------------------------------------------------------- linear regression
@pytest.mark.parametrize("l2", [0.0, 0.5, 10.0])
def test_linear_regression_matches_sklearn(l2):
    X = rng.normal(size=(80, 5)) + 3
    y = X @ rng.normal(size=5) - 2 + rng.normal(scale=0.1, size=80)
    ours = LinearRegression(l2=l2).fit(X, y)
    ref = (linear_model.Ridge(alpha=l2) if l2 else linear_model.LinearRegression()).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-6, atol=1e-8)
    assert np.isclose(ours.intercept_, ref.intercept_, rtol=1e-6)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-6)


def test_linear_regression_collinear_no_crash():
    X = rng.normal(size=(50, 2))
    X = np.column_stack([X, X[:, 0] * 2])  # exactly collinear
    y = X[:, 0] + X[:, 1]
    m = LinearRegression().fit(X, y)
    np.testing.assert_allclose(m.predict(X), y, atol=1e-8)


# ------------------------------------------------------------------------------- logistic regression
def test_logistic_uses_your_autograd(monkeypatch):
    calls = {"n": 0}
    orig = forge.autograd.Tensor.backward

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(forge.autograd.Tensor, "backward", counting)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    LogisticRegression(max_iter=20).fit(X, y)
    assert calls["n"] >= 20, "LogisticRegression must train with forge.autograd (one backward per step)"


def test_logistic_digits_multiclass():
    X, y = datasets.load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(lr=0.05, l2=1e-3, max_iter=300).fit(sc.transform(Xtr), ytr)
    acc = accuracy_score(yte, m.predict(sc.transform(Xte)))
    assert acc >= 0.95, f"test accuracy {acc:.3f}"
    P = m.predict_proba(sc.transform(Xte))
    assert P.shape == (len(Xte), 10)
    np.testing.assert_allclose(P.sum(1), 1)
    assert m.coef_.shape == (64, 10) and m.intercept_.shape == (10,)
    assert m.loss_history_[-1] < m.loss_history_[0] / 5


def test_logistic_string_labels_binary():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    labels = np.where(y == 1, "benign", "malignant")
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, random_state=1)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(lr=0.05, l2=1e-2, max_iter=300).fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    assert set(pred) <= {"benign", "malignant"}
    assert accuracy_score(yte, pred) >= 0.95


# ------------------------------------------------------------------------------- metrics
def test_roc_auc_with_ties_matches_sklearn():
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)  # rounding creates many ties
    assert np.isclose(roc_auc_score(y, s), skm.roc_auc_score(y, s), atol=1e-12)
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert roc_auc_score([0, 1], [0.5, 0.5]) == 0.5


def test_roc_auc_is_fast():
    import time
    y = rng.integers(0, 2, 400_000)
    s = rng.normal(size=400_000)
    t = time.perf_counter()
    roc_auc_score(y, s)
    assert time.perf_counter() - t < 3.0, "must be O(n log n)"


def test_classification_metrics():
    yt = rng.integers(0, 3, 200)
    yp = np.where(rng.random(200) < 0.7, yt, rng.integers(0, 3, 200))
    np.testing.assert_array_equal(confusion_matrix(yt, yp), skm.confusion_matrix(yt, yp))
    np.testing.assert_array_equal(confusion_matrix(yt, yp, labels=np.array([2, 0])),
                                  skm.confusion_matrix(yt, yp, labels=[2, 0]))
    assert np.isclose(accuracy_score(yt, yp), skm.accuracy_score(yt, yp))
    yb, pb = (yt == 1).astype(int), (yp == 1).astype(int)
    p, r, f = precision_recall_f1(yb, pb)
    assert np.allclose([p, r, f], [skm.precision_score(yb, pb), skm.recall_score(yb, pb), skm.f1_score(yb, pb)])
    assert precision_recall_f1(np.array([1, 1]), np.array([0, 0])) == (0.0, 0.0, 0.0)


def test_regression_and_prob_metrics():
    y = rng.normal(size=100)
    yhat = y + rng.normal(scale=0.3, size=100)
    assert np.isclose(mean_squared_error(y, yhat), skm.mean_squared_error(y, yhat))
    assert np.isclose(r2_score(y, yhat), skm.r2_score(y, yhat))
    yb = rng.integers(0, 2, 100)
    p = rng.random(100)
    p[0], p[1] = 0.0, 1.0
    assert np.isclose(log_loss(yb, p), skm.log_loss(yb, np.clip(p, 1e-15, 1 - 1e-15)))


# ------------------------------------------------------------------------------- model selection
def test_kfold_matches_sklearn_unshuffled():
    X = np.zeros((23, 2))
    for (a_tr, a_te), (b_tr, b_te) in zip(KFold(5).split(X), SkKFold(5).split(X)):
        np.testing.assert_array_equal(np.sort(a_te), b_te)
        np.testing.assert_array_equal(np.sort(a_tr), b_tr)


def test_kfold_shuffled_partitions():
    X = np.zeros((101, 1))
    tests = [te for _, te in KFold(7, shuffle=True, random_state=3).split(X)]
    allidx = np.concatenate(tests)
    assert sorted(allidx.tolist()) == list(range(101))
    assert not np.array_equal(allidx, np.arange(101))
    for tr, te in KFold(7, shuffle=True, random_state=3).split(X):
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == 101


def test_train_test_split():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    assert len(Xte) == 5 and len(Xtr) == 15
    np.testing.assert_array_equal(Xtr[:, 0] // 2, ytr)  # rows stay aligned with labels
    assert sorted(np.concatenate([ytr, yte]).tolist()) == list(range(20))
    assert np.array_equal(train_test_split(X, y, random_state=0)[3], train_test_split(X, y, random_state=0)[3])


def test_cross_val_score_fresh_models():
    X, y = datasets.load_iris(return_X_y=True)
    created = []

    def make():
        m = LogisticRegression(lr=0.1, max_iter=200)
        created.append(m)
        return m
    scores = cross_val_score(make, X, y, cv=5)
    assert len(scores) == 5 and len({id(m) for m in created}) == 5
    assert scores.mean() > 0.9
