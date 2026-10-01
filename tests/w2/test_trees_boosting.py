"""Week 2 autograder: CART trees, gradient boosting, random forest (stretch)."""
import time

import numpy as np
import pytest
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier as SkDTC
from sklearn.tree import DecisionTreeRegressor as SkDTR

from forge.ml import (DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingClassifier,
                      GradientBoostingRegressor, LinearRegression, RandomForestClassifier, train_test_split)
from forge.ml.metrics import accuracy_score, mean_squared_error
from tests.helpers import need_data

rng = np.random.default_rng(0)


def california():
    (path,) = need_data("california.npz")
    d = np.load(path)
    return train_test_split(d["X"], d["y"], test_size=0.2, random_state=0)


def test_tree_regressor_1d_matches_sklearn_exactly():
    X = np.sort(rng.uniform(0, 6, 300))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(scale=0.2, size=300)
    Xq = np.linspace(-1, 7, 500)[:, None]
    for depth, leaf in [(1, 1), (3, 1), (5, 10), (None, 1)]:
        ours = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf).fit(X, y)
        ref = SkDTR(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X, y)
        np.testing.assert_allclose(ours.predict(Xq), ref.predict(Xq), atol=1e-9, err_msg=f"depth={depth} leaf={leaf}")
        assert ours.get_depth() == ref.get_depth()
        assert ours.get_n_leaves() == ref.get_n_leaves()


def test_tree_classifier_fits_xor_fully():
    X = rng.uniform(-1, 1, size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    t = DecisionTreeClassifier().fit(X, y)
    assert accuracy_score(y, t.predict(X)) == 1.0


def test_tree_classifier_quality_and_api():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for crit in ["gini", "entropy"]:
        ours = DecisionTreeClassifier(max_depth=4, criterion=crit).fit(Xtr, ytr)
        ref = SkDTC(max_depth=4, criterion=crit, random_state=0).fit(Xtr, ytr)
        assert accuracy_score(ytr, ours.predict(Xtr)) >= accuracy_score(ytr, ref.predict(Xtr)) - 0.01
        assert accuracy_score(yte, ours.predict(Xte)) >= accuracy_score(yte, ref.predict(Xte)) - 0.04
        assert ours.get_depth() <= 4
    P = ours.predict_proba(Xte)
    assert P.shape == (len(Xte), 2)
    np.testing.assert_allclose(P.sum(1), 1)


def test_tree_constraints_respected():
    X, y = datasets.load_iris(return_X_y=True)
    t = DecisionTreeClassifier(min_samples_leaf=7).fit(X, y)
    _, counts = np.unique(t.apply(X), return_counts=True)
    assert counts.min() >= 7
    assert len(counts) == t.get_n_leaves()
    t2 = DecisionTreeClassifier(min_samples_split=40).fit(X, y)
    assert t2.get_n_leaves() < DecisionTreeClassifier().fit(X, y).get_n_leaves()
    assert DecisionTreeClassifier(max_depth=0).fit(X, y).get_n_leaves() == 1


def test_tree_string_labels():
    X, y = datasets.load_iris(return_X_y=True)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    t = DecisionTreeClassifier(max_depth=3).fit(X, names)
    assert set(t.predict(X)) <= set(names)
    assert accuracy_score(names, t.predict(X)) > 0.95


def test_tree_split_search_is_vectorized():
    Xtr, _, ytr, _ = california()
    t = time.perf_counter()
    DecisionTreeRegressor(max_depth=8).fit(Xtr, ytr)
    elapsed = time.perf_counter() - t
    assert elapsed < 15, f"depth-8 tree on {len(Xtr)} rows took {elapsed:.1f}s -- vectorize the split search"


def test_gbm_regressor_beats_linear_on_california():
    Xtr, Xte, ytr, yte = california()
    lin = mean_squared_error(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)) ** 0.5
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    rmse = mean_squared_error(yte, g.predict(Xte)) ** 0.5
    assert rmse < 0.60, f"GBM test RMSE {rmse:.3f}"
    assert rmse < 0.85 * lin, f"GBM {rmse:.3f} vs linear {lin:.3f}"
    assert len(g.train_loss_) == 100 and len(g.estimators_) == 100
    assert all(b <= a + 1e-12 for a, b in zip(g.train_loss_, g.train_loss_[1:])), "train MSE must never go up"


def test_gbm_regressor_subsample_is_random_but_seeded():
    X = rng.normal(size=(300, 4))
    y = X[:, 0] ** 2 + X[:, 1]
    a = GradientBoostingRegressor(n_estimators=20, subsample=0.5, random_state=1).fit(X, y).predict(X)
    b = GradientBoostingRegressor(n_estimators=20, subsample=0.5, random_state=1).fit(X, y).predict(X)
    c = GradientBoostingRegressor(n_estimators=20, subsample=0.5, random_state=2).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_gbm_classifier_newton_steps():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    g = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=2).fit(Xtr, ytr)
    assert accuracy_score(yte, g.predict(Xte)) >= 0.94
    p0 = ytr.mean()
    assert np.isclose(g.init_, np.log(p0 / (1 - p0)))
    # Newton leaf values make boosting converge far faster than plain residual means would:
    assert g.train_loss_[9] < 0.35, f"loss after 10 stages {g.train_loss_[9]:.3f} -- are leaf values Newton steps?"
    assert g.train_loss_[-1] < g.train_loss_[0] / 4
    P = g.predict_proba(Xte)
    np.testing.assert_allclose(P.sum(1), 1)
    np.testing.assert_allclose(np.log(P[:, 1] / P[:, 0]), g.decision_function(Xte), atol=1e-8)


@pytest.mark.stretch
def test_random_forest_beats_single_tree():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=2)
    tree = accuracy_score(yte, DecisionTreeClassifier().fit(Xtr, ytr).predict(Xte))
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, rf.predict(Xte)) > tree
