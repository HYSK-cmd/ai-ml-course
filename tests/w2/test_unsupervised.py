"""Week 2 autograder: k-means, GMM/EM, PCA."""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import adjusted_rand_score

from forge.ml import PCA, GaussianMixture, KMeans, kmeans_plusplus

rng = np.random.default_rng(0)


def blobs(n=600, centers=5, std=0.6, seed=0):
    return datasets.make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=seed)


def test_kmeans_plusplus_properties():
    X, _ = blobs()
    C = kmeans_plusplus(X, 5, np.random.default_rng(0))
    assert C.shape == (5, 2)
    assert all(any(np.array_equal(c, x) for x in X) for c in C), "centers must be rows of X"
    assert len({tuple(c) for c in C}) == 5
    # D^2 sampling: with 999 points at the origin and one at 100, the 2nd center is ~always the outlier
    Y = np.zeros((1000, 1))
    Y[0] = 100.0
    hits = sum(np.any(kmeans_plusplus(Y, 2, np.random.default_rng(s)) == 100.0) for s in range(20))
    assert hits == 20


def test_kmeans_recovers_blobs_and_matches_sklearn_inertia():
    X, y = blobs()
    km = KMeans(5, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.99
    ref = SkKMeans(5, n_init=10, random_state=0).fit(X)
    assert km.inertia_ <= ref.inertia_ * 1.01
    np.testing.assert_array_equal(km.predict(X), km.labels_)
    d = ((X[:, None] - km.cluster_centers_[None]) ** 2).sum(-1)
    assert np.isclose(km.inertia_, d.min(1).sum())


def test_kmeans_vectorized_speed():
    import time
    X = rng.normal(size=(50_000, 16))
    t = time.perf_counter()
    KMeans(20, n_init=1, max_iter=30, random_state=0).fit(X)
    assert time.perf_counter() - t < 20


def test_gmm_em_monotone_and_recovers_means():
    means = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    covs = [np.array([[1.0, 0.3], [0.3, 0.5]]), np.eye(2) * 0.4, np.array([[0.6, -0.2], [-0.2, 1.0]])]
    X = np.vstack([rng.multivariate_normal(m, c, 300) for m, c in zip(means, covs)])
    g = GaussianMixture(3, max_iter=200, tol=1e-8, random_state=0).fit(X)
    h = np.array(g.lower_bound_history_)
    assert len(h) >= 2
    assert np.all(np.diff(h) >= -1e-9), "EM must never decrease the log-likelihood"
    dist = np.linalg.norm(means[:, None] - g.means_[None], axis=-1)  # (true, fitted)
    match = dist.argmin(1)
    assert sorted(match.tolist()) == [0, 1, 2], f"components don't match the true clusters: {g.means_}"
    assert dist[np.arange(3), match].max() < 0.2
    assert np.isclose(g.weights_.sum(), 1)
    assert g.covariances_.shape == (3, 2, 2)
    # score_samples must be the true mixture log-density given the fitted parameters
    dens = sum(w * multivariate_normal(m, c).pdf(X[:50]) for w, m, c in zip(g.weights_, g.means_, g.covariances_))
    np.testing.assert_allclose(g.score_samples(X[:50]), np.log(dens), rtol=1e-6)
    assert np.isclose(g.score(X), g.score_samples(X).mean())


def test_gmm_numerically_stable_far_points():
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(8, 1, (100, 2))])
    g = GaussianMixture(2, random_state=0).fit(X)
    far = np.array([[1e4, -1e4], [-500.0, 300.0]])
    P = g.predict_proba(far)
    assert np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1)
    assert np.all(np.isfinite(g.score_samples(far)))


def test_pca_matches_sklearn_up_to_sign():
    X, _ = datasets.load_digits(return_X_y=True)
    ours = PCA(10).fit(X)
    ref = SkPCA(10, svd_solver="full").fit(X)
    signs = np.sign((ours.components_ * ref.components_).sum(1))
    np.testing.assert_allclose(ours.components_ * signs[:, None], ref.components_, atol=1e-6)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-6)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-6)
    Z = ours.transform(X)
    np.testing.assert_allclose(Z * signs, ref.transform(X), atol=1e-6)
    np.testing.assert_allclose(ours.components_ @ ours.components_.T, np.eye(10), atol=1e-10)


def test_pca_full_rank_reconstruction():
    X = rng.normal(size=(50, 6))
    p = PCA(6)
    np.testing.assert_allclose(p.inverse_transform(p.fit_transform(X)), X, atol=1e-10)
