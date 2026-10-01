"""Week 2 -- classical ML from scratch. sklearn-style API: ``fit(X, y)`` returns self, attributes that
are learned end with ``_``. Only numpy (and your forge.autograd / forge.optim) allowed here --
scikit-learn is used ONLY inside the tests as a reference."""
from forge.ml.cluster import KMeans, kmeans_plusplus
from forge.ml.decomposition import PCA
from forge.ml.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from forge.ml.linear import LinearRegression, LogisticRegression
from forge.ml.mixture import GaussianMixture
from forge.ml.model_selection import KFold, cross_val_score, train_test_split
from forge.ml.preprocessing import StandardScaler
from forge.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "PCA",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GaussianMixture",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KFold",
    "KMeans",
    "LinearRegression",
    "LogisticRegression",
    "RandomForestClassifier",
    "StandardScaler",
    "cross_val_score",
    "kmeans_plusplus",
    "train_test_split",
]
