# src/plastic_id/preprocessing.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.decomposition import PCA


def make_pca(n: int, whiten: bool = True):
    """Return a PCA transformer that clips n_components to ≤ n_features."""
    from sklearn.decomposition import PCA

    class _SafePCA(PCA):
        def fit(self, X, y=None):
            max_comp = min(n, X.shape[1])
            self.n_components_ = max_comp  # store for downstream logic
            return super().fit(X, y)

    return _SafePCA(
        n_components=min(n, 1),  # dummy, replaced in `fit`
        whiten=whiten,
        random_state=0,
    )


class RowNormalizer(BaseEstimator, TransformerMixin):
    """Divide each spectrum by its L2 norm."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norm, 1e-12)


class RowSNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate per row."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        return (X - mean) / np.maximum(std, 1e-12)


def make_pca(n_comp: int):
    return PCA(n_components=n_comp, whiten=True, random_state=0)
