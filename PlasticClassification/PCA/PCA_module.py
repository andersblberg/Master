# PCA/PCA_module.py

from sklearn.decomposition import PCA

def apply_pca(X, n_components=2):
    """
    Applies PCA to reduce the dimensionality of the feature matrix X.
    Returns the transformed features and the explained variance ratio.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance
