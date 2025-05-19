import numpy as np


def add_gaussian_noise(X: np.ndarray, pct: float, rng=None) -> np.ndarray:
    """
    Element-wise Gaussian noise with σ = pct % of each value.
    """
    rng = np.random.default_rng(rng)
    sigma = pct / 100.0
    noise = rng.normal(0.0, sigma, size=X.shape) * X
    return X + noise
