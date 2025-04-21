import numpy as np

def add_gaussian_noise(X: np.ndarray, pct: float, rng=None):
    """Return X with element‑wise Gaussian noise ±pct of each value."""
    rng = np.random.default_rng(rng)
    sigma = pct / 100.0
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape) * X
    return X + noise