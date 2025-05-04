import numpy as np
from plastic_id.evaluation.noise import add_gaussian_noise


def test_noise_stats():
    X = np.ones((10, 5))
    Y = add_gaussian_noise(X, pct=1.0, rng=0)
    # mean should stay ≈1 after small noise; 0.05 tolerance is plenty
    assert np.allclose(Y.mean(), 1.0, atol=0.05)
