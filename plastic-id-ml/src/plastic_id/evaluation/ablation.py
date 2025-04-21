import numpy as np
from itertools import combinations

CHANNEL_IDX = {
    940: 0, 1050: 1, 1200: 2, 1300: 3, 1450: 4, 1550: 5, 1650: 6, 1720: 7,
}

def drop_channels(X: np.ndarray, channels):
    idx = [CHANNEL_IDX[w] for w in channels]
    return np.delete(X, idx, axis=1)

def gen_single_drop():
    for ch in CHANNEL_IDX:  # yields tuples (channel, kept_mask_fn)
        yield (ch, lambda X, ch=ch: drop_channels(X, [ch]))

def gen_pair_drop():
    for pair in combinations(CHANNEL_IDX, 2):
        yield (pair, lambda X, pair=pair: drop_channels(X, pair))