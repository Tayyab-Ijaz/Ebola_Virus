import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evd.config import load_config  # noqa: E402
from evd.data import Dataset, load_dataset  # noqa: E402
from evd.nested import ModelCfg  # noqa: E402

# reduced grid: same code paths as the primary grid, a fraction of the time
SMALL = ModelCfg(k_grid=(1, 3, 5, 10), c_grid=(0.3,), l1_grid=(0.5,), max_iter=5000)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def real(cfg):
    return load_dataset(cfg)


def make_noise(seed, n=40, G=1500):
    """Negative-binomial counts with NO relationship to the (random, balanced) labels."""
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(3.0, 1.2, G))
    depth = np.exp(rng.normal(0, 0.2, (n, 1)))
    m = mu[None, :] * depth
    counts = rng.negative_binomial(5, 5 / (5 + m))
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(y)
    return Dataset.from_arrays(counts, y)
