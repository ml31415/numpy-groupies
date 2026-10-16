import numpy as np

from numpy_groupies.tests.helpers import random_values
from numpy_groupies.utils import check_dtype, unpack


def test_check_dtype():
    dtype = check_dtype(None, "mean", np.arange(10, dtype=int), 10)
    assert np.issubdtype(dtype, np.floating)


def test_unpack(rng):
    """Keep this test, in case unpack might get reimplemented again at some point."""
    group_idx = np.arange(10)
    rng.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_long(rng):
    group_idx = np.repeat(np.arange(10000), 20)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])
