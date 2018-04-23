import numpy as np

from ..utils_numpy import check_dtype, unpack

def test_check_dtype():
    dtype = check_dtype(None, "mean", np.arange(10, dtype=int), 10)
    assert np.issubdtype(dtype, np.floating)


def test_unpack():
    """Keep this test, in case unpack might get reimplemented again at some point."""
    group_idx = np.arange(10)
    np.random.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    vals = np.random.randn(np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_unpack_long():
    group_idx = np.repeat(np.arange(10000), 20)
    a = np.arange(group_idx.size, dtype=int)
    a = np.random.randn(np.max(group_idx) + 1)
    vals = np.random.randn(np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])
