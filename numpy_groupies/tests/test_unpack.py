import numpy as np

from .. import unpack, aggregate


def test_compare():
    group_idx = np.arange(10)
    np.random.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    a = np.random.randn(group_idx.size)
    np.testing.assert_array_equal(unpack(group_idx, a), a[group_idx])


def test_simple():
    group_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    vals = aggregate(group_idx, np.arange(group_idx.size))
    unpacked = unpack(group_idx, vals)
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21, 30, 30, 30]))


def test_incontiguous_a():
    group_idx = np.array([5, 5, 3, 3, 1, 1, 4, 4])
    vals = aggregate(group_idx, np.arange(group_idx.size))
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_incontiguous_b():
    group_idx = np.array([5, 5, 12, 5, 9, 12, 9])
    x = np.array([1, 2, 3, 24, 15, 6, 17])
    vals = aggregate(group_idx, x)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])


def test_long():
    group_idx = np.repeat(np.arange(10000), 20)
    a = np.arange(group_idx.size, dtype=int)
    vals = aggregate(group_idx, a)
    np.testing.assert_array_equal(unpack(group_idx, vals), vals[group_idx])
