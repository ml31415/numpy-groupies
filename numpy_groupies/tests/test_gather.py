"""The gather of a result: the result of every group, back on the input.

Every backend has its own gather - the fancy index of numpy, the jitted loop
of numba and the plain python loop of the pure python implementation - and
they all answer the same.
"""

import numpy as np

from numpy_groupies.aggregate_numpy import unpack as unpack_numpy
from numpy_groupies.aggregate_purepy import unpack as unpack_purepy
from numpy_groupies.tests.helpers import random_values


def test_unpack(rng):
    """Keep this test, in case unpack might get reimplemented again at some point."""
    group_idx = np.arange(10)
    rng.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 3)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack_numpy(group_idx, vals), vals[group_idx])


def test_unpack_long(rng):
    group_idx = np.repeat(np.arange(10000), 20)
    vals = random_values(rng, np.max(group_idx) + 1)
    np.testing.assert_array_equal(unpack_numpy(group_idx, vals), vals[group_idx])


def test_the_pure_python_gather_answers_the_same(rng):
    """The gather of the pure python implementation is the plain python loop."""
    group_idx = rng.integers(0, 7, size=100)
    vals = random_values(rng, 7)

    np.testing.assert_array_equal(unpack_purepy(group_idx, vals), vals[group_idx])
    # the result of the pure python implementation is a list - the gather of a
    # list is a list as well
    assert isinstance(unpack_purepy(group_idx, list(vals)), list)


def test_the_numba_gather_answers_the_same(rng):
    """The gather of the numba implementation is a jitted loop."""
    from numpy_groupies.aggregate_numba import unpack as unpack_numba

    group_idx = rng.integers(0, 7, size=100)
    vals = random_values(rng, 7)

    np.testing.assert_array_equal(unpack_numba(group_idx, vals), vals[group_idx])
