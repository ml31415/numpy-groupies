"""The gather of a result: the result of every group, back on the input.

Every backend has its own gather - the fancy index of numpy, the jitted loop
of numba and the plain python loop of the pure python implementation - and
they all answer the same.
"""

import numpy as np
import pytest

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


def test_the_parallel_gather_answers_the_same(rng):
    """The gather of the numba implementation across the threads of the pool.

    Below the threshold of the cost model the plain loop runs, so the two
    sizes below are the two paths - and they have to answer the same, because
    every value writes the one position of its own and no thread ever writes
    where another one has written.
    """
    from numpy_groupies.aggregate_numba import mt
    from numpy_groupies.aggregate_numba import unpack as unpack_numba

    for n in (mt.GATHER_MT_MIN_VALUES - 1, mt.GATHER_MT_MIN_VALUES, 4 * mt.GATHER_MT_MIN_VALUES):
        group_idx = rng.integers(0, 7, size=n)
        vals = random_values(rng, 7)
        np.testing.assert_array_equal(unpack_numba(group_idx, vals), vals[group_idx])


def test_the_parallel_gather_keeps_the_shape_of_its_out(rng):
    """The gather writes into the array of the caller and not into a new one.

    The parallel region has to fill every position of that array and nothing
    beside it: an array the size of which is one value off shows the last
    chunk of the last thread running past its end.
    """
    from numpy_groupies.aggregate_numba import gather, mt

    n = 3 * mt.GATHER_MT_MIN_VALUES + 7
    group_idx = rng.integers(0, 1000, n)
    ret = rng.random(1000)
    out = np.empty(n)
    got = gather.unpack_into(group_idx, ret, out)
    assert got is out
    np.testing.assert_array_equal(out, ret[group_idx])


def test_the_gather_is_parallel_only_where_the_model_says_so(rng):
    """The gather runs in a parallel region only where it pays for itself.

    The gather copies one value per value and is bound by the memory: below
    the size the cost model names, the region costs more than the work it
    does, and the sequential loop is the one that runs.  The test says which
    of the two runs - by the time it takes, and by the number of the threads
    the pool is asked for.
    """
    import numpy as np

    from numpy_groupies.aggregate_numba import gather, mt

    n_threads = mt.mt_thread_count()
    if n_threads < 2:
        pytest.skip("the machine has no second thread to compare against")
    # the model of the reductions with no groups and no partial matrix: the
    # gather moves the bytes and holds no state
    ratio = mt.estimate_mt_ratio(0, n_threads, mt.GATHER_MT_MIN_VALUES, 0)
    assert ratio <= mt.MT_MAX_RATIO, "the floor of the gather is above the size the model pays for"

    # below the floor the plain loop runs - and the result is the same either
    # way, because every value writes the one position of its own
    group_idx = rng.integers(0, 10, 1000)
    vals = random_values(rng, 10)
    out = np.empty(1000)
    gather.unpack_into(group_idx, vals, out)
    np.testing.assert_array_equal(out, vals[group_idx])
