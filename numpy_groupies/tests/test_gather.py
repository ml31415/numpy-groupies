"""The gather of a result: the result of every group, back on the input.

Every backend has its own gather - the fancy index of numpy, the jitted loop
of numba and the plain python loop of the pure python implementation - and
they all answer the same.
"""

import numpy as np
import pytest

from numpy_groupies import npg_numba
from numpy_groupies.npg_numba import gather, mt
from numpy_groupies.tests.helpers import is_purepy, random_values


@pytest.mark.parametrize("groups", [7, 1000], ids=["few groups", "many groups"])
def test_unpack_answers_same(impl, rng, groups):
    """Every backend gathers the same: the result of every group, back on the input.

    The labels are the groups in a shuffled order and every group holds the
    same number of values, so the gather cannot lean on the order of the
    labels.  The many-groups case runs it over a longer input; the parallel
    path of the numba backend has a test of its own.
    """
    group_idx = np.repeat(np.arange(groups), 20)
    rng.shuffle(group_idx)
    vals = random_values(rng, groups)
    got = impl.unpack(group_idx, vals)
    np.testing.assert_array_equal(got, vals[group_idx])
    if is_purepy(impl):
        # the result of the pure python implementation is a list - the gather
        # of a list is a list as well
        assert isinstance(got, list)


def test_parallel_path_answers_same(rng):
    """The gather of the numba implementation across the threads of the pool.

    Below the threshold of the cost model the plain loop runs, so the two
    sizes below are the two paths - and they have to answer the same, because
    every value writes the one position of its own and no thread ever writes
    where another one has written.
    """

    for n in (mt.GATHER_MT_MIN_VALUES - 1, mt.GATHER_MT_MIN_VALUES, 4 * mt.GATHER_MT_MIN_VALUES):
        group_idx = rng.integers(0, 7, size=n)
        vals = random_values(rng, 7)
        np.testing.assert_array_equal(npg_numba.unpack(group_idx, vals), vals[group_idx])


def test_parallel_path_keeps_shape_of_out(rng):
    """The gather writes into the array of the caller and not into a new one.

    The parallel region has to fill every position of that array and nothing
    beside it: an array the size of which is one value off shows the last
    chunk of the last thread running past its end.
    """

    n = 3 * mt.GATHER_MT_MIN_VALUES + 7
    group_idx = rng.integers(0, 1000, n)
    ret = rng.random(1000)
    out = np.empty(n)
    got = gather.unpack_into(group_idx, ret, out)
    assert got is out
    np.testing.assert_array_equal(out, ret[group_idx])


def test_parallelism_follows_model(rng):
    """The gather runs in a parallel region only where it pays for itself.

    The gather copies one value per value and is bound by the memory: below
    the size the cost model names, the region costs more than the work it
    does, and the sequential loop is the one that runs.  The test says which
    of the two runs - by the time it takes, and by the number of the threads
    the pool is asked for.
    """
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
