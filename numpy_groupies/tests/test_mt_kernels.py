"""The parallel path of the numba kernels.

The parallel path splits the data into as many chunks as there are threads,
reduces them separately and folds their states afterwards.  The tests below
say three things about it: it gives the same result as the sequential path,
the result with one thread is the one of the sequential path bit for bit,
and the number of the threads can be steered.
"""

import numpy as np
import pytest

from numpy_groupies.aggregate_numba import aggregate, mt
from numpy_groupies.registry import DEFAULT_FILL_VALUE

# the operations that can be reduced in chunks - the others need the values
# of a group together (the order statistics) or one output per value (the
# cumulatives)
CHUNKABLE = [
    "sum",
    "prod",
    "len",
    "all",
    "any",
    "allnan",
    "anynan",
    "max",
    "min",
    "mean",
    "sumofsquares",
    "var",
    "std",
]


@pytest.fixture
def data():
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    return group_idx, values


@pytest.fixture(autouse=True)
def restore_the_thread_count():
    """Every test that steers the number of the threads gives it back."""
    before = mt.mt_thread_count()
    yield
    mt.set_thread_count(before if before == mt._pool_threads else None)


def test_the_thread_count_can_be_steered():
    assert mt.mt_thread_count() >= 1
    mt.set_thread_count(3)
    assert mt.mt_thread_count() == 3
    mt.set_thread_count(1)
    assert mt.mt_thread_count() == 1
    mt.set_thread_count(None)
    assert mt.mt_thread_count() >= 1


@pytest.mark.parametrize("func", CHUNKABLE)
def test_the_parallel_path_gives_the_result_of_the_sequential_one(func, data):
    group_idx, values = data
    mt.set_thread_count(4)
    parallel = aggregate(group_idx, values, func=func)
    mt.set_thread_count(1)
    sequential = aggregate(group_idx, values, func=func)
    np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("func", CHUNKABLE)
def test_one_thread_gives_the_sequential_result(func, data):
    group_idx, values = data
    mt.set_thread_count(1)
    one = aggregate(group_idx, values, func=func)
    mt.set_thread_count(4)
    many = aggregate(group_idx, values, func=func)
    mt.set_thread_count(1)
    sequential = aggregate(group_idx, values, func=func)
    # with one thread the sequential path runs, and it is the one that has
    # always run - bit for bit
    np.testing.assert_allclose(one, sequential, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(many, sequential, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("func", CHUNKABLE)
def test_the_number_of_the_threads_does_not_change_the_result(func, data):
    group_idx, values = data
    results = []
    for threads in (2, 3, 4, 6):
        mt.set_thread_count(threads)
        results.append(aggregate(group_idx, values, func=func))
    # the chunks are as many as the threads, so the sums of the floats add up
    # in a different order - the results agree to the last bits, and the
    # extremes, the counts and the logic agree exactly
    for result in results[1:]:
        np.testing.assert_allclose(results[0], result, rtol=1e-12, equal_nan=True)


def test_too_few_values_take_the_sequential_path():
    # below the threshold of the cost model the parallel path is not used, and
    # the result is the one of the sequential path bit for bit
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 1000)
    values = rng.random(1000)
    mt.set_thread_count(4)
    parallel = aggregate(group_idx, values, func="sum")
    mt.set_thread_count(1)
    sequential = aggregate(group_idx, values, func="sum")
    np.testing.assert_allclose(parallel, sequential, rtol=0, atol=0, equal_nan=True)


def test_the_nan_variants_give_the_same_result():
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    values[group_idx == 2] = np.nan
    # allnan and anynan are the nan variants themselves - they have no twin
    for func in ("sum", "prod", "len", "all", "any", "max", "min", "mean", "var", "std"):
        mt.set_thread_count(4)
        parallel = aggregate(group_idx, values, func="nan" + func)
        mt.set_thread_count(1)
        sequential = aggregate(group_idx, values, func="nan" + func)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_complex_input_gives_the_same_result():
    # the complex input has no order, so the extremes compare the way the
    # kernel compares them - and the fold of the states has to as well
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000) + 1j * rng.random(200_000)
    for func in ("sum", "max", "min", "mean", "var", "std"):
        mt.set_thread_count(4)
        parallel = aggregate(group_idx, values, func=func)
        mt.set_thread_count(1)
        sequential = aggregate(group_idx, values, func=func)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_groups_without_a_value_keep_the_fill_value():
    # a group that no chunk received a value for stays untouched - the fold
    # of the states has to keep that, and the custom fill value has to reach
    # such a group
    rng = np.random.default_rng(100)
    # only ten of the hundred groups receive a value
    group_idx = rng.integers(0, 10, 200_000)
    values = rng.random(200_000)
    for func in ("sum", "prod", "max", "min", "mean"):
        for fill in (DEFAULT_FILL_VALUE, -1.0):
            mt.set_thread_count(4)
            parallel = aggregate(group_idx, values, func=func, size=100, fill_value=fill)
            mt.set_thread_count(1)
            sequential = aggregate(group_idx, values, func=func, size=100, fill_value=fill)
            np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_dispersion_honours_the_ddof():
    # the ddof is the argument of the operation of its own - the parallel
    # driver takes it like the sequential one
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    for ddof in (0, 1, 2):
        mt.set_thread_count(4)
        parallel = aggregate(group_idx, values, func="var", ddof=ddof)
        mt.set_thread_count(1)
        sequential = aggregate(group_idx, values, func="var", ddof=ddof)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_dispersion_keeps_the_result_of_the_integer_input():
    # the integer input is centered on the first value of its group, and
    # every chunk has its own first value - the fold of the two states has
    # to correct for the two anchors
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.integers(-1000, 1000, 200_000)
    for func in ("var", "std"):
        mt.set_thread_count(1)
        one = aggregate(group_idx, values, func=func)
        mt.set_thread_count(4)
        many = aggregate(group_idx, values, func=func)
        np.testing.assert_allclose(one, many, rtol=1e-12, equal_nan=True)


def test_a_fault_of_the_indices_is_complained_about():
    # the parallel path cannot raise out of its region, so the kernel
    # remembers the fault and the driver raises the very same complaint
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    group_idx[17] = -1
    mt.set_thread_count(4)
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func="sum")
    # an index beyond the size that was asked for
    group_idx[17] = 100
    with pytest.raises(ValueError, match="too large for size 100"):
        aggregate(group_idx, values, func="sum", size=100)
