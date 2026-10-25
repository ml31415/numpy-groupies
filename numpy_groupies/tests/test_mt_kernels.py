"""The parallel path of the numba kernels.

The parallel path splits the data into chunks - twice as many as there are
threads - reduces every chunk into a state of its own and folds the states
afterwards.  The tests below say three things about it: it gives the same
result as the sequential path, the result with one thread is the one of the
sequential path bit for bit, and the number of the threads can be steered.

The last of these was not true for a while and could not be noticed: the
number of the threads was read once, when the wrappers of the kernels were
built at the import of the package, so every later call to
``numba.set_num_threads``
changed the number the *model* saw and not the number the kernels ran with.
The tests below that steer the number of the threads were comparing a call
with itself.
"""

import numba as nb
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


def _thread_counts() -> list[int]:
    """The numbers of the threads the tests try: one, two and the pool.

    Two threads are enough for every test here, and the pool of the machine
    is as many as numba was started with - a machine with two of them (a
    notebook, a container) runs the tests as well.
    """
    counts = [1, 2]
    pool = nb.config.NUMBA_NUM_THREADS
    if pool > 2:
        counts.append(pool)
    return counts


@pytest.fixture
def data():
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    return group_idx, values


@pytest.fixture(autouse=True)
def restore_the_thread_count():
    """Every test that steers the number of the threads gives it back.

    The number of the threads is the one of the pool of numba - the caller
    steers it with ``numba.set_num_threads`` and the kernels use what they
    find there.
    """
    before = nb.get_num_threads()
    yield
    nb.set_num_threads(before)


def test_the_thread_count_can_be_steered():
    # the number of the threads is the one of the pool of numba, and the
    # kernels use what they find there - there is no knob of our own
    assert mt.mt_thread_count() == nb.get_num_threads()
    nb.set_num_threads(2)
    assert mt.mt_thread_count() == 2
    nb.set_num_threads(1)
    assert mt.mt_thread_count() == 1
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
    assert mt.mt_thread_count() == nb.config.NUMBA_NUM_THREADS


@pytest.mark.parametrize("func", CHUNKABLE)
def test_the_parallel_path_gives_the_result_of_the_sequential_one(func, data):
    group_idx, values = data
    nb.set_num_threads(2)
    parallel = aggregate(group_idx, values, func=func)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func)
    np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("func", CHUNKABLE)
def test_one_thread_gives_the_sequential_result(func, data):
    group_idx, values = data
    nb.set_num_threads(1)
    one = aggregate(group_idx, values, func=func)
    nb.set_num_threads(2)
    many = aggregate(group_idx, values, func=func)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func)
    # with one thread the sequential path runs, and it is the one that has
    # always run - bit for bit
    np.testing.assert_allclose(one, sequential, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(many, sequential, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("func", CHUNKABLE)
def test_the_number_of_the_threads_does_not_change_the_result(func, data):
    group_idx, values = data
    results = []
    for threads in _thread_counts():
        nb.set_num_threads(threads)
        results.append(aggregate(group_idx, values, func=func))
    # the chunks are twice as many as the threads and the number of them
    # changes with the number of the threads, so the sums of the floats add up
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
    nb.set_num_threads(2)
    parallel = aggregate(group_idx, values, func="sum")
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func="sum")
    np.testing.assert_allclose(parallel, sequential, rtol=0, atol=0, equal_nan=True)


def test_the_nan_variants_give_the_same_result():
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    values[group_idx == 2] = np.nan
    # allnan and anynan are the nan variants themselves - they have no twin
    for func in ("sum", "prod", "len", "all", "any", "max", "min", "mean", "var", "std"):
        nb.set_num_threads(2)
        parallel = aggregate(group_idx, values, func="nan" + func)
        nb.set_num_threads(1)
        sequential = aggregate(group_idx, values, func="nan" + func)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_complex_input_gives_the_same_result():
    # the complex input has no order, so the extremes compare the way the
    # kernel compares them - and the fold of the states has to as well
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000) + 1j * rng.random(200_000)
    for func in ("sum", "max", "min", "mean", "var", "std"):
        nb.set_num_threads(2)
        parallel = aggregate(group_idx, values, func=func)
        nb.set_num_threads(1)
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
            nb.set_num_threads(2)
            parallel = aggregate(group_idx, values, func=func, size=100, fill_value=fill)
            nb.set_num_threads(1)
            sequential = aggregate(group_idx, values, func=func, size=100, fill_value=fill)
            np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_dispersion_honours_the_ddof():
    # the ddof is the argument of the operation of its own - the parallel
    # driver takes it like the sequential one
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    for ddof in (0, 1, 2):
        nb.set_num_threads(2)
        parallel = aggregate(group_idx, values, func="var", ddof=ddof)
        nb.set_num_threads(1)
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
        nb.set_num_threads(1)
        one = aggregate(group_idx, values, func=func)
        nb.set_num_threads(2)
        many = aggregate(group_idx, values, func=func)
        np.testing.assert_allclose(one, many, rtol=1e-12, equal_nan=True)


def test_a_fault_of_the_indices_is_complained_about():
    # the parallel path cannot raise out of its region, so the kernel
    # remembers the fault and the driver raises the very same complaint
    rng = np.random.default_rng(100)
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    group_idx[17] = -1
    nb.set_num_threads(2)
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func="sum")
    # an index beyond the size that was asked for
    group_idx[17] = 100
    with pytest.raises(ValueError, match="too large for size 100"):
        aggregate(group_idx, values, func="sum", size=100)


@pytest.mark.parametrize("func", CHUNKABLE)
def test_fewer_threads_than_the_pool_has_do_not_lose_a_chunk(func, data):
    # the number of the threads a call asks for and the number of the threads
    # the pool has are two numbers: a call for two threads on a machine whose
    # pool has six still hands its chunks to those six, and a state per thread
    # indexed by the id of the thread that took a chunk is then read out of
    # its range - silently, because the read is inside a compiled function.
    # The chunks that fall off the end of the states are lost, and the result
    # is a sum over less than the values of the group.
    #
    # The state belongs to the chunk and not to the thread, so the number of
    # the states is the number of the chunks and the number of the threads
    # does not matter.  The test says that the result does not depend on it.
    group_idx, values = data
    if nb.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("the machine has no second thread to compare against")
    nb.set_num_threads(2)
    few = aggregate(group_idx, values, func=func)
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
    many = aggregate(group_idx, values, func=func)
    np.testing.assert_allclose(few, many, rtol=1e-12, equal_nan=True)


def test_the_number_of_the_threads_of_a_call_reaches_the_kernel(data):
    # the number of the threads is asked of every call: a number captured when
    # the wrapper of the operation was built would be the number the package
    # was imported with, whatever the caller does with numba.set_num_threads after
    # that - and the sequential path would be unreachable
    import numba as nb

    group_idx, values = data
    if nb.get_num_threads() < 2:
        pytest.skip("the machine has no second thread to compare against")
    nb.set_num_threads(1)
    one = aggregate(group_idx, values, func="sum")
    # the parallel path adds the values of a group in the order of the chunks
    # and not in the order of the values, which the last bits of a float sum
    # do notice - a result identical to the one of the single thread is a
    # result of the single thread
    for threads in _thread_counts()[1:]:  # the ones above one
        nb.set_num_threads(threads)
        many = aggregate(group_idx, values, func="sum")
        assert not np.array_equal(one, many)
