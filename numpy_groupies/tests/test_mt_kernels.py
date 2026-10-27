"""The parallel paths of the numba kernels.

The kernels that can be reduced in chunks split the data into chunks - twice
as many as there are threads - reduce every chunk into a state of its own
and fold the states afterwards.  The tests of the first section say three
things about it: it gives the same result as the sequential path, the result
with one thread is the one of the sequential path bit for bit, and the
number of the threads can be steered.

The last of these was not true for a while and could not be noticed: the
number of the threads was read once, when the wrappers of the kernels were
built at the import of the package, so every later call to
``numba.set_num_threads``
changed the number the *model* saw and not the number the kernels ran with.
The tests below that steer the number of the threads were comparing a call
with itself.

The kernels that cannot be reduced in chunks - the order statistics - cut
the GROUPS into ranges instead, and the loop over the groups of one range is
the loop of the sequential kernel.  The tests of the second section say
something stronger about them: the parallel path does not agree with the
sequential one to the last bits, it gives the very same bytes.  And no shape
of the groups hides a group that no range worked on - which is where a range
that is one group too short, or one too long, would show.
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

# the operations that parallelize over the groups and not over the values
GROUP_PARALLEL = [
    "median",
    "nanmedian",
    "quantile",
    "nanquantile",
    "sort",
    "nunique",
    "nannunique",
]

N_GROUPS = 100
N_VALUES = 200_000


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


def _write_protected(*arrays: np.ndarray) -> None:
    """The arrays of a fixture are write protected.

    The fixtures are shared by the tests of the module, so a test that writes
    into one of them fails at that place rather than in whatever test reads
    the data next.
    """
    for array in arrays:
        array.setflags(write=False)


@pytest.fixture(scope="module")
def data(rng):
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    _write_protected(group_idx, values)
    return group_idx, values


def _group_data(
    rng, n_values: int = N_VALUES, n_groups: int = N_GROUPS, nans: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    group_idx = rng.integers(0, n_groups, n_values)
    values = rng.integers(-1000, 1000, n_values).astype(np.float64)
    if nans:
        values[::10] = np.nan
    return group_idx, values


@pytest.fixture(scope="module")
def group_data(rng) -> tuple[np.ndarray, np.ndarray]:
    """A shape the cost model of the group path sends to the parallel path.

    The values carry nan values, which the nan variants filter out and the
    plain ones answer with a poisoned cell.
    """
    return _group_data(rng)


@pytest.fixture(scope="module")
def group_clean(rng) -> tuple[np.ndarray, np.ndarray]:
    """The same, without the nan values - for the comparisons that a nan in
    either of the two results would break."""
    return _group_data(rng, nans=False)


@pytest.fixture(scope="module")
def group_skewed(rng) -> tuple[np.ndarray, np.ndarray]:
    """The same number of values, in groups of wildly different size.

    A zipf distribution over the groups, so a handful of them hold most of the
    values.  The ranges are cut after the number of the *values* and not after
    the number of the groups, and this is the shape that shows whether they
    really are: ranges of an equal number of groups would give one thread all
    the big groups, and the call would be as parallel as its longest range.
    """
    weights = 1.0 / np.arange(1, N_GROUPS + 1)
    weights /= weights.sum()
    group_idx = rng.choice(N_GROUPS, size=N_VALUES, p=weights).astype(np.intp)
    values = rng.integers(-1000, 1000, N_VALUES).astype(np.float64)
    values[::10] = np.nan
    _write_protected(group_idx, values)
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


def test_too_few_values_take_the_sequential_path(rng):
    # below the threshold of the cost model the parallel path is not used, and
    # the result is the one of the sequential path bit for bit
    group_idx = rng.integers(0, 100, 1000)
    values = rng.random(1000)
    nb.set_num_threads(2)
    parallel = aggregate(group_idx, values, func="sum")
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func="sum")
    np.testing.assert_allclose(parallel, sequential, rtol=0, atol=0, equal_nan=True)


def test_the_nan_variants_give_the_same_result(rng):
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


def test_the_complex_input_gives_the_same_result(rng):
    # the complex input has no order, so the extremes compare the way the
    # kernel compares them - and the fold of the states has to as well
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000) + 1j * rng.random(200_000)
    for func in ("sum", "max", "min", "mean", "var", "std"):
        nb.set_num_threads(2)
        parallel = aggregate(group_idx, values, func=func)
        nb.set_num_threads(1)
        sequential = aggregate(group_idx, values, func=func)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_groups_without_a_value_keep_the_fill_value(rng):
    # a group that no chunk received a value for stays untouched - the fold
    # of the states has to keep that, and the custom fill value has to reach
    # such a group
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


def test_the_dispersion_honours_the_ddof(rng):
    # the ddof is the argument of the operation of its own - the parallel
    # driver takes it like the sequential one
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.random(200_000)
    for ddof in (0, 1, 2):
        nb.set_num_threads(2)
        parallel = aggregate(group_idx, values, func="var", ddof=ddof)
        nb.set_num_threads(1)
        sequential = aggregate(group_idx, values, func="var", ddof=ddof)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-12, equal_nan=True)


def test_the_dispersion_keeps_the_result_of_the_integer_input(rng):
    # the integer input is centered on the first value of its group, and
    # every chunk has its own first value - the fold of the two states has
    # to correct for the two anchors
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.integers(-1000, 1000, 200_000)
    for func in ("var", "std"):
        nb.set_num_threads(1)
        one = aggregate(group_idx, values, func=func)
        nb.set_num_threads(2)
        many = aggregate(group_idx, values, func=func)
        np.testing.assert_allclose(one, many, rtol=1e-12, equal_nan=True)


def test_a_fault_of_the_indices_is_complained_about(rng):
    # the parallel path cannot raise out of its region, so the kernel
    # remembers the fault and the driver raises the very same complaint
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


# ---------------------------------------------------------------------------
# the path over the GROUPS: the order statistics
#
# Every group belongs to exactly one range and is worked on by one thread,
# with the same code the sequential kernel runs - so there is no second order
# in which anything could add up, and the comparisons below are held to the
# very same bytes.


def _same(got: np.ndarray, wanted: np.ndarray, rtol: float = 0) -> None:
    """The two results have to be the same, nan cells included.

    ``assert_allclose`` and not ``assert_array_equal``: the fill value of the
    order statistics is a nan, and a nan compares unequal to itself - the
    comparison that is about two results of the same operation has to call
    them equal.  ``rtol`` of zero is the bit for bit comparison, the one the
    parallel path of these operations is held to.
    """
    np.testing.assert_allclose(got, wanted, rtol=rtol, atol=0, equal_nan=True)


def _both_ways(group_idx: np.ndarray, values: np.ndarray, func: str, **kwargs: object) -> tuple:
    """The result of the operation with two threads and the one with one of
    them - the two paths side by side."""
    nb.set_num_threads(2)
    parallel = aggregate(group_idx, values, func=func, **kwargs)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func, **kwargs)
    return parallel, sequential


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_group_path_gives_the_result_of_the_sequential_one(func, group_data):
    group_idx, values = group_data
    parallel, sequential = _both_ways(group_idx, values, func)
    # not "to the last bits": every group is worked on by the one thread that
    # was given its range, with the same code the sequential kernel runs, so
    # there is no second order in which anything could add up
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_number_of_the_threads_does_not_change_the_group_result(func, group_data):
    group_idx, values = group_data
    results = []
    # the pool of numba takes the number of the threads it was started with,
    # and no more
    for threads in _thread_counts():
        nb.set_num_threads(threads)
        results.append(aggregate(group_idx, values, func=func))
    # the ranges are the units of work and nothing else: every group belongs to
    # exactly one of them whichever number they are cut into, so the result
    # does not depend on the number at all
    for result in results[1:]:
        _same(results[0], result, 0)


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_groups_of_wildly_different_size_give_the_same_result(func, group_skewed):
    group_idx, values = group_skewed
    # the biggest group of this input holds a fifth of all the values: a range
    # that ends in the middle of it, or a range that is given nothing but
    # empty groups, would show here and only here
    assert np.bincount(group_idx).max() > N_VALUES // 10
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_untouched_groups_of_the_group_path_keep_the_fill_value(func, group_data):
    # the groups beyond the last index of the input and the ones the input
    # leaves empty: the ranges are cut after the counts, so a range can easily
    # hold nothing but groups that are empty
    group_idx, values = group_data
    # the sort writes one value per value and not one per group, so it has no
    # cell a fill value could stand in
    per_group = func != "sort"
    for fill in (DEFAULT_FILL_VALUE, -7.0):
        parallel, sequential = _both_ways(group_idx, values, func, size=500, fill_value=fill)
        _same(parallel, sequential, 0)
        assert parallel.shape == ((500,) if per_group else values.shape)
        if fill != DEFAULT_FILL_VALUE and per_group:
            # the fill value has to reach the groups of both paths alike
            assert (parallel[N_GROUPS:] == -7.0).all()


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("func", ["quantile", "nanquantile"])
def test_the_quantile_of_every_quantile(func, q, group_data):
    group_idx, values = group_data
    parallel, sequential = _both_ways(group_idx, values, func, q=q)
    _same(parallel, sequential, 0)


def test_the_median_is_the_quantile_of_a_half(group_data):
    # two kernels that are built separately and have to agree, one with the
    # middle index and one with the quantile of it
    group_idx, values = group_data
    nb.set_num_threads(2)
    _same(aggregate(group_idx, values, func="median"), aggregate(group_idx, values, func="quantile", q=0.5))


@pytest.mark.parametrize("func", ["median", "nanmedian", "quantile", "nanquantile"])
def test_the_result_of_every_group_is_the_result_of_that_group(func, group_clean):
    # every group, and with them the blocks of one and of two values that the
    # ranges run into at their edges: the two values of a group on either side
    # of a boundary of the ranges is the case a range that is one too short
    # cannot survive
    group_idx, values = group_clean
    nb.set_num_threads(2)
    kwargs = {"q": 0.5} if func.endswith("quantile") else {}
    got = aggregate(group_idx, values, func=func, **kwargs)
    wanted = np.array([np.quantile(values[group_idx == g], 0.5) for g in range(N_GROUPS)])
    _same(got, wanted, 1e-12)


@pytest.mark.parametrize("func", ["nunique", "nannunique"])
def test_the_distinct_values_are_counted_once(func, group_data):
    group_idx, values = group_data
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)
    keep = ~np.isnan(values) if func.startswith("nan") else np.ones(values.shape, bool)
    wanted = np.array([len(np.unique(values[(group_idx == g) & keep])) for g in range(N_GROUPS)])
    _same(parallel, wanted, 0)


@pytest.mark.parametrize("reverse", [False, True])
def test_the_sort_is_sorted_within_every_group(reverse, group_clean):
    group_idx, values = group_clean
    parallel, sequential = _both_ways(group_idx, values, "sort", reverse=reverse)
    _same(parallel, sequential, 0)
    for g in (0, 37, 99):
        ordered = np.diff(parallel[group_idx == g])
        # the default is the ascending order, reverse is the descending one
        assert ((ordered <= 0) if reverse else (ordered >= 0)).all()


def test_the_sort_keeps_the_values_of_its_group(group_clean):
    # the same values in every group, only put in order - the ranges of the
    # threads may not move a value across the border of a group
    group_idx, values = group_clean
    nb.set_num_threads(2)
    got = aggregate(group_idx, values, func="sort")
    for g in (0, 37, 99):
        _same(np.sort(got[group_idx == g]), np.sort(values[group_idx == g]))


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_group_path_complains_about_a_fault_of_the_indices(func, group_data):
    # the ranges cannot raise out of their region, so the complaint of the
    # sequential path has to be made before a single one of them is cut
    group_idx, values = group_data
    # the labels of the fixture are write protected - the test writes into a
    # copy of them
    group_idx = group_idx.copy()
    group_idx[17] = -1
    nb.set_num_threads(2)
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func=func)
    group_idx[17] = N_GROUPS
    with pytest.raises(ValueError, match=f"too large for size {N_GROUPS}"):
        aggregate(group_idx, values, func=func, size=N_GROUPS)


# ---------------------------------------------------------------------------
# the moments: the skewness and the kurtosis
#
# Their state is the running central moment of Welford, and two states of
# two chunks of the data cannot be added up like the sums of the mean can:
# the moments of a chunk are the moments about the mean of that chunk, and
# the merge of Chan et al. is what puts the two together.
#
# The merge is not the same arithmetic as the accumulation of one value at a
# time, so the result of the parallel path is not the one of the sequential
# path bit for bit - it is the one of the sequential path to the last bits of
# the moments.  The kurtosis is a ratio of two moments and the skewness a
# ratio of a moment and a power of another one, and a ratio of two nearly
# equal numbers is where the difference shows: the tests below compare the
# two results with a tolerance and not with the last bits.


MOMENTS = ["skew", "kurt", "nanskew", "nankurt"]


def _reference_moment(group_idx: np.ndarray, values: np.ndarray, func: str, n_groups: int) -> np.ndarray:
    """The moment of every group, computed the long way round.

    The central moments of the group are taken from the deviations of its
    values from their mean - the plain definition, without the recurrence
    that keeps the running mean.
    """
    wanted = 3 if func.endswith("skew") else 4
    out = np.full(n_groups, np.nan)
    for g in range(n_groups):
        block = values[group_idx == g]
        if func.startswith("nan"):
            block = block[~np.isnan(block)]
        n = len(block)
        if n < wanted:
            continue
        deviations = block - block.mean()
        m2 = (deviations**2).sum()
        if m2 == 0:
            # a group whose values are all the same has no spread - and no
            # skewness or kurtosis either, which pandas answers with a zero
            out[g] = 0.0
            continue
        if func.endswith("skew"):
            m3 = (deviations**3).sum()
            out[g] = n * np.sqrt(n - 1) / (n - 2) * m3 / m2**1.5
        else:
            m4 = (deviations**4).sum()
            out[g] = n * (n + 1) * (n - 1) * m4 / ((n - 2) * (n - 3) * m2**2) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return out


def test_the_moments_take_the_parallel_path(data):
    # the tests below compare the two paths with each other, and they would
    # compare the sequential path with itself if the cost model kept the
    # call single threaded - the state of the moments is the widest of the
    # package, and the model counts it
    _group_idx, values = data
    assert mt.should_use_mt(N_GROUPS, 2, len(values), 5)


@pytest.mark.parametrize("func", MOMENTS)
def test_the_moments_give_the_result_of_the_sequential_one(func, data):
    group_idx, values = data
    parallel, sequential = _both_ways(group_idx, values, func)
    np.testing.assert_allclose(parallel, sequential, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("func", MOMENTS)
def test_the_moments_of_every_group_are_the_moments_of_that_group(func, group_data):
    # the reference of the moments is the definition itself - the merge of
    # the states of the chunks has to give the moment of the group and not
    # the moment of a chunk of it
    group_idx, values = group_data
    parallel, _ = _both_ways(group_idx, values, func)
    wanted = _reference_moment(group_idx, values, func, N_GROUPS)
    np.testing.assert_allclose(parallel, wanted, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("func", MOMENTS)
def test_one_thread_gives_the_sequential_result_of_the_moments(func, data):
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
    np.testing.assert_allclose(many, sequential, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("func", ["skew", "kurt"])
def test_the_moments_keep_the_result_of_the_integer_input(func, rng):
    # the integer input is tracked by the extrema that feed the exactness
    # gate, and the extrema of the chunks have to be folded as well
    group_idx = rng.integers(0, 100, 200_000)
    values = rng.integers(-1000, 1000, 200_000)
    parallel, sequential = _both_ways(group_idx, values, func)
    np.testing.assert_allclose(parallel, sequential, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_the_fold_of_the_moments_takes_over_the_extrema_of_an_empty_chunk():
    # A chunk that received no value at all has no extrema: its scratch is
    # never seeded, and it holds whatever the allocator left there.  The
    # fold of two such states has to take the extrema of the one that has
    # values, and it has to leave the ones of the other alone.
    from numpy_groupies.aggregate_numba import averaging

    def state(count: int, low: int, high: int) -> np.ndarray:
        return averaging.MomentState(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.array([count], np.int64),
            np.array([low]),
            np.array([high]),
        )

    # the accumulator has no values: the extrema of the other chunk are
    # taken over as they are, the ones of the allocator left behind are not
    # compared with anything - the garbage is the way round here, so that a
    # comparison of the two would keep it
    acc = state(0, -999, 999)
    averaging._moment_combine(acc, state(5, 7, -7), 0.0, 0)
    assert acc.low[0] == 7
    assert acc.high[0] == -7
    assert acc.count[0] == 5

    # the other chunk has no values: its extrema are left out of the fold
    acc = state(5, 7, -7)
    averaging._moment_combine(acc, state(0, 999, -999), 0.0, 0)
    assert acc.low[0] == 7
    assert acc.high[0] == -7
    assert acc.count[0] == 5


@pytest.mark.parametrize("func", MOMENTS)
def test_the_moments_keep_the_fill_value_of_the_empty_groups(func, rng):
    # the groups that received no value keep the fill value, and the fill
    # value has to reach them in both paths alike
    group_idx = rng.integers(0, 10, 200_000)
    values = rng.random(200_000)
    for fill in (DEFAULT_FILL_VALUE, -7.0):
        parallel, sequential = _both_ways(group_idx, values, func, size=100, fill_value=fill)
        np.testing.assert_allclose(parallel, sequential, rtol=1e-9, atol=1e-9, equal_nan=True)
        if fill != DEFAULT_FILL_VALUE:
            assert (parallel[10:] == -7.0).all()


@pytest.mark.parametrize("func", MOMENTS)
def test_the_moments_refuse_the_complex_input(func, rng):
    # the skewness and the kurtosis are not defined for the complex values,
    # and the complaint has to be the same one whichever path the call takes
    group_idx = rng.integers(0, 10, 1000)
    values = rng.random(1000) + 1j * rng.random(1000)
    for threads in (1, 2):
        nb.set_num_threads(threads)
        with pytest.raises(TypeError, match="not defined for complex values"):
            aggregate(group_idx, values, func=func)


# ---------------------------------------------------------------------------
# the first and the last: the values that depend on the order of the input
#
# They cannot be reduced in chunks the way the sums are: the value of a
# group is the one that comes first (or last) in the input.  Their parallel
# path splits the data into chunks, lets every chunk look for the value of
# its own and folds the chunks in the order of their index - the first chunk
# that has a value for a group wins for the first, the last one for the
# last.
#
# The result is the one of the sequential path bit for bit: every chunk looks
# at its own values in the order they come, and the fold picks the value of
# the chunk that comes first (or last) in the input.


ORDERED = ["first", "last", "nanfirst", "nanlast"]


def test_the_ordered_values_take_the_parallel_path(data):
    # the tests below compare the two paths with each other, and they would
    # compare the sequential path with itself if the cost model kept the
    # call single threaded
    _group_idx, values = data
    assert mt.should_use_mt(N_GROUPS, 2, len(values), 1)


@pytest.mark.parametrize("func", ORDERED)
def test_the_ordered_values_give_the_result_of_the_sequential_one(func, data):
    group_idx, values = data
    parallel, sequential = _both_ways(group_idx, values, func)
    # not "to the last bits": every value of a group is looked for by the
    # one chunk that has it, and the fold picks the chunk that comes first
    # (or last) in the input - the very same value the sequential path
    # finds
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", ORDERED)
def test_the_ordered_value_of_every_group_is_the_value_of_that_group(func, data):
    # the reference is the definition itself: the first (or the last) value
    # of the group in the order of the input
    group_idx, values = data
    parallel, _ = _both_ways(group_idx, values, func)
    wanted = []
    for g in range(N_GROUPS):
        block = values[group_idx == g]
        if func.startswith("nan"):
            # the nan variants look for the first (the last) value that is
            # not a nan
            block = block[~np.isnan(block)]
        if block.size == 0:
            wanted.append(np.nan)
        elif func.endswith("first"):
            wanted.append(block[0])
        else:
            wanted.append(block[-1])
    _same(parallel, np.array(wanted), 0)


@pytest.mark.parametrize("func", ORDERED)
def test_the_ordered_values_with_nans_give_the_result_of_the_sequential_one(func, group_data):
    # die nan-Varianten lassen die NaNs aus, die plain-Varianten nicht - und
    # beide muessen auf dem parallelen Pfad das sagen, was der sequentielle
    # sagt
    group_idx, values = group_data
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", ORDERED)
def test_one_thread_gives_the_sequential_result_of_the_ordered_values(func, data):
    group_idx, values = data
    nb.set_num_threads(1)
    one = aggregate(group_idx, values, func=func)
    nb.set_num_threads(2)
    many = aggregate(group_idx, values, func=func)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func)
    # with one thread the sequential path runs, and it is the one that has
    # always run - bit for bit
    _same(one, sequential, 0)
    _same(many, sequential, 0)


@pytest.mark.parametrize("func", ORDERED)
def test_the_ordered_values_keep_the_fill_value_of_the_empty_groups(func, rng):
    # the groups that no value reached keep the fill value, and the fill
    # value has to reach them in both paths alike
    group_idx = rng.integers(0, 10, 200_000)
    values = rng.random(200_000)
    for fill in (DEFAULT_FILL_VALUE, -7.0):
        parallel, sequential = _both_ways(group_idx, values, func, size=100, fill_value=fill)
        _same(parallel, sequential, 0)
        if fill != DEFAULT_FILL_VALUE:
            assert (parallel[10:] == -7.0).all()


@pytest.mark.parametrize("func", ORDERED)
def test_the_ordered_values_complain_about_a_fault_of_the_indices(func, data):
    # the parallel path cannot raise out of its region, so the complaint of
    # the sequential path has to be made after it
    group_idx, values = data
    group_idx = group_idx.copy()
    group_idx[17] = -1
    nb.set_num_threads(2)
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func=func)


# ---------------------------------------------------------------------------
# the callable of the caller: the generic aggregation loop
#
# The loop applies the callable of the caller to the values of every group.
# The counting sort that places the values of a group together stays single
# threaded - the values have to be together where the callable is called on
# them - and the loop over the groups, which is the expensive part, is cut
# into ranges the threads work on one after another.
#
# Every group belongs to exactly one range and is worked on by one thread,
# with the same code the sequential loop runs - so the result is the one of
# the sequential loop bit for bit.


CUSTOM = [
    lambda x: x.sum(),
    lambda x: float(np.sort(x)[len(x) // 2]),
    lambda x: x.max() - x.min(),
]


def _both_ways_with_the_pool(group_idx: np.ndarray, values: np.ndarray, func, **kwargs: object) -> tuple:
    """The result of the callable with the whole pool of threads and with one.

    The path over the groups needs three threads to pay for the region of
    the pool - the model of the package says so - so a machine with fewer of
    them cannot run the test.
    """
    threads = nb.config.NUMBA_NUM_THREADS
    if threads < 4:
        pytest.skip("the machine has too few threads for the path over the groups")
    nb.set_num_threads(threads)
    parallel = aggregate(group_idx, values, func=func, **kwargs)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func, **kwargs)
    return parallel, sequential


def test_the_callable_of_the_caller_takes_the_parallel_path(group_data):
    # the tests below compare the two paths with each other, and they would
    # compare the sequential path with itself if the cost model kept the
    # call single threaded
    _group_idx, values = group_data
    threads = nb.config.NUMBA_NUM_THREADS
    if threads < 4:
        pytest.skip("the machine has too few threads for the path over the groups")
    assert mt.should_use_group_mt(N_GROUPS, threads, len(values), 18.0)


@pytest.mark.parametrize("func", CUSTOM)
def test_the_callable_of_the_caller_gives_the_result_of_the_sequential_one(func, group_data):
    group_idx, values = group_data
    parallel, sequential = _both_ways_with_the_pool(group_idx, values, func)
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", CUSTOM)
def test_the_callable_of_the_caller_gives_the_value_of_every_group(func, group_data):
    # the reference is the callable applied to the values of the group, the
    # long way round
    group_idx, values = group_data
    parallel, _ = _both_ways_with_the_pool(group_idx, values, func)
    wanted = np.array([func(values[group_idx == g]) for g in range(N_GROUPS)])
    _same(parallel, wanted, 0)


@pytest.mark.parametrize("func", CUSTOM)
def test_the_callable_of_the_caller_complains_about_a_fault_of_the_indices(func, group_data):
    # the counting sort that places the values stays single threaded, and it
    # is the one that complains about the indices - the same complaint in
    # both paths
    group_idx, values = group_data
    group_idx = group_idx.copy()
    group_idx[17] = -1
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func=func)
    group_idx[17] = N_GROUPS
    with pytest.raises(ValueError, match="too large"):
        aggregate(group_idx, values, func=func, size=N_GROUPS)


@pytest.mark.parametrize("func", CUSTOM)
def test_the_number_of_the_threads_does_not_change_the_result_of_the_callable(func, group_data):
    group_idx, values = group_data
    results = []
    for threads in _thread_counts():
        nb.set_num_threads(threads)
        results.append(aggregate(group_idx, values, func=func))
    # the ranges are the units of work and nothing else: every group belongs
    # to exactly one of them whichever number they are cut into
    for result in results[1:]:
        _same(results[0], result, 0)


# ---------------------------------------------------------------------------
# the argmax and the argmin: the index of the best value of every group
#
# They cannot be reduced in chunks the way the sums are: the value of a
# group is the one that wins, and the winner of two chunks is not the sum of
# the two.  Their parallel path splits the data into chunks, lets every
# chunk look for the best value of every group, and folds the chunks in the
# order of their index - the chunk with the better value wins, and a tie is
# won by the chunk that comes first, the way the sequential kernel keeps the
# first of two equal values.
#
# The result is the one of the sequential path bit for bit.


ARGUMENTS = ["argmax", "argmin", "nanargmax", "nanargmin"]


def _reference_arg(group_idx: np.ndarray, values: np.ndarray, func: str, n_groups: int) -> np.ndarray:
    """The index of the best value of every group, the long way round.

    The plain variants are poisoned by a nan: a group that has one keeps
    the fill value, and the nan variants leave the nans out.
    """
    out = np.full(n_groups, -1, dtype=np.int64)
    for g in range(n_groups):
        where = np.where(group_idx == g)[0]
        block = values[where]
        if func.startswith("nan"):
            keep = ~np.isnan(block)
            where = where[keep]
            block = block[keep]
        elif np.isnan(block).any():
            # a nan poisons its group
            continue
        if block.size == 0:
            continue
        best = block.max() if "max" in func else block.min()
        # the first of the values that are the best one
        out[g] = where[np.argmax(block == best)]
    return out


def test_the_arguments_take_the_parallel_path(data):
    # the tests below compare the two paths with each other, and they would
    # compare the sequential path with itself if the cost model kept the
    # call single threaded
    _group_idx, values = data
    assert mt.should_use_mt(N_GROUPS, 2, len(values), 2)


@pytest.mark.parametrize("func", ARGUMENTS)
def test_the_arguments_give_the_result_of_the_sequential_one(func, data):
    group_idx, values = data
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", ARGUMENTS)
def test_the_arguments_keep_the_first_of_two_equal_values(func, rng):
    # two equal values in a group: the first of them wins - and that is the
    # delicate part of the fold of the chunks, where a tie is won by the
    # chunk that comes first
    group_idx = rng.integers(0, N_GROUPS, 200_000)
    # a handful of values only, so that the groups hold many equal values
    values = np.round(rng.random(200_000) * 3)
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)
    wanted = _reference_arg(group_idx, values, func, N_GROUPS)
    _same(parallel, wanted, 0)


@pytest.mark.parametrize("func", ARGUMENTS)
def test_the_argument_of_every_group_is_the_index_of_its_best_value(func, group_data):
    # the reference is the definition itself: the index of the best value of
    # the group, and the first of them if there are several
    group_idx, values = group_data
    parallel, _ = _both_ways(group_idx, values, func)
    wanted = _reference_arg(group_idx, values, func, N_GROUPS)
    _same(parallel, wanted, 0)


@pytest.mark.parametrize("func", ARGUMENTS)
def test_one_thread_gives_the_sequential_result_of_the_arguments(func, data):
    group_idx, values = data
    nb.set_num_threads(1)
    one = aggregate(group_idx, values, func=func)
    nb.set_num_threads(2)
    many = aggregate(group_idx, values, func=func)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func)
    # with one thread the sequential path runs, and it is the one that has
    # always run - bit for bit
    _same(one, sequential, 0)
    _same(many, sequential, 0)


@pytest.mark.parametrize("func", ARGUMENTS)
def test_the_arguments_keep_the_fill_value_of_the_empty_groups(func, rng):
    group_idx = rng.integers(0, 10, 200_000)
    values = rng.random(200_000)
    for fill in (DEFAULT_FILL_VALUE, -7.0):
        parallel, sequential = _both_ways(group_idx, values, func, size=100, fill_value=fill)
        _same(parallel, sequential, 0)
        if fill != DEFAULT_FILL_VALUE:
            assert (parallel[10:] == -7.0).all()


@pytest.mark.parametrize("func", ARGUMENTS)
def test_the_arguments_complain_about_a_fault_of_the_indices(func, data):
    # the parallel path cannot raise out of its region, so the complaint of
    # the sequential path has to be made after it
    group_idx, values = data
    group_idx = group_idx.copy()
    group_idx[17] = -1
    nb.set_num_threads(2)
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func=func)
