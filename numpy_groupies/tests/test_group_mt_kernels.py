"""The parallel path of the operations that parallelize over the GROUPS.

The order statistics (the median, the quantile, the sort, the number of the
distinct values) cannot be reduced in chunks: the values of a group have to be
together where they are worked on.  These operations cut the *groups* into
ranges instead, and the loop over the groups of one range is the loop of the
sequential kernel - so the tests below say something stronger than the ones of
``test_mt_kernels``: the parallel path does not agree with the sequential one
to the last bits, it gives the very same bytes.

Three things are said about it: the result is the one of the sequential path
bit for bit, the number of the threads does not touch it, and no shape of the
groups hides a group that no range worked on - which is where a range that is
one group too short, or one too long, would show.
"""

import numba as nb
import numpy as np
import pytest

from numpy_groupies.aggregate_numba import aggregate
from numpy_groupies.registry import DEFAULT_FILL_VALUE

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


def _same(got: np.ndarray, wanted: np.ndarray, rtol: float = 0) -> None:
    """The two results have to be the same, nan cells included.

    ``assert_allclose`` and not ``assert_array_equal``: the fill value of the
    order statistics is a nan, and a nan compares unequal to itself - the
    comparison that is about two results of the same operation has to call
    them equal.  ``rtol`` of zero is the bit for bit comparison, the one the
    parallel path of these operations is held to.
    """
    np.testing.assert_allclose(got, wanted, rtol=rtol, atol=0, equal_nan=True)


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


def _data(
    n_values: int = N_VALUES, n_groups: int = N_GROUPS, seed: int = 100, nans: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    group_idx = rng.integers(0, n_groups, n_values)
    values = rng.integers(-1000, 1000, n_values).astype(np.float64)
    if nans:
        values[::10] = np.nan
    return group_idx, values


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    """A shape the cost model sends over to the parallel path."""
    return _data()


@pytest.fixture
def clean() -> tuple[np.ndarray, np.ndarray]:
    """The same, without the nan values - for the comparisons that a nan in
    either of the two results would break."""
    return _data(nans=False)


@pytest.fixture
def skewed() -> tuple[np.ndarray, np.ndarray]:
    """The same number of values, in groups of wildly different size.

    A zipf distribution over the groups, so a handful of them hold most of the
    values.  The ranges are cut after the number of the *values* and not after
    the number of the groups, and this is the shape that shows whether they
    really are: ranges of an equal number of groups would give one thread all
    the big groups, and the call would be as parallel as its longest range.
    """
    rng = np.random.default_rng(7)
    weights = 1.0 / np.arange(1, N_GROUPS + 1)
    weights /= weights.sum()
    group_idx = rng.choice(N_GROUPS, size=N_VALUES, p=weights).astype(np.intp)
    values = rng.integers(-1000, 1000, N_VALUES).astype(np.float64)
    values[::10] = np.nan
    return group_idx, values


def _both_ways(group_idx: np.ndarray, values: np.ndarray, func: str, **kwargs: object) -> tuple:
    """The result of the operation with four threads and the one with one of
    them - the two paths side by side."""
    nb.set_num_threads(2)
    parallel = aggregate(group_idx, values, func=func, **kwargs)
    nb.set_num_threads(1)
    sequential = aggregate(group_idx, values, func=func, **kwargs)
    return parallel, sequential


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_parallel_path_gives_the_result_of_the_sequential_one(func, data):
    group_idx, values = data
    parallel, sequential = _both_ways(group_idx, values, func)
    # not "to the last bits": every group is worked on by the one thread that
    # was given its range, with the same code the sequential kernel runs, so
    # there is no second order in which anything could add up
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_number_of_the_threads_does_not_change_the_result(func, data):
    group_idx, values = data
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
def test_groups_of_wildly_different_size_give_the_same_result(func, skewed):
    group_idx, values = skewed
    # the biggest group of this input holds a fifth of all the values: a range
    # that ends in the middle of it, or a range that is given nothing but
    # empty groups, would show here and only here
    assert np.bincount(group_idx).max() > N_VALUES // 10
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_the_groups_without_a_value_keep_the_fill_value(func, data):
    # the groups beyond the last index of the input and the ones the input
    # leaves empty: the ranges are cut after the counts, so a range can easily
    # hold nothing but groups that are empty
    group_idx, values = data
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
def test_the_quantile_of_every_quantile(func, q, data):
    group_idx, values = data
    parallel, sequential = _both_ways(group_idx, values, func, q=q)
    _same(parallel, sequential, 0)


def test_the_median_is_the_quantile_of_a_half(data):
    # two kernels that are built separately and have to agree, one with the
    # middle index and one with the quantile of it
    group_idx, values = data
    nb.set_num_threads(2)
    _same(aggregate(group_idx, values, func="median"), aggregate(group_idx, values, func="quantile", q=0.5))


@pytest.mark.parametrize("func", ["median", "nanmedian", "quantile", "nanquantile"])
def test_the_result_of_every_group_is_the_result_of_that_group(func, clean):
    # every group, and with them the blocks of one and of two values that the
    # ranges run into at their edges: the two values of a group on either side
    # of a boundary of the ranges is the case a range that is one too short
    # cannot survive
    group_idx, values = clean
    nb.set_num_threads(2)
    kwargs = {"q": 0.5} if func.endswith("quantile") else {}
    got = aggregate(group_idx, values, func=func, **kwargs)
    wanted = np.array([np.quantile(values[group_idx == g], 0.5) for g in range(N_GROUPS)])
    _same(got, wanted, 1e-12)


@pytest.mark.parametrize("func", ["nunique", "nannunique"])
def test_the_distinct_values_are_counted_once(func, data):
    group_idx, values = data
    parallel, sequential = _both_ways(group_idx, values, func)
    _same(parallel, sequential, 0)
    keep = ~np.isnan(values) if func.startswith("nan") else np.ones(values.shape, bool)
    wanted = np.array([len(np.unique(values[(group_idx == g) & keep])) for g in range(N_GROUPS)])
    _same(parallel, wanted, 0)


@pytest.mark.parametrize("reverse", [False, True])
def test_the_sort_is_sorted_within_every_group(reverse, clean):
    group_idx, values = clean
    parallel, sequential = _both_ways(group_idx, values, "sort", reverse=reverse)
    _same(parallel, sequential, 0)
    for g in (0, 37, 99):
        ordered = np.diff(parallel[group_idx == g])
        # the default is the ascending order, reverse is the descending one
        assert ((ordered <= 0) if reverse else (ordered >= 0)).all()


def test_the_sort_keeps_the_values_of_its_group(clean):
    # the same values in every group, only put in order - the ranges of the
    # threads may not move a value across the border of a group
    group_idx, values = clean
    nb.set_num_threads(2)
    got = aggregate(group_idx, values, func="sort")
    for g in (0, 37, 99):
        _same(np.sort(got[group_idx == g]), np.sort(values[group_idx == g]))


@pytest.mark.parametrize("func", GROUP_PARALLEL)
def test_a_fault_of_the_indices_is_complained_about(func, data):
    # the ranges cannot raise out of their region, so the complaint of the
    # sequential path has to be made before a single one of them is cut
    group_idx, values = data
    group_idx[17] = -1
    nb.set_num_threads(2)
    with pytest.raises(ValueError, match="negative indices"):
        aggregate(group_idx, values, func=func)
    group_idx[17] = N_GROUPS
    with pytest.raises(ValueError, match=f"too large for size {N_GROUPS}"):
        aggregate(group_idx, values, func=func, size=N_GROUPS)
