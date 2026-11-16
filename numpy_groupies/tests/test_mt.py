"""The decision whether a kernel runs multithreaded."""

import numba as nb
import numpy as np
import pytest

from numpy_groupies.npg_numba import averaging, mt, reductions, specials
from numpy_groupies.npg_numba.compose import StateLayout

# The machine the cases below are built for: a cache of thirty-two
# megabytes and three physical cores.  The tests name the machine
# explicitly, because the decision of the model depends on the machine - the
# one of the run is not the one the model was fitted on, and a test that
# reads the cache of the machine it runs on fails on every other one.
_CACHE = 32 << 20
_CORES = 3
# the bytes of the state of a group of the sum
_BYTES_PER_GROUP = 8
_MACHINE = mt.MachineLimits(_CACHE, _CORES)


# the decision of the chunked path: the groups, the threads and the values
# against the answer the gate gives
_CHUNKED_CASES = [
    # one thread: nothing is shared, and the region costs more than the work
    pytest.param(1000, 1, 10**9, False, id="one-thread"),
    # too few values: the region costs more than the work it does
    pytest.param(1000, 4, mt.MT_MIN_VALUES - 1, False, id="too-few-values"),
    # many values and few groups: the work pays for the region
    pytest.param(1000, 4, 10**7, True, id="many-values-few-groups"),
    # as many groups as values, the state still in the last-level cache: the
    # partial matrix costs more than the data
    pytest.param(10**6, 4, 10**7, False, id="many-groups-in-cache"),
    # a state beyond the cache: the sequential state misses the cache either
    # way, the duplicated states of the parallel path add no pressure of their
    # own and the threads hide the latency of the scatter
    pytest.param(_CACHE // _BYTES_PER_GROUP + 1, 4, 10**7, True, id="many-groups-dram-bound"),
]


@pytest.mark.parametrize(("n_groups", "n_threads", "n_values", "expected"), _CHUNKED_CASES)
def test_chunked_path_is_decided_by_model(n_groups, n_threads, n_values, expected):
    assert mt.should_use_mt(n_groups, n_threads, n_values, 1, _BYTES_PER_GROUP, _CACHE) is expected


def test_the_threads_whose_state_fits_the_cache():
    """The states of the threads are as many times the state of a kernel as
    there are threads - the cache of the machine holds so many of them."""
    # the state of four threads fits into the cache, the state of five does
    groups = (_CACHE // 4) // _BYTES_PER_GROUP
    assert mt.threads_that_fit(groups, _BYTES_PER_GROUP, _CACHE) == 4
    # a state of thirty two bytes per group holds a quarter as many
    assert mt.threads_that_fit(groups, 4 * _BYTES_PER_GROUP, _CACHE) == 1
    # and the state of one group alone always fits
    assert mt.threads_that_fit(1, _BYTES_PER_GROUP, _CACHE) >= 1


def test_the_plan_uses_the_physical_cores():
    """The threads of the pool beyond the cores of the machine share a core
    with another thread: they bring no bandwidth of their own, but a state
    of their own."""
    machine = mt.MachineLimits(_CACHE, 2)
    # a state that fits the cache twice over: the model takes the cores of
    # the machine, never the hardware threads of the pool
    plan = mt.mt_thread_plan(1000, 6, 10**7, 1, _BYTES_PER_GROUP, machine)
    assert plan <= 2


def test_the_plan_picks_the_threads_that_win():
    """Over the candidates the model of the per-group costs decides.

    The plan takes the threads the model estimates the best - the same
    number, whatever the machine the test runs on.
    """
    # a state that fits the cache many times over: the model decides, and
    # not the cache
    groups = 10**6
    plan = mt.mt_thread_plan(groups, 6, 10**8, 1, _BYTES_PER_GROUP, _MACHINE)
    best = min(range(2, _CORES + 1), key=lambda t: float(mt.estimate_mt_ratio(groups, t, 10**8, 1)))
    assert plan == best
    # ten million values in a hundred thousand groups: the same
    assert mt.mt_thread_plan(100_000, 6, 10**7, 1, _BYTES_PER_GROUP, _MACHINE) == 2


def test_the_plan_falls_back_to_the_sequential_path():
    """Where the model sees no gain, the sequential path runs - its result is
    the parallel one's bit for bit."""
    # too few values: the region costs more than the work it does
    assert mt.mt_thread_plan(1000, 6, mt.MT_MIN_VALUES - 1, 1, _BYTES_PER_GROUP, _MACHINE) == 1
    # one thread only: a parallel region would be pure overhead
    assert mt.mt_thread_plan(1000, 1, 10**7, 1, _BYTES_PER_GROUP, _MACHINE) == 1
    # one core only: the same
    assert mt.mt_thread_plan(1000, 6, 10**7, 1, _BYTES_PER_GROUP, mt.MachineLimits(_CACHE, 1)) == 1


# the decision of the path that cuts the groups into ranges: the same model,
# with the cost of a value of its own
_GROUP_CASES = [
    pytest.param(1000, 1, 10**9, False, id="one-thread"),
    pytest.param(1000, 4, mt.GROUP_MT_MIN_VALUES - 1, False, id="too-few-values"),
    pytest.param(1000, 4, mt.GROUP_MT_MIN_VALUES, True, id="from-the-floor-on"),
    # a third of these kernels stays single threaded, so two threads leave
    # less than the margin of the model: the call would run the parallel
    # region, pay for it and still be near the sequential time
    pytest.param(1000, 2, 10**6, False, id="two-threads"),
    pytest.param(1000, 6, 10**6, True, id="six-threads"),
]


@pytest.mark.parametrize(("n_groups", "n_threads", "n_values", "expected"), _GROUP_CASES)
def test_group_path_is_decided_by_model(n_groups, n_threads, n_values, expected):
    assert mt.should_use_group_mt(n_groups, n_threads, n_values, 18.0) is expected


@pytest.mark.parametrize(
    ("cache_limit", "expected"),
    [
        # a state of ten million groups is 80 MB: it misses a small cache and
        # fits a big one
        pytest.param(8 << 20, True, id="small-cache"),
        pytest.param(256 << 20, False, id="big-cache"),
    ],
)
def test_the_cache_of_the_machine_decides_the_regime(cache_limit, expected):
    """The state of the kernel is measured against the cache of the machine.

    The limit is handed into the call, so that the size of the cache of the
    machine is not baked into the compiled artifact - the cache of the
    compiles is the user-wide one, and a home shared between machines would
    serve the artifacts of the other.
    """
    assert mt.is_dram_bound(10**7, 4, 10**8, 8, cache_limit) is expected


def _measured_state_bytes(init, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    """The bytes the state of an operation holds per group, measured.

    Two states of the call, one with two groups and one with a single one:
    the arrays per group grow with the number of the groups, the constants
    beside them do not, and the difference of the two sizes is the bytes of
    a group.
    """
    bigger = init(2, 2, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill)
    smaller = init(1, 1, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill)

    def total(state):
        return sum(field.nbytes for field in state if isinstance(field, np.ndarray))

    return total(bigger) - total(smaller)


@pytest.mark.parametrize(
    ("operation", "init", "expected_f64", "expected_f32"),
    [
        (reductions.sum, reductions._sum_init, 9, 5),
        (averaging.mean, averaging._mean_init, 16, 12),
        (averaging.var, averaging._std_init, 32, 20),
        (averaging.skew, averaging._moment_init, 48, 28),
    ],
)
def test_the_declared_state_bytes_match_the_state(operation, init, expected_f64, expected_f32):
    """The bytes an operation declares are the bytes its state holds.

    The declaration is what the cost model measures the state against the
    cache of the machine with - a guess of eight bytes per number would be
    wrong by a factor of two for the float32 and the complex input, and in
    the direction that lets the parallel path into a state that does not fit.
    """
    handle = operation.plain if hasattr(operation, "plain") else operation
    declared = handle.state_bytes_per_group_of
    assert declared is not None
    f64 = np.dtype("float64")
    f32 = np.dtype("float32")

    f64_layout = StateLayout(f64, f64, f64)
    f32_layout = StateLayout(f32, f32, f32)
    assert declared(f64_layout) == expected_f64
    assert declared(f32_layout) == expected_f32
    # and the declaration says what the state holds - measured with a fill
    # value of its own, which is the case the bookkeeping of the groups
    # without a value runs in
    assert declared(f64_layout) == _measured_state_bytes(init, f64, f64, f64, np.nan, 0.0)
    assert declared(f32_layout) == _measured_state_bytes(init, f32, f32, f32, np.nan, 0.0)


def test_cache_size_is_read_from_the_machine():
    """The size of the last-level cache, as the machine names it."""
    assert mt.cache_size_bytes() > 0
    assert mt.CACHE_SIZE == mt.cache_size_bytes()


def test_decision_does_not_depend_on_machine():
    # nothing is measured at run time - the answer is the same every time
    first = mt.should_use_mt(1000, 4, 10**6, 1, 8, mt.CACHE_SIZE)
    second = mt.should_use_mt(1000, 4, 10**6, 1, 8, mt.CACHE_SIZE)
    assert first == second


def test_ratio_gets_better_with_more_values():
    # the cost of the groups is fixed, so it amortizes over the values
    few = mt.estimate_mt_ratio(1000, 4, 10**5)
    many = mt.estimate_mt_ratio(1000, 4, 10**7)
    assert many < few


def test_ratio_gets_worse_with_more_groups():
    few = mt.estimate_mt_ratio(1000, 4, 10**6)
    many = mt.estimate_mt_ratio(10**5, 4, 10**6)
    assert many > few


def test_ratio_gets_worse_with_more_threads():
    # the partial matrix of every chunk has to be written and read again
    few = mt.estimate_mt_ratio(10**4, 2, 10**6)
    many = mt.estimate_mt_ratio(10**4, 6, 10**6)
    assert many > few


def test_estimate_is_positive():
    assert mt.estimate_seq_time_us(1000, 10**6) > 0
    assert mt.estimate_mt_time_us(1000, 4, 10**6) > 0
    assert np.isfinite(mt.estimate_mt_ratio(1000, 4, 10**6))


def test_core_counts_are_sane():
    assert 1 <= mt.physical_core_count() <= mt.logical_core_count()


def test_thread_count_is_within_pool():
    assert 1 <= mt.mt_thread_count() <= nb.get_num_threads()


# --- the operations that parallelize over the groups and not over the values ---
#
# The order statistics cannot be reduced in chunks: the values of a group have
# to be together where they are worked on.  They cut the *groups* into ranges
# instead - the counting sort stays single threaded, only the loop over the
# groups is shared out.


def test_one_thread_means_no_group_parallel_path():
    assert not mt.should_use_group_mt(1000, 1, 10**9, 18.0)


def test_too_few_values_mean_no_group_parallel_path():
    assert not mt.should_use_group_mt(1000, 4, mt.GROUP_MT_MIN_VALUES - 1, 18.0)


def test_group_ratio_gets_better_with_more_threads():
    few = mt.estimate_group_mt_ratio(1000, 2, 10**6, 18.0)
    many = mt.estimate_group_mt_ratio(1000, 6, 10**6, 18.0)
    assert many < few


def test_group_path_needs_fewer_values_than_chunked_one():
    # the operations that cut their groups into ranges cost orders of
    # magnitude more per value than the sum: the fixed cost of a parallel
    # region is a far smaller share of their time, so they parallelize where
    # the reductions still run single threaded
    assert not mt.should_use_mt(1000, 4, mt.GROUP_MT_MIN_VALUES, 1, 8, mt.CACHE_SIZE)
    assert mt.should_use_group_mt(1000, 4, mt.GROUP_MT_MIN_VALUES, 18.0)


def test_cheap_operation_needs_more_values_to_amortize_region():
    # the fixed cost of the parallel region against what a value costs: the
    # cheaper the operation, the more values it has to gather before the
    # region is paid for
    cheap = mt.estimate_group_mt_ratio(1000, 4, mt.GROUP_MT_MIN_VALUES, 5.0)
    expensive = mt.estimate_group_mt_ratio(1000, 4, mt.GROUP_MT_MIN_VALUES, 68.0)
    assert cheap > expensive


def test_group_ratio_of_one_thread_is_no_gain():
    # one thread: nothing is shared, and what is left is the overhead of the
    # region - the answer of the model is above one, and the answer of the
    # gate is no
    assert mt.estimate_group_mt_ratio(1000, 1, 10**6, 18.0) > 1.0
    assert not mt.should_use_group_mt(1000, 1, 10**6, 18.0)


def test_ranges_are_more_than_threads():
    # a thread that is done with its range takes one that is left instead of
    # waiting for the slowest core, so the ranges have to outnumber the
    # threads - with a pool of one thread the sequential path runs, and
    # nothing is cut at all
    pools = {2, nb.get_num_threads()}
    for threads in (t for t in pools if t > 1):
        assert mt.group_chunk_count(threads, 1000) > threads
    assert mt.group_chunk_count(1, 1000) == 1
    # and never more than there are groups to work on
    assert mt.group_chunk_count(8, 2) <= 2


def test_group_estimate_is_positive():
    assert mt.estimate_group_mt_ratio(1000, 4, 10**6, 18.0) > 0
    assert np.isfinite(mt.estimate_group_mt_ratio(1000, 4, 10**6, 18.0))


def test_kernel_test_shape_reaches_parallel_path():
    # the tests of the group parallel kernels are only about the parallel path
    # if the shape they use is a shape the model sends there - a threshold that
    # creeps above the test data would leave every one of them a test of the
    # sequential kernel, passing and empty.  The select is the cheapest of the
    # operations that cut their groups into ranges.

    for n_groups in (100, 1000, 10_000):
        assert mt.should_use_group_mt(n_groups, 4, 200_000, specials._NS_PER_VALUE_SELECT)
    assert not mt.should_use_group_mt(100, 4, 1000, specials._NS_PER_VALUE_SELECT)
