"""The decision whether a kernel runs multithreaded."""

import numpy as np

from numpy_groupies.aggregate_numba import mt


def test_one_thread_means_no_parallel_path():
    assert not mt.should_use_mt(1000, 1, 10**9)


def test_too_few_values_mean_no_parallel_path():
    assert not mt.should_use_mt(1000, 4, mt.MT_MIN_VALUES - 1)


def test_many_values_with_few_groups_use_the_parallel_path():
    assert mt.should_use_mt(1000, 4, 10**7)


def test_many_groups_mean_no_parallel_path():
    # as many groups as values: the partial matrix costs more than the data
    assert not mt.should_use_mt(10**7, 4, 10**7)


def test_the_decision_does_not_depend_on_the_machine():
    # nothing is measured at run time - the answer is the same every time
    first = mt.should_use_mt(1000, 4, 10**6)
    second = mt.should_use_mt(1000, 4, 10**6)
    assert first == second


def test_the_ratio_gets_better_with_more_values():
    # the cost of the groups is fixed, so it amortizes over the values
    few = mt.estimate_mt_ratio(1000, 4, 10**5)
    many = mt.estimate_mt_ratio(1000, 4, 10**7)
    assert many < few


def test_the_ratio_gets_worse_with_more_groups():
    few = mt.estimate_mt_ratio(1000, 4, 10**6)
    many = mt.estimate_mt_ratio(10**5, 4, 10**6)
    assert many > few


def test_the_ratio_gets_worse_with_more_threads():
    # the partial matrix of every chunk has to be written and read again
    few = mt.estimate_mt_ratio(10**4, 2, 10**6)
    many = mt.estimate_mt_ratio(10**4, 6, 10**6)
    assert many > few


def test_the_estimate_is_positive():
    assert mt.estimate_seq_time_us(1000, 10**6) > 0
    assert mt.estimate_mt_time_us(1000, 4, 10**6) > 0
    assert np.isfinite(mt.estimate_mt_ratio(1000, 4, 10**6))


def test_the_core_counts_are_sane():
    assert 1 <= mt.physical_core_count() <= mt.logical_core_count()


def test_the_thread_count_is_within_the_pool():
    import numba as nb

    assert 1 <= mt.mt_thread_count() <= nb.get_num_threads()


# --- the operations that parallelize over the groups and not over the values ---
#
# The order statistics cannot be reduced in chunks: the values of a group have
# to be together where they are worked on.  They cut the *groups* into ranges
# instead, and the decision is a model of its own - the counting sort that
# places the values stays single threaded, and only the loop over the groups
# is shared out.


def test_one_thread_means_no_group_parallel_path():
    assert not mt.should_use_group_mt(1000, 1, 10**9, 18.0)


def test_too_few_values_mean_no_group_parallel_path():
    assert not mt.should_use_group_mt(1000, 4, mt.GROUP_MT_MIN_VALUES - 1, 18.0)


def test_the_group_path_is_used_from_the_floor_of_values_on():
    # the estimate of these operations is nearly flat in the number of the
    # values (the fixed cost of the parallel region is a few microseconds
    # against the milliseconds they take), so the floor is what decides
    assert not mt.should_use_group_mt(1000, 4, mt.GROUP_MT_MIN_VALUES - 1, 18.0)
    assert mt.should_use_group_mt(1000, 4, mt.GROUP_MT_MIN_VALUES, 18.0)


def test_two_threads_are_not_enough_for_the_group_path():
    # a third of these kernels stays single threaded, so two threads leave
    # less than the margin of the model: the call would run the parallel
    # region, pay for it and still be near the sequential time
    assert not mt.should_use_group_mt(1000, 2, 10**6, 18.0)
    assert mt.should_use_group_mt(1000, 6, 10**6, 18.0)


def test_the_group_ratio_gets_better_with_more_threads():
    few = mt.estimate_group_mt_ratio(1000, 2, 10**6, 18.0)
    many = mt.estimate_group_mt_ratio(1000, 6, 10**6, 18.0)
    assert many < few


def test_the_group_path_needs_fewer_values_than_the_chunked_one():
    # the operations that cut their groups into ranges cost 18 to 68
    # nanoseconds a value against the half a nanosecond of the sum: the fixed
    # cost of a parallel region is a thousand times smaller a share of their
    # time, so they parallelize where the reductions still run single threaded
    assert not mt.should_use_mt(1000, 4, mt.GROUP_MT_MIN_VALUES)
    assert mt.should_use_group_mt(1000, 4, mt.GROUP_MT_MIN_VALUES, 18.0)


def test_an_cheap_operation_needs_more_values_to_amortize_the_region():
    # the fixed cost of the parallel region against what a value costs: the
    # cheaper the operation, the more values it has to gather before the
    # region is paid for
    cheap = mt.estimate_group_mt_ratio(1000, 4, mt.GROUP_MT_MIN_VALUES, 5.0)
    expensive = mt.estimate_group_mt_ratio(1000, 4, mt.GROUP_MT_MIN_VALUES, 68.0)
    assert cheap > expensive


def test_the_group_ratio_of_one_thread_is_no_gain():
    # one thread: nothing is shared, and what is left is the overhead of the
    # region - the answer of the model is above one, and the answer of the
    # gate is no
    assert mt.estimate_group_mt_ratio(1000, 1, 10**6, 18.0) > 1.0
    assert not mt.should_use_group_mt(1000, 1, 10**6, 18.0)


def test_the_ranges_are_more_than_the_threads():
    # a thread that is done with its range takes one that is left instead of
    # waiting for the slowest core, so the ranges have to outnumber the threads
    for threads in (2, 4, 6, 7):
        assert mt.group_chunk_count(threads, 1000) > threads
    # and never more than there are groups to work on
    assert mt.group_chunk_count(8, 2) <= 2


def test_the_group_estimate_is_positive():
    assert mt.estimate_group_mt_ratio(1000, 4, 10**6, 18.0) > 0
    assert np.isfinite(mt.estimate_group_mt_ratio(1000, 4, 10**6, 18.0))


def test_the_shape_of_the_kernel_tests_is_one_the_parallel_path_is_used_for():
    # the tests of the group parallel kernels are only about the parallel path
    # if the shape they use is a shape the model sends there - a threshold that
    # creeps above the test data would leave every one of them a test of the
    # sequential kernel, passing and empty.  The select is the cheapest of the
    # operations that cut their groups into ranges, so what is enough for it is
    # enough for all of them (the sort costs nearly four times as much a value).
    from numpy_groupies.aggregate_numba import specials

    for n_groups in (100, 1000, 10_000):
        assert mt.should_use_group_mt(n_groups, 4, 200_000, specials._NS_PER_VALUE_SELECT)
    assert not mt.should_use_group_mt(100, 4, 1000, specials._NS_PER_VALUE_SELECT)
