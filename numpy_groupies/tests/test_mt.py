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
