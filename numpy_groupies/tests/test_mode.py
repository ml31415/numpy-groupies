"""The tests of the mode: the most frequent value of every group.

The mode answers one value per group - the smallest of the values that
share the highest count, the way ``scipy.stats.mode`` breaks a tie.  Like
the topk it has no nan variant: the missing values are in no selection - a
missing value is never the most frequent value of a group - and a group
that holds nothing but them has no mode at all, so it takes the fill value.

The backends are measured against the group loop of the reference module by
the sweeps of the suite - this module holds the rules of the mode itself.
"""

import numpy as np
import pytest

import numpy_groupies as npg
import numpy_groupies.aggregate_numba as npgb
from numpy_groupies.tests.helpers import is_purepy


def test_the_mode_answers_the_most_frequent_value_of_every_group(impl):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [1.0, 2.0, 5.0])


def test_a_tie_answers_the_smallest_value(impl):
    """The values that share the highest count hand the answer to the
    smallest of them, like ``scipy.stats.mode`` breaks a tie."""
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    a = np.array([7.0, -5.0, 7.0, -5.0, 4.0, 4.0, 9.0])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [-5.0, 4.0])


def test_the_missing_values_are_in_no_selection(impl):
    """A nan is never the most frequent value of a group: it does not count,
    and the mode of the values that are left is the answer."""
    group_idx = np.array([0, 0, 0, 0, 1])
    a = np.array([np.nan, 2.0, 2.0, np.nan, 3.0])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [2.0, 3.0])


def test_a_group_of_nothing_but_missing_values_takes_the_fill_value(impl):
    """A group that holds nothing but nans has no mode at all - and neither
    has one the labels never name."""
    group_idx = np.array([0, 0])
    a = np.array([np.nan, 2.0])
    answer = impl.aggregate(group_idx, a, func="mode", size=3)
    np.testing.assert_array_equal(answer, [2.0, np.nan, np.nan])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode", size=3, fill_value=-1.0), [2.0, -1.0, -1.0])


def test_the_mode_keeps_the_dtype_of_the_values(impl):
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([3, 1, 2, 2], dtype=np.int32)
    answer = impl.aggregate(group_idx, a, func="mode")
    if not is_purepy(impl):
        assert answer.dtype == np.int32
    np.testing.assert_array_equal(answer, [1, 2])


def test_the_mode_of_an_integer_input_fills_with_zero(impl):
    """An integer output holds no nan, so a group without a mode is filled
    with the zero the package fills an integer result with."""
    group_idx = np.array([0, 0])
    a = np.array([3, 1], dtype=np.int64)
    answer = impl.aggregate(group_idx, a, func="mode", size=3)
    np.testing.assert_array_equal(answer, [1, 0, 0])


def test_the_mode_of_a_boolean_input(impl):
    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array([True, True, False, False, True, False])
    answer = impl.aggregate(group_idx, a, func="mode")
    np.testing.assert_array_equal(answer, [True, False])


def test_the_mode_of_complex_values_ties_lexicographically(impl):
    """The values are ordered the way numpy orders complex ones - the real
    part first, the imaginary part only on a tie."""
    group_idx = np.array([0, 0, 0, 1])
    a = np.array([1 + 9j, 3 + 1j, 1 + 2j, 2 + 2j])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [1 + 2j, 2 + 2j])


def test_the_mode_of_the_temporal_values(impl):
    """The temporal values are counted like any other value, and their NaT
    is in no selection."""
    if is_purepy(impl):
        # the pure python backend knows no temporal types
        pytest.skip("the pure python implementation has no temporal types")
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-01", "NaT", "2011-01-01", "2011-01-05"], dtype="datetime64[s]")
    answer = impl.aggregate(group_idx, a, func="mode")
    assert answer.dtype == a.dtype
    np.testing.assert_array_equal(answer, np.array(["2011-01-01", "2011-01-05"], dtype="datetime64[s]"))


@pytest.mark.parametrize(
    "values", [[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0, 0.0]], ids=["minus-first", "plus-first", "three"]
)
def test_the_sign_of_a_zero_is_the_sign_of_the_first_one(impl, values):
    """A zero of a group keeps the sign of the first zero of the group.

    The two zeros are the same value - ``0.0 == -0.0`` - and every backend
    answers the one of them, but the sign of the answer is the sign of the
    first zero the group holds.  The numpy backend orders the values of the
    group and takes the first of the longest run, the numba one counts them
    in its hash table - where the two zeros share their slot - and the pure
    python one counts them in its dict, which keeps the first key.
    """
    group_idx = np.zeros(len(values), dtype=int)
    a = np.array(values)
    answer = np.asarray(impl.aggregate(group_idx, a, func="mode"))
    assert np.signbit(answer[0]) == np.signbit(a[0])


def test_the_mode_of_the_package_is_the_one_of_the_best_implementation():
    import numpy_groupies.aggregate_numpy as npgn

    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    np.testing.assert_array_equal(npg.mode(group_idx, a), npgn.aggregate(group_idx, a, func="mode"))


def test_the_mode_of_the_kernel_answers_the_same_as_the_call():
    """The bare kernel of the numba backend and the python-level call of the
    backend answer the same values."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    np.testing.assert_array_equal(npgb.mode(group_idx, a), npgb.aggregate(group_idx, a, func="mode"))


def test_the_mode_of_the_numba_backend_goes_into_the_array_of_the_caller():
    """The out= of the numba backend takes the values of the mode."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    out = np.empty(3)
    answer = npgb.aggregate(group_idx, a, func="mode", out=out)
    assert answer is out
    np.testing.assert_array_equal(out, [1.0, 2.0, 5.0])


def test_the_mode_of_the_numba_backend_of_float16_runs_on_float32():
    """Numba has no float16, so the values run on float32 and the answer is
    rounded back - like the other operations of the backend."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 5.0, 4.0], dtype=np.float16)
    answer = npgb.aggregate(group_idx, a, func="mode")
    assert answer.dtype == np.float16
    np.testing.assert_array_equal(answer, np.array([5.0, 4.0], dtype=np.float16))
