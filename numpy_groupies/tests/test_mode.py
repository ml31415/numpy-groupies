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
from numpy_groupies.tests.helpers import is_numba


def test_answers_most_frequent_value_of_every_group(impl):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [1.0, 2.0, 5.0])


def test_tie_answers_smallest_value(impl):
    """The values that share the highest count hand the answer to the
    smallest of them, like ``scipy.stats.mode`` breaks a tie."""
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    a = np.array([7.0, -5.0, 7.0, -5.0, 4.0, 4.0, 9.0])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [-5.0, 4.0])


def test_missing_values_are_in_no_selection(impl):
    """A nan is never the most frequent value of a group: it does not count,
    and the mode of the values that are left is the answer."""
    group_idx = np.array([0, 0, 0, 0, 1])
    a = np.array([np.nan, 2.0, 2.0, np.nan, 3.0])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [2.0, 3.0])


def test_group_of_nothing_but_missing_values_takes_fill_value(impl):
    """A group that holds nothing but nans has no mode at all - and neither
    has one the labels never name."""
    group_idx = np.array([0, 0])
    a = np.array([np.nan, 2.0])
    answer = impl.aggregate(group_idx, a, func="mode", size=3)
    np.testing.assert_array_equal(answer, [2.0, np.nan, np.nan])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode", size=3, fill_value=-1.0), [2.0, -1.0, -1.0])


def test_keeps_dtype_of_values(impl):
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([3, 1, 2, 2], dtype=np.int32)
    answer = np.asarray(impl.aggregate(group_idx, a, func="mode"))
    assert answer.dtype == np.int32
    np.testing.assert_array_equal(answer, [1, 2])


def test_integer_input_fills_with_zero(impl):
    """An integer output holds no nan, so a group without a mode is filled
    with the zero the package fills an integer result with."""
    group_idx = np.array([0, 0])
    a = np.array([3, 1], dtype=np.int64)
    answer = impl.aggregate(group_idx, a, func="mode", size=3)
    np.testing.assert_array_equal(answer, [1, 0, 0])


def test_boolean_input(impl):
    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array([True, True, False, False, True, False])
    answer = impl.aggregate(group_idx, a, func="mode")
    np.testing.assert_array_equal(answer, [True, False])


def test_complex_values_tie_lexicographically(impl):
    """The values are ordered the way numpy orders complex ones - the real
    part first, the imaginary part only on a tie."""
    group_idx = np.array([0, 0, 0, 1])
    a = np.array([1 + 9j, 3 + 1j, 1 + 2j, 2 + 2j])
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="mode"), [1 + 2j, 2 + 2j])


def test_orders_temporal_values(impl):
    """The temporal values are counted like any other value, and their NaT
    is in no selection."""
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-01", "NaT", "2011-01-01", "2011-01-05"], dtype="datetime64[s]")
    answer = np.asarray(impl.aggregate(group_idx, a, func="mode"))
    assert answer.dtype == a.dtype
    np.testing.assert_array_equal(answer, np.array(["2011-01-01", "2011-01-05"], dtype="datetime64[s]"))


@pytest.mark.parametrize(
    "values", [[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0, 0.0]], ids=["minus-first", "plus-first", "three"]
)
def test_sign_of_zero_is_sign_of_first_one(impl, values):
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


def test_package_and_implementation_agree(impl):
    """The mode of the package - the one of the best implementation - is the
    mode of every implementation."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    np.testing.assert_array_equal(npg.mode(group_idx, a), impl.mode(group_idx, a))


def test_function_answers_same_as_call(impl):
    """The function of the implementation - the bare kernel of the numba one -
    and the call of the backend answer the same values."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    np.testing.assert_array_equal(impl.mode(group_idx, a), impl.aggregate(group_idx, a, func="mode"))


def test_out_argument_takes_answer_of_caller(impl):
    """The out= of the numba backend takes the values of the mode."""
    if not is_numba(impl):
        pytest.skip("only the numba backend gathers into the array of the caller")
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 5.0])
    out = np.empty(3)
    answer = impl.aggregate(group_idx, a, func="mode", out=out)
    assert answer is out
    np.testing.assert_array_equal(out, [1.0, 2.0, 5.0])


def test_float16_runs_on_float32(impl):
    """Numba has no float16, so the values run on float32 and the answer is
    rounded back - like the other operations of the backend."""
    if not is_numba(impl):
        pytest.skip("only the numba backend runs the values on float32")
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 5.0, 4.0], dtype=np.float16)
    answer = impl.aggregate(group_idx, a, func="mode")
    assert answer.dtype == np.float16
    np.testing.assert_array_equal(answer, np.array([5.0, 4.0], dtype=np.float16))
