"""The tests of the topk: the k largest (or smallest) values of every group.

The answer of the topk is a matrix with a row per group and k columns, so
the tests compare the rows, not the values one by one.

The backends are measured against the group loop of the reference module by
the sweeps of the suite - this module holds the rules of the topk itself.
"""

import numpy as np
import pytest

import numpy_groupies as npg
import numpy_groupies.aggregate_numba as npgb
from numpy_groupies.tests.helpers import is_purepy


def _rows(answer):
    """The answer as a matrix - the pure python backend hands out a list of
    lists, which is a matrix of its own."""
    if isinstance(answer, list) and answer and isinstance(answer[0], list):
        return np.array([np.asarray(row) for row in answer])
    return np.asarray(answer)


def test_the_topk_answers_the_largest_values_of_every_group(impl):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2))
    assert answer.shape == (3, 2)
    np.testing.assert_array_equal(answer[0], [9.0, 5.0])
    np.testing.assert_array_equal(answer[1], [7.0, 3.0])
    np.testing.assert_array_equal(answer[2], [4.0, np.nan])


def test_the_topk_answers_the_smallest_values_when_it_is_asked_to(impl):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2, largest=False))
    np.testing.assert_array_equal(answer[0], [5.0, 9.0])
    np.testing.assert_array_equal(answer[1], [1.0, 3.0])
    np.testing.assert_array_equal(answer[2], [4.0, np.nan])


def test_a_group_with_too_few_values_fills_the_rest(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0, np.nan])
    np.testing.assert_array_equal(answer[1], [4.0, np.nan, np.nan])


def test_a_group_that_holds_no_value_at_all_is_filled(impl):
    group_idx = np.array([0, 0])
    a = np.array([5.0, 9.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2, size=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0])
    np.testing.assert_array_equal(answer[1], [np.nan, np.nan])
    np.testing.assert_array_equal(answer[2], [np.nan, np.nan])


def test_the_missing_values_are_in_no_selection(impl):
    """A nan is neither the largest nor the smallest value of a group.

    The sort of a group puts the missing values last, so they are in no
    selection - and a group that holds nothing but them has no value to
    name at all.
    """
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array([5.0, np.nan, 9.0, np.nan, np.nan])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0, np.nan])
    np.testing.assert_array_equal(answer[1], [np.nan, np.nan, np.nan])


def test_k_of_zero_answers_no_value_at_all(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=0))
    assert answer.shape == (2, 0)


def test_a_negative_k_is_refused(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    with pytest.raises(ValueError, match="k must not be negative"):
        impl.aggregate(group_idx, a, func="topk", k=-1)


def test_the_topk_keeps_the_dtype_of_the_values(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5, 9, 4], dtype=np.int32)
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=1))
    np.testing.assert_array_equal(answer, [[9], [4]])


def test_the_topk_of_an_integer_input_fills_with_zero(impl):
    """An integer output holds no nan, so the places a group has none are
    filled with the zero the package fills an integer result with."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5, 9, 4], dtype=np.int32)
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9, 5, 0])
    np.testing.assert_array_equal(answer[1], [4, 0, 0])


def test_the_topk_of_the_temporal_values(impl):
    """The temporal values are ordered like any other value, and their NaT is
    in no selection."""
    if is_purepy(impl):
        # the pure python backend knows no temporal types
        pytest.skip("the pure python implementation has no temporal types")
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "2011-01-05"], dtype="datetime64[s]")
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2))
    np.testing.assert_array_equal(answer[0], np.array(["2011-01-03", "2011-01-01"], dtype="datetime64[s]"))
    np.testing.assert_array_equal(answer[1], np.array(["2011-01-05", "NaT"], dtype="datetime64[s]"))


def test_the_topk_needs_a_flat_group_index(impl):
    """A multidimensional group index has no place in a matrix of groups."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match="the topk needs a flat group index"):
        impl.aggregate(group_idx, a, func="topk", k=1, axis=0)


def test_the_topk_of_the_package_is_the_one_of_the_best_implementation():
    import numpy_groupies.aggregate_numpy as npgn

    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    np.testing.assert_array_equal(npg.topk(group_idx, a, k=2), npgn.aggregate(group_idx, a, func="topk", k=2))


def test_the_topk_of_the_kernel_answers_the_same_as_the_call():
    """The bare kernel of the numba backend and the python-level call of the
    backend answer the same rows."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    np.testing.assert_array_equal(npgb.topk(group_idx, a, 2), npgb.aggregate(group_idx, a, func="topk", k=2))


def test_the_topk_of_the_ticks_answers_the_ticks():
    """The bare kernel of the topk of the ticks works on the int64 ticks, and
    it leaves the NaT of a group out."""
    ticks = np.array([5, -9223372036854775808, 9, 3], dtype=np.int64)
    group_idx = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(
        npgb.specials.tick_topk.plain.stub(group_idx, ticks, 2),
        np.array([[5, 0], [9, 3]], dtype=np.int64),
    )


def test_the_topk_of_a_temporal_input_can_be_asked_for_in_ticks():
    import numpy_groupies.aggregate_numpy as npgn

    group_idx = np.array([0, 0, 1])
    a = np.array(["2011-01-03", "2011-01-01", "2011-01-05"], dtype="datetime64[s]")
    ticks = npgn.aggregate(group_idx, a, func="topk", k=1, dtype=np.int64)
    np.testing.assert_array_equal(ticks, np.array([[1294012800], [1294185600]]))


def test_the_topk_of_the_numba_backend_goes_into_the_array_of_the_caller():
    """The out= of the numba backend takes the matrix of the topk."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    out = np.empty((3, 2))
    answer = npgb.aggregate(group_idx, a, func="topk", k=2, out=out)
    assert answer is out
    np.testing.assert_array_equal(out[0], [9.0, 5.0])
    np.testing.assert_array_equal(out[2], [4.0, np.nan])


def test_the_topk_of_the_numba_backend_of_float16_runs_on_float32():
    """Numba has no float16, so the values run on float32 and the answer is
    rounded back - like the other operations of the backend."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0], dtype=np.float16)
    answer = npgb.aggregate(group_idx, a, func="topk", k=1)
    assert answer.dtype == np.float16
    np.testing.assert_array_equal(answer, np.array([[9.0], [4.0]], dtype=np.float16))


def test_the_values_at_the_border_are_the_ones_the_group_holds(impl):
    """The k-th value of the group is the border of the selection.

    A group whose values are equal at the border has as many of them as the
    selection needs - the kernel fills the places the better values leave
    with the border itself, and it has to be a value the group holds.
    """
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    a = np.array([7.0, 7.0, 7.0, 1.0, 5.0, 5.0, 9.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [7.0, 7.0, 7.0])
    np.testing.assert_array_equal(answer[1], [9.0, 5.0, 5.0])


def test_the_values_at_the_border_are_the_ones_the_group_holds_for_the_smallest(impl):
    """The border of the selection of the smallest values is the k-th one."""
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    a = np.array([7.0, 7.0, 7.0, 1.0, 5.0, 5.0, 9.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2, largest=False))
    np.testing.assert_array_equal(answer[0], [1.0, 7.0])
    np.testing.assert_array_equal(answer[1], [5.0, 5.0])


def test_a_group_index_of_more_than_one_dimension_is_refused(impl):
    """The answer of the topk is a matrix with a row per group - a group
    index of more than one dimension names its groups in more than one way,
    and has no place in it.

    The values are two-dimensional as well, so that the complaint is about
    the group index and not about the shape of the values.
    """
    group_idx = np.array([[0, 0], [1, 1]])
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        impl.aggregate(group_idx, a, func="topk", k=1)
