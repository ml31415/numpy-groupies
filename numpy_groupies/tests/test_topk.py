"""The tests of the topk: the k largest (or smallest) values of every group.

The answer of the topk is a matrix with a row per group and k columns, so
the tests compare the rows, not the values one by one.

The backends are measured against the group loop of the reference module by
the sweeps of the suite - this module holds the rules of the topk itself.
"""

import numpy as np
import pytest

import numpy_groupies as npg
from numpy_groupies.npg_numba import specials
from numpy_groupies.tests.helpers import is_numba, is_purepy
from numpy_groupies.tests.reference import aggregate_ref, aggregate_ref_along_axis


def _rows(answer):
    """The answer as a matrix - the pure python backend hands out a list of
    lists, which is a matrix of its own."""
    if isinstance(answer, list) and answer and isinstance(answer[0], list):
        return np.array([np.asarray(row) for row in answer])
    return np.asarray(answer)


def test_answers_largest_values_of_every_group(impl):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2))
    assert answer.shape == (3, 2)
    np.testing.assert_array_equal(answer[0], [9.0, 5.0])
    np.testing.assert_array_equal(answer[1], [7.0, 3.0])
    np.testing.assert_array_equal(answer[2], [4.0, np.nan])


def test_answers_smallest_values_on_request(impl):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2, largest=False))
    np.testing.assert_array_equal(answer[0], [5.0, 9.0])
    np.testing.assert_array_equal(answer[1], [1.0, 3.0])
    np.testing.assert_array_equal(answer[2], [4.0, np.nan])


def test_group_with_too_few_values_fills_rest(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0, np.nan])
    np.testing.assert_array_equal(answer[1], [4.0, np.nan, np.nan])


def test_group_without_values_is_filled(impl):
    group_idx = np.array([0, 0])
    a = np.array([5.0, 9.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=2, size=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0])
    np.testing.assert_array_equal(answer[1], [np.nan, np.nan])
    np.testing.assert_array_equal(answer[2], [np.nan, np.nan])


def test_missing_values_are_in_no_selection(impl):
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


def test_k_of_zero_answers_no_value(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=0))
    assert answer.shape == (2, 0)


def test_negative_k_is_refused(impl):
    """No group answers a negative number of values, in the same words."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    with pytest.raises(ValueError, match="k must not be negative"):
        impl.aggregate(group_idx, a, func="topk", k=-1)
    with pytest.raises(ValueError, match="k must not be negative"):
        aggregate_ref(group_idx, a, func="topk", k=-1)


def test_keeps_dtype_of_values(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([5, 9, 4], dtype=np.int32)
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=1))
    np.testing.assert_array_equal(answer, [[9], [4]])


def test_integer_input_fills_with_zero(impl):
    """An integer output holds no nan, so the places a group has none are
    filled with the zero the package fills an integer result with."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5, 9, 4], dtype=np.int32)
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9, 5, 0])
    np.testing.assert_array_equal(answer[1], [4, 0, 0])


def test_orders_temporal_values(impl):
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


def test_axis_of_multidimensional_value_is_refused(impl):
    """A multidimensional group index has no place in a matrix of groups.

    Every backend says the very same words for the very same call, and it says
    them before anything else about the call is complained about: the axis of a
    value of more than one dimension makes a group index of as many dimensions
    as the value has, and the rows of the topk are groups of one.
    """
    a = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match="the topk needs a flat group index"):
        impl.aggregate(np.array([0, 0, 1, 1, 1, 2]), a, func="topk", k=1, axis=0)
    with pytest.raises(ValueError, match="the topk needs a flat group index"):
        impl.aggregate(np.array([0, 1]), a, func="topk", k=1, axis=-1)


def test_axis_of_flat_value_changes_nothing(impl):
    """The labels of the only axis of a value are its plain labels.

    The axis of a one-dimensional value makes the group index neither deeper
    nor wider, so the topk answers the same rows it answers without it - the
    two spellings are the same call, in every backend and with the reference.
    """
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    plain = _rows(impl.aggregate(group_idx, a, func="topk", k=2))
    for axis in (0, -1):
        np.testing.assert_array_equal(_rows(impl.aggregate(group_idx, a, func="topk", k=2, axis=axis)), plain)
    np.testing.assert_array_equal(_rows(aggregate_ref_along_axis(group_idx, a, func="topk", axis=0, k=2)), plain)


def test_more_than_matrix_is_refused_by_reference():
    """The reference measures the matrix of the topk and nothing above it.

    Labels of more than one dimension - and with them the sequence of sizes
    they ask for - and an axis of a value of more than one dimension would all
    place the rows of the selection over more than one dimension, which the
    package refuses as well.  The reference refuses them instead of flattening
    the groups into the rows of a matrix it has no right to expect.
    """
    group_idx = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    a = np.array([5.0, 9.0, 1.0, 7.0])
    with pytest.raises(NotImplementedError, match="the topk answers a matrix"):
        aggregate_ref(group_idx, a, func="topk", size=(2, 2), k=2)
    with pytest.raises(NotImplementedError, match="the topk answers a matrix"):
        aggregate_ref_along_axis(np.array([0, 1, 0]), np.zeros((3, 2)), func="topk", axis=0, k=2)


# the refusals of the topk are a rule of the call, and not of a backend: the
# same bad call says the same words in every one of them - and the reference
# refuses what the package refuses, with the words of the package
_BAD_GROUP_IDX = (
    pytest.param(np.array([[0, 1, 0, 1], [0, 0, 1, 1]]), np.arange(4.0), {}, id="2d labels without a size"),
    pytest.param(
        np.array([[0, 1, 0, 1], [0, 0, 1, 1]]), np.arange(4.0), {"size": (2, 2)}, id="2d labels with a size sequence"
    ),
    pytest.param(
        np.array([[0, 1, 0, 1], [0, 0, 1, 1]]), np.arange(4.0), {"size": [2, 2]}, id="2d labels with a size list"
    ),
    pytest.param(np.array([0, 1, 0]), np.zeros((3, 4)), {"axis": 0}, id="an axis of a 2d value"),
    pytest.param(np.array([0, 1, 0, 0]), np.zeros((3, 4)), {"axis": -1}, id="the last axis of a 2d value"),
)


@pytest.mark.parametrize(("group_idx", "a", "kwargs"), _BAD_GROUP_IDX)
def test_flat_group_index_rule_is_same_in_every_backend(impl, group_idx, a, kwargs):
    """A call the matrix of the topk cannot answer is refused alike.

    Labels of more than one dimension - the ones an axis of a multi-dimensional
    value makes - name the groups of the matrix in more ways than it has rows,
    so every backend refuses them in the words of the package.  The pure python
    implementation knows no axis at all, and says so before any rule of a
    function is asked.
    """
    if is_purepy(impl) and "axis" in kwargs:
        pytest.skip("the pure python implementation has no axis")
    with pytest.raises(ValueError, match="the topk needs a flat group index"):
        impl.aggregate(group_idx, a, func="topk", k=1, **kwargs)


@pytest.mark.parametrize(
    "k",
    [1.5, True, "2", np.array([2]), [2]],
    ids=["fraction", "flag", "text", "array", "list"],
)
def test_k_naming_no_count_is_refused_in_same_words(impl, k):
    """The count of the values is a whole number, in every backend.

    Before this was asked for itself, a fraction reached the numpy backend and
    came back in the words of numpy, the compiled kernels of the numba backend
    failed to type it in a wall of their own talk, and the pure python one
    multiplied a list by it - three complaints, and none of them about k.
    """
    with pytest.raises(TypeError, match="k must be a whole number of values"):
        impl.aggregate(np.array([0, 0, 1]), np.array([5.0, 9.0, 4.0]), func="topk", k=k)
    with pytest.raises(TypeError, match="k must be a whole number of values"):
        aggregate_ref(np.array([0, 0, 1]), np.array([5.0, 9.0, 4.0]), func="topk", k=k)


def test_package_and_implementation_agree(impl):
    """The topk of the package - the one of the best implementation - is the
    topk of every implementation."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    np.testing.assert_array_equal(npg.topk(group_idx, a, k=2), impl.topk(group_idx, a, k=2))


def test_function_answers_same_as_call(impl):
    """The function of the implementation - the bare kernel of the numba one -
    and the call of the backend answer the same rows."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    np.testing.assert_array_equal(impl.topk(group_idx, a, k=2), impl.aggregate(group_idx, a, func="topk", k=2))


def test_tick_kernel_answers_ticks():
    """The bare kernel of the topk of the ticks works on the int64 ticks, and
    it leaves the NaT of a group out."""

    ticks = np.array([5, -9223372036854775808, 9, 3], dtype=np.int64)
    group_idx = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(
        specials.tick_topk.plain.stub(group_idx, ticks, 2),
        np.array([[5, 0], [9, 3]], dtype=np.int64),
    )


def test_temporal_input_can_be_asked_for_in_ticks(impl, skip_purepy):
    """The ticks of the temporal values are the int64 of the moments."""
    group_idx = np.array([0, 0, 1])
    a = np.array(["2011-01-03", "2011-01-01", "2011-01-05"], dtype="datetime64[s]")
    ticks = impl.aggregate(group_idx, a, func="topk", k=1, dtype=np.int64)
    np.testing.assert_array_equal(ticks, np.array([[1294012800], [1294185600]]))


def test_out_argument_takes_matrix_of_caller(impl):
    """The out= of the numba backend takes the matrix of the topk."""
    if not is_numba(impl):
        pytest.skip("only the numba backend gathers into the array of the caller")
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    out = np.empty((3, 2))
    answer = impl.aggregate(group_idx, a, func="topk", k=2, out=out)
    assert answer is out
    np.testing.assert_array_equal(out[0], [9.0, 5.0])
    np.testing.assert_array_equal(out[2], [4.0, np.nan])


def test_float16_runs_on_float32(impl):
    """Numba has no float16, so the values run on float32 and the answer is
    rounded back - like the other operations of the backend."""
    if not is_numba(impl):
        pytest.skip("only the numba backend runs the values on float32")
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0], dtype=np.float16)
    answer = impl.aggregate(group_idx, a, func="topk", k=1)
    assert answer.dtype == np.float16
    np.testing.assert_array_equal(answer, np.array([[9.0], [4.0]], dtype=np.float16))


@pytest.mark.parametrize(
    ("largest", "k", "expected"),
    [
        (True, 3, [[7.0, 7.0, 7.0], [9.0, 5.0, 5.0]]),
        (False, 2, [[1.0, 7.0], [5.0, 5.0]]),
    ],
    ids=["largest", "smallest"],
)
def test_border_values_come_from_group(impl, largest, k, expected):
    """The k-th value of the group is the border of the selection.

    A group whose values are equal at the border has as many of them as the
    selection needs - the kernel fills the places the better values leave
    with the border itself, and it has to be a value the group holds.  The
    border of the selection of the smallest values is the k-th one of them.
    """
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    a = np.array([7.0, 7.0, 7.0, 1.0, 5.0, 5.0, 9.0])
    answer = _rows(impl.aggregate(group_idx, a, func="topk", k=k, largest=largest))
    for group, row in enumerate(expected):
        np.testing.assert_array_equal(answer[group], row)


def test_group_index_of_more_than_one_dimension_is_refused(impl):
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
