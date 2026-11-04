"""The tests of the topk: the k largest (or smallest) values of every group.

The answer of the topk is a matrix with a row per group and k columns, so
the tests compare the rows, not the values one by one.
"""

import numpy as np
import pytest

import numpy_groupies as npg
import numpy_groupies.aggregate_numba as npgb
import numpy_groupies.aggregate_numpy as npgn
import numpy_groupies.aggregate_purepy as npgp

# the three backends of the package, and the package itself
BACKENDS = [npgn, npgb, npgp]


def _as_matrix(answer):
    """The answer of a backend as a matrix - the pure python one hands out a
    list of lists, which is a matrix of its own."""
    if isinstance(answer, list) and answer and isinstance(answer[0], list):
        return np.array([np.asarray(row) for row in answer])
    return np.asarray(answer)


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_the_topk_answers_the_largest_values_of_every_group(backend):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=2))
    assert answer.shape == (3, 2)
    np.testing.assert_array_equal(answer[0], [9.0, 5.0])
    np.testing.assert_array_equal(answer[1], [7.0, 3.0])
    np.testing.assert_array_equal(answer[2], [4.0, np.nan])


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_the_topk_answers_the_smallest_values_when_it_is_asked_to(backend):
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.array([5.0, 9.0, 1.0, 7.0, 3.0, 4.0])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=2, largest=False))
    np.testing.assert_array_equal(answer[0], [5.0, 9.0])
    np.testing.assert_array_equal(answer[1], [1.0, 3.0])
    np.testing.assert_array_equal(answer[2], [4.0, np.nan])


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_a_group_with_too_few_values_fills_the_rest(backend):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0, np.nan])
    np.testing.assert_array_equal(answer[1], [4.0, np.nan, np.nan])


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_a_group_that_holds_no_value_at_all_is_filled(backend):
    group_idx = np.array([0, 0])
    a = np.array([5.0, 9.0])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=2, size=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0])
    np.testing.assert_array_equal(answer[1], [np.nan, np.nan])
    np.testing.assert_array_equal(answer[2], [np.nan, np.nan])


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_the_missing_values_are_in_no_selection(backend):
    """A nan is neither the largest nor the smallest value of a group.

    The sort of a group puts the missing values last, so they are in no
    selection - and a group that holds nothing but them has no value to
    name at all.
    """
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array([5.0, np.nan, 9.0, np.nan, np.nan])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9.0, 5.0, np.nan])
    np.testing.assert_array_equal(answer[1], [np.nan, np.nan, np.nan])


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_k_of_zero_answers_no_value_at_all(backend):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=0))
    assert answer.shape == (2, 0)


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_a_negative_k_is_refused(backend):
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0])
    with pytest.raises(ValueError, match="k must not be negative"):
        backend.aggregate(group_idx, a, func="topk", k=-1)


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_the_topk_keeps_the_dtype_of_the_values(backend):
    group_idx = np.array([0, 0, 1])
    a = np.array([5, 9, 4], dtype=np.int32)
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=1))
    np.testing.assert_array_equal(answer, [[9], [4]])


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_the_topk_of_an_integer_input_fills_with_zero(backend):
    """An integer output holds no nan, so the places a group has none are
    filled with the zero the package fills an integer result with."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5, 9, 4], dtype=np.int32)
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=3))
    np.testing.assert_array_equal(answer[0], [9, 5, 0])
    np.testing.assert_array_equal(answer[1], [4, 0, 0])


@pytest.mark.parametrize("backend", [npgn, npgb], ids=["numpy", "numba"])
def test_the_topk_of_the_temporal_values(backend):
    """The temporal values are ordered like any other value, and their NaT is
    in no selection."""
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "2011-01-05"], dtype="datetime64[s]")
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=2))
    np.testing.assert_array_equal(answer[0], np.array(["2011-01-03", "2011-01-01"], dtype="datetime64[s]"))
    np.testing.assert_array_equal(answer[1], np.array(["2011-01-05", "NaT"], dtype="datetime64[s]"))


def test_the_backends_answer_the_same_thing():
    """The three backends answer the same rows, for every k and both
    directions - the random data holds nans, empty groups and groups of
    every size."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(1, 60))
        group_idx = rng.integers(0, 7, n)
        a = rng.random(n)
        a[rng.random(n) < 0.15] = np.nan
        for k in (0, 1, 2, 3, 7, 50):
            for largest in (True, False):
                answers = [
                    _as_matrix(backend.aggregate(group_idx, a, func="topk", k=k, largest=largest))
                    for backend in BACKENDS
                ]
                for other in answers[1:]:
                    np.testing.assert_array_equal(answers[0], other)


def test_the_topk_of_the_package_is_the_one_of_the_best_implementation():
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
        npgb.tick_topk(group_idx, ticks, 2),
        np.array([[5, 0], [9, 3]], dtype=np.int64),
    )


def test_the_topk_of_a_temporal_input_can_be_asked_for_in_ticks():
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


@pytest.mark.parametrize("backend", [npgn, npgb], ids=["numpy", "numba"])
def test_the_topk_needs_a_flat_group_index(backend):
    """A multidimensional group index has no place in a matrix of groups."""
    group_idx = np.array([0, 0, 1, 1, 1, 2])
    a = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match="the topk needs a flat group index"):
        backend.aggregate(group_idx, a, func="topk", k=1, axis=0)


@pytest.mark.parametrize("backend", BACKENDS, ids=["numpy", "numba", "purepy"])
def test_the_topk_of_complex_values_uses_the_order_of_numpy(backend):
    """Numpy orders complex values lexicographically - the real part first,
    the imaginary one second - and the topk keeps that order."""
    group_idx = np.array([0, 0, 1])
    a = np.array([3 + 1j, 1 + 5j, 2 + 0j])
    answer = _as_matrix(backend.aggregate(group_idx, a, func="topk", k=2))
    np.testing.assert_array_equal(answer[0], np.array([3 + 1j, 1 + 5j]))
    np.testing.assert_array_equal(answer[1], np.array([2 + 0j, np.nan + np.nan * 1j]))


def test_the_topk_of_the_numba_backend_of_float16_runs_on_float32():
    """Numba has no float16, so the values run on float32 and the answer is
    rounded back - like the other operations of the backend."""
    group_idx = np.array([0, 0, 1])
    a = np.array([5.0, 9.0, 4.0], dtype=np.float16)
    answer = npgb.aggregate(group_idx, a, func="topk", k=1)
    assert answer.dtype == np.float16
    np.testing.assert_array_equal(answer, np.array([[9.0], [4.0]], dtype=np.float16))
