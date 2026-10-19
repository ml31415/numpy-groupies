"""Per-function value semantics, run against every backend.

The reductions that need more than one pass over a group (a mean, a variance,
a median, a trapezoid), the cumulative ones and the sort - each against the
reference result of the numpy function of the same name, or against the
group loop of :mod:`.reference` where a numpy function is not the definition.
"""

import itertools

import numpy as np
import pytest

from numpy_groupies.tests.helpers import (
    assert_same_result,
    func_list,
    random_values,
)
from numpy_groupies.tests.reference import aggregate_ref


def test_len(impl, group_size=5):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(group_size)
    a = np.arange(group_idx.size)
    res = impl.aggregate(group_idx, a, func="len")
    ref = impl.aggregate(group_idx, 1, func="sum")
    if isinstance(res, np.ndarray):
        assert issubclass(res.dtype.type, np.integer)
    else:
        assert isinstance(res[0], int)
    np.testing.assert_array_equal(res, ref)
    group_idx = np.arange(0, 100, dtype=int).repeat(group_size)
    a = np.arange(group_idx.size)
    res = impl.aggregate(group_idx, a, func=len)
    if isinstance(res, np.ndarray):
        assert np.all(res == group_size)
    else:
        assert all(x == group_size for x in res)


@pytest.mark.parametrize("first_last", ["first", "last"])
def test_first_last(impl, first_last):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(5)
    a = np.arange(group_idx.size)
    res = impl.aggregate(group_idx, a, func=first_last, fill_value=-1)
    ref = np.zeros(np.max(group_idx) + 1)
    ref.fill(-1)
    ref[::2] = np.arange(0 if first_last == "first" else 4, group_idx.size, 5, dtype=int)
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64], ids=["float16", "float32", "float64"])
@pytest.mark.parametrize(["func", "ddof"], list(itertools.product(["var", "std"], [0, 1, 2])))
def test_ddof(impl, rng, func, ddof, dtype, size=20):
    group_idx = np.zeros(20, dtype=int)
    a = random_values(rng, group_idx.size).astype(dtype)
    res = impl.aggregate(group_idx, a, func, ddof=ddof)
    ref_func = {"std": np.std, "var": np.var}.get(func)
    ref = ref_func(a.astype(np.float64), ddof=ddof)
    # the tolerance follows the input precision - a float16 keeps roughly
    # three decimal digits, and the pure python implementation accumulates its
    # squares in float16 steps, so the reduced statistic may sit a hundredth
    # off the exact value; float32 keeps seven
    tol = {np.float16: 1e-2, np.float32: 1e-6, np.float64: 1e-10}[dtype]
    assert abs(res[0] - ref) < tol


def test_mean(impl):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.arange(len(group_idx))

    res = impl.aggregate(group_idx, a, func="mean")
    np.testing.assert_allclose(res, [1.5, np.nan, np.nan, 5.5], equal_nan=True)


def test_argmin_argmax_nonans(impl):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.array([4, 4, 3, 1, 10, 9, 9, 11])

    res = impl.aggregate(group_idx, a, func="argmax", fill_value=-1)
    np.testing.assert_array_equal(res, [0, -1, -1, 7])

    res = impl.aggregate(group_idx, a, func="argmin", fill_value=-1)
    np.testing.assert_array_equal(res, [3, -1, -1, 5])


def test_min_max_inf(impl):
    # https://github.com/ml31415/numpy-groupies/issues/40
    res = impl.aggregate(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([-np.inf, 0, -np.inf, -np.inf, 0, 0]),
        func="max",
    )
    np.testing.assert_array_equal(res, [-np.inf, 0, 0])

    res = impl.aggregate(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([np.inf, 0, np.inf, np.inf, 0, 0]),
        func="min",
    )
    np.testing.assert_array_equal(res, [np.inf, 0, 0])


def test_argmin_argmax_inf(impl):
    # https://github.com/ml31415/numpy-groupies/issues/40
    res = impl.aggregate(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([-np.inf, 0, -np.inf, -np.inf, 0, 0]),
        func="argmax",
        fill_value=-1,
    )
    np.testing.assert_array_equal(res, [0, 1, 5])

    res = impl.aggregate(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([np.inf, 0, np.inf, np.inf, 0, 0]),
        func="argmin",
        fill_value=-1,
    )
    np.testing.assert_array_equal(res, [0, 1, 5])


def test_cumsum(impl):
    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    ref = np.array([3, 4, 5, 6, 15, 9, 15, 22, 7, 0, 15, 17, 6, 14, 31, 39])

    res = impl.aggregate(group_idx, a, func="cumsum")
    np.testing.assert_array_equal(res, ref)


def test_cumsum_accuracy(impl):
    array = np.array([0.00000000e00, 0.00000000e00, 0.00000000e00, 3.27680000e04, 9.99999975e-06])
    group_idx = np.array([0, 0, 0, 0, 1])

    actual = impl.aggregate(group_idx, array, axis=-1, func="cumsum")
    expected = array
    np.testing.assert_allclose(actual, expected)


def test_cummax(impl):
    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    ref = np.array([3, 4, 4, 3, 9, 9, 9, 9, 7, 0, 8, 9, 4, 8, 9, 9])

    res = impl.aggregate(group_idx, a, func="cummax")
    np.testing.assert_array_equal(res, ref)


def test_trapezoid(impl):
    # https://github.com/ml31415/numpy-groupies/issues/54 - the integral
    # follows the order the samples appear in, one sample integrates to zero
    group_idx = np.array([0, 0, 0, 1, 1, 2, 4, 4, 4])
    a = np.array([1.0, 5.0, 2.0, 2.0, 8.0, 7.0, 9.0, 3.0, 4.0])
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, func="trapezoid", fill_value=-1.0),
        [6.5, 5.0, 0.0, -1.0, 9.5],
    )
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, func="trapezoid", dx=2.0, fill_value=-1.0),
        [13.0, 10.0, 0.0, -1.0, 19.0],
    )


@pytest.mark.parametrize("order", ["normal", "reverse"])
def test_list_ordering(impl, order):
    group_idx = np.repeat(np.arange(5), 4)
    a = np.arange(group_idx.size)
    if order == "reverse":
        a = a[::-1]
    ref = a[:4]

    res = impl.aggregate(group_idx, a, func=list)
    np.testing.assert_array_equal(np.array(res[0]), ref)


@pytest.mark.parametrize("order", ["normal", "reverse"])
def test_sort(impl, order):
    group_idx = np.array([3, 3, 3, 2, 2, 2, 1, 1, 1])
    a = np.array([3, 2, 1, 3, 4, 5, 5, 10, 1])
    ref_normal = np.array([1, 2, 3, 3, 4, 5, 1, 5, 10])
    ref_reverse = np.array([3, 2, 1, 5, 4, 3, 10, 5, 1])
    reverse = order == "reverse"
    ref = ref_reverse if reverse else ref_normal

    res = impl.aggregate(group_idx, a, func="sort", reverse=reverse)
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize(
    ("a", "ref"),
    [
        (np.array([-128, 5, 0, -3], dtype=np.int8), [5, 0, -3, -128]),
        (np.array([3, 1, 0, 200], dtype=np.uint8), [200, 3, 1, 0]),
        (np.array([1, 0, 2**63, 2], dtype=np.uint64), [2**63, 2, 1, 0]),
        (np.array([True, False, True]), [True, True, False]),
    ],
    ids=["int8-min", "uint8-zero", "uint64-zero", "bool"],
)
def test_sort_reverse_descends(impl, a, ref):
    # reversing must not come from negating the values: the smallest value of a
    # signed dtype survives negation as itself, a zero of an unsigned one stays
    # the smallest key, and booleans cannot be negated at all
    group_idx = np.zeros(a.size, dtype=int)
    res = impl.aggregate(group_idx, a, func="sort", reverse=True)
    np.testing.assert_array_equal(np.asarray(res), ref)


@pytest.mark.parametrize("size", [1, 2, 3, 8, 16, 31, 32, 33, 40])
@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float32, np.float16, np.int64, np.uint8, np.bool_],
    ids=["float64", "float32", "float16", "int64", "uint8", "bool"],
)
def test_sort_of_a_small_group_matches_the_reference(impl, size, dtype):
    # a block of a few dozen values is ordered by insertion sort in the flat
    # array instead of a library sort of a view - the values, the nan at their
    # end and the descending order of them all have to come out the same, and
    # the sizes here straddle the limit that switches the two passes.  The
    # reference is the group loop, so even the numpy backend is compared
    # against something it does not share a sort with
    rnd = np.random.RandomState(size)
    if dtype is np.bool_:
        a = rnd.randint(0, 2, size).astype(dtype)
    elif dtype is np.int64:
        a = rnd.randint(0, 100, size).astype(dtype)
    elif dtype is np.uint8:
        a = rnd.randint(0, 20, size).astype(dtype)
    else:
        a = rnd.random_sample(size).astype(dtype)
        a[rnd.random_sample(size) < 0.25] = np.nan
    group_idx = np.zeros(size, dtype=int)

    for reverse in (False, True):
        res = impl.aggregate(group_idx, a, func="sort", reverse=reverse)
        ref = aggregate_ref(group_idx, a, func="sort", reverse=reverse)
        np.testing.assert_array_equal(res, ref)


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.parametrize("func", func_list)
def test_matches_the_group_loop(impl, func, block):
    """every function over ``block.a``, which holds no nan, against the group loop
    of :mod:`.reference`.

    The backend and the loop share no code: no bincount, no ufunc trick, no
    compiled kernel, one computation per group.
    """
    got = impl.aggregate(block.group_idx, block.a, func=func, size=block.size)
    want = aggregate_ref(block.group_idx, block.a, func=func, size=block.size)
    assert_same_result(got, want, func)
