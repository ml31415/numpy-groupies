"""What a nan does to the group it sits in.

The plain functions propagate it, their nan counterparts skip the value, and
neither may let it reach into a neighbouring group.  A group of nothing but
nans is an empty group for the nan counterparts, so it takes the fill value.
"""

import itertools
import warnings

import numpy as np
import pytest

from numpy_groupies import aggregate_np, aggregate_purepy
from numpy_groupies.tests.helpers import (
    func_list,
    random_values,
)
from numpy_groupies.tests.reference import aggregate_ref
from numpy_groupies.tests.test_collect import assert_same_result


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64], ids=["float16", "float32", "float64"])
@pytest.mark.parametrize("func", ["sum", "prod", "mean", "var", "std", "min", "max", "all", "any"])
def test_nan_input(impl, rng, func, dtype, groups=100):
    group_idx = np.arange(0, groups, dtype=int).repeat(5)
    a = random_values(rng, group_idx.size).astype(dtype)
    a[::2] = np.nan

    if func in ("all", "any"):
        ref = np.ones(groups, dtype=bool)
    else:
        ref = np.full(groups, np.nan, dtype=dtype)
    res = impl.aggregate(group_idx, a, func=func)
    np.testing.assert_array_equal(res, ref)


def test_nan_input_len(impl, rng, groups=100, group_size=5):
    group_idx = np.arange(0, groups, dtype=int).repeat(group_size)
    a = random_values(rng, len(group_idx))
    a[::2] = np.nan
    ref = np.full(groups, group_size, dtype=int)
    res = impl.aggregate(group_idx, a, func=len)
    np.testing.assert_array_equal(res, ref)


def test_nan_form_of_a_group_of_only_nan():
    # filtering the nans out can leave nothing to unpack, and an all-nan group
    # then takes the fill value (called directly - purepy's nan forms are skipped)
    funcs = (
        "nansum",
        "nanprod",
        "nanmean",
        "nanmedian",
        "nanmin",
        "nanmax",
        "nanlen",
        "nanfirst",
        "nanlast",
        "nanvar",
        "nanstd",
        "nanall",
        "nanany",
        "nanargmin",
        "nanargmax",
    )
    cases = (
        # every value a nan: nothing survives the filtering
        (np.array([0, 0]), np.array([np.nan, np.nan])),
        # one group of nothing but nans, one with a value left
        (np.array([0, 0, 1]), np.array([np.nan, np.nan, 1.0])),
        # the same with complex values, where a nan in either part counts
        (np.array([0, 0]), np.array([complex(np.nan, 1), complex(1, np.nan)])),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an all-nan group is worth a numpy warning
        for group_idx, a in cases:
            for func in funcs:
                res = np.asarray(aggregate_purepy.aggregate(group_idx, a, func=func, size=2))
                expected = np.asarray(aggregate_np(group_idx, a, func=func, size=2))
                # only the group of nothing but nans - in a group with values the
                # pure python implementation reports the index inside the group
                np.testing.assert_array_equal(res[:1], expected[:1])


def test_nan_len(impl, rng):
    group_idx = np.arange(0, 20, 2, dtype=int).repeat(5)
    a = random_values(rng, group_idx.size)
    a[::4] = np.nan
    a[::5] = np.nan
    res = impl.aggregate(group_idx, a, func="nanlen")
    ref = impl.aggregate(group_idx[~np.isnan(a)], 1, func="sum")
    if isinstance(res, np.ndarray):
        assert issubclass(res.dtype.type, np.integer)
    else:
        assert isinstance(res[0], int)
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize(["first_last", "nanoffset"], list(itertools.product(["nanfirst", "nanlast"], [0, 2, 4])))
def test_nan_first_last(impl, first_last, nanoffset):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(5)
    a = np.arange(group_idx.size, dtype=float)

    a[nanoffset::5] = np.nan
    res = impl.aggregate(group_idx, a, func=first_last, fill_value=-1)
    ref = np.zeros(np.max(group_idx) + 1)
    ref.fill(-1)

    if first_last == "nanfirst":
        ref_offset = 1 if nanoffset == 0 else 0
    else:
        ref_offset = 3 if nanoffset == 4 else 4
    ref[::2] = np.arange(ref_offset, group_idx.size, 5, dtype=int)
    np.testing.assert_array_equal(res, ref)


def test_min_max_nans(impl):
    # a nan of a group wins over the other values, at every position, as in
    # np.min and np.max - the builtin reductions did it only for the first one
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([4, 3, np.nan, 1, 5, 9, 2, 7])

    res = impl.aggregate(group_idx, a, func="min", fill_value=np.nan)
    np.testing.assert_array_equal(res, [np.nan, 2])

    res = impl.aggregate(group_idx, a, func="max", fill_value=np.nan)
    np.testing.assert_array_equal(res, [np.nan, 9])


def test_argmin_argmax_nans(impl):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.array([4, 4, 3, 1, np.nan, 1, 2, 3])

    res = impl.aggregate(group_idx, a, func="argmax", fill_value=-1)
    np.testing.assert_array_equal(res, [0, -1, -1, -1])

    res = impl.aggregate(group_idx, a, func="argmin", fill_value=-1)
    np.testing.assert_array_equal(res, [3, -1, -1, -1])


def test_nanargmin_nanargmax_nans(impl):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.array([4, 4, np.nan, 1, np.nan, np.nan, np.nan, np.nan])

    res = impl.aggregate(group_idx, a, func="nanargmax", fill_value=-1)
    np.testing.assert_array_equal(res, [0, -1, -1, -1])

    res = impl.aggregate(group_idx, a, func="nanargmin", fill_value=-1)
    np.testing.assert_array_equal(res, [3, -1, -1, -1])


def test_nanargmin_nanargmax_nonans(impl):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.array([4, 4, 3, 1, 10, 9, 9, 11])

    res = impl.aggregate(group_idx, a, func="nanargmax", fill_value=-1)
    np.testing.assert_array_equal(res, [0, -1, -1, 7])

    res = impl.aggregate(group_idx, a, func="nanargmin", fill_value=-1)
    np.testing.assert_array_equal(res, [3, -1, -1, 5])


def test_cumsum_nan(impl):
    # https://github.com/ml31415/numpy-groupies/issues/91 - nans propagate
    # within their own group only, other groups are unaffected
    group_idx = np.array([1, 1, 1, 0, 0])
    a = np.array([[5.0, 6.0, 7.0, 8.0, 9.0], [0.0, 0.0, 0.0, np.nan, 0.0]])
    res = impl.aggregate(group_idx, a, func="cumsum", axis=-1)
    ref = np.array([[5.0, 11.0, 18.0, 8.0, 17.0], [0.0, 0.0, 0.0, np.nan, np.nan]])
    np.testing.assert_array_equal(res, ref)

    res = impl.aggregate(np.array([0, 0, 0, 1]), np.array([np.nan, 1.0, 2.0, 5.0]), func="cumsum")
    np.testing.assert_array_equal(res, [np.nan, np.nan, np.nan, 5.0])


def test_nancumsum(impl):
    # https://github.com/ml31415/numpy-groupies/issues/79
    group_idx = [0, 0, 0, 1, 1, 0, 0]
    a = [2, 2, np.nan, 2, 2, 2, 2]
    ref = [2.0, 4.0, 4.0, 2.0, 4.0, 6.0, 8.0]

    res = impl.aggregate(group_idx, a, func="nancumsum")
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("func", ["cumsum", "nancumsum"])
@pytest.mark.parametrize("large", [2.0**60, np.inf, -np.inf, np.nan])
def test_cumsum_groups_are_independent(impl, dtype, func, large):
    group_idx = np.array([0, 1, 0, 1, 2, 2])
    a = np.array([large, 1, large, 2, 3, 4], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a.imag = [large, 2, large, 3, 4, 5]
    original = a.copy()
    expected = np.empty_like(a)
    for group in range(3):
        positions = np.flatnonzero(group_idx == group)
        expected[positions] = getattr(np, func)(a[positions])
    result = impl.aggregate(group_idx, a, func=func, dtype=np.dtype(dtype))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(a, original)


def test_median_nan(impl):
    # plain median propagates nans within their group (like np.median),
    # nanmedian skips them; groups without any member keep the fill value
    group_idx = np.array([0, 0, 0, 1, 1, 2, 4, 4, 4])
    a = np.array([1.0, 5.0, np.nan, 2.0, 8.0, 7.0, 9.0, np.nan, 3.0])
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, "median", fill_value=-1.0),
        [np.nan, 5.0, 7.0, -1.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, "nanmedian", fill_value=-1.0),
        [3.0, 5.0, 7.0, -1.0, 6.0],
    )


def test_trapezoid_nan(impl):
    # plain trapezoid is poisoned by nans within their group, while
    # nantrapezoid skips them and bridges the gap they leave
    group_idx = np.array([0, 0, 0, 1, 1, 2, 4, 4, 4])
    a = np.array([1.0, 5.0, np.nan, 2.0, 8.0, 7.0, 9.0, np.nan, 3.0])
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, func="trapezoid", fill_value=-1.0),
        [np.nan, 5.0, 0.0, -1.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, func="nantrapezoid", fill_value=-1.0),
        [3.0, 5.0, 0.0, -1.0, 6.0],
    )


@pytest.mark.parametrize("nan_inds", (None, ([1, 4, 5], Ellipsis), (1, (0, 1, 2, 3))))
@pytest.mark.parametrize("ddof", (0, 1))
@pytest.mark.parametrize("func", ("nanvar", "nanstd"))
def test_var_with_nan_fill_value(impl, ddof, nan_inds, func):
    a = np.ones((12, 5))
    group_idx = np.zeros(a.shape[-1:], dtype=int)

    if nan_inds is not None:
        a[nan_inds] = np.nan

    with warnings.catch_warnings():
        # Filter RuntimeWarning: Degrees of freedom <= 0 for slice.
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = getattr(np, func)(a, keepdims=True, axis=-1, ddof=ddof)

    actual = impl.aggregate(group_idx, a, axis=-1, fill_value=np.nan, func=func, ddof=ddof)
    np.testing.assert_equal(actual, expected)


def test_sort_leaves_nan_at_the_end(impl):
    # numpy orders a nan as the largest value, so it ends up last - and since
    # the descending pass only flips the ordered part of a group, it stays last
    # there too.  The builtin sorted the pure python implementation orders with
    # compares a nan neither way, which leaves it wherever it happened to sit.
    group_idx = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([np.nan, 2.0, 1.0, np.nan, np.nan, 0.5])

    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="sort"), [1.0, 2.0, np.nan, np.nan, 0.5, np.nan])
    np.testing.assert_array_equal(
        impl.aggregate(group_idx, a, func="sort", reverse=True), [2.0, 1.0, np.nan, np.nan, 0.5, np.nan]
    )


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.parametrize("func", func_list)
def test_matches_the_group_loop(impl, func, block):
    """``block.a_nans``, the same values with nans in them, against the group loop.

    The plain functions let the nan reach the result, their nan counterparts
    skip the value, and a group of nothing but nans is an empty group for them -
    the rule the loop spells out one group at a time.
    """
    got = impl.aggregate(block.group_idx, block.a_nans, func=func, size=block.size)
    want = aggregate_ref(block.group_idx, block.a_nans, func=func, size=block.size)
    assert_same_result(got, want, func)
