"""Aggregating along an axis, and multi-dimensional labels.

``group_idx`` may label the entries of one axis of ``a`` instead of its first
dimension, and it may hold several rows of labels, which then address a group
by coordinate.  Both are checked against the numpy reference over every axis,
memory order and number of dimensions.
"""

import itertools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from numpy_groupies.registry import (
    funcs_collecting,
)
from numpy_groupies.tests.helpers import (
    assert_same_result,
    func_list,
    random_group_idx,
    random_values,
)
from numpy_groupies.tests.reference import _kurt_ref, _skew_ref, _skip_nans, aggregate_ref, trapezoid_ref


def _quiet_reference(func: Callable[..., Any], a: npt.NDArray[Any], *args: Any, **kwargs: Any) -> Any:
    """The numpy reference, without its warning about the all-nan slices.

    Numpy warns when it averages or quantiles a slice that holds nothing but
    nans - the reference is asked for exactly that here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(a, *args, **kwargs)


@pytest.mark.parametrize(
    "func",
    [np.argmin, "argmin", np.argmax, "argmax", np.nanargmin, "nanargmin", np.nanargmax, "nanargmax"],
    ids=[
        "argmin-callable",
        "argmin-name",
        "argmax-callable",
        "argmax-name",
        "nanargmin-callable",
        "nanargmin-name",
        "nanargmax-callable",
        "nanargmax-name",
    ],
)
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("order", ["C", "F"])
def test_arg_alias_axis_coordinates(impl, rng, func, axis, order):
    """NumPy aliases report positions along the requested axis, including gaps.

    Both spellings of every alias are asked, since the bug behind this test was
    one code path resolving the name of a ``func`` and the path next to it not
    doing it: the position has to come out along the axis that was asked for,
    whether the caller wrote ``"argmin"`` or ``np.argmin``.
    """
    a = np.array(rng.normal(size=(3, 4, 5)), order=order)
    name = func if isinstance(func, str) else func.__name__
    if name in ("nanargmin", "nanargmax"):
        a[0, 0, 0] = np.nan
    normalized_axis = axis % a.ndim
    labels = 1 - np.arange(a.shape[axis]) % 2
    output_shape = list(a.shape)
    output_shape[axis] = 3
    expected = np.full(output_shape, -1)
    for index in np.ndindex(*output_shape):
        positions = np.flatnonzero(labels == index[normalized_axis])
        if positions.size:
            values = [a[index[:normalized_axis] + (int(p),) + index[normalized_axis + 1 :]] for p in positions]
            expected[index] = positions[getattr(np, name)(values)]

    actual = impl.aggregate(labels, a, func=func, axis=axis, order=order, size=3)
    np.testing.assert_array_equal(actual, expected)


def _other_shape(size, axis):
    """The shape of the positions beside the aggregated axis."""
    return tuple(s for i, s in enumerate(size) if i != axis)


def _collecting_shape(size, axis):
    """The shape of the result of a collecting function along an axis.

    The axis dimension holds the groups, the other ones the positions beside
    it - one group per label, so the dimension is of the groups alone.
    """
    shape = list(size)
    shape[axis] = 1
    return tuple(shape)


def _distinct_of(values, count_nans):
    """the number of the distinct values of a slice of an axis.

    The nans are one value among themselves, and the nan variant of the
    function leaves them out.
    """
    nans = np.isnan(values)
    distinct = len(np.unique(values[~nans]))
    return distinct + (1 if count_nans and nans.any() else 0)


# The axis of the aggregation is what the case is about, not the function: the
# labels of a one-dimensional array and the labels of the last axis of a
# two-dimensional one are asked for every function, the labels of the first
# axis of a two-dimensional one for a representative set of them.
_AXIS_FUNCS = (
    "sum",
    "prod",
    "mean",
    "median",
    "quantile",
    "var",
    "std",
    "min",
    "max",
    "len",
    "nanlen",
    "nunique",
    "cumsum",
    "sort",
    "nanmean",
    "array",
    # the functions whose expectation the test writes itself - the first and
    # the last value of a group, the truth of the values that are left and the
    # running extreme of a slice
    "first",
    "last",
    "nanfirst",
    "nanlast",
    "nanall",
    "nanany",
    "cummax",
    "cummin",
)

_AXIS_CASES = (
    [pytest.param((12,), 0, func, id=f"{func}-(12,)-ax0") for func in func_list]
    + [pytest.param((12, 5), 1, func, id=f"{func}-(12,5)-ax1") for func in func_list]
    + [pytest.param((12, 5), 0, func, id=f"{func}-(12,5)-ax0") for func in _AXIS_FUNCS]
)


@pytest.mark.parametrize(("size", "axis", "func"), _AXIS_CASES)
def test_along_axis(impl, rng, func, size, axis):
    group_idx = np.zeros(size[axis], dtype=int)
    a = random_values(rng, size)

    # add some NaNs to test out nan-skipping
    if "nan" in func and "nanarg" not in func:
        a[[1, 4, 5], ...] = np.nan
    elif "nanarg" in func and a.ndim > 1:
        a[[1, 4, 5], 1] = np.nan
    nans = np.isnan(a)
    if func in ["any", "all", "nanany", "nanall"]:
        a = a > 0.5

    # construct expected values for all cases
    if func == "len":
        expected = np.array(size[axis])
    elif func == "nanlen":
        expected = np.array((~np.isnan(a)).sum(axis=axis))
    elif func == "anynan":
        expected = np.isnan(a).any(axis=axis)
    elif func == "allnan":
        expected = np.isnan(a).all(axis=axis)
    elif func in ("nanall", "nanany"):
        # the truth of the values that are left - the nans are no value of
        # their own and are left out
        truth = np.where(nans, False, a)
        expected = np.all(truth, axis=axis) if func == "nanall" else np.any(truth, axis=axis)
    elif func in ("first", "last", "nanfirst", "nanlast"):
        # the first or the last value of the slice, the nan variants leaving
        # the nans out
        if func == "first":
            expected = np.take(a, 0, axis=axis)
        elif func == "last":
            expected = np.take(a, -1, axis=axis)
        else:
            take = 0 if func == "nanfirst" else -1

            def first_of(v):
                values = v[~np.isnan(v)]
                # a slice of nothing but nans is an empty group: it takes the
                # fill value of the function
                return values[take] if values.size else np.nan

            expected = np.apply_along_axis(first_of, axis, a)
    elif func in ("cummax", "cummin"):
        # the running extreme of the slice, as numpy accumulates it
        operation = np.maximum if func == "cummax" else np.minimum
        expected = operation.accumulate(a, axis=axis)
    elif func == "sumofsquares":
        expected = np.sum(a * a, axis=axis)
    elif func == "nansumofsquares":
        expected = np.nansum(a * a, axis=axis)
    elif func == "trapezoid":
        expected = trapezoid_ref(a, axis=axis)
    elif func == "nantrapezoid":
        # numpy has no nan-aware trapezoid - nantrapezoid integrates the
        # samples that are left, bridging over the gaps the nans leave
        expected = np.apply_along_axis(lambda v: trapezoid_ref(v[~np.isnan(v)]), axis, a)
    elif func == "quantile":
        expected = _quiet_reference(np.quantile, a, 0.5, axis=axis)
    elif func == "nanquantile":
        expected = _quiet_reference(np.nanquantile, a, 0.5, axis=axis)
    elif func == "percentile":
        expected = _quiet_reference(np.percentile, a, 50.0, axis=axis)
    elif func == "nanpercentile":
        expected = _quiet_reference(np.nanpercentile, a, 50.0, axis=axis)
    elif func in ("skew", "kurt", "nanskew", "nankurt"):
        # numpy has no skewness of its own - the reference of the package
        # computes it, and the nan variants leave the nans out
        ref = _skew_ref if "skew" in func else _kurt_ref
        if func.startswith("nan"):
            ref = _skip_nans(ref)
        expected = np.apply_along_axis(lambda v: ref(v, np.zeros(v.size, dtype=np.intp)), axis, a)
    elif func in ("nunique", "nannunique"):
        # numpy has no count of the distinct values along an axis, so the
        # slices are counted one by one - the nans are one value among
        # themselves, and the nan variant leaves them out
        expected = np.apply_along_axis(lambda v: _distinct_of(v, func == "nunique"), axis, a)
    elif func in funcs_collecting:
        # the values of every group, along the axis: the labels are all zero,
        # so the one group holds the values of one position along the axis
        expected = np.empty(_collecting_shape(size, axis), dtype=object)
        for index in np.ndindex(*_other_shape(size, axis)):
            where = index[:axis] + (slice(None),) + index[axis:]
            expected[index[:axis] + (0,) + index[axis:]] = a[where]
    else:
        expected = _quiet_reference(getattr(np, func), a, axis=axis)

    # no fill_value needed - the function specific defaults match numpy
    actual = impl.aggregate(group_idx, a, axis=axis, func=func)
    assert actual.ndim == a.ndim

    # argmin, argmax don't support keepdims, so we can't use that to construct expected
    # instead we squeeze out the extra dims in actual.
    assert_same_result(actual.squeeze(), expected.squeeze(), func)


def test_not_last_axis_reduction(impl):
    group_idx = np.array([1, 2, 2, 0, 1])
    a = np.array([[1.0, 2.0], [4.0, 4.0], [5.0, 2.0], [np.nan, 3.0], [8.0, 7.0]])
    func = "nanmax"
    fill_value = np.nan
    axis = 0
    actual = impl.aggregate(group_idx, a, axis=axis, func=func, fill_value=fill_value)
    expected = np.array([[np.nan, 3.0], [8.0, 7.0], [5.0, 4.0]])
    np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize("axis", (0, 1))
def test_along_axis_multidim(impl, axis):
    # https://github.com/ml31415/numpy-groupies/issues/74
    a = np.arange(15, dtype=float).reshape(3, 5)
    labels = np.array([1, 1, 0, 2, 2]) if axis == 1 else np.array([2, 0, 2])
    ref = impl.aggregate(labels, a, axis=axis, func="sum")

    # a single row/column of labels broadcasts over the remaining axes,
    # so it must give identical results to the equivalent 1d group_idx
    group_idx = labels.reshape(1, -1) if axis == 1 else labels.reshape(-1, 1)
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, axis=axis, func="sum"), ref)

    # labels may also vary along the non-aggregated axes
    if axis == 1:
        group_idx = np.array([[0, 1, 1, 2, 2], [2, 2, 1, 1, 0], [1, 1, 0, 0, 2]])
        expected = np.stack([np.bincount(row, weights=vals, minlength=3) for row, vals in zip(group_idx, a)])
    else:
        group_idx = np.array([[0, 1, 2, 2, 1], [1, 1, 0, 2, 1], [2, 0, 2, 0, 0]])
        expected = np.zeros((3, 5))
        for jj in range(a.shape[1]):
            for lbl in range(3):
                mask = group_idx[:, jj] == lbl
                if mask.any():
                    expected[lbl, jj] = a[mask, jj].sum()
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, axis=axis, func="sum", size=3), expected)


@pytest.mark.parametrize(
    "func", ["cumsum", np.cumsum, "nancumsum", np.nancumsum, "sort", np.sort, "cumprod", "cummin", "cummax"]
)
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("order", ["C", "F"])
def test_one_to_one_axis_shape(impl, func, axis, order):
    rows = np.array([[3.0, 1.0, 5.0, 2.0], [20.0, 40.0, 10.0, 30.0]])
    data = rows if axis == -1 else rows.T
    labels = np.array([0, 1, 0, 1])
    name = func if isinstance(func, str) else func.__name__
    expected = np.empty_like(rows)
    for row in range(2):
        for group in range(2):
            positions = np.flatnonzero(labels == group)
            values = rows[row, positions]
            if name in ("cummin", "cummax"):
                operation = np.minimum if name == "cummin" else np.maximum
                values = operation.accumulate(values)
            else:
                values = getattr(np, name)(values)
            expected[row, positions] = values
    if axis == 0:
        expected = expected.T

    actual = impl.aggregate(labels, data, func=func, axis=axis, order=order)

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == data.shape


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("func", ["sum", "mean", "argmax"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_axis_output_order_preserves_groups(impl, shape, broadcast, func, order):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    labels = np.array([0, 1, 0, 1])
    if broadcast:
        labels = np.broadcast_to(labels, shape).copy()
        labels = (labels + np.arange(np.prod(shape[:-1])).reshape(shape[:-1] + (1,))) % 2
    expected = np.empty(shape[:-1] + (2,))
    for loc in np.ndindex(shape[:-1]):
        row_labels = labels[loc] if broadcast else labels
        for group in range(2):
            positions = np.flatnonzero(row_labels == group)
            values = data[loc][positions]
            if func == "argmax":
                value = positions[np.argmax(values)]
            else:
                value = getattr(np, func)(values)
            expected[loc + (group,)] = value

    actual = impl.aggregate(labels, data, func=func, axis=-1, order=order)

    np.testing.assert_allclose(actual, expected)
    if func != "argmax":
        assert actual.flags.f_contiguous if order == "F" else actual.flags.c_contiguous


@pytest.mark.parametrize(["ndim", "order"], list(itertools.product([1, 2, 3], ["C", "F"])))
def test_ndim_indexing(impl, rng, ndim, order, outsize=10):
    nindices = int(outsize**ndim)
    outshape = tuple([outsize] * ndim)
    group_idx = random_group_idx(rng, (ndim, nindices), outsize)
    a = random_values(rng, group_idx.shape[1])
    res = impl.aggregate(group_idx, a, size=outshape, order=order)
    if ndim > 1 and order == "F":
        # 1d arrays always return False here
        assert np.isfortran(res)
    else:
        assert not np.isfortran(res)
    assert res.shape == outshape


@pytest.mark.parametrize("size", [None, (10, 2)])
def test_ndim_group_idx(impl, size):
    group_idx = np.vstack((np.repeat(np.arange(10), 10), np.repeat([0, 1], 50)))
    impl.aggregate(group_idx, 1, size=size)


def test_holes_in_multidim_size(impl):
    # multidimensional labels with the last column cut off, leaving group
    # (2, 3) without any item (issue #32)
    group_idx = np.array(list(itertools.product(range(3), range(4)))).T
    group_idx, a = group_idx[:, :-1], np.arange(11, dtype=float)

    for func, expected in (("sum", 0.0), ("prod", 1.0), ("mean", np.nan), ("argmax", -1)):
        res = impl.aggregate(group_idx, a, func=func, size=(3, 4))
        assert res[2, 3] == expected or expected != expected

    res = impl.aggregate(group_idx, a, func="array", size=(3, 4))
    assert len(res[2, 3]) == 0


@pytest.mark.parametrize("func", ["sort", "cumsum", "cumprod"])
def test_one_out_per_in_cannot_fill(impl, func):
    # sort and the cum-functions emit one value per item, so they have
    # nothing to fill a hole with - but they need to say so clearly
    group_idx = np.array(list(itertools.product(range(3), range(4)))).T[:, :-1]
    with pytest.raises(ValueError, match="cannot fill"):
        impl.aggregate(group_idx, np.arange(11, dtype=float), func=func, size=(3, 4))


def test_argreduction_nD_array_1D_idx(impl):
    # https://github.com/ml31415/numpy-groupies/issues/41
    group_idx = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=int)
    a = np.array([[1] * 12, [1] * 12])
    actual = impl.aggregate(group_idx, a, axis=-1, func="argmax")
    expected = np.array([[0, 5, 2], [0, 5, 2]])
    np.testing.assert_equal(actual, expected)


def test_argreduction_negative_fill_value(impl):
    group_idx = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=int)
    a = np.array([[1] * 12, [np.nan] * 12])
    actual = impl.aggregate(group_idx, a, axis=-1, fill_value=-1, func="argmax")
    expected = np.array([[0, 5, 2], [-1, -1, -1]])
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(["ndim", "order"], list(itertools.product([2, 3], ["C", "F"])))
def test_matches_the_group_loop(impl, block, ndim, order):
    """labels of several rows address a group by coordinate, and the group loop
    reaches the same values by flattening the coordinates.

    Which layout the result comes back in is the business of
    ``test_ndim_indexing``; only the values are asked for here.
    """
    rows = [block.group_idx] + [block.group_idx % n for n in (2, 3)][: ndim - 1]
    size = (block.size, 2, 3)[:ndim]
    group_idx = np.vstack(rows)

    got = impl.aggregate(group_idx, block.a, func="sum", size=size, order=order, fill_value=0)
    want = aggregate_ref(group_idx, block.a, func="sum", size=size, fill_value=0)
    assert got.shape == size
    np.testing.assert_allclose(got, want, rtol=10**-10)
