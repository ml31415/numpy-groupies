"""Aggregating along an axis, and multi-dimensional labels.

``group_idx`` may label the entries of one axis of ``a`` instead of its first
dimension, and it may hold several rows of labels, which then address a group
by coordinate.  Both are checked against the numpy reference over every axis,
memory order and number of dimensions.
"""

import itertools
import warnings

import numpy as np
import pytest

from numpy_groupies.tests.helpers import func_list, is_purepy, random_group_idx, random_values, reference_trapezoid


@pytest.mark.parametrize(
    ("size", "axis"),
    [((12,), 0), ((12, 5), 0), ((12, 5), 1)],
    ids=["(12,)-ax0", "(12,5)-ax0", "(12,5)-ax1"],
)
@pytest.mark.parametrize("func", func_list)
def test_along_axis(impl, rng, func, size, axis):
    group_idx = np.zeros(size[axis], dtype=int)
    a = random_values(rng, size)

    # add some NaNs to test out nan-skipping
    if "nan" in func and "nanarg" not in func:
        a[[1, 4, 5], ...] = np.nan
    elif "nanarg" in func and a.ndim > 1:
        a[[1, 4, 5], 1] = np.nan
    if func in ["any", "all"]:
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
    elif func == "sumofsquares":
        expected = np.sum(a * a, axis=axis)
    elif func == "nansumofsquares":
        expected = np.nansum(a * a, axis=axis)
    elif func == "trapezoid":
        expected = reference_trapezoid(a, axis=axis)
    elif func == "nantrapezoid":
        # numpy has no nan-aware trapezoid - nantrapezoid integrates the
        # samples that are left, bridging over the gaps the nans leave
        expected = np.apply_along_axis(lambda v: reference_trapezoid(v[~np.isnan(v)]), axis, a)
    else:
        with warnings.catch_warnings():
            # the numpy reference warns about the all-nan slices it averages
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = getattr(np, func)(a, axis=axis)

    # no fill_value needed - the function specific defaults match numpy
    actual = impl.aggregate(group_idx, a, axis=axis, func=func)
    assert actual.ndim == a.ndim

    # argmin, argmax don't support keepdims, so we can't use that to construct expected
    # instead we squeeze out the extra dims in actual.
    np.testing.assert_allclose(actual.squeeze(), expected)


@pytest.mark.deselect_if(func=is_purepy)
def test_not_last_axis_reduction(impl):
    group_idx = np.array([1, 2, 2, 0, 1])
    a = np.array([[1.0, 2.0], [4.0, 4.0], [5.0, 2.0], [np.nan, 3.0], [8.0, 7.0]])
    func = "nanmax"
    fill_value = np.nan
    axis = 0
    actual = impl.aggregate(group_idx, a, axis=axis, func=func, fill_value=fill_value)
    expected = np.array([[np.nan, 3.0], [8.0, 7.0], [5.0, 4.0]])
    np.testing.assert_allclose(expected, actual)


@pytest.mark.deselect_if(func=is_purepy)
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


@pytest.mark.deselect_if(func=is_purepy)
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


@pytest.mark.deselect_if(func=is_purepy)
@pytest.mark.parametrize("size", [None, (10, 2)])
def test_ndim_group_idx(impl, size):
    group_idx = np.vstack((np.repeat(np.arange(10), 10), np.repeat([0, 1], 50)))
    impl.aggregate(group_idx, 1, size=size)


@pytest.mark.deselect_if(func=is_purepy)
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


@pytest.mark.deselect_if(func=is_purepy)
@pytest.mark.parametrize("func", ["sort", "cumsum", "cumprod"])
def test_one_out_per_in_cannot_fill(impl, func):
    # sort and the cum-functions emit one value per item, so they have
    # nothing to fill a hole with - but they need to say so clearly
    group_idx = np.array(list(itertools.product(range(3), range(4)))).T[:, :-1]
    with pytest.raises(ValueError, match="cannot fill"):
        impl.aggregate(group_idx, np.arange(11, dtype=float), func=func, size=(3, 4))


@pytest.mark.deselect_if(func=is_purepy)
def test_argreduction_nD_array_1D_idx(impl):
    # https://github.com/ml31415/numpy-groupies/issues/41
    group_idx = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=int)
    a = np.array([[1] * 12, [1] * 12])
    actual = impl.aggregate(group_idx, a, axis=-1, func="argmax")
    expected = np.array([[0, 5, 2], [0, 5, 2]])
    np.testing.assert_equal(actual, expected)


@pytest.mark.deselect_if(func=is_purepy)
def test_argreduction_negative_fill_value(impl):
    group_idx = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=int)
    a = np.array([[1] * 12, [np.nan] * 12])
    actual = impl.aggregate(group_idx, a, axis=-1, fill_value=-1, func="argmax")
    expected = np.array([[0, 5, 2], [-1, -1, -1]])
    np.testing.assert_equal(actual, expected)
