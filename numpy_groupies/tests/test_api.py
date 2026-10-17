"""The calling contract of ``aggregate``, run against every backend.

Arguments are validated the same way everywhere, the shape of the output
follows ``size`` (or the labels), an absent group takes the fill value of its
function, and whatever is not a plain group reduction (a list, a callable, a
single value instead of a column) behaves like it does in numpy.
"""

import numpy as np
import pytest

from numpy_groupies.tests.helpers import (
    func_list,
    is_purepy,
    keeps_the_fill_value_as_given,
    random_values,
)
from numpy_groupies.tests.reference import aggregate_ref
from numpy_groupies.utils import default_fill_value

# the one-out-per-item functions (cum*, sort) have nothing to fill an absent
# group with - see test_one_out_per_in_cannot_fill in test_axis.py
default_fill_funcs = tuple(fn for fn in func_list if "cum" not in fn)
default_fill_funcs += ("first", "last", "nanfirst", "nanlast", "array")

# every spelling of "one value instead of a column": python numbers carry no
# dtype at all, numpy scalars and 0d arrays carry their own
scalar_input_values = [
    pytest.param(1, id="python int"),
    pytest.param(2.5, id="python float"),
    pytest.param(True, id="python bool"),
    pytest.param(np.float64(2.5), id="float64"),
    pytest.param(np.float32(2.5), id="float32"),
    pytest.param(np.int64(3), id="int64"),
    pytest.param(np.int32(3), id="int32"),
    pytest.param(np.bool_(True), id="bool"),
    pytest.param(np.complex64(2.5 + 1j), id="complex64"),
    pytest.param(np.array(2.5), id="0d float64"),
    pytest.param(np.array(np.float32(2.5)), id="0d float32"),
    pytest.param(np.array(True), id="0d bool"),
]


def test_preserve_missing(impl):
    res = impl.aggregate(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int))
    np.testing.assert_array_equal(res, np.array([101, 206, 0, 208]))
    if not isinstance(res, list):
        assert "int" in res.dtype.name


@pytest.mark.parametrize("group_idx_type", [int, "uint32", "uint64"])
def test_uint_group_idx(impl, group_idx_type):
    group_idx = np.array([1, 1, 2, 2, 2, 2, 4, 4], dtype=group_idx_type)
    res = impl.aggregate(group_idx, np.ones(group_idx.size), dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 2, 4, 0, 2]))
    if not isinstance(res, list):
        assert "int" in res.dtype.name


def test_start_with_offset(impl):
    group_idx = np.array([1, 1, 2, 2, 2, 2, 4, 4])
    res = impl.aggregate(group_idx, np.ones(group_idx.size), dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 2, 4, 0, 2]))
    if not isinstance(res, list):
        assert "int" in res.dtype.name


def test_start_with_offset_prod(impl):
    group_idx = np.array([2, 2, 4, 4, 4, 7, 7, 7])
    res = impl.aggregate(group_idx, group_idx, func=np.prod, dtype=int)
    np.testing.assert_array_equal(res, np.array([1, 1, 4, 1, 64, 1, 1, 343]))


def test_create_lists(impl):
    res = impl.aggregate(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int), func=list)
    np.testing.assert_array_equal(np.array(res[0]), np.array([101]))
    assert len(res[2]) == 0
    np.testing.assert_array_equal(np.array(res[3]), np.array([103, 105]))


def test_item_counting(impl):
    group_idx = np.array([0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 5, 4, 3, 8, 8])
    a = np.arange(group_idx.size)
    res = impl.aggregate(group_idx, a, func=lambda x: len(x) > 1)
    np.testing.assert_array_equal(res, np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))


@pytest.mark.parametrize("a_dtype", ["float64", "int64", "float16"], ids=["float", "int", "float16"])
@pytest.mark.parametrize("func", default_fill_funcs, ids=lambda x: getattr(x, "__name__", x))
def test_default_fill_value(impl, func, a_dtype):
    # group 1 has no items at all, so it takes the function specific default
    group_idx = np.array([0, 0, 2, 2])
    a = np.array([1.0, 2.0, 4.0, 8.0], dtype=a_dtype)

    res = impl.aggregate(group_idx, a, func=func, size=3)
    if isinstance(res, list) and isinstance(res[0], list):
        # the array function gives a list of lists, which no ndarray accepts
        assert len(res[1]) == 0
        return

    res = np.asarray(res)
    expected = default_fill_value(func, res.dtype)
    if isinstance(expected, list):
        assert len(res[1]) == 0
    else:
        # nan is only equal to itself in the eyes of this test
        assert res[1] == expected or (res[1] != res[1] and expected != expected)


@pytest.mark.parametrize(
    ["func", "dtype", "expected"],
    [
        ("sum", "int64", 0),
        ("prod", "int64", 1),
        ("mean", "float64", np.nan),
        ("median", "float64", np.nan),
        ("var", "float64", np.nan),
        ("trapezoid", "float64", 0),
        ("argmax", "int64", -1),
        ("nanargmin", "int64", -1),
        ("first", "float64", np.nan),
        # integer output cannot hold nan, so those keep the conservative 0
        ("first", "int64", 0),
        ("max", "int64", 0),
        ("all", "bool", False),
        ("array", "object", []),
    ],
)
def test_default_fill_value_table(func, dtype, expected):
    # comparing by repr because neither nan nor the empty sequence compare
    # equal to themselves
    assert repr(default_fill_value(func, dtype)) == repr(expected)


@pytest.mark.parametrize(["func", "fill_value"], [(np.array, None), (np.sum, -1)], ids=["array", "sum"])
def test_fill_value(impl, func, fill_value):
    group_idx = np.array([0, 2, 2], dtype=int)
    res = impl.aggregate(
        group_idx,
        np.arange(len(group_idx), dtype=int),
        func=func,
        fill_value=fill_value,
    )
    assert res[1] == fill_value


@pytest.mark.parametrize("value", scalar_input_values)
@pytest.mark.parametrize("func", ["sum", "prod", "mean", "var", "std"])
def test_scalar_input(impl, func, value):
    group_idx = np.arange(0, 100, dtype=int).repeat(5)
    if func not in ("sum", "prod", "len"):
        with pytest.raises((ValueError, NotImplementedError)):
            impl.aggregate(group_idx, value, func=func)
    else:
        res = impl.aggregate(group_idx, value, func=func)
        ref = impl.aggregate(group_idx, np.full(group_idx.size, value), func=func)
        np.testing.assert_array_equal(res, ref)


@pytest.mark.deselect_if(func=is_purepy)
def test_custom_callable(impl, rng):
    def custom_callable(x):
        return x.sum()

    size = (10,)
    axis = -1

    group_idx = np.zeros(size, dtype=int)
    a = random_values(rng, size)

    expected = a.sum(axis=axis, keepdims=True)
    actual = impl.aggregate(group_idx, a, axis=axis, func=custom_callable, fill_value=0)
    assert actual.ndim == a.ndim

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_array_ordering(impl, order, size=10):
    mat = np.zeros((size, size), order=order, dtype=float)
    mat.flat[:] = np.arange(size * size)
    assert impl.aggregate(np.zeros(size, dtype=int), mat[0, :], order=order)[0] == sum(range(size))


def test_no_negative_indices(impl):
    for pos in (0, 10, -1):
        group_idx = np.arange(5).repeat(5)
        group_idx[pos] = -1
        with pytest.raises(ValueError):
            impl.aggregate(group_idx, np.arange(len(group_idx)))


def test_empty_group_idx_raises(impl):
    with pytest.raises(ValueError, match="must not be empty"):
        impl.aggregate(np.array([], dtype=int), np.array([], dtype=float))


def test_parameter_missing(impl):
    with pytest.raises(TypeError):
        impl.aggregate(np.arange(5))


def test_shape_mismatch(impl):
    with pytest.raises(ValueError):
        impl.aggregate(np.array((1, 2, 3)), np.array((1, 2)))


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.deselect_if(func=keeps_the_fill_value_as_given)
@pytest.mark.parametrize("fill_value", [0, 1, np.nan])
@pytest.mark.parametrize("func", func_list)
def test_the_fill_value_matches_the_group_loop(impl, func, fill_value, block):
    """what an absent group is filled with, and which fill value is refused.

    Two of the block's labels name no value, so the fill value shows up in the
    result - and one that does not fit the dtype of the result is refused, which
    the group loop refuses before it computes anything.
    """
    labels, a = block.group_idx, block.a
    kwargs = {"func": func, "size": block.size, "fill_value": fill_value}
    try:
        want = aggregate_ref(labels, a, **kwargs)
    except ValueError:
        with pytest.raises(ValueError):
            impl.aggregate(labels, a, **kwargs)
    else:
        np.testing.assert_allclose(impl.aggregate(labels, a, **kwargs), want, rtol=10**-10)
