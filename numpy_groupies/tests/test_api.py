"""The calling contract of ``aggregate``, run against every backend.

Arguments are validated the same way everywhere, the shape of the output
follows ``size`` (or the labels), an absent group takes the fill value of its
function, and whatever is not a plain group reduction (a list, a callable, a
single value instead of a column) behaves like it does in numpy.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from numpy_groupies import aggregate_purepy
from numpy_groupies.fills import default_fill_value
from numpy_groupies.functions import default_fill_values, fill_value_key
from numpy_groupies.registry import function_names
from numpy_groupies.tests.helpers import (
    assert_same_result,
    func_list,
    is_numba,
    random_values,
)
from numpy_groupies.tests.reference import aggregate_ref

# the one-out-per-item functions (cum*, sort) have nothing to fill an absent
# group with - see test_one_out_per_in_cannot_fill in test_axis.py
# the topk answers a row per group, and the test below compares the row with
# the fill value - one value against many - so the topk is measured by the
# test of the fill value against the group loop instead
default_fill_funcs = tuple(fn for fn in func_list if "cum" not in fn and fn not in ("sort", "topk"))

# every spelling of "one value instead of a column": python numbers carry no
# dtype at all, numpy scalars and 0d arrays carry their own
scalar_input_values = [
    pytest.param(1, id="python int"),
    pytest.param(2.5, id="python float"),
    pytest.param(True, id="python bool"),
    pytest.param(np.float64(2.5), id="float64"),
    pytest.param(np.float32(2.5), id="float32"),
    pytest.param(np.int64(3), id="int64"),
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
        # the counts and the flags
        ("len", "int64", 0),
        ("nunique", "int64", 0),
        ("all", "bool", False),
        ("any", "bool", False),
        ("allnan", "bool", False),
        ("anynan", "bool", False),
        # the sums and the products
        ("sum", "int64", 0),
        ("prod", "int64", 1),
        ("sumofsquares", "int64", 0),
        ("trapezoid", "float64", 0),
        # the positions
        ("argmax", "int64", -1),
        ("argmin", "int64", -1),
        # the sequences
        ("array", "object", []),
        ("sort", "object", []),
        # the running values
        ("cumsum", "float64", 0),
        ("cumprod", "float64", 1),
        ("cummax", "float64", 0),
        ("cummin", "float64", 0),
        # the averages and the order statistics: a nan, where the answer can
        # hold one
        ("mean", "float64", np.nan),
        ("median", "float64", np.nan),
        ("quantile", "float64", np.nan),
        ("percentile", "float64", np.nan),
        ("var", "float64", np.nan),
        ("std", "float64", np.nan),
        ("skew", "float64", np.nan),
        ("kurt", "float64", np.nan),
        ("min", "float64", np.nan),
        ("max", "float64", np.nan),
        ("first", "float64", np.nan),
        ("last", "float64", np.nan),
        ("topk", "float64", np.nan),
        ("mode", "float64", np.nan),
        # an integer output cannot hold a nan, so those keep the
        # conservative 0
        ("mean", "int64", 0),
        ("median", "int64", 0),
        ("var", "int64", 0),
        ("std", "int64", 0),
        ("skew", "int64", 0),
        ("kurt", "int64", 0),
        ("min", "int64", 0),
        ("max", "int64", 0),
        ("first", "int64", 0),
        ("last", "int64", 0),
        ("topk", "int64", 0),
        ("mode", "int64", 0),
        # the nan variants inherit the default of the function they skip the
        # nans of
        ("nanmean", "float64", np.nan),
        ("nanmedian", "float64", np.nan),
        ("nanquantile", "float64", np.nan),
        ("nanpercentile", "float64", np.nan),
        ("nanvar", "float64", np.nan),
        ("nanstd", "float64", np.nan),
        ("nanskew", "float64", np.nan),
        ("nankurt", "float64", np.nan),
        ("nanmin", "float64", np.nan),
        ("nanmax", "float64", np.nan),
        ("nanfirst", "float64", np.nan),
        ("nanlast", "float64", np.nan),
        ("nansum", "int64", 0),
        ("nanprod", "int64", 1),
        ("nansumofsquares", "int64", 0),
        ("nantrapezoid", "float64", 0),
        ("nanlen", "int64", 0),
        ("nannunique", "int64", 0),
        ("nanargmin", "int64", -1),
        ("nanargmax", "int64", -1),
        ("nanall", "bool", False),
        ("nanany", "bool", False),
        ("nancumsum", "float64", 0),
        ("nancumprod", "float64", 1),
        ("nancummax", "float64", 0),
        ("nancummin", "float64", 0),
    ],
)
def test_default_fill_value_table(func, dtype, expected):
    """The default of every function of the package, written out.

    The table is the policy of the package: a nan where the answer can hold
    one, a zero where it cannot, and the value that makes the operation
    neutral - the zero of a sum, the one of a product, the false of a flag.
    Every function of the registry has an entry here, so a function that is
    added without a default is a hole in the table.
    """
    # comparing by repr because neither nan nor the empty sequence compare
    # equal to themselves
    assert repr(default_fill_value(func, dtype)) == repr(expected)


def test_every_function_of_the_registry_has_a_default():
    """Every function of the package resolves to a default of its own.

    The table above names them all - a function that is added to the registry
    without a default would be answered with the nan of the generic rule,
    which is no policy at all.  The nan variants share the default of the
    function they skip the nans of.
    """
    for name in function_names():
        assert fill_value_key(name) in default_fill_values, name


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
@pytest.mark.parametrize("func", ["sum", "prod", "mean"])
def test_scalar_input(impl, func, value):
    group_idx = np.arange(0, 100, dtype=int).repeat(5)
    if func not in ("sum", "prod"):
        with pytest.raises((ValueError, NotImplementedError)):
            impl.aggregate(group_idx, value, func=func)
    else:
        res = impl.aggregate(group_idx, value, func=func)
        ref = impl.aggregate(group_idx, np.full(group_idx.size, value), func=func)
        np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize(
    "func",
    ["sum", "mean", "min", "max", "median", "std", "var", "first", "last", "cumsum", "array", "sort"],
)
def test_the_input_arrays_are_not_touched(impl, rng, func):
    """No implementation may write into the data it aggregates.

    The data belongs to the caller, and a write into it would change the
    results of the calls that follow - silently.  The promise holds whether
    the caller hands out writable or read-only arrays.
    """
    group_idx = rng.integers(0, 4, 100)
    a = rng.random(100)

    for readonly in (False, True):
        idx = group_idx.copy()
        values = a.copy()
        if readonly:
            idx.flags.writeable = False
            values.flags.writeable = False
        before = (idx.tobytes(), values.tobytes())
        impl.aggregate(idx, values, func=func, size=4)
        assert (idx.tobytes(), values.tobytes()) == before


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


def test_custom_callable_gets_the_values_in_the_order_of_the_input(impl):
    """The values of a group reach the callable in the order they appear in
    the input - the backends group them in different ways (the numba one
    counts them into blocks, the numpy ones sort them), and the order has to
    come out the same everywhere."""
    group_idx = np.array([0, 1, 0, 1, 0])
    a = np.array([1.0, 10.0, 2.0, 11.0, 3.0])

    def first_and_last(x):
        return x[0] + 10 * x[-1]

    actual = impl.aggregate(group_idx, a, func=first_and_last)
    np.testing.assert_allclose(actual, [31.0, 120.0])


@dataclass
class _spread:
    """The spread of a group, written as an object of one's own.

    A dataclass has its ``__hash__`` taken away by the ``__eq__`` that
    ``eq=True`` gives it, so it can be no key of a dict - which is what makes it
    interesting here: the dispatch table cannot be asked about it at all.
    """

    def __call__(self, x):
        return float(np.max(x) - np.min(x))


class _spread_object:
    """The same aggregation as an object of one's own that is hashable.

    Hashable enough to be a key of the table, and still no function that numba
    could compile: hashability is not the question numba asks.
    """

    def __call__(self, x):
        return float(np.max(x) - np.min(x))


def test_a_callable_that_cannot_be_a_key(impl):
    # not being a key of the dispatch table is no mistake of the func, it is what
    # makes it one of the caller's own: numpy and pure python run it, and numba,
    # which compiles its func, says which kind of func it compiles instead
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1.0, 3.0, 7.0, 2.0])
    if is_numba(impl):
        with pytest.raises(TypeError, match="can only compile a function"):
            impl.aggregate(group_idx, a, func=_spread(), size=2)
    else:
        res = np.asarray(impl.aggregate(group_idx, a, func=_spread(), size=2))
        np.testing.assert_allclose(res, [2.0, 5.0])


def test_a_callable_that_cannot_be_a_key_along_an_axis(impl):
    # the axis path names the function it validates against, and a func that
    # names no function is the caller's own there as anywhere else
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1.0, 3.0, 7.0, 2.0])
    if is_numba(impl):
        with pytest.raises(TypeError, match="can only compile a function"):
            impl.aggregate(group_idx, a, func=_spread(), axis=0)
    else:
        res = np.asarray(impl.aggregate(group_idx, a, func=_spread(), axis=0))
        np.testing.assert_allclose(res, [2.0, 5.0])


def test_a_callable_object_is_still_no_function_for_numba(impl):
    # what ran before keeps running in numpy and pure python, and numba refuses
    # the hashable object as it refused the other one - in its own words, which
    # name the object it was handed rather than the table that could not hold it
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1.0, 3.0, 7.0, 2.0])
    if is_numba(impl):
        with pytest.raises(TypeError):
            impl.aggregate(group_idx, a, func=_spread_object(), size=2)
    else:
        res = np.asarray(impl.aggregate(group_idx, a, func=_spread_object(), size=2))
        np.testing.assert_allclose(res, [2.0, 5.0])


@pytest.mark.parametrize(
    "func",
    [42, "no_such_function", [1, 2], {"a": 1}, np.array([1.0])],
    ids=["int", "unknown name", "list", "dict", "array"],
)
def test_a_func_that_names_nothing_says_the_same_everywhere(impl, func):
    # whether a func cannot be a key of the dispatch table or only is no key of
    # it makes no difference to the answer: the table is no mistake the caller
    # made, and no implementation is entitled to report it as one
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1.0, 3.0, 7.0, 2.0])
    with pytest.raises(ValueError, match="neither a valid function string nor a callable object"):
        impl.aggregate(group_idx, a, func=func, size=2)


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


# the input the validation of the package refuses - the words are the ones of
# ``numpy_groupies.validation``, and every path into a computation says the
# same ones.  The pure python implementation knows no axis and no sequence of
# sizes, so it refuses those with a ``NotImplementedError`` of its own - the
# proxy of the ``impl`` fixture turns that into a skip.
_VALIDATION_CASES = [
    # (name, group_idx, a, kwargs, error, message)
    ("axis too large", np.array([0, 1]), np.zeros((2, 2)), {"axis": 5}, ValueError, "axis arg too large"),
    (
        "a of two dimensions without an axis",
        np.array([0, 1]),
        np.zeros((2, 2)),
        {},
        ValueError,
        "must be scalar or 1 dimensional",
    ),
    (
        "labels of another shape than the values",
        np.zeros((2, 3), dtype=int),
        np.zeros((2, 2)),
        {"axis": 0},
        ValueError,
        "cannot be broadcast",
    ),
    (
        "a sequence of sizes with flat labels",
        np.array([0, 1]),
        np.array([1.0, 2.0]),
        {"size": (2,)},
        ValueError,
        "output size must be scalar",
    ),
    (
        "a sequence of sizes of another length",
        np.zeros((2, 2), dtype=int),
        np.array([1.0, 2.0]),
        {"size": (2, 2, 2)},
        ValueError,
        "output dimensions specified",
    ),
    (
        "labels that are no numbers",
        np.array([0.5, 1.0]),
        np.array([1.0, 2.0]),
        {},
        TypeError,
        "must be of integer type",
    ),
    (
        "labels that are flags",
        np.array([True, False]),
        np.array([1.0, 2.0]),
        {},
        TypeError,
        "must be of integer type",
    ),
    (
        "labels that are whole floats in a list",
        [0.0, 1.0],
        [1.0, 2.0],
        {},
        TypeError,
        "must be of integer type",
    ),
    (
        "labels that are flags in a list",
        [True, False],
        [1.0, 2.0],
        {},
        TypeError,
        "must be of integer type",
    ),
    (
        "labels beyond the size",
        np.array([0, 5]),
        np.array([1.0, 2.0]),
        {"size": 2},
        ValueError,
        "too large",
    ),
]


@pytest.mark.parametrize(
    ("name", "group_idx", "a", "kwargs", "error", "message"),
    _VALIDATION_CASES,
    ids=[case[0] for case in _VALIDATION_CASES],
)
def test_a_refused_input_is_refused_in_the_words_of_the_package(impl, name, group_idx, a, kwargs, error, message):
    """The validation of the input says the same thing in every path.

    The words are the ones of ``numpy_groupies.validation`` - the negative
    indices, the labels that are no numbers, the labels beyond the size - so
    that a caller of any of the backends reads one complaint and not three.
    """
    with pytest.raises(error, match=message):
        impl.aggregate(group_idx, a, func="sum", **kwargs)


@pytest.mark.parametrize(
    "values",
    [{"0": 1, "1": 2}, {1, 2}, frozenset({1, 2}), iter([1, 2]), (x for x in (1, 2))],
    ids=["dict", "set", "frozenset", "iterator", "generator"],
)
def test_the_pure_python_path_refuses_what_is_no_sequence_of_values(values):
    """The values of a call are a sequence of values.

    A mapping and a set are no such sequence: iterating over them yields
    their keys, and the answer would be the one of the keys instead of the
    one of the values - silently.  A generator is spent by the first look at
    it.  The pure python path reads the values as a python sequence, so it
    is the path that has to say so.
    """
    with pytest.raises(ValueError, match="must be scalar or 1 dimensional"):
        aggregate_purepy.aggregate(np.array([0, 1]), values, func="sum", size=2)


def test_the_out_argument_is_the_one_of_the_numba_backend(impl):
    """``out=`` is the numba business, and the others say so.

    The protocol of the public ``aggregate`` promises it: only the numba
    implementation honours the argument, the others raise it.
    """
    group_idx = np.array([0, 0, 1])
    a = np.array([1.0, 2.0, 3.0])
    out = np.empty(2)
    if is_numba(impl):
        ret = impl.aggregate(group_idx, a, func="sum", out=out)
        assert ret is out
        np.testing.assert_array_equal(out, [3.0, 3.0])
    else:
        with pytest.raises(NotImplementedError, match="out="):
            impl.aggregate(group_idx, a, func="sum", out=out)


@pytest.mark.parametrize("fill_value", [0, np.nan])
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
        assert_same_result(impl.aggregate(labels, a, **kwargs), want, func)


def test_the_arg_reductions_name_a_position_in_the_input(impl):
    """the position an arg reduction reports is one in the input, not in an
    array the backend had to build on the way to it, and a group holding a nan
    has no extreme to point at - so it names no position at all and takes the
    fill value of its function.
    """
    group_idx = np.array([1, 2, 0, 1, 0, 2])  # the labels are out of order and
    a = np.array([np.nan, 5.0, 3.0, 7.0, 1.0, 9.0])  # a nan poisons group 1
    for func, expected in (
        ("argmin", [4, -1, 1]),  # group 0: 1.0 at 4, group 2: 5.0 at 1
        ("argmax", [2, -1, 5]),  # group 0: 3.0 at 2, group 2: 9.0 at 5
        ("nanargmin", [4, 3, 1]),  # group 1 keeps 7.0 at 3 once the nan is gone
        ("nanargmax", [2, 3, 5]),
    ):
        got = np.asarray(impl.aggregate(group_idx, a, func=func, size=3))
        np.testing.assert_array_equal(got, expected, err_msg=func)


@pytest.mark.parametrize("func", ["sum", "mean", "all", "len", "argmax", "min", "sort"])
def test_none_is_refused_where_no_result_holds_it(impl, func):
    """``fill_value=None`` is a value, and only an object result holds one.

    The sentinel that asks for the default of a function is ``DEFAULT_FILL_VALUE``
    and the suite pins ``None`` as the value of an empty sequence result
    (`test_fill_value`).  Asked of a number, a flag, a count, a position or the
    values of a sort, the request used to be answered three different ways: the
    numpy implementation complained about an ``isnan`` of an object, the numba one
    failed to type a kernel, and the pure python one wrote the ``None`` into a
    column of floats, which made the result an object array.
    """
    labels = np.array([0, 2])
    a = np.array([1.0, 2.0])  # group 1 is never named
    with pytest.raises(TypeError, match="fills an object result"):
        impl.aggregate(labels, a, func=func, size=3, fill_value=None)
