"""Tests for complex input, that are run against all implemented versions of aggregate."""

import numpy as np
import pytest

from .. import aggregate_np, aggregate_ufunc
from . import _deselect_not_implemented, _deselect_purepy, aggregate_numba

# functions with a well-defined complex result, following numpy semantics - the
# order statistics included, as numpy orders complex values lexicographically
complex_funcs = (
    "sum",
    "prod",
    "mean",
    "var",
    "std",
    "trapezoid",
    "sumofsquares",
    "first",
    "last",
    "nansum",
    "nanprod",
    "nanmean",
    "nanvar",
    "nanstd",
    "nanmedian",
    "nantrapezoid",
    "nansumofsquares",
    "cumsum",
    "all",
    "any",
    "allnan",
    "anynan",
    "len",
    "nanlen",
    "nanfirst",
    "nanlast",
    "min",
    "max",
    "nanmin",
    "nanmax",
    "argmin",
    "argmax",
    "nanargmin",
    "nanargmax",
)
# their complex result is the real squared magnitude, not complex a*a
complex_real_out_funcs = {"var", "std", "sumofsquares", "nanvar", "nanstd", "nansumofsquares"}
# these count, test or point at a position, so their result is not a number of
# the input type
complex_bool_funcs = {"all", "any", "allnan", "anynan"}
complex_arg_funcs = ("argmin", "argmax", "nanargmin", "nanargmax")
complex_int_funcs = {"len", "nanlen", *complex_arg_funcs}
# order statistics over the values themselves, as opposed to their positions
complex_order_funcs = ("min", "max", "nanmin", "nanmax")
# conjugating the input does not simply conjugate the result for these: it
# flips the order of two values sharing a real part
complex_asymmetric_funcs = complex_order_funcs + complex_arg_funcs
# functions whose output dtype a complex input does not fix
complex_dtype_funcs = tuple(complex_real_out_funcs | complex_bool_funcs | complex_int_funcs)


def _not_nan(vals):
    """numpy's isnan of a complex value is true when either part is nan"""
    return ~(np.isnan(vals.real) | np.isnan(vals.imag))


def _trapezoid_ref(vals):
    # np.trapz was renamed to np.trapezoid in numpy 2.0
    func_ref = getattr(np, "trapezoid", None) or np.trapz
    return func_ref(vals)


def _complex_reference(func, group_idx, a, size):
    """group-wise application of the numpy function (or its python equivalent)"""
    if func == "cumsum":
        # one value per input item, written back to the position of the item
        ret = np.empty(a.shape, dtype=np.result_type(a, np.float64))
        for grp in range(size):
            pos = np.flatnonzero(group_idx == grp)
            ret[pos] = np.cumsum(a[pos])
        return ret
    ret = []
    for grp in range(size):
        vals = a[group_idx == grp]
        if func in complex_arg_funcs:
            # numpy reports the position inside the group, the implementations
            # report it inside the input
            pos = np.flatnonzero(group_idx == grp)
            ret.append(pos[getattr(np, func)(vals)])
        elif func == "first":
            ret.append(vals[0])
        elif func == "last":
            ret.append(vals[-1])
        elif func == "nanfirst":
            ret.append(vals[_not_nan(vals)][0])
        elif func == "nanlast":
            ret.append(vals[_not_nan(vals)][-1])
        elif func == "sumofsquares":
            ret.append(np.sum(np.abs(vals) ** 2))
        elif func == "nansumofsquares":
            ret.append(np.sum(np.abs(vals[_not_nan(vals)]) ** 2))
        elif func in ("len", "nanlen"):
            ret.append(len(vals) if func == "len" else int(_not_nan(vals).sum()))
        elif func == "allnan":
            ret.append(bool(np.all(np.isnan(vals))))
        elif func == "anynan":
            ret.append(bool(np.any(np.isnan(vals))))
        elif func == "trapezoid":
            ret.append(_trapezoid_ref(vals))
        elif func == "nantrapezoid":
            ret.append(_trapezoid_ref(vals[_not_nan(vals)]))
        else:
            ret.append(getattr(np, func)(vals))
    return ret


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("a_dtype", [np.complex64, np.complex128], ids=["complex64", "complex128"])
@pytest.mark.parametrize("func", complex_funcs, ids=str)
def test_complex(aggregate_all, func, a_dtype):
    if aggregate_all.__name__.endswith("purepy") and func in complex_arg_funcs:
        pytest.skip("the pure python implementation reports the index inside the group")
    group_idx = np.array([0, 1, 1, 2, 2, 2])
    a = np.array([1 + 2j, 3 + 1j, 5 + 5j, -2 + 0j, 4 - 3j, 6 - 6j]).astype(a_dtype)

    res = aggregate_all(group_idx, a, func=func, size=3)
    expected = _complex_reference(func, group_idx, a, 3)

    res = np.asarray(res)
    if aggregate_all.__name__.endswith("purepy"):
        # the pure python implementation computes with python numbers, which
        # are always double precision
        pass
    elif func in complex_bool_funcs:
        assert res.dtype == np.dtype(bool)
    elif func in complex_int_funcs:
        assert np.issubdtype(res.dtype, np.integer)
    elif func in complex_real_out_funcs:
        # the squared magnitude is real, with the real counterpart dtype of
        # the input (complex64 -> float32), exactly like np.var
        expected_real_dtype = {np.complex64: np.float32, np.complex128: np.float64}[a_dtype]
        assert res.dtype == expected_real_dtype
    else:
        assert res.dtype == a_dtype
    np.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.parametrize("a_dtype", [np.complex64, np.complex128], ids=["complex64", "complex128"])
def test_complex_median(aggregate_all, a_dtype):
    # traps: by magnitude these would mediate to -1.5+0j and 2+0j, by the real
    # part alone to 1-1j - only numpy's lexicographic order answers 1+4j, 1+9j
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1])
    a = np.array([1 + 9j, 1 - 1j, 2 + 0j, -5 + 0j, 1 + 9j, 1 - 1j, 2 + 0j]).astype(a_dtype)

    res = aggregate_all(group_idx, a, func="median", size=2)
    res = np.asarray(res)
    if not aggregate_all.__name__.endswith("purepy"):
        assert res.dtype == a_dtype
    expected = [np.median(vals) for vals in (a[:4], a[4:])]
    np.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.parametrize("a_dtype", [np.complex64, np.complex128], ids=["complex64", "complex128"])
def test_complex_sort(aggregate_all, a_dtype):
    # sorting follows numpy's lexicographic complex order
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([3 + 1j, 1 + 2j, 1 - 5j, 1 + 5j]).astype(a_dtype)

    res = np.asarray(aggregate_all(group_idx, a, func="sort"))
    assert res.dtype == a_dtype
    expected = np.concatenate([np.sort(a[group_idx == grp]) for grp in range(2)])
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("a_dtype", [np.complex64, np.complex128], ids=["complex64", "complex128"])
def test_complex_empty_group(aggregate_all, a_dtype):
    # an empty group takes the nan default, which complex dtypes can hold
    group_idx = np.array([0, 0, 2, 2])
    a = np.array([1 + 2j, 3 + 1j, 5 + 5j, 4 - 3j]).astype(a_dtype)

    res = np.asarray(aggregate_all(group_idx, a, func="mean", size=3))
    if not aggregate_all.__name__.endswith("purepy"):
        # the pure python implementation computes with python numbers, which
        # are always double precision
        assert res.dtype == a_dtype
    # the nan default of a complex output is a nan in *both* parts, so all
    # implementations fill an empty group identically
    assert np.isnan(res[1].real) and np.isnan(res[1].imag)


def test_complex_nan_handling(aggregate_all):
    group_idx = np.array([0, 0, 0, 1])
    a = np.array([1 + 2j, np.nan + 0j, 3 + 1j, 5 + 5j])

    res = np.asarray(aggregate_all(group_idx, a, func="nanmean", size=2))
    np.testing.assert_allclose(res, [np.mean([1 + 2j, 3 + 1j]), 5 + 5j], rtol=1e-5)

    # the plain mean stays poisoned by the nan, like numpy
    res = np.asarray(aggregate_all(group_idx, a, func="mean", size=2))
    assert np.isnan(res[0].real) and np.isnan(res[0].imag)
    np.testing.assert_allclose(res[1], 5 + 5j)


# every complex function but cumsum, which yields one value per input item and
# so cannot be indexed per group
complex_nan_funcs = tuple(func for func in complex_funcs if func != "cumsum")
# and without the order statistics, which conjugating the input does not survive
complex_symmetric_funcs = tuple(func for func in complex_funcs if func not in complex_asymmetric_funcs)


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("func", complex_funcs, ids=str)
def test_complex_matches_equivalent_real_input(aggregate_all, func):
    # values with a zero imaginary part have to aggregate exactly like the very
    # same values as floats - representing them as complex may not change a result
    rng = np.random.default_rng(11)
    group_idx = np.repeat(np.arange(4), 5)
    x = rng.random(group_idx.size) * 10 - 5
    if func.startswith("nan"):
        x[::4] = np.nan

    expected = np.asarray(aggregate_all(group_idx, x, func=func, size=4))
    res = np.asarray(aggregate_all(group_idx, x.astype(np.complex128), func=func, size=4))
    np.testing.assert_allclose(res, expected.astype(np.complex128), rtol=1e-9, equal_nan=True)


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("func", complex_symmetric_funcs, ids=str)
def test_complex_conjugation_symmetry(aggregate_all, func):
    # negating every imaginary part may only negate the imaginary part of the
    # result - the functions measuring a squared magnitude ignore it entirely
    rng = np.random.default_rng(12)
    group_idx = np.repeat(np.arange(3), 4)
    a = (rng.random(group_idx.size) * 4 - 2) + 1j * (rng.random(group_idx.size) * 4 - 2)

    res = np.asarray(aggregate_all(group_idx, a, func=func, size=3))
    conj = np.asarray(aggregate_all(group_idx, a.conj(), func=func, size=3))
    if func in complex_real_out_funcs | complex_bool_funcs | complex_int_funcs:
        np.testing.assert_allclose(conj, res, rtol=1e-6)
    else:
        np.testing.assert_allclose(conj, np.conj(res), rtol=1e-6)


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("func", complex_nan_funcs, ids=str)
def test_complex_nan_in_either_part(aggregate_all, func):
    # a nan in the imaginary part has to hit the result exactly like a nan in
    # the real part - the nan check looks at both parts
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([1 + 2j, 2 + 3j, np.nan + 1j, 4 + 5j, 5 + 6j, 6 + 7j, 7 + 8j, np.nan + 9j])

    res = np.asarray(aggregate_all(group_idx, a, func=func, size=2))
    imag_nan = np.asarray(aggregate_all(group_idx, a.conj(), func=func, size=2))
    if func in complex_bool_funcs | complex_int_funcs:
        # truthiness and counting do not care where the nan sits
        np.testing.assert_array_equal(imag_nan, res)
        return
    if func in complex_arg_funcs:
        # these report an index, which cannot conjugate - the nan only shows up
        # in which index is (not) chosen
        np.testing.assert_array_equal(imag_nan, res)
        return
    if func in ("first", "last"):
        # these pick a position, so the value simply follows the conjugation
        np.testing.assert_allclose(imag_nan, np.conj(res), rtol=1e-9)
        return
    if not func.startswith("nan"):
        # the plain functions propagate the nan, whichever part it hides in,
        # while their nan counterparts skip it in both parts alike
        assert np.isnan(res[0]).any() and np.isnan(imag_nan[0]).any()
    if func in complex_real_out_funcs:
        np.testing.assert_allclose(imag_nan, res, rtol=1e-6, equal_nan=True)
    else:
        np.testing.assert_allclose(imag_nan, np.conj(res), rtol=1e-6, equal_nan=True)


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, bool], ids=["float64", "float32", "int64", "bool"])
@pytest.mark.parametrize("func", ["sum", "prod", "mean", "median", "first", "last", "trapezoid", "cumsum"])
def test_complex_rejects_non_complex_dtype(aggregate_all, func, dtype):
    # a non-complex dtype would silently throw the imaginary part away
    group_idx = np.array([0, 0, 1, 1, 2])
    a = np.array([1 + 2j, 3 - 1j, -2 + 4j, 5 + 5j, 6 - 6j])

    with pytest.raises(TypeError) as raised:
        aggregate_all(group_idx, a, func=func, size=3, dtype=dtype)
    if np.dtype(dtype) is not np.dtype(bool):
        # bool is refused one step earlier, as a result dtype too narrow
        assert "cannot aggregate complex values into the non-complex dtype" in str(raised.value)


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize(
    ("func", "dtype"),
    [
        ("var", np.float32),
        ("std", np.float64),
        ("sumofsquares", np.float64),
        ("nanvar", np.float64),
        ("len", np.int64),
        ("all", bool),
    ],
)
def test_complex_accepts_the_real_result_dtypes(aggregate_all, func, dtype):
    # var, std and sumofsquares measure a real squared magnitude, and the
    # counting and testing functions never were complex to begin with
    group_idx = np.array([0, 0, 1, 1, 2])
    a = np.array([1 + 2j, 3 - 1j, -2 + 4j, 5 + 5j, 6 - 6j]).astype(np.complex64)

    res = np.asarray(aggregate_all(group_idx, a, func=func, size=3, dtype=dtype))
    if not aggregate_all.__name__.endswith("purepy"):
        # the pure python implementation ignores the dtype argument entirely
        assert res.dtype == np.dtype(dtype)


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("func", ["sum", "prod", "mean", "var", "std", "min", "max"])
def test_complex_scalar(aggregate_all, func):
    # a scalar is one and the same value in every group: sum and prod can do
    # something with that, everything reducing over an array of them cannot
    group_idx = np.arange(0, 20, dtype=int).repeat(5)
    if func in ("sum", "prod"):
        res = np.asarray(aggregate_all(group_idx, 2 + 3j, func=func))
        expected = np.asarray(aggregate_all(group_idx, np.full(group_idx.size, 2 + 3j), func=func))
        np.testing.assert_allclose(res, expected)
    else:
        with pytest.raises((ValueError, NotImplementedError)):
            aggregate_all(group_idx, 2 + 3j, func=func)


# the value a group missing from group_idx is filled with, for complex input -
# a nan in *both* parts wherever numpy's own default is a nan at all
complex_fill_cases = [
    ("mean", np.nan),
    ("median", np.nan),
    ("first", np.nan),
    ("last", np.nan),
    ("min", np.nan),
    ("max", np.nan),
    ("nanmean", np.nan),
    ("nanmedian", np.nan),
    ("var", np.nan),
    ("std", np.nan),
    ("nanvar", np.nan),
    ("nanstd", np.nan),
    ("sum", 0),
    ("nansum", 0),
    ("prod", 1),
    ("len", 0),
    ("nanlen", 0),
    ("trapezoid", 0),
    ("sumofsquares", 0),
    ("nansumofsquares", 0),
    ("all", 0),
    ("any", 0),
]


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize(("func", "fill"), complex_fill_cases)
def test_complex_default_fill_value(aggregate_all, func, fill):
    group_idx = np.array([0, 0, 2, 2, 2])
    a = np.array([1 + 2j, 3 + 1j, 5 + 5j, 4 - 3j, 6 + 1j])

    res = np.asarray(aggregate_all(group_idx, a, func=func, size=3))
    if not np.isnan(fill):
        np.testing.assert_allclose(res[1], fill)
    elif func in complex_real_out_funcs:
        # a real result has no imaginary part to poison
        assert np.isnan(res[1])
    else:
        assert np.isnan(res[1].real) and np.isnan(res[1].imag)


def test_complex_explicit_fill_value(aggregate_all):
    group_idx = np.array([0, 0, 2, 2])
    a = np.array([1 + 2j, 3 + 1j, 5 + 5j, 4 - 3j])

    res = np.asarray(aggregate_all(group_idx, a, func="mean", size=3, fill_value=1 - 2j))
    np.testing.assert_allclose(res[1], 1 - 2j)

    # a real fill value is a complex one with a zero imaginary part
    res = np.asarray(aggregate_all(group_idx, a, func="mean", size=3, fill_value=1.0))
    np.testing.assert_allclose(res[1], 1 + 0j)


@pytest.mark.deselect_if(func=_deselect_purepy)
@pytest.mark.parametrize("axis", (0, 1))
def test_complex_axis(aggregate_all, axis):
    # the labels run along the requested axis, the remaining axes are kept
    group_idx = np.array([1, 0, 0, 1])
    rng = np.random.default_rng(14)
    shape = (4, 3) if axis == 0 else (3, 4)
    a = rng.random(shape) + 1j * rng.random(shape)

    res = np.asarray(aggregate_all(group_idx, a, func="sum", axis=axis))
    groups = np.unique(group_idx)
    if axis == 0:
        expected = np.stack([a[group_idx == grp].sum(axis=0) for grp in groups])
    else:
        expected = np.stack([a[:, group_idx == grp].sum(axis=1) for grp in groups], axis=1)
    np.testing.assert_allclose(res, expected, rtol=1e-9)
    assert res.dtype == a.dtype


@pytest.mark.deselect_if(func=_deselect_not_implemented)
@pytest.mark.parametrize("func", ["sum", "mean", "var", "max", "first", "last"])
def test_complex_non_contiguous_input(aggregate_all, func):
    # a strided view is as valid an input as a contiguous array
    rng = np.random.default_rng(13)
    base = rng.random(40) + 1j * rng.random(40)
    a = base[::2]  # 20 values, stride 2 - and not contiguous
    assert not a.flags["C_CONTIGUOUS"] and not a.flags["F_CONTIGUOUS"]
    group_idx = np.repeat(np.arange(5), 4)

    expected = np.asarray(aggregate_all(group_idx, a.copy(), func=func, size=5))
    res = np.asarray(aggregate_all(group_idx, a, func=func, size=5))
    np.testing.assert_allclose(res, expected, rtol=1e-9)


def test_complex_list_input(aggregate_all):
    group_idx = np.array([0, 0, 1, 1, 1])
    a = [1 + 2j, 3 - 1j, -2 + 4j, 5 + 5j, 6 - 6j]

    res = np.asarray(aggregate_all(group_idx, a, func="mean", size=2))
    np.testing.assert_allclose(res, [np.mean(a[:2]), np.mean(a[2:])], rtol=1e-5)
    assert res.dtype == np.complex128


@pytest.mark.parametrize("func", complex_order_funcs + complex_arg_funcs)
def test_complex_order_functions_follow_numpy_lexicographic(aggregate_all, func):
    # the traps tell magnitude, real part alone and lexicographic order apart;
    # -0.0 against 0.0 pins that a tie keeps the first of the two, like np.min
    group_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    a = np.array(
        [
            3 + 0j,  # of the smaller magnitude, but of the larger real part
            1 + 9j,
            1 + 9j,  # same real part: the imaginary parts have to decide
            1 - 1j,
            2 + 5j,  # same magnitude, same real part
            2 - 5j,
            -0.0 + 0j,  # equal by ==, but not equal in bits
            0.0 - 0j,
            float("inf") + 1j,  # an infinite real part beats every finite one
            3 - 2j,
            complex(float("inf"), float("inf")),  # the largest value there is -
            complex(float("inf"), float("inf")),  # the reduction may not start below it
        ]
    )
    if aggregate_all.__name__.endswith("purepy") and func in complex_arg_funcs:
        pytest.skip("the pure python implementation reports the index inside the group")

    res = np.asarray(aggregate_all(group_idx, a, func=func, size=6))
    expected = _complex_reference(func, group_idx, a, 6)
    np.testing.assert_array_equal(res, np.asarray(expected))


@pytest.mark.parametrize("func", complex_order_funcs + complex_arg_funcs)
def test_complex_order_functions_are_not_rejected_by_numba(func):
    # the numba kernels used to refuse complex values here; they compare the two
    # parts themselves now, so they have to agree with the numpy implementation
    if aggregate_numba is None:
        pytest.skip("numba implementation not available")
    group_idx = np.array([0, 0, 1, 1, 2, 2])
    a = np.array([1 + 9j, 1 - 1j, 3 + 0j, 2 + 5j, -0.0 + 0j, 0.0 - 0j])

    res = np.asarray(aggregate_numba.aggregate(group_idx, a, func=func, size=3))
    expected = np.asarray(aggregate_np(group_idx, a, func=func, size=3))
    np.testing.assert_array_equal(res, expected)


@pytest.mark.deselect_if(func=_deselect_purepy)
def test_complex_nan_in_one_part_of_the_order_functions(aggregate_all):
    # a nan in either part is a nan of the value, wherever it sits in it.  The
    # value is built with complex(): `np.nan * 1j` puts a nan in both parts
    group_idx = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([1 + 2j, 2 + 3j, complex(1.0, np.nan), 4 + 5j, 5 + 6j, 6 + 7j, 7 + 8j, np.nan + 9j])

    if aggregate_all.__name__.endswith("pandas"):
        pytest.skip("pandas decides on its own what a nan in a complex value is")

    for func, expected in (("argmin", [-1, -1]), ("argmax", [-1, -1])):
        np.testing.assert_array_equal(np.asarray(aggregate_all(group_idx, a, func=func, size=2)), expected)
    for func, expected in (("nanargmin", [0, 4]), ("nanargmax", [3, 6])):
        np.testing.assert_array_equal(np.asarray(aggregate_all(group_idx, a, func=func, size=2)), expected)
    # min and max propagate the value itself, as np.min and np.max do
    for func in ("min", "max"):
        np.testing.assert_array_equal(
            np.asarray(aggregate_all(group_idx, a, func=func, size=2)), [complex(1.0, np.nan), complex(np.nan, 9.0)]
        )
    np.testing.assert_array_equal(np.asarray(aggregate_all(group_idx, a, func="nanmin", size=2)), [1 + 2j, 5 + 6j])
    np.testing.assert_array_equal(np.asarray(aggregate_all(group_idx, a, func="nanmax", size=2)), [4 + 5j, 7 + 8j])


def test_complex_order_functions_use_the_extreme_value_of_the_order():
    # the extreme of a complex dtype is inf+infj, not inf+0j: the reductions
    # seed with it, and the nan variants mask the nans with it
    group_idx = np.array([0, 1])
    a = np.array([complex(np.inf, np.inf), complex(-np.inf, -np.inf)])
    # each group holds one value, so min, max and the indices are all determined
    for func in ("min", "max", "nanmin", "nanmax"):
        np.testing.assert_array_equal(np.asarray(aggregate_np(group_idx, a, func=func, size=2)), a)
    for func in ("argmin", "argmax", "nanargmin", "nanargmax"):
        np.testing.assert_array_equal(np.asarray(aggregate_np(group_idx, a, func=func, size=2)), [0, 1])
    # the ufunc backend reduces min and max only - it has no arg functions and
    # no nan variants to get the seed wrong with
    for func in ("min", "max"):
        np.testing.assert_array_equal(np.asarray(aggregate_ufunc(group_idx, a, func=func, size=2)), a)

    # masking with inf+0j would leave no value equal to the found minimum
    impls = [aggregate_np] + ([aggregate_numba.aggregate] if aggregate_numba is not None else [])
    both_in_one_group = np.array([0, 0])
    for impl in impls:
        for values, func in (
            ([complex(np.nan, 0), complex(np.inf, np.inf)], "nanargmin"),
            ([complex(np.nan, 0), complex(np.inf, np.inf)], "nanargmax"),
            ([complex(np.nan, 0), complex(-np.inf, -np.inf)], "nanargmin"),
            ([complex(np.nan, 0), complex(-np.inf, -np.inf)], "nanargmax"),
        ):
            res = np.asarray(impl(both_in_one_group, np.array(values), func=func, size=1))
            np.testing.assert_array_equal(res, [1])
