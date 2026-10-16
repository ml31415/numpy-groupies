"""Tests for the njit-callable kernels of ``aggregate_numba.kernels``.

The kernels share their compiled drivers with the python-level
``aggregate()``, so value-parity with the (elsewhere exhaustively tested)
layer-2 results is the contract here; the explicitly documented deviations
(njit-only dtype approximations, fill defaults, pure-python refusal) get
their own tests.
"""

import numba as nb
import numpy as np
import pytest

from .. import aggregate_numba, aggregate_numpy
from ..aggregate_numba import kernels

GROUP_IDX = np.array([0, 0, 1, 1, 1, 3])  # group 2 stays empty
FLOAT_A = np.array([3.0, 1.0, np.nan, 2.0, 5.0, 4.0])
INT_A = np.array([3, 1, 7, 2, 5, 4], dtype=np.int64)
BOOL_A = np.array([True, False, True, True, False, True])

# (kernel name, kwargs, values) - the kernels run inside a jitted caller and
# are compared against the python-level aggregate() of the same name
_FLOAT_OPS = [
    ("sum", {}),
    ("nansum", {}),
    ("prod", {}),
    ("nanprod", {}),
    ("mean", {}),
    ("nanmean", {}),
    ("var", {"ddof": 1}),
    ("nanvar", {"ddof": 1}),
    ("std", {}),
    ("nanstd", {}),
    ("min", {}),
    ("nanmin", {}),
    ("max", {}),
    ("nanmax", {}),
    ("first", {}),
    ("nanfirst", {}),
    ("last", {}),
    ("nanlast", {}),
    ("median", {}),
    ("nanmedian", {}),
    ("trapezoid", {"ddof": 1.0}),
    ("nantrapezoid", {"ddof": 1.0}),
    ("sumofsquares", {}),
    ("nansumofsquares", {}),
    ("argmin", {}),
    ("nanargmin", {}),
    ("argmax", {}),
    ("nanargmax", {}),
    ("cumsum", {}),
    ("nancumsum", {}),
    ("cummax", {}),
    ("nancummax", {}),
    ("sort", {}),
    ("sort", {"ddof": 1}),
]
_INT_OPS = [
    ("sum", {}),
    ("prod", {}),
    ("len", {}),
    ("nanlen", {}),
    ("min", {}),
    ("max", {}),
    ("first", {}),
    ("last", {}),
    ("median", {}),
    ("argmin", {}),
    ("argmax", {}),
    ("cumsum", {}),
    ("cumprod", {}),
    ("cummax", {}),
    ("cummin", {}),
    ("sort", {}),
    ("sort", {"ddof": 1}),
]
_BOOL_OPS = [
    ("all", {}),
    ("nanall", {}),
    ("any", {}),
    ("nanany", {}),
    ("sort", {}),
    ("sort", {"ddof": 1}),
]


def _kernel_result(name, values, ddof=0):
    """Call the kernel of ``name`` from jitted code (fill default via None)."""
    stub = getattr(kernels, "aggregate_" + name)

    @nb.njit
    def caller(group_idx, a, ddof, size, fill_value):
        return stub(group_idx, a, ddof, size, fill_value)

    return caller(GROUP_IDX, values, ddof, -1, None)


def _layer2_result(name, values, ddof=0):
    if name in ("trapezoid", "nantrapezoid"):
        # the python-level wrapper carries the sample spacing as dx
        return aggregate_numba.aggregate(GROUP_IDX, values, func=name, dx=ddof)
    if name == "sort":
        # ... and the reverse flag of the sort kernel as its third argument
        return aggregate_numba.aggregate(GROUP_IDX, values, func=name, reverse=bool(ddof))
    return aggregate_numba.aggregate(GROUP_IDX, values, func=name, ddof=ddof)


@pytest.mark.parametrize("name,kwargs", _FLOAT_OPS)
def test_float_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, FLOAT_A, **kwargs)
    result = _kernel_result(name, FLOAT_A, **kwargs)
    np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize("name,kwargs", _INT_OPS)
def test_int_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, INT_A, **kwargs)
    result = _kernel_result(name, INT_A, **kwargs)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("name,kwargs", _BOOL_OPS)
def test_bool_ops_match_layer2(name, kwargs):
    expected = _layer2_result(name, BOOL_A, **kwargs)
    result = _kernel_result(name, BOOL_A, **kwargs)
    np.testing.assert_array_equal(result, expected)


def test_njit_composition():
    """Kernels compose within jitted code - the motivating use case."""

    @nb.njit
    def zscore(group_idx, values):
        m = kernels.aggregate_mean(group_idx, values)
        s = kernels.aggregate_std(group_idx, values)
        return (values - m[group_idx]) / s[group_idx]

    result = zscore(np.array([0, 0, 1, 1]), np.array([1.0, 3.0, 10.0, 14.0]))
    np.testing.assert_allclose(result, [-1.0, 1.0, -1.0, 1.0])


def test_pure_python_call_raises():
    with pytest.raises(NotImplementedError, match="jitted"):
        kernels.aggregate_mean(GROUP_IDX, FLOAT_A)


def test_median_rejects_complex_from_njit():
    # the median driver partitions the values and numba has no order for
    # complex dtypes - the kernel refuses to compile, aggregate() hands
    # complex medians to the numpy implementation instead
    @nb.njit
    def wrapped(group_idx, a):
        return kernels.aggregate_median(group_idx, a)

    with pytest.raises(nb.TypingError):
        wrapped(np.array([0, 1]), np.array([1 + 2j, 3 + 4j]))


def test_median_complex_falls_back_to_numpy():
    # the layer-2 wrapper keeps working for complex input via the handle's
    # python-level fallback - it must agree with the numpy backend exactly
    group_idx = np.array([0, 0, 1])
    a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    ret = aggregate_numba.aggregate(group_idx, a, func="median")
    assert ret.dtype == np.complex128
    np.testing.assert_array_equal(ret, aggregate_numpy.aggregate(group_idx, a, func="median"))


def test_nanmedian_complex_falls_back_to_numpy():
    group_idx = np.array([0, 0, 1])
    a = np.array([1 + 2j, np.nan * 1j, 5 + 6j])
    ret = aggregate_numba.aggregate(group_idx, a, func="nanmedian")
    np.testing.assert_array_equal(ret, aggregate_numpy.aggregate(group_idx, a, func="nanmedian"))


def test_sort_orders_complex_from_njit_like_numpy():
    # the kernel jitted code reaches orders complex like numpy does: numba
    # compares the real part first, the imaginary part on a tie, and leaves a
    # value it cannot compare with - one with a nan in either part, even when
    # the real parts differ - behind everything it is reached by
    @nb.njit
    def wrapped(group_idx, a):
        return kernels.aggregate_sort(group_idx, a)

    for a in (
        np.array([1 + 5j, 1 + complex(np.nan, 0), 3 + 0j, 1 + 2j]),
        np.array([3 + 0j, 1 + 2j, 1 + complex(np.nan, 0), 1 + 5j]),
        np.array([1 + complex(0, np.nan), 2 + 1j, 2 + 0j, 1 + 0j]),
    ):
        group_idx = np.zeros(a.size, dtype=int)
        np.testing.assert_array_equal(wrapped(group_idx, a), aggregate_numpy.aggregate(group_idx, a, func="sort"))


def test_sum_bool_widens_int64():
    """Documented njit-kernel deviation: bool input widens to int64 instead of
    the python-level length-dependent overflow guesses."""

    @nb.njit
    def caller(group_idx, a):
        return kernels.aggregate_sum(group_idx, a)

    assert caller(GROUP_IDX, BOOL_A).dtype == np.int64


def test_min_default_fill_depends_on_output_dtype():
    """An omitted fill_value resolves like DEFAULT_FILL_VALUE: nan for inexact
    outputs, 0 otherwise - never a garbage nan cast into an integer array."""

    @nb.njit
    def caller(group_idx, a):
        return kernels.aggregate_min(group_idx, a)

    group_idx = np.array([0, 0, 2, 2])  # group 1 empty
    result = caller(group_idx, np.array([5.0, 2.0, 7.0, 3.0]))
    assert np.isnan(result[1])
    result = caller(group_idx, np.array([5, 2, 7, 3], dtype=np.int64))
    assert result[1] == 0


def test_fill_value_passthrough():
    """An explicit fill_value reaches the untouched groups unchanged."""

    @nb.njit
    def caller(group_idx, a, fill_value):
        return kernels.aggregate_sum(group_idx, a, 0, -1, fill_value)

    result = caller(GROUP_IDX, INT_A, 42)
    assert result[2] == 42


def test_size_too_small_raises():

    @nb.njit
    def caller(group_idx, a, size):
        return kernels.aggregate_sum(group_idx, a, 0, size)

    with pytest.raises(ValueError, match="too large"):
        caller(GROUP_IDX, FLOAT_A, 1)


def test_integer_mean_overflow_from_njit():
    """The exactness gates run inside the kernels as well."""

    @nb.njit
    def caller(group_idx, a):
        return kernels.aggregate_mean(group_idx, a)

    with pytest.raises(OverflowError, match="exact mean"):
        caller(np.array([0, 0]), np.array([2**60, 2**60 + 1], dtype=np.int64))


@pytest.mark.parametrize(
    "kernel_name, values, expected_dtype",
    [
        ("aggregate_mean", FLOAT_A.astype(np.float32), np.float32),
        ("aggregate_mean", INT_A, np.float64),
        ("aggregate_var", (FLOAT_A + 1j * FLOAT_A).astype(np.complex64), np.float32),
        ("aggregate_var", INT_A, np.float64),
        ("aggregate_cumsum", INT_A.astype(np.int8), np.int64),
        ("aggregate_min", INT_A, np.int64),
    ],
)
def test_dtype_rules(kernel_name, values, expected_dtype):
    stub = getattr(kernels, kernel_name)

    @nb.njit
    def caller(group_idx, a):
        return stub(group_idx, a)

    assert caller(GROUP_IDX, values).dtype == expected_dtype


def test_layer1_layer2_share_the_driver():
    """The python-level aggregate() and the kernels use one compiled artifact
    - spot-check that both paths return bit-identical values."""
    name, values = "var", FLOAT_A
    expected = _layer2_result(name, values, ddof=1)
    result = _kernel_result(name, values, ddof=1)
    assert result.dtype == expected.dtype
    assert result.tobytes() == expected.tobytes()
