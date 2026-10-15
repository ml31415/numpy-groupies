"""Integer precision policy for mean/var/std, uniform across backends.

``var``/``std``/``nanvar``/``nanstd`` center each group on its first value, so
a large common offset can no longer erase small deviations (previously numpy
silently returned 0.0 and numba returned garbage from int64 wraparound). When
the values span the whole signed 64-bit range, centering would itself wrap and
every backend raises ``OverflowError`` instead.

``mean``/``nanmean`` raise ``OverflowError`` once the exact integer sums could
leave the +/-2**53 window that float64 represents exactly - again for the same
inputs in every backend, so the results never depend on the implementation.
"""

import numpy as np
import pytest


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("func", ["var", "std", "nanvar", "nanstd"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_variance_large_offset(impl, dtype, func, ddof):
    values = np.array([2**60, 2**60 + 1, 2**60 + 7, 2**60 + 10], dtype=dtype)
    labels = np.array([0, 0, 1, 1])
    expected = np.array([0.25, 2.25]) * (2 if ddof else 1)
    if func.endswith("std"):
        expected = np.sqrt(expected)
    original = values.copy()
    actual = impl.aggregate(labels, values, func=func, ddof=ddof, size=3, fill_value=7)
    np.testing.assert_array_equal(actual, np.r_[expected, 7])
    np.testing.assert_array_equal(values, original)


def test_integer_variance_uint64_above_half_span(impl):
    # values beyond 2**63 cannot be cast to int64 directly - the deviation has
    # to come out of the two's-complement bit pattern of the uint64 difference
    values = np.array([2**63 + 10**18, 2**63 + 10**18 + 4], dtype=np.uint64)
    actual = impl.aggregate(np.array([0, 0]), values, func="var", size=1)
    np.testing.assert_array_equal(actual, [4.0])


def test_integer_variance_int32_offset(impl):
    # small dtypes pass the exactness gate (4 * 2**31 < 2**53), but their
    # squared deviations still dominate the result - the backends must agree
    # on the float64 rounding of the uncentered path
    values = np.array([2**31 - 1, -(2**31) + 3, 2**31 - 1, -(2**31) + 7], dtype=np.int32)
    actual = impl.aggregate(np.array([0, 0, 1, 1]), values, func="var", size=2)
    expected = np.array([(2**31 - 2.0) ** 2, (2**31 - 4.0) ** 2])
    np.testing.assert_allclose(actual, expected, rtol=1e-15)


def test_integer_variance_small_values_uncentered(impl):
    # below the exactness gate the plain float64 path is unchanged
    values = np.array([100, 103, -50, -46], dtype=np.int16)
    actual = impl.aggregate(np.array([0, 0, 1, 1]), values, func="var", size=2)
    np.testing.assert_array_equal(actual, [2.25, 4.0])


def test_integer_variance_offset_2d(impl):
    values = np.array([[2**60, 2**60 + 1], [2**60 + 10, 2**60 + 13]], dtype=np.int64)
    actual = impl.aggregate(np.array([0, 0]), values, axis=1, func="var")
    np.testing.assert_array_equal(actual, [[0.25], [2.25]])


@pytest.mark.parametrize("func", ["var", "std"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_variance_full_span_raises(impl, dtype, func):
    limits = np.iinfo(dtype)
    values = np.array([limits.min, limits.max], dtype=dtype)
    with pytest.raises(OverflowError, match="span"):
        impl.aggregate(np.array([0, 0]), values, func=func, size=1)
    np.testing.assert_array_equal(values, np.array([limits.min, limits.max]))


@pytest.mark.parametrize("func", ["mean", "nanmean"])
def test_integer_mean_cancellation_raises(impl, func):
    maximum = np.iinfo(np.int64).max
    values = np.array([maximum, -maximum + 1, maximum, maximum, maximum], dtype=np.int64)
    original = values.copy()
    with pytest.raises(OverflowError, match="exact mean"):
        impl.aggregate(np.array([0, 0, 1, 1, 1]), values, func=func, size=3)
    np.testing.assert_array_equal(values, original)


def test_integer_mean_beyond_exact_window_raises(impl):
    # (2**53 + 1) is not representable in float64, so even this tiny group has
    # no exact float64 mean
    values = np.array([2**53 + 1, 0, 0], dtype=np.int64)
    with pytest.raises(OverflowError, match="exact mean"):
        impl.aggregate(np.array([0, 0, 0]), values, func="mean", size=1)


def test_integer_mean_gate_boundary(impl):
    # 2 * 2**52 == 2**53 stays inside the exact window and returns the exact
    # mean; a third value pushes the possible sum beyond it and is refused
    values = np.array([2**52, 2**52], dtype=np.int64)
    np.testing.assert_array_equal(impl.aggregate(np.array([0, 0]), values, func="mean", size=1), [2**52])
    values = np.array([2**52, 2**52, 2**52], dtype=np.int64)
    with pytest.raises(OverflowError, match="exact mean"):
        impl.aggregate(np.array([0, 0, 0]), values, func="mean", size=1)


def test_integer_mean_small_values_exact(impl):
    # small integers pass the gate and keep their exact float64 means
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    actual = impl.aggregate(np.array([0, 0, 0, 1, 1, 1]), values, func="mean", size=2)
    np.testing.assert_array_equal(actual, [2.0, 5.0])
