"""The dtype and precision policy, uniform across backends.

The output dtype is guessed from the input and the function, and a group
reduction never wraps silently: narrow integer inputs widen, an integer sum
stays exact beyond what float64 partial sums could represent, and the mean or
variance of integers either comes out exact or raises.  ``var`` and ``std``
center each group on its first value, so a large common offset can no longer
erase small deviations - and when the values span the whole signed range,
centering would itself wrap, which every backend refuses with an
``OverflowError``.  ``mean`` and ``nanmean`` raise as soon as the exact sums
could leave the +/-2**53 window that float64 represents exactly, so a result
never depends on which backend computed it.
"""

import numpy as np
import pytest

from numpy_groupies import aggregate_numpy, aggregate_purepy
from numpy_groupies.utils_np import minimum_dtype

float_funcs = [np.std, np.var, np.mean, np.median, np.nanmedian]
if hasattr(np, "trapezoid"):
    float_funcs.append(np.trapezoid)


@pytest.mark.parametrize("func", ["sumofsquares", "nansumofsquares"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32])
def test_numpy_integer_sumofsquares(func, dtype):
    maximum = min(int(np.iinfo(dtype).max), 50000)
    a = np.array([maximum, maximum - 1, maximum // 2, 1], dtype=dtype)
    group_idx = np.array([0, 0, 1, 1])
    expected = [sum(int(value) ** 2 for value in a[:2]), sum(int(value) ** 2 for value in a[2:]), -1]
    result = aggregate_numpy.aggregate(group_idx, a, func=func, size=3, fill_value=-1)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("func", ["sumofsquares", "nansumofsquares"])
def test_numpy_integer_sumofsquares_float_output(func):
    a = np.array([[20000, -20000], [10000, -10000]], dtype=np.int16)
    result = aggregate_numpy.aggregate(np.array([0, 0]), a, func=func, axis=1, dtype=np.float64)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[800000000], [200000000]])


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32])
def test_sum_dtype_guess_survives_single_group(impl, dtype):
    # a single group receives every value - the sum overflow guess has to be
    # based on the input length, not on the (here smaller) number of output
    # groups, or the result silently wraps in the input dtype
    maximum = int(np.iinfo(dtype).max)
    group_idx = np.array([0, 0])
    a = np.array([maximum, maximum], dtype=dtype)
    # detected output size: one single group - the worst case for the guess
    res = impl.aggregate(group_idx, a, func="sum")
    np.testing.assert_array_equal(res, [2 * maximum])
    # an explicitly larger output size with fill values must not change this
    res = impl.aggregate(group_idx, a, func="sum", size=3, fill_value=-1)
    np.testing.assert_array_equal(res, [2 * maximum, -1, -1])
    if not isinstance(res, list):
        assert res.dtype == minimum_dtype(2 * maximum, np.dtype(dtype))


def test_sum_bool_dtype_guess_ignores_empty_groups(impl):
    # empty groups inflate the output size, but never the number of values
    # that get summed into one group - the bool sum guess stays narrow
    res = impl.aggregate(np.array([0, 0]), np.array([True, True]), func="sum", size=300)
    values = np.asarray(res)
    assert values[0] == 2
    assert not values[1:].any()
    if not isinstance(res, list):
        assert res.dtype == np.int8


def test_purepy_integer_sum_does_not_overflow():
    """Integer group sums in purepy do not wrap when the result overflows
    the input dtype.

    A Python int in the group keeps the accumulation in Python-int arithmetic
    and bypasses the wrapping numpy scalar path; integer groups therefore
    sum without the fixed dtype limits.
    """
    res = aggregate_purepy.aggregate([0, 0], [1, np.uint64(2**64 - 1)], func="sum")
    assert res[0] == 2**64
    res = aggregate_purepy.aggregate([0, 0], np.array([2**64 - 1, 2**64 - 1], dtype=np.uint64), func="sum")
    assert res[0] == 2**65 - 2


@pytest.mark.parametrize("floatfunc", float_funcs, ids=lambda x: x.__name__)
def test_float_enforcement(impl, floatfunc):
    group_idx = np.arange(10).repeat(3)
    a = np.arange(group_idx.size)
    res = impl.aggregate(group_idx, a, func=floatfunc)
    if not isinstance(res, list):
        assert "float" in res.dtype.name
    assert np.all(np.array(res) > 0)


@pytest.mark.parametrize("func", ["median", "nanmedian"])
@pytest.mark.parametrize(
    ("a", "expected"),
    [
        (np.array([True, True]), 1.0),
        (np.array([False, False]), 0.0),
        (np.array([100, 100], dtype=np.int8), 100.0),
        (np.array([200, 200], dtype=np.uint8), 200.0),
        (np.array([30000, 30000], dtype=np.int16), 30000.0),
        (np.array([np.int64(2**62), np.int64(2**62)]), 2.0**62),
    ],
    ids=["bool", "bool_zero", "int8", "uint8", "int16", "int64"],
)
def test_median_even_group_promotes_integral_middles(impl, func, a, expected):
    # the two middles may not be summed in the dtype of the input, which
    # overflows for narrow integers and makes bool addition a logical or
    group_idx = np.array([0, 0])
    np.testing.assert_allclose(impl.aggregate(group_idx, a, func, size=1), [expected])


@pytest.mark.parametrize("func", ["median", "nanmedian"])
@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32])
def test_median_narrow_dtypes_match_numpy(impl, rng, func, dtype):
    # group sizes from one to five, so both the odd and the even branch of the
    # median are checked against numpy's own promotion rules
    counts = [1, 2, 3, 4, 5, 2, 4]
    group_idx = np.repeat(np.arange(len(counts)), counts)
    if dtype is np.bool_:
        a = rng.integers(0, 2, group_idx.size, dtype=bool)
    else:
        a = rng.integers(0, np.iinfo(dtype).max, group_idx.size).astype(dtype)
    expected = [np.median(a[group_idx == grp]) for grp in range(len(counts))]
    np.testing.assert_allclose(
        impl.aggregate(group_idx, a, func, size=len(counts)),
        expected,
    )


@pytest.mark.parametrize("func", ["sum", "nansum"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("fill_value", [0, 7])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_wide_integer_grouped_sum(impl, func, dtype, fill_value, explicit_output):
    """An integer group sum that fits the output dtype is exact, even when
    the partial sums exceed float64 precision.

    bincount accumulates its weights in float64, so a partial sum with a
    magnitude above 2**53 loses integer precision before it completes. The
    numpy backend routes wide-integer groups through an exact accumulation
    in the resolved output dtype; the test runs the same expectation across
    every backend so each impl proves it has not regressed to float64
    partial sums. The parametrize covers int64/uint64 input and output
    dtypes, with and without an explicit dtype request, and two fill
    values for untouched groups.
    """
    groups = np.array([0, 2, 0, 2])
    values = np.array([2**60 + 1, 2**53 + 1, 2**60 + 2, 2**53 + 2], dtype=dtype)
    before = values.copy()
    result = np.asarray(
        impl.aggregate(
            groups,
            values,
            func=func,
            size=4,
            fill_value=fill_value,
            dtype=dtype if explicit_output else None,
        )
    )
    expected = [sum(int(v) for v in values[groups == g]) if g in groups else fill_value for g in range(4)]
    np.testing.assert_array_equal(result, np.array(expected, dtype=result.dtype))
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("func", ["sum", "nansum"])
@pytest.mark.parametrize(
    "values, groups, expected",
    [
        # a sum of INT64_MIN and INT64_MAX yields -1; float64 partial sums
        # round both ends away to zero and lose the cancellation remainder.
        pytest.param(
            np.array([2**60 + 1, np.iinfo(np.int64).min, -(2**60), np.iinfo(np.int64).max]),
            np.array([0, 1, 0, 1]),
            [1, -1],
            id="cancellation",
        ),
        # values that stay within float64 precision must keep landing on
        # the bincount path so ordinary int64 inputs are not slowed down.
        pytest.param(
            np.array([-(2**52), 2**52], dtype=np.int64),
            np.zeros(2, dtype=int),
            [0],
            id="within_float_precision",
        ),
    ],
)
def test_integer_grouped_sum_cancellation(impl, func, values, groups, expected):
    """Integer group sums stay exact at the dtype edge - both on the
    INT64_MIN cancellation boundary and for value sets that fit float
    precision and must therefore keep using the bincount path.
    """
    result = np.asarray(impl.aggregate(groups, values, func=func, size=len(expected)))
    np.testing.assert_array_equal(result, np.array(expected, dtype=result.dtype))


@pytest.mark.parametrize("func", ["sum", "nansum"])
@pytest.mark.parametrize("axis", [1])
def test_integer_grouped_sum_axis(impl, func, axis):
    """Integer group sums are exact along an axis too; purepy does not
    implement axis and is skipped.
    """
    groups = np.zeros(2, dtype=int)
    values = np.array([[2**60 + 1, -(2**60)], [2**53 + 1, -(2**53)]])
    result = np.asarray(impl.aggregate(groups, values, func=func, axis=axis))
    np.testing.assert_array_equal(result, [[1], [1]])


@pytest.mark.parametrize("base", [2**53, 2**63, 2**64 - 8])
@pytest.mark.parametrize("func", ["argmin", "argmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize("axis", [None, -1])
def test_uint64_argreductions_preserve_order(impl, base, func, axis):
    """uint64 ordering is preserved through argmin/argmax and their nan
    variants, even on value sets that float64 cannot distinguish.

    The numpy backend used to lower the comparison dtype to float64 on
    NaN-bearing paths, which folded all large uint64 values to the same
    bucket and broke the argreduction. The invariant is shared across
    backends so the test runs generically; purepy does not implement axis.
    """
    group_idx = np.array([0, 0, 2, 2, 2, 2])
    a = np.array([base + 1, base, base + 2, base + 3, base + 2, base + 3], dtype=np.uint64)
    expected = np.array([1, -1, 2]) if func.endswith("min") else np.array([0, -1, 3])
    if axis is not None:
        a = np.stack([a, a])
        expected = np.stack([expected, expected])
    result = np.asarray(impl.aggregate(group_idx, a, func=func, axis=axis, fill_value=-1))
    np.testing.assert_array_equal(result, expected)


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


@pytest.mark.parametrize("func", ["min", "max", "argmin", "argmax"])
def test_bool_min_max_and_arg_reductions(impl, func):
    # a bool input has a smallest and a largest value, and the arg reductions
    # name their position - the numpy backend used to refuse the bool dtype
    # with a ValueError from np.finfo
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([True, False, False, True])
    expected = {"min": [False, False], "max": [True, True], "argmin": [1, 2], "argmax": [0, 3]}[func]
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func=func, size=2), expected)
