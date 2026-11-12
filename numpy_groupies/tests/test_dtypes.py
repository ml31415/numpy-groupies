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

from numpy_groupies.dtypes import (
    minimum_dtype,
    resolve_output_dtype,
)
from numpy_groupies.tests.helpers import is_purepy

float_funcs = [np.std, np.var, np.mean, np.median, np.nanmedian]
if hasattr(np, "trapezoid"):
    float_funcs.append(np.trapezoid)


@pytest.mark.parametrize("func", ["sumofsquares", "nansumofsquares"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32])
def test_integer_sumofsquares(impl, func, dtype):
    """The sum of the squares of the integers is exact in the accumulator of
    their sign, in every backend.

    The squares of unsigned values are unsigned - a negative sum of squares
    is no answer at all, and the squares of the widest uint32 leave the
    int64.
    """
    maximum = min(int(np.iinfo(dtype).max), 50000)
    a = np.array([maximum, maximum - 1, maximum // 2, 1], dtype=dtype)
    group_idx = np.array([0, 0, 1, 1])
    expected = [sum(int(value) ** 2 for value in a[:2]), sum(int(value) ** 2 for value in a[2:])]
    if np.dtype(dtype).kind == "u":
        result = impl.aggregate(group_idx, a, func=func, size=2)
        if not is_purepy(impl):
            # the pure python implementation carries no dtype of its own
            assert result.dtype == np.uint64
    else:
        expected.append(-1)
        result = impl.aggregate(group_idx, a, func=func, size=3, fill_value=-1)
        if not is_purepy(impl):
            assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)
    # the square of the widest value of the dtype is exact in the
    # accumulator - one value per group, for the sum of two squares of the
    # widest uint32 would leave even the uint64
    edge = np.array([np.iinfo(dtype).max], dtype=dtype)
    edge_result = impl.aggregate(np.array([0]), edge, func=func, size=1)
    if not is_purepy(impl):
        assert edge_result.dtype == (np.uint64 if np.dtype(dtype).kind == "u" else np.int64)
    np.testing.assert_array_equal(edge_result, int(np.iinfo(dtype).max) ** 2)


@pytest.mark.parametrize("func", ["sumofsquares", "nansumofsquares"])
def test_integer_sumofsquares_float_output(impl, func):
    a = np.array([[20000, -20000], [10000, -10000]], dtype=np.int16)
    result = impl.aggregate(np.array([0, 0]), a, func=func, axis=1, dtype=np.float64)
    assert np.asarray(result).dtype == np.float64
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
    # the fill value has to fit into the dtype of the answer, which keeps the
    # sign of the input: an unsigned input answers with an unsigned dtype
    fill = -1 if np.dtype(dtype).kind == "i" else 2**31
    res = impl.aggregate(group_idx, a, func="sum", size=3, fill_value=fill)
    np.testing.assert_array_equal(res, [2 * maximum, fill, fill])
    if not is_purepy(impl):
        # the pure python implementation carries no dtype of its own
        assert res.dtype == minimum_dtype(2 * maximum, np.dtype(dtype))


def test_sum_bool_dtype_guess_ignores_empty_groups(impl):
    # empty groups inflate the output size, but never the number of values
    # that get summed into one group - the bool sum guess stays narrow
    res = impl.aggregate(np.array([0, 0]), np.array([True, True]), func="sum", size=300)
    values = np.asarray(res)
    assert values[0] == 2
    assert not values[1:].any()
    if not is_purepy(impl):
        assert res.dtype == np.int8


def test_integer_sums_wrap_in_accumulator(impl):
    """The sums of the integers are computed in the accumulator of their
    sign, and they wrap there - the way numpy's own sums wrap in int64 and
    uint64.

    The accumulator is the floor of the ladder: a narrow integer is widened
    before it is summed, and the wide one wraps - quietly, as numpy does it.

    The pure python implementation is the exception: it accumulates in the
    python integers, which do not wrap - the widening there is exact.
    """
    if is_purepy(impl):
        pytest.skip("the pure python implementation accumulates in python integers, which do not wrap")
    # the uint64 accumulator wraps: the sum of the two largest values is the
    # largest value again, the one below the end of the range
    res = impl.aggregate(np.array([0, 0]), np.array([2**64 - 1, 2**64 - 1], dtype=np.uint64), func="sum", size=1)
    np.testing.assert_array_equal(res, [2**64 - 2])
    # and so does the int64 one: the sum of the two largest signed values is
    # the largest negative one
    res = impl.aggregate(np.array([0, 0]), np.array([2**63 - 1, 2**63 - 1], dtype=np.int64), func="sum", size=1)
    np.testing.assert_array_equal(res, [-2])


@pytest.mark.parametrize("func", ["sum", "nansum"])
@pytest.mark.parametrize("large", [10**18, 2**63, 2**64 - 2])
@pytest.mark.parametrize("fill_value", [0, 7])
def test_uint64_sum_keeps_integer_precision(impl, func, large, fill_value):
    values = np.array([large, 3, 1, 4], dtype=np.uint64)
    groups = np.array([0, 2, 0, 2])
    result = impl.aggregate(groups, values, func=func, size=4, fill_value=fill_value)
    # Comparing float arrays to float-cast expectations hides lost low bits.
    assert [int(value) for value in result] == [large + 1, fill_value, 7, fill_value]
    if not isinstance(result, list):
        assert result.dtype == np.uint64


@pytest.mark.parametrize("floatfunc", float_funcs, ids=lambda x: x.__name__)
def test_float_enforcement(impl, floatfunc):
    group_idx = np.arange(10).repeat(3)
    a = np.arange(group_idx.size)
    res = impl.aggregate(group_idx, a, func=floatfunc)
    if not is_purepy(impl):
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
    partial sums - over int64/uint64 input and output dtypes, with and
    without an explicit dtype request, and two fill values.
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
def test_integer_mean_extreme_cancellation_exact(impl, func):
    # the sums cancel to a small exact mean although the values reach the
    # edges of int64 - the limb sums hold the total exactly where float64
    # accumulation would round
    maximum = np.iinfo(np.int64).max
    values = np.array([maximum, -maximum + 1, maximum, maximum, maximum], dtype=np.int64)
    original = values.copy()
    actual = impl.aggregate(np.array([0, 0, 1, 1, 1]), values, func=func, size=3)
    np.testing.assert_array_equal(actual, [0.5, maximum, np.nan])
    np.testing.assert_array_equal(values, original)


def test_integer_mean_beyond_exact_window_exact(impl):
    # (2**53 + 1) is not representable in float64 - the mean is still the
    # exact quotient of the exact total, not a float64 rounding of it
    values = np.array([2**53 + 1, 0, 0], dtype=np.int64)
    actual = impl.aggregate(np.array([0, 0, 0]), values, func="mean", size=1)
    np.testing.assert_array_equal(actual, [(2**53 + 1) / 3])


def test_integer_mean_half_to_even_beyond_exact_window(impl):
    # an integer output rounds the exact quotient half to even: the sum of
    # (2**53 + 1) and 2 is odd, the tie of (2**53 + 3) / 2 rounds up to the
    # even neighbour - although the values themselves are not floats.
    values = np.array([2**53 + 1, 2], dtype=np.int64)
    actual = impl.aggregate(np.array([0, 0]), values, func="mean", size=1, dtype=np.int64)
    np.testing.assert_array_equal(actual, [2**52 + 2])


def test_integer_mean_gate_boundary(impl):
    # 2 * 2**52 == 2**53 stays inside the exact window and returns the exact
    # mean; a third value leaves it and the answer stays exact as well - the
    # window no longer decides between an answer and a refusal
    values = np.array([2**52, 2**52], dtype=np.int64)
    np.testing.assert_array_equal(impl.aggregate(np.array([0, 0]), values, func="mean", size=1), [2**52])
    values = np.array([2**52, 2**52, 2**52], dtype=np.int64)
    np.testing.assert_array_equal(impl.aggregate(np.array([0, 0, 0]), values, func="mean", size=1), [2**52])


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


INTEGER_DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("n", [1, 2, 8, 1000, 10**6])
def test_sum_of_integers_answers_with_integer(dtype, n):
    """The sum of integers answers with an integer, whatever the count.

    The ladder of the output dtype climbs with the number of the values
    (bool -> int8 -> int32 -> int64, uint8 -> uint32 -> uint64) and keeps
    the sign of the input: a signed input never lands in an unsigned dtype,
    where a negative sum would read as a huge positive one, and an unsigned
    one does not become signed.  numpy keeps the sign as well (it sums into
    int64 or uint64), but it has no ladder at all - the sum overflows there.
    """
    out = resolve_output_dtype(None, "sum", np.dtype(dtype), n)
    assert out.kind in "iub", f"{dtype} at n={n} answers {out}"
    if dtype.startswith("u"):
        assert out.kind in "ub", f"{dtype} at n={n} answers {out}"
    else:
        assert out.kind in "ib", f"{dtype} at n={n} answers {out}"


@pytest.mark.parametrize("dtype", ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"])
def test_ladder_ends_in_float64_and_never_in_float32(dtype):
    """A value that fits into no integer dtype answers with float64.

    The mantissa of float32 is 24 bits - too coarse for the sum it would
    carry - and the ladder has no float before it.  The int64 and the uint64
    are not part of the ladder: they keep their dtype and wrap, like numpy.
    """
    out = resolve_output_dtype(None, "sum", np.dtype(dtype), 10**30)
    assert out == np.dtype(np.float64)


def test_sum_of_small_integers_answers_with_ladder_dtype(impl):
    """The dtype of the answer is the one of the ladder, in every backend.

    The pure python implementation carries no dtype of its own - it answers
    with the python integers it accumulates.
    """
    if is_purepy(impl):
        pytest.skip("the pure python implementation carries no dtype")
    a = np.array([100, 100, 100], dtype=np.uint8)
    actual = impl.aggregate(np.array([0, 0, 0]), a, func="sum", size=1)
    assert np.asarray(actual).dtype == np.dtype(np.uint32)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_dtype_policy_of_running_values_and_products(dtype):
    """The dtype policy of the answers that accumulate: the running sums
    climb the ladder of the sums (the same rung for the same values), the
    products land in the accumulator of the input's own sign, and the
    functions that hand the values of the group out keep the dtype of the
    values.  The ladder is numpy's accumulator (int64 for the signed values
    and bool, uint64 for the unsigned ones) with a 32-bit rung in the
    middle, and the widening never crosses the sign: an unsigned input
    never lands in a signed dtype, where half of its range would read as
    negative.
    """
    a_dtype = np.dtype(dtype)
    n = 1000
    # the running sums climb the ladder of the sums, whatever the size of a
    # single group - their growth is linear, and the count is known
    for func in ("sum", "nansum", "cumsum", "nancumsum"):
        assert resolve_output_dtype(None, func, a_dtype, n) == resolve_output_dtype(None, "sum", a_dtype, n), func
    # the products grow exponentially - no rung of the ladder can promise
    # them anything, and the wide accumulator of the input's own sign is the
    # floor
    accumulator = np.promote_types(a_dtype, np.uint if a_dtype.kind == "u" else np.int_)
    for func in ("prod", "nanprod", "cumprod", "nancumprod"):
        assert resolve_output_dtype(None, func, a_dtype, n) == accumulator, func
    # the extrema and the sorted values hand out the values of the group
    # themselves - their dtype is the dtype of the values
    for func in ("min", "max", "sort", "cummin", "cummax"):
        assert resolve_output_dtype(None, func, a_dtype, n) == a_dtype, func


@pytest.mark.parametrize(
    "dtype",
    ["int8", "int16", "int32", "uint8", "uint16", "uint32"],
)
def _the_narrow_values(dtype):
    """the values whose exact answer leaves the dtype of the input.

    Three values whose sum does, a pair whose product does, and the widest
    value of the dtype, whose square does.
    """
    a_dtype = np.dtype(dtype)
    maximum = int(np.iinfo(a_dtype).max)
    return a_dtype, maximum, np.array([maximum, maximum, maximum // 2], dtype=a_dtype)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "uint8", "uint16", "uint32"])
@pytest.mark.parametrize(
    ("func", "values_of", "wanted_of"),
    [
        # the sum of the three values
        ("sum", lambda maximum: np.array([maximum, maximum, maximum // 2]), lambda maximum: 2 * maximum + maximum // 2),
        # the product of the widest value and two small ones - the wrap of the
        # input dtype would answer a 1
        ("prod", lambda maximum: np.array([maximum, 2, 3]), lambda maximum: 6 * maximum),
        # the square of the widest value - the squares of the widest uint32
        # leave the int64
        ("sumofsquares", lambda maximum: np.array([maximum]), lambda maximum: maximum**2),
    ],
    ids=["sum", "prod", "sumofsquares"],
)
def test_narrow_integer_values_are_never_wrapped_in_their_own_dtype(impl, dtype, func, values_of, wanted_of):
    """No backend hands back a value that wrapped in the dtype of the input.

    The answers of the narrow integers are widened before they are computed -
    the values of the answer are the exact ones the ladder's rung holds,
    whatever backend computed them.  The pure python backend accumulates in
    python numbers and answers exact values of its own; the comparison here
    is the value of the answer, not its dtype.  The int64 and the uint64 are
    not part of this promise: they are the end of the ladder, and their wrap
    is numpy's own.
    """
    a_dtype, maximum, _ = _the_narrow_values(dtype)
    values = values_of(maximum).astype(a_dtype)
    group_idx = np.zeros(values.size, dtype=int)
    wanted = wanted_of(maximum)
    got = impl.aggregate(group_idx, values, func=func, size=1)
    np.testing.assert_array_equal(np.asarray(got), wanted, err_msg=f"{func} over {dtype} wrapped")


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "uint8", "uint16", "uint32"])
def test_narrow_running_sums_are_never_wrapped_in_their_own_dtype(impl, dtype):
    """The running sum of the narrow integers is widened before it is computed.

    The last value of the running sum is the sum of the group, and it is the
    exact one the ladder's rung holds.
    """
    _a_dtype, maximum, values = _the_narrow_values(dtype)
    group_idx = np.zeros(values.size, dtype=int)
    wanted = 2 * maximum + maximum // 2
    got = np.asarray(impl.aggregate(group_idx, values, func="cumsum", size=1))
    np.testing.assert_array_equal(got[-1], wanted, err_msg=f"cumsum over {dtype} wrapped")
