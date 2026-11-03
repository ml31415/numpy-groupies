"""The datetime64 and timedelta64 support of the numpy and numba backends.

Temporal input never reaches the kernels as datetime values: the wrapper of
``utils_np`` takes it apart into its int64 ticks and turns the results back.
The tests pin the semantics that emerge - the NaT propagation of the plain
reductions, the skipping of the nan variants, the exact means on ticks that
float64 cannot represent, the unit of the answer, and the refusals numpy
shares - and they run them over the aggregate entry points of both numpy
based backends. The pure python backend knows no temporal types: without
numpy there are none, so its parameterizations are skipped here.
"""

from fractions import Fraction

import numba as nb
import numpy as np
import pytest

from numpy_groupies.tests.helpers import impl_name

# the datetime64 cases run on a coarse unit and on nanoseconds - the ns ticks
# are the ones float64 cannot represent exactly
DTYPES = ["M8[s]", "M8[ns]", "m8[s]", "m8[ns]"]


@pytest.fixture(autouse=True)
def _skip_purepy(impl):
    if impl_name(impl) == "purepy":
        pytest.skip("the pure python backend knows no datetime64 or timedelta64 values")


def is_datetime(a):
    return np.asarray(a).dtype.kind == "M"


# ---------------------------------------------------------------------------
# the reductions and their NaT semantics


@pytest.mark.parametrize("unit", ["s", "ns"])
@pytest.mark.parametrize("func", ["min", "max", "first", "last"])
def test_order_reductions_propagate_nat(impl, unit, func):
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "2011-02-01", "2011-01-15"], dtype=f"M8[{unit}]")
    res = impl.aggregate(group_idx, a, func=func, size=2)
    assert res.dtype == a.dtype
    expected = {
        # numpy propagates the NaT of a group into both extremes
        "min": ["NaT", "2011-01-15"],
        "max": ["NaT", "2011-02-01"],
        "first": ["2011-01-03", "2011-02-01"],
        "last": ["2011-01-01", "2011-01-15"],
    }[func]
    np.testing.assert_array_equal(res, np.array(expected, dtype=a.dtype))


@pytest.mark.parametrize("unit", ["s", "ns"])
@pytest.mark.parametrize("func", ["nanmin", "nanmax", "nanfirst", "nanlast"])
def test_nan_order_reductions_skip_nat(impl, unit, func):
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "2011-02-01", "NaT"], dtype=f"M8[{unit}]")
    res = impl.aggregate(group_idx, a, func=func, size=2)
    expected = {
        "nanmin": ["2011-01-01", "2011-02-01"],
        "nanmax": ["2011-01-03", "2011-02-01"],
        "nanfirst": ["2011-01-03", "2011-02-01"],
        "nanlast": ["2011-01-01", "2011-02-01"],
    }[func]
    np.testing.assert_array_equal(res, np.array(expected, dtype=a.dtype))


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_argmin_and_argmax_name_the_nat_position(impl, unit):
    # numpy answers the position of the first NaT of a group for both
    # argreductions - as the smallest tick for argmin, and through the plan
    # for argmax - and the nan variants skip it
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "NaT", "2011-02-01"], dtype=f"M8[{unit}]")
    assert list(impl.aggregate(group_idx, a, func="argmin", size=2)) == [1, 3]
    assert list(impl.aggregate(group_idx, a, func="argmax", size=2)) == [1, 3]
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [2, 4]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [0, 4]


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_allnan_and_anynan_read_nat(impl, unit):
    group_idx = np.array([0, 0, 1, 1, 2])
    a = np.array(["2011-01-01", "NaT", "2011-02-01", "2011-03-01", "NaT"], dtype=f"M8[{unit}]")
    assert list(impl.aggregate(group_idx, a, func="allnan", size=3)) == [False, False, True]
    assert list(impl.aggregate(group_idx, a, func="anynan", size=3)) == [True, False, True]


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_all_and_any_treat_nat_as_falsy(impl, unit):
    group_idx = np.array([0, 0, 1, 2])
    a = np.array(["2011-01-01", "NaT", "NaT", "2011-02-01"], dtype=f"M8[{unit}]")
    # NaT compares above zero as a tick, but it is no value at all
    assert list(impl.aggregate(group_idx, a, func="any", size=3)) == [True, False, True]
    assert list(impl.aggregate(group_idx, a, func="all", size=3)) == [False, False, True]


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_nunique_counts_nat_once(impl, unit):
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-01", "NaT", "2011-01-01", "NaT"], dtype=f"M8[{unit}]")
    assert list(impl.aggregate(group_idx, a, func="nunique", size=2)) == [2, 1]
    assert list(impl.aggregate(group_idx, a, func="nannunique", size=2)) == [1, 0]


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_len_ignores_nothing(impl, unit):
    group_idx = np.array([0, 0, 1, 1])
    a = np.array(["2011-01-01", "NaT", "NaT", "2011-02-01"], dtype=f"M8[{unit}]")
    assert list(impl.aggregate(group_idx, a, func="len", size=2)) == [2, 2]
    assert list(impl.aggregate(group_idx, a, func="nanlen", size=2)) == [1, 1]


# ---------------------------------------------------------------------------
# the exact mean


def _round_half_even(fr):
    q, r = divmod(fr.numerator, fr.denominator)
    if 2 * r > fr.denominator or (2 * r == fr.denominator and q % 2 == 1):
        q += 1
    return q


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_datetime_mean_rounds_half_to_even(impl, unit):
    # two ticks of a group one day apart: their mean is the half day, which
    # the tick answer rounds half to even (down here, the even neighbour)
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2011-01-01", "2011-01-02", "2011-01-02", "2011-01-01", "2011-01-03"], dtype=f"M8[{unit}]")
    res = impl.aggregate(group_idx, a, func="mean", size=2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(res, np.array(["2011-01-01T16:00:00", "2011-01-02"], dtype=a.dtype))
    ticks = np.asarray(a).view(np.int64)
    want = _round_half_even(Fraction(int(ticks[3]) + int(ticks[4]), 2))
    assert int(np.asarray(res).view(np.int64)[1]) == want


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_timedelta_mean_rounds_half_to_even(impl, unit):
    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array([1, 2, 1, 2, 4], dtype=f"m8[{unit}]")
    res = impl.aggregate(group_idx, a, func="mean", size=2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(res, np.array([2, 2], dtype=a.dtype))
    # the half of a second rounds to the even neighbour
    a_tie = np.array([1, 2], dtype=f"m8[{unit}]")
    res_tie = impl.aggregate(np.array([0, 0]), a_tie, func="mean", size=1)
    assert int(np.asarray(res_tie).view(np.int64)[0]) == 2


def test_datetime_mean_is_exact_beyond_float64(impl):
    # ns ticks around 2**62: their sums leave the window float64 represents
    # exactly - the limb sums of the wrapper keep the mean exact, and the
    # python integers are the yardstick
    rng = np.random.default_rng(42)
    group_idx = rng.integers(0, 8, size=200)
    ticks = rng.integers(2**62, 2**62 + 10**7, size=200, dtype=np.int64)
    a = ticks.view("M8[ns]")
    res = impl.aggregate(group_idx, a, func="mean", size=8)
    for g in range(8):
        mask = group_idx == g
        want = _round_half_even(Fraction(int(ticks[mask].astype(object).sum()), int(mask.sum())))
        assert int(np.asarray(res).view(np.int64)[g]) == want


def test_datetime_mean_negative_ticks_exact(impl):
    # the ticks before 1970 are negative - the limb sums of the signed
    # pattern are corrected by the count of the negatives
    rng = np.random.default_rng(7)
    group_idx = rng.integers(0, 5, size=120)
    ticks = -rng.integers(0, 2**62, size=120, dtype=np.int64)
    a = ticks.view("M8[ns]")
    res = impl.aggregate(group_idx, a, func="mean", dtype=np.int64, size=5)
    for g in range(5):
        mask = group_idx == g
        want = _round_half_even(Fraction(int(ticks[mask].astype(object).sum()), int(mask.sum())))
        assert int(res[g]) == want


@pytest.mark.parametrize("func", ["mean", "nanmean"])
def test_mean_nat_semantics(impl, func):
    group_idx = np.array([0, 0, 0, 1, 1, 2])
    a = np.array(["2011-01-01", "2011-01-03", "NaT", "2011-02-01", "2011-02-03", "NaT"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func=func, size=3)
    if func == "mean":
        # a single NaT poisons its group
        np.testing.assert_array_equal(res, np.array(["NaT", "2011-02-02", "NaT"], dtype=a.dtype))
    else:
        np.testing.assert_array_equal(res, np.array(["2011-01-02", "2011-02-02", "NaT"], dtype=a.dtype))


def test_timedelta_mean_nat_semantics(impl):
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1, "NaT", 3, 5], dtype="m8[s]")
    np.testing.assert_array_equal(
        impl.aggregate(group_idx, a, func="mean", size=2), np.array(["NaT", 4], dtype=a.dtype)
    )
    np.testing.assert_array_equal(impl.aggregate(group_idx, a, func="nanmean", size=2), np.array([1, 4], dtype=a.dtype))


def test_mean_default_dtype_is_float_of_the_exact_quotient(impl):
    # an explicitly requested float64 holds the exact quotient, not the
    # rounded tick
    ticks = np.array([2**62, 2**62, 2**62 + 1], dtype=np.int64)
    a = ticks.view("M8[ns]")
    res = impl.aggregate(np.array([0, 0, 0]), a, func="mean", size=1, dtype=np.float64)
    assert res[0] == float(Fraction(int(ticks.astype(object).sum()), 3))


def test_mean_float_dtype_poisons_the_nat_groups(impl):
    # the float answer of a group a NaT poisons is no quotient at all - nan
    ticks = np.array([2**62, 2**62 + 2, 0, 2**62], dtype=np.int64)
    a = ticks.view("M8[ns]")
    a[1] = np.datetime64("NaT", "ns")
    group_idx = np.array([0, 0, 1, 1])
    res = impl.aggregate(group_idx, a, func="mean", size=2, dtype=np.float64)
    assert res[1] == float(Fraction(int(ticks[2]) + int(ticks[3]), 2))
    assert res[0] != res[0]  # nan


def test_nanmean_float_dtype_leaves_the_nat_out(impl):
    # the nan variant of the float answer averages the values it has
    ticks = np.array([2**62, 2**62 + 4, 2**62 + 8, 2**62], dtype=np.int64)
    a = ticks.view("M8[ns]")
    a[1] = np.datetime64("NaT", "ns")
    group_idx = np.array([0, 0, 0, 1])
    res = impl.aggregate(group_idx, a, func="nanmean", size=2, dtype=np.float64)
    assert res[0] == float(Fraction(int(ticks[0]) + int(ticks[2]), 2))
    assert res[1] == float(ticks[3])


def test_mean_of_a_spanning_group_with_nat_answers_nat(impl):
    # a NaT and a value of this epoch: the span of the ticks alone would
    # overflow the deviations of the kernel, and the limb sums take over -
    # the poison of the NaT survives the fallback
    group_idx = np.array([0, 0, 1])
    a = np.array(["1970-01-01", "NaT", "2011-01-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="mean", size=2)
    np.testing.assert_array_equal(res, np.array(["NaT", "2011-01-01"], dtype=a.dtype))


# ---------------------------------------------------------------------------
# the order statistics and the sums of the durations


def test_timedelta_median(impl):
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array([120, 31, 59, 1, 2], dtype="m8[s]")
    res = impl.aggregate(group_idx, a, func="median", size=2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(res, np.array([59, 2], dtype=a.dtype))
    # an even group averages its middles exactly, half to even - the tie of
    # (1 + 4) / 2 rounds down to the even neighbour
    res_even = impl.aggregate(np.array([0, 0]), np.array([1, 4], dtype="m8[s]"), func="median", size=1)
    assert int(np.asarray(res_even).view(np.int64)[0]) == 2
    # and the tie the other way round rounds up to the even neighbour
    res_even = impl.aggregate(np.array([0, 0]), np.array([2, 5], dtype="m8[s]"), func="median", size=1)
    assert int(np.asarray(res_even).view(np.int64)[0]) == 4


def test_timedelta_median_nat_poisons(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array([1, "NaT", 7], dtype="m8[s]")
    np.testing.assert_array_equal(
        impl.aggregate(group_idx, a, func="median", size=2), np.array(["NaT", 7], dtype=a.dtype)
    )
    np.testing.assert_array_equal(
        impl.aggregate(group_idx, a, func="nanmedian", size=2), np.array([1, 7], dtype=a.dtype)
    )


@pytest.mark.parametrize("q, want", [(0.0, 31), (0.25, 45), (0.5, 59), (0.75, 89), (1.0, 120)])
def test_timedelta_quantile_linear(impl, q, want):
    # like numpy's linear method: 31 + q * (120 - 31) on the sorted group
    group_idx = np.array([0, 0, 0])
    a = np.array([120, 31, 59], dtype="m8[s]")
    res = impl.aggregate(group_idx, a, func="quantile", q=q, size=1)
    assert int(np.asarray(res).view(np.int64)[0]) == want
    res_pct = impl.aggregate(group_idx, a, func="percentile", q=100 * q, size=1)
    assert int(np.asarray(res_pct).view(np.int64)[0]) == want


@pytest.mark.parametrize("func", ["sum", "nansum"])
def test_timedelta_sum(impl, func):
    group_idx = np.array([0, 0, 1, 1, 2])
    a = np.array([1, 2, "NaT", 5, "NaT"], dtype="m8[s]")
    res = impl.aggregate(group_idx, a, func=func, size=3)
    assert res.dtype == a.dtype
    if func == "sum":
        # numpy adds NaT into a temporal sum, and the answer is NaT
        np.testing.assert_array_equal(res, np.array([3, "NaT", "NaT"], dtype=a.dtype))
    else:
        np.testing.assert_array_equal(res, np.array([3, 5, 0], dtype=a.dtype))


@pytest.mark.parametrize("func", ["cumsum", "nancumsum"])
def test_timedelta_cumsum_propagates_nat(impl, func):
    # numpy propagates NaT through the accumulation of temporal input, the
    # nan variant included
    group_idx = np.array([0, 0, 0, 0])
    a = np.array([1, "NaT", 5, 7], dtype="m8[s]")
    res = impl.aggregate(group_idx, a, func=func, size=1)
    np.testing.assert_array_equal(res, np.array([1, "NaT", "NaT", "NaT"], dtype=a.dtype))


def test_datetime_sort_puts_nat_last(impl):
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "NaT"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="sort", size=2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(res, np.array(["2011-01-01", "2011-01-03", "NaT", "NaT"], dtype="M8[s]"))


def test_timedelta_cummax_implemented_or_skipped(impl):
    group_idx = np.array([0, 0, 0, 0])
    a = np.array([5, "NaT", 1, 7], dtype="m8[s]")
    try:
        res = impl.aggregate(group_idx, a, func="cummax", size=1)
    except NotImplementedError:
        pytest.skip("this backend has no cummax")
    np.testing.assert_array_equal(res, np.array([5, "NaT", "NaT", "NaT"], dtype=a.dtype))


def test_array_collects_the_values(impl):
    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array(["2011-01-01", "NaT", "2011-03-01", "2011-02-01", "NaT"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="array", size=2)
    assert len(res) == 2
    np.testing.assert_array_equal(np.asarray(res[0]), np.array(["2011-01-01", "NaT"], dtype="M8[s]"))
    np.testing.assert_array_equal(np.asarray(res[1]), np.array(["2011-03-01", "2011-02-01", "NaT"], dtype="M8[s]"))


# ---------------------------------------------------------------------------
# the dtype and the fill value of the answer


@pytest.mark.parametrize("unit", ["M8[s]", "M8[ns]", "m8[s]", "m8[ns]"])
def test_unit_of_input_is_unit_of_answer(impl, unit):
    group_idx = np.array([0, 0, 1, 1])
    if unit.startswith("M8"):
        a = np.array(["2011-01-01", "2011-06-01", "NaT", "2011-03-01"], dtype=unit)
    else:
        a = np.array([10, 20, "NaT", 15], dtype=unit)
    for func in ["min", "max", "mean", "first", "last", "sort"]:
        assert impl.aggregate(group_idx, a, func=func, size=2).dtype == np.dtype(unit)


def test_empty_groups_answer_nat_by_default(impl):
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-02-01"], dtype="M8[s]")
    for func in ["min", "max", "mean", "first", "last"]:
        res = impl.aggregate(group_idx, a, func=func, size=4)
        assert np.isnat(res[2])
        assert np.isnat(res[3])


def test_user_fill_value_dates_the_empties(impl):
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-02-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="min", size=2, fill_value=np.datetime64("2000-01-01", "s"))
    assert res[1] == np.datetime64("2000-01-01", "s")


def test_user_fill_value_int_is_a_tick(impl):
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-02-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="min", size=2, fill_value=0)
    assert res[1] == np.datetime64("1970-01-01", "s")


def test_int64_dtype_answers_the_ticks(impl):
    group_idx = np.array([0, 0, 1])
    a = np.array(["2011-01-01", "2011-03-01", "2011-02-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="min", dtype=np.int64, size=2)
    assert res.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(res, np.asarray(a).view(np.int64)[[0, 2]])


def test_cross_unit_dtype_rescales(impl):
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-03-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="min", dtype="M8[ms]", size=1)
    assert res.dtype == np.dtype("M8[ms]")
    assert res[0] == np.datetime64("2011-01-01", "ms")


def test_timedelta_std_is_refused_like_numpy(impl):
    group_idx = np.array([0, 0])
    a = np.array([1, 3], dtype="m8[s]")
    with pytest.raises(TypeError, match="not supported"):
        impl.aggregate(group_idx, a, func="std", size=1)


# ---------------------------------------------------------------------------
# the refusals - numpy raises for all of these as well


@pytest.mark.parametrize(
    "func",
    [
        "sum",
        "prod",
        "cumsum",
        "cumprod",
        "median",
        "quantile",
        "percentile",
        "var",
        "std",
        "skew",
        "kurt",
        "trapezoid",
        "sumofsquares",
    ],
)
def test_datetime_refusals(impl, func):
    kwargs = {"q": 0.5} if func in ("quantile", "percentile") else {}
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-03-01"], dtype="M8[s]")
    with pytest.raises(TypeError, match="not supported"):
        impl.aggregate(group_idx, a, func=func, size=1, **kwargs)


@pytest.mark.parametrize(
    "func",
    [
        "prod",
        "cumprod",
        "var",
        "std",
        "skew",
        "kurt",
        "trapezoid",
        "sumofsquares",
    ],
)
def test_timedelta_refusals(impl, func):
    group_idx = np.array([0, 0])
    a = np.array([1, 3], dtype="m8[s]")
    with pytest.raises(TypeError, match="not supported"):
        impl.aggregate(group_idx, a, func=func, size=1)


def test_custom_callable_sees_the_values(impl):
    # a callable of the caller's own sees the values as they came - and its
    # result comes back in the dtype the call asks for
    if impl_name(impl) != "numpy":
        pytest.skip("the compiled callables of the numba backend take numeric values only")
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-03-01"], dtype="M8[s]")
    seen = impl.aggregate(group_idx, a, func=lambda x: x[0], size=1, dtype="M8[s]")
    assert seen[0] == np.datetime64("2011-01-01", "s")


# ---------------------------------------------------------------------------
# scalar temporal input


def test_timedelta_scalar_sum(impl):
    res = impl.aggregate(np.array([0]), np.timedelta64(5, "s"), func="sum", size=1)
    assert res[0] == np.timedelta64(5, "s")


# ---------------------------------------------------------------------------
# the bare numba kernels
#
# The kernels are the low-level entry of the numba backend: they take flat
# one-dimensional input and answer in compiled code.  The temporal dtypes
# reach them as they are - the comparisons of compiled code order datetime64
# and timedelta64 values, and ``val != val`` detects NaT like a nan - so the
# selection and the reduction kernels aggregate them directly.  The ones that
# cannot (the arithmetic of the moments, the partitions, the truthiness) say
# so by name.


@pytest.mark.parametrize(
    "func, expected",
    [
        # NaT is the smallest tick, so it wins the minimum of its group -
        # numpy answers NaT for the minimum of a group that holds one - and
        # it poisons the maximum of its group
        ("min", ["NaT", "NaT"]),
        ("max", ["NaT", "NaT"]),
        ("nanmin", ["2011-01-01", "2011-02-01"]),
        ("nanmax", ["2011-01-03", "2011-05-01"]),
        ("first", ["2011-01-03", "2011-02-01"]),
        ("last", ["2011-01-01", "2011-05-01"]),
        ("nanfirst", ["2011-01-03", "2011-02-01"]),
        ("nanlast", ["2011-01-01", "2011-05-01"]),
    ],
)
def test_bare_kernel_aggregates_datetime_values(func, expected):
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array(
        ["2011-01-03", "NaT", "2011-01-01", "2011-02-01", "NaT", "2011-05-01"],
        dtype="M8[s]",
    )
    kernel = getattr(aggregate_numba, func)
    res = kernel(group_idx, a, 2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(res, np.array(expected, dtype="M8[s]"))


@pytest.mark.parametrize(
    "func, expected",
    [
        ("argmin", [1, 4]),
        ("argmax", [1, 4]),
        ("nanargmin", [2, 3]),
        ("nanargmax", [0, 5]),
    ],
)
def test_bare_kernel_names_the_nat_position(func, expected):
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array(
        ["2011-01-03", "NaT", "2011-01-01", "2011-02-01", "NaT", "2011-05-01"],
        dtype="M8[s]",
    )
    kernel = getattr(aggregate_numba, func)
    assert list(kernel(group_idx, a, 2)) == expected


@pytest.mark.parametrize("func", ["allnan", "anynan", "len", "nanlen"])
def test_bare_kernel_counts_the_temporal_values(func):
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "2011-02-01", "NaT"], dtype="M8[s]")
    kernel = getattr(aggregate_numba, func)
    res = kernel(group_idx, a, 2)
    expected = {
        "allnan": [False, False],
        "anynan": [True, True],
        "len": [3, 2],
        "nanlen": [2, 1],
    }[func]
    assert list(res) == expected


def test_bare_kernel_sums_the_durations():
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array([5, "NaT", 3, 1, 7], dtype="m8[s]")
    # the plain sum carries the NaT (numpy adds it into the sum), the nan
    # variant skips it
    np.testing.assert_array_equal(aggregate_numba.sum(group_idx, a, 2), np.array(["NaT", 8], dtype="m8[s]"))
    np.testing.assert_array_equal(aggregate_numba.nansum(group_idx, a, 2), np.array([8, 8], dtype="m8[s]"))


def test_bare_kernel_accumulates_the_durations_with_the_nat():
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array([5, "NaT", 3, 1, 7], dtype="m8[s]")
    # the NaT poisons the rest of its own group and of no other one
    np.testing.assert_array_equal(
        aggregate_numba.cumsum(group_idx, a),
        np.array([5, "NaT", "NaT", 1, 8], dtype="m8[s]"),
    )


@pytest.mark.parametrize("func", ["mean", "nanmean", "median", "quantile", "sort", "var", "std"])
def test_bare_kernel_refuses_what_it_cannot_compute(func):
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0])
    a = np.array(["2011-01-03", "2011-01-01"], dtype="M8[s]")
    kernel = getattr(aggregate_numba, func)
    with pytest.raises((TypeError, nb.TypingError), match="takes no datetime64"):
        kernel(group_idx, a, 2)


def test_bare_kernel_refuses_the_mean_of_the_durations():
    # numba's division of durations truncates, and the package rounds the
    # mean of a duration half to even - only aggregate() computes that
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0])
    a = np.array([1, 2], dtype="m8[s]")
    with pytest.raises((TypeError, nb.TypingError), match="takes no timedelta64"):
        aggregate_numba.mean(group_idx, a, 2)


def test_the_bare_kernels_and_the_aggregate_entry_agree():
    # the low-level entry and the high-level one answer the same temporal
    # questions the same way
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 0, 1, 1, 2])
    a = np.array(
        ["2011-01-03", "NaT", "2011-01-01", "2011-02-01", "NaT", "2011-05-01"],
        dtype="M8[s]",
    )
    for func in ["min", "max", "nanmin", "nanmax", "first", "last", "nanfirst", "nanlast"]:
        kernel = getattr(aggregate_numba, func)
        bare = kernel(group_idx, a, 3)
        called = aggregate_numba.aggregate(group_idx, a, func=func, size=3)
        np.testing.assert_array_equal(bare, called)


# ---------------------------------------------------------------------------
# the mean of the ticks
#
# The mean of a temporal input is the mean of its ticks: the number of units
# since the epoch.  Their sum leaves the window float64 represents exactly,
# so the mean is computed on the deviations from the first value of the
# group - the kernel of the numba backend does that in its own pass, and the
# numpy backend falls back to the exact limb sums when the values of a group
# reach too far apart for their sum to stay inside int64.


def test_the_mean_of_the_ticks_is_exact_and_centered(impl):
    # the values of a group are close to each other - their sum fits into
    # int64, and the mean of the ticks is the exact quotient of it
    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array(
        [
            1577836800000000000,
            1577836800000000001,
            1577836800000000002,
            1577836800000000000,
            1577836800000000004,
            1577836800000000000,
        ],
        dtype="M8[ns]",
    )
    res = impl.aggregate(group_idx, a, func="mean", size=2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(
        res, np.array(["2020-01-01T00:00:00.000000001", "2020-01-01T00:00:00.000000001"], dtype=a.dtype)
    )


def test_the_mean_of_the_ticks_keeps_the_unit(impl):
    group_idx = np.array([0, 0, 0])
    a = np.array([1, 2, 4], dtype="m8[ns]")
    res = impl.aggregate(group_idx, a, func="mean", size=1)
    assert res.dtype == a.dtype
    # 7 / 3 = 2.333... - the tick answer rounds half to even, so 2
    np.testing.assert_array_equal(res, np.array([2], dtype="m8[ns]"))


def test_the_parallel_path_of_the_mean_answers_the_same(impl):
    # the parallel path splits the data into chunks, and every chunk measures
    # its deviations from its own origin - the mean must not depend on the
    # origin, and the chunks of the data know their values by their place in
    # the chunk, not by their place in the data.  The data is big enough for
    # the parallel path to run at all.
    rng = np.random.default_rng(3)
    size = 300_000
    groups = 500
    group_idx = rng.integers(0, groups, size).astype(np.int64)
    ticks = rng.integers(2**62, 2**62 + 10**9, size, dtype=np.int64)
    a = ticks.view("M8[ns]")
    # every fifth value is the NaT of its group
    a[::5] = np.datetime64("NaT", "ns")
    parallel = impl.aggregate(group_idx, a, func="mean", size=groups)
    if hasattr(impl, "tick_mean"):
        # the sequential kernel of the numba backend is the yardstick
        sequential = impl.tick_mean(group_idx, ticks, groups)
        np.testing.assert_array_equal(np.asarray(parallel).view(np.int64), sequential)
    # and the answer is the exact mean of the ticks, NaT and all
    for g in range(groups):
        mask = group_idx == g
        values = ticks[mask]
        if not np.isnat(a[mask]).any():
            want = _round_half_even(Fraction(int(values.astype(object).sum()), int(mask.sum())))
            assert int(np.asarray(parallel).view(np.int64)[g]) == want


def test_the_bare_tick_mean_poisons_the_nat_of_a_group():
    # the kernel sees the ticks alone - the tick of the NaT is the smallest
    # one, and the plain kernel takes its group over with it, while the nan
    # kernel leaves it out
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 1, 1])
    ticks = np.array([10, np.iinfo(np.int64).min, 3, 5], dtype=np.int64)
    plain = aggregate_numba.tick_mean(group_idx, ticks, 2)
    np.testing.assert_array_equal(plain, np.array([np.iinfo(np.int64).min, 4], dtype=np.int64))
    nan = aggregate_numba.nantick_mean(group_idx, ticks, 2)
    np.testing.assert_array_equal(nan, np.array([10, 4], dtype=np.int64))


def test_timedelta_median_of_huge_ticks_is_exact(impl):
    # ticks around 2**62 carry more bits than the 53 of a float64 mantissa:
    # the mean of the two middles of an even group rounds the answer off the
    # tick grid where float64 averages it, as np.median and the numeric
    # kernel of the numba backend do - the exact integer mean of the ticks
    # does not
    rng = np.random.default_rng(7)
    group_idx = rng.integers(0, 40, 400).astype(np.int64)
    ticks = rng.integers(2**62, 2**62 + 10**6, 400, dtype=np.int64)
    a = ticks.view("m8[ns]")
    res = impl.aggregate(group_idx, a, func="median", size=40)
    counts = np.bincount(group_idx, minlength=40)
    for g in range(40):
        part = np.sort(ticks[group_idx == g])
        m = counts[g]
        want = (
            int(part[m // 2])
            if m % 2 == 1
            else _round_half_even(Fraction(int(part[m // 2 - 1]) + int(part[m // 2]), 2))
        )
        assert int(np.asarray(res).view(np.int64)[g]) == want
    # and the nan variant answers the same over the NaT it skips
    a[::3] = np.datetime64("NaT", "ns")
    res_nan = impl.aggregate(group_idx, a, func="nanmedian", size=40)
    good = ~np.isnat(a)
    res_nan_ref = impl.aggregate(group_idx[good], a[good], func="nanmedian", size=40)
    np.testing.assert_array_equal(np.asarray(res_nan).view(np.int64), np.asarray(res_nan_ref).view(np.int64))


def test_the_parallel_path_of_the_median_answers_the_same(impl):
    # the groups are cut into ranges the threads work on alone - the answer
    # must not depend on where the cut falls.  The data is big enough for the
    # parallel path to run at all.
    rng = np.random.default_rng(5)
    size = 300_000
    groups = 500
    group_idx = rng.integers(0, groups, size).astype(np.int64)
    ticks = rng.integers(-(2**62), 2**62, size, dtype=np.int64)
    a = ticks.view("m8[ns]")
    a[::5] = np.datetime64("NaT", "ns")
    parallel = impl.aggregate(group_idx, a, func="median", size=groups)
    parallel_nan = impl.aggregate(group_idx, a, func="nanmedian", size=groups)
    if hasattr(impl, "tick_median"):
        # the sequential kernel of the numba backend is the yardstick - the
        # plain one averages the NaT tick like any other, so its groups that
        # hold a NaT are compared as the kernel answers them, and the rest
        # as the plain answer of the aggregation (the NaT of the wrapper)
        nat_groups = np.zeros(groups, dtype=bool)
        np.maximum.at(nat_groups, group_idx[a.view(np.int64) == np.iinfo(np.int64).min], True)
        sequential = impl.tick_median(group_idx, ticks, groups)
        sequential_nan = impl.nantick_median(group_idx, ticks, groups)
        np.testing.assert_array_equal(
            np.asarray(parallel).view(np.int64)[~nat_groups], np.asarray(sequential)[~nat_groups]
        )
        # the groups that hold a NaT answer the NaT of the wrapper
        np.testing.assert_array_equal(
            np.asarray(parallel).view(np.int64)[nat_groups], np.full(int(nat_groups.sum()), np.iinfo(np.int64).min)
        )
        np.testing.assert_array_equal(np.asarray(parallel_nan).view(np.int64), sequential_nan)


def test_the_bare_tick_median_poisons_the_nat_of_a_group():
    # the plain kernel sees the NaT tick as the smallest value and averages
    # it like any other - the caller knows the groups that hold a NaT and
    # overwrites them with the NaT; the nan kernel leaves the NaT out on its
    # own and answers the middle of the ticks that remain
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 1, 1])
    ticks = np.array([np.iinfo(np.int64).min, 10, 3, 5], dtype=np.int64)
    plain = aggregate_numba.tick_median(group_idx, ticks, 2)
    # what the kernel sees: the exact mean of the NaT tick and the 10
    want = _round_half_even(Fraction(int(np.iinfo(np.int64).min) + 10, 2))
    assert int(plain[0]) == want
    assert int(plain[1]) == 4
    nan = aggregate_numba.nantick_median(group_idx, ticks, 2)
    np.testing.assert_array_equal(nan, np.array([10, 4], dtype=np.int64))
    # an even group whose middles sum beyond the window of int64 - the exact
    # average of (2**62, 2**62 + 1) is a half above 2**62, and half to even
    # keeps the even neighbour
    edge = aggregate_numba.nantick_median(np.array([0, 0]), np.array([2**62, 2**62 + 1], dtype=np.int64), 1)
    assert int(edge[0]) == 2**62
