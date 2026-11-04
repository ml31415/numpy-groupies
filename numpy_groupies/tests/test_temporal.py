"""The datetime64 and timedelta64 support of the numpy and numba backends.

Temporal input never reaches the kernels as datetime values: the wrapper of
``temporal`` takes it apart into its int64 ticks and turns the results back.
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
from numpy_groupies.tests.reference import aggregate_ref

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


def test_the_plain_argreductions_of_a_temporal_group_follow_numpy(impl):
    # numpy answers the position of the first NaT of a group for both
    # argreductions - the group holding one has no minimum or maximum to
    # name, and the NaT is the value it names.  The package follows that,
    # like it follows numpy everywhere else.
    group_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    a = np.array(
        ["2020-03-15", "NaT", "2020-01-01", "2020-05-05", "2020-02-02", "2020-04-04", "2020-06-06", "NaT"],
        dtype="datetime64[s]",
    )
    # the NaT of the group 0 sits at the position 1, the one of the group 2 at
    # the position 7 - and the group 1 holds none, so it answers the position
    # of its own minimum and maximum
    for func in ("argmin", "argmax"):
        ours = list(impl.aggregate(group_idx, a, func=func, size=3))
        numpy = [int(getattr(np, func)(a[group_idx == g])) + int(np.flatnonzero(group_idx == g)[0]) for g in range(3)]
        assert ours == numpy


def test_the_nan_argreductions_of_a_temporal_group_skip_the_nat(impl):
    # the nan forms skip the missing value, and that is a deliberate
    # deviation from numpy: np.nanargmin and np.nanargmax do not skip a NaT
    # of a datetime64, they answer its position like the plain forms, while
    # np.nanmin and np.nanmax of the very same dtype do skip it.  numpy
    # closed the report of that inconsistency as fixed (numpy/numpy#5222),
    # while its own closing output still shows the nanarg forms answering
    # the NaT position where the float nanarg skips - so the contract of the
    # nan forms wins here: they skip, for every dtype.
    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array(["2020-03-15", "NaT", "2020-01-01", "2020-05-05", "2020-02-02", "2020-04-04"], dtype="datetime64[s]")
    # the nan forms name the extreme among the values that are left: the
    # smallest of the group 0 is 2020-01-01 at the position 2, the largest
    # is 2020-03-15 at the position 0 - and the group 1 holds no NaT, so it
    # names the position of its own extremes, 2020-02-02 and 2020-05-05
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [2, 4]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [0, 3]
    # and the float nanarg skips the nan as well - the two agree
    f = np.array([2.0, np.nan, 1.0, 4.0, 3.0, 5.0])
    for func in ("nanargmin", "nanargmax"):
        ours = list(impl.aggregate(group_idx, f, func=func, size=2))
        numpy = [int(getattr(np, func)(f[group_idx == g])) + int(np.flatnonzero(group_idx == g)[0]) for g in range(2)]
        assert ours == numpy


def test_the_nan_argreductions_of_a_temporal_group_differ_from_numpy(impl):
    # the deviation itself, pinned: numpy answers the position of the NaT
    # for the nanarg forms of a datetime64, and the package names the
    # extreme among the values that are left.  If numpy ever changes that,
    # this test says so - and the deviation can go with it.
    group_idx = np.array([0, 0, 0])
    a = np.array(["2020-03-15", "NaT", "2020-01-01"], dtype="datetime64[s]")
    assert np.nanargmin(a) == 1  # numpy names the NaT
    assert np.nanmin(a) == np.datetime64("2020-01-01", "s")  # and skips it here
    # the package skips it in both - the nanarg form names the smallest of
    # the values that are left, the position 2
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=1)) == [2]
    # the plain form follows numpy and names the NaT
    assert list(impl.aggregate(group_idx, a, func="argmin", size=1)) == [1]


@pytest.mark.parametrize("trial", range(25))
def test_the_nan_argreductions_of_a_temporal_group_match_the_reference(impl, trial):
    # the reference of the suite computes the nan argreductions the way the
    # contract of the nan forms says: it filters the missing values out and
    # takes the plain argmin over the values that are left, at the positions
    # they sit at in the input - a construction numpy's own nanargmin of a
    # datetime64 cannot stand for, since it does not skip.  Both backends
    # match it over data drawn at random, NaT included, whole NaT groups
    # included - those answer the fill value on both sides.
    rng = np.random.default_rng(11 + trial)
    n = int(rng.integers(1, 30))
    groups = int(rng.integers(1, 6))
    group_idx = rng.integers(0, groups, n)
    a = rng.integers(-(10**7), 10**7, n).astype("M8[s]")
    a[rng.random(n) < 0.3] = np.datetime64("NaT", "s")
    for func in ("nanargmin", "nanargmax"):
        ours = np.asarray(impl.aggregate(group_idx, a, func=func, size=groups))
        want = np.asarray(aggregate_ref(group_idx, a, func=func, size=groups))
        np.testing.assert_array_equal(ours, want)


def test_the_nan_argreductions_of_a_temporal_group_answer_the_fill_of_an_empty_one(impl):
    # a group that holds nothing but NaT has nothing to name, and answers the
    # fill value - like the plain forms of a group without any item
    group_idx = np.array([0, 0, 1, 1])
    a = np.array(["NaT", "NaT", "2020-01-01", "2020-02-01"], dtype="datetime64[s]")
    # the group 0 holds nothing but NaT and answers the fill value - the
    # group 1 holds 2020-01-01 at the position 2 and 2020-02-01 at the
    # position 3, so it names the one of its extremes
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [-1, 2]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [-1, 3]


def test_the_nan_argreductions_of_a_temporal_group_keep_the_positions_of_the_input(impl, request):
    # the positions the nan argreductions name are the ones of the input -
    # the values they skip do not shift them
    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array(
        ["NaT", "2020-01-01", "2020-03-01", "NaT", "2020-02-01"],
        dtype="datetime64[s]",
    )
    # the group 0 holds 2020-01-01 at the position 1, the group 1 holds
    # 2020-02-01 at the position 4 - the NaT of the group 1 at the position 3
    # is skipped, and the position of 2020-03-01 at the position 2 is the
    # largest of the values that are left
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [1, 4]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [1, 2]
    # the numpy backend stands in a sentinel for the NaT of a row while it
    # reduces - the smallest tick for the nanargmax and the largest one for
    # the nanargmin - and both sentinels are timestamps of their own: the
    # largest ns tick is 2262-04-11T23:47:16.854775807, the smallest one
    # 1677-09-21T00:12:43.145224193.  A group that holds such a value gets
    # the position of the NaT answered where the tie of the sentinel keeps
    # the first of the equal values.  The numba backend has no sentinel: it
    # skips the NaT where it finds it.
    if impl_name(impl) == "numpy":
        # the mark is strict: should the backend answer the right positions
        # one day, the test fails and asks for the mark to go - a silent
        # xfail would hide the fix
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="the sentinels of the numpy backend collide with timestamps of their own",
            )
        )
    hi, lo = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    group_idx = np.array([0, 0, 1, 1])
    a = np.array(
        [np.datetime64(hi, "ns"), "NaT", "NaT", np.datetime64(lo + 1, "ns")],
        dtype="M8[ns]",
    )
    # the nanargmin of the group 0 names the position of the largest tick -
    # the one the sentinel of the backend would stand in for - and the
    # nanargmax of the group 1 the one of the smallest tick but the NaT
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [0, 3]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [0, 3]


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


def test_the_average_of_two_ticks_is_exact_over_the_whole_int64_range():
    # the mean of the two middles of an even group must not take the sum of
    # the two: pairs at the ends of the int64 range sum past it, and the
    # wraparound answers a tick of no value of the group.  The average is
    # built from the bits the two share and the bits they differ in, and it
    # rounds the half remainder to the even neighbour - here it is checked
    # against the python integers, over the corners of the range, the ties
    # of half to even and twenty thousand random pairs.
    from numpy_groupies.aggregate_numba.specials import _tick_pair_average

    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max

    def want(a: int, b: int) -> int:
        # the reference reads the sum as a python integer, where nothing wraps
        return _round_half_even(Fraction(a + b, 2))

    corners = [lo, lo + 1, lo + 2, -2, -1, 0, 1, 2, 2**61 - 1, 2**61, 2**62 - 1, 2**62, 2**62 + 1, hi - 1, hi]
    for a in corners:
        for b in corners:
            assert int(_tick_pair_average(np.int64(a), np.int64(b))) == want(a, b)
    # the ties, spelled out: a half rounds to the even of the two neighbours
    for a, b, mean in [(-1, 2, 0), (-2, 1, 0), (1, 2, 2), (2, 3, 2), (-3, -2, -2), (0, 1, 0), (-1, 0, 0)]:
        assert int(_tick_pair_average(np.int64(a), np.int64(b))) == mean, (a, b)
    # the two ends of the range together, and each with itself - the sum of
    # the pair would wrap in both cases
    assert int(_tick_pair_average(np.int64(lo), np.int64(hi))) == 0
    assert int(_tick_pair_average(np.int64(hi), np.int64(hi))) == hi
    assert int(_tick_pair_average(np.int64(lo), np.int64(lo))) == lo
    # and twenty thousand pairs drawn over the whole range
    rng = np.random.default_rng(11)
    first = [int(v) for v in rng.integers(lo, hi, 20_000, dtype=np.int64)]
    second = [int(v) for v in rng.integers(lo, hi, 20_000, dtype=np.int64)]
    for a, b in zip(first, second):
        assert int(_tick_pair_average(np.int64(a), np.int64(b))) == want(a, b), (a, b)


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


@pytest.mark.parametrize("q, want", [(0.0, 31), (0.25, 45), (0.5, 59), (0.75, 90), (1.0, 120)])
def test_timedelta_quantile_linear(impl, q, want):
    # numpy's linear method on the sorted group [31, 59, 120]: the quantile
    # of 0.75 falls between the 59 and the 120, and numpy takes the upper
    # one minus the half of the difference - 120 - 30 = 90
    group_idx = np.array([0, 0, 0])
    a = np.array([120, 31, 59], dtype="m8[s]")
    res = impl.aggregate(group_idx, a, func="quantile", q=q, size=1)
    assert int(np.asarray(res).view(np.int64)[0]) == want
    assert int(np.asarray(res).view(np.int64)[0]) == int(np.quantile(a, q).view("m8[s]").astype(np.int64))
    res_pct = impl.aggregate(group_idx, a, func="percentile", q=100 * q, size=1)
    assert int(np.asarray(res_pct).view(np.int64)[0]) == want


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "nearest", "midpoint"])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 1 / 3, 0.5, 0.6, 0.75, 0.9, 1.0])
def test_the_temporal_quantile_answers_like_numpy(impl, method, q):
    # numpy computes the quantile of a temporal value (numpy/numpy#11620,
    # closed with numpy/numpy#11627): it interpolates between the two ticks
    # the quantile falls between, in the arithmetic of the ticks - the
    # difference of the two is a duration, its product with the fraction is
    # truncated towards zero, and the answer is a tick again.  The two
    # backends answer the same as numpy does, tick for tick, over values of
    # every size - the ticks of a nanosecond timestamp are around 1.6e18,
    # far beyond the window a float64 holds exactly.
    rng = np.random.default_rng(5)
    group_idx = rng.integers(0, 3, 24)
    scale = 10**18
    ticks = rng.integers(-scale, scale, 24).astype(np.int64)
    for unit in ("M8[ns]", "m8[ns]"):
        a = ticks.view(unit)
        res = impl.aggregate(group_idx, a, func="quantile", q=q, method=method, size=3)
        for g in range(3):
            mask = group_idx == g
            if not mask.any():
                continue
            want = np.quantile(a[mask], q, method=method)
            assert res[g] == want, (unit, method, q, g)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "nearest", "midpoint"])
def test_the_temporal_percentile_answers_like_numpy(impl, method):
    # the percentile is the quantile of the fraction of a hundred
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2020-01-01", "2020-05-01", "2020-03-01", "2020-02-01", "2020-06-01"], dtype="M8[s]")
    for q in (0.0, 25.0, 50.0, 75.0, 100.0):
        res = impl.aggregate(group_idx, a, func="percentile", q=q, method=method, size=2)
        for g in range(2):
            mask = group_idx == g
            want = np.percentile(a[mask], q, method=method)
            assert res[g] == want, (method, q, g)


def test_the_datetime_quantile_propagates_the_nat(impl):
    # the plain quantile of numpy propagates the NaT of its group, the nan
    # variant leaves it out
    group_idx = np.array([0, 0, 0, 1, 1, 1])
    a = np.array(["2020-01-01", "NaT", "2020-03-01", "2020-01-01", "2020-02-01", "2020-04-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="quantile", q=0.5, size=2)
    assert np.isnat(res[0])
    assert res[1] == np.datetime64("2020-02-01", "s")
    res_nan = impl.aggregate(group_idx, a, func="nanquantile", q=0.5, size=2)
    assert res_nan[0] == np.datetime64("2020-01-31", "s")  # the middle of the two values that are left
    assert res_nan[1] == np.datetime64("2020-02-01", "s")


def test_the_temporal_quantile_refuses_an_unknown_method(impl):
    # the methods numpy knows are the ones the package knows
    group_idx = np.array([0, 0])
    a = np.array(["2020-01-01", "2020-02-01"], dtype="M8[s]")
    with pytest.raises(ValueError, match="unknown interpolation method"):
        impl.aggregate(group_idx, a, func="quantile", q=0.5, method="weibull", size=1)


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
        "var",
        "std",
        "skew",
        "kurt",
        "trapezoid",
        "sumofsquares",
    ],
)
def test_datetime_refusals(impl, func):
    # the quantile and the percentile are not among them: numpy computes
    # them of a datetime64 (numpy/numpy#11620, closed with
    # numpy/numpy#11627), while the median it cannot - it adds two
    # middles, and numpy has no addition of datetime64 values
    kwargs = {"q": 0.5} if func in ("quantile", "percentile") else {}
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-03-01"], dtype="M8[s]")
    ops = getattr(impl, "ops", None)
    if ops is not None and ops._op_dict.get(func) is None:
        # a function the backend does not implement is one it cannot refuse
        # for a dtype either - the missing implementation is the answer
        with pytest.raises(NotImplementedError):
            impl.aggregate(group_idx, a, func=func, size=1, **kwargs)
    else:
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
    ops = getattr(impl, "ops", None)
    if ops is not None and ops._op_dict.get(func) is None:
        # a function the backend does not implement is one it cannot refuse
        # for a dtype either - the missing implementation is the answer
        with pytest.raises(NotImplementedError):
            impl.aggregate(group_idx, a, func=func, size=1)
    else:
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


@pytest.mark.parametrize("func", ["sort", "var", "std"])
def test_bare_kernel_refuses_what_it_cannot_compute(func):
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0])
    a = np.array(["2011-01-03", "2011-01-01"], dtype="M8[s]")
    kernel = getattr(aggregate_numba, func)
    with pytest.raises((TypeError, nb.TypingError), match="takes no datetime64"):
        kernel(group_idx, a, 2)


@pytest.mark.parametrize(
    "func,args,expected",
    [
        # the mean of the ticks, rounded half to even - the mean of the
        # values themselves would be the business of a division of durations,
        # which numba truncates
        ("mean", (2,), "2011-01-02T00:00:00"),
        ("nanmean", (2,), "2011-01-02T00:00:00"),
        ("median", (2,), "2011-01-02T00:00:00"),
        ("nanmedian", (2,), "2011-01-02T00:00:00"),
        ("quantile", (0.5, "linear", 2), "2011-01-02T00:00:00"),
        ("nanquantile", (0.5, "linear", 2), "2011-01-02T00:00:00"),
        ("percentile", (50.0, "linear", 2), "2011-01-02T00:00:00"),
        ("nanpercentile", (50.0, "linear", 2), "2011-01-02T00:00:00"),
    ],
)
def test_bare_kernel_answers_the_temporal_input_on_the_ticks(func, args, expected):
    """The kernel of the mean, the median and the quantile answers the
    temporal input: it hands the ticks to the kernel of the ticks and views
    the answer back into the unit of the input."""
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0])
    a = np.array(["2011-01-03", "2011-01-01"], dtype="M8[s]")
    kernel = getattr(aggregate_numba, func)
    res = kernel(group_idx, a, *args)
    assert res.dtype == a.dtype
    # the second group holds no value - it answers the NaT of the unit
    np.testing.assert_array_equal(res, np.array([expected, "NaT"], dtype="M8[s]"))


def test_bare_kernel_answers_the_mean_of_the_durations_on_the_ticks():
    # numba's division of durations truncates, and the package rounds the
    # mean of a duration half to even - the kernel of the mean answers the
    # durations on their ticks, where the rounding is the one of numpy
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0])
    a = np.array([1, 2], dtype="m8[s]")
    res = aggregate_numba.mean(group_idx, a, 2)
    assert res.dtype == a.dtype
    # the mean of the ticks 1 and 2 is 1.5 - rounded half to even, that is 2
    np.testing.assert_array_equal(res, np.array([2, "NaT"], dtype="m8[s]"))


def test_the_jitted_kernel_answers_the_temporal_input():
    """The kernel of the mean is called from jitted code with temporal
    values - the ticks are the int64 the kernel works on."""
    from numpy_groupies import aggregate_numba

    @nb.njit
    def jitted_mean(group_idx, values):
        return aggregate_numba.mean(group_idx, values)

    group_idx = np.array([0, 0])
    a = np.array(["2011-01-03", "2011-01-01"], dtype="M8[s]")
    res = jitted_mean(group_idx, a)
    np.testing.assert_array_equal(res, np.array(["2011-01-02T00:00:00"], dtype="M8[s]"))


@pytest.mark.parametrize("unit", ["M8[s]", "m8[s]"])
def test_the_sort_of_numba_makes_no_move_when_a_nat_is_in_the_values(unit):
    # why the bare sort kernel refuses temporal input: the sort of numba
    # itself does not survive a NaT.  Its comparison makes the NaT
    # incomparable - both < and > answer False where a value takes part -
    # and the sort makes no move at all, it hands the input back, while
    # numpy sorts the NaT to the end of the array (numpy/numpy#12629,
    # fixed by numpy/numpy#12658).  Without a NaT the sort of numba is
    # correct, and its partition answers the NaT the way numpy does - so
    # the machinery is there, the comparison of the sort just does not use
    # it.  This test is the watchman of the fix: when numba sorts a
    # temporal array holding a NaT like numpy does, the assertion about the
    # untouched input fails - and the bare sort kernel of this package can
    # take the values then.
    from numpy_groupies import aggregate_numba

    @nb.njit
    def nb_sort(x):
        return np.sort(x)

    @nb.njit
    def nb_argsort(x):
        return np.argsort(x)

    @nb.njit
    def nb_lt(a, b):
        return a < b

    # the NaT of the unit of the values: the constructor of the dtype would
    # build a NaT without a unit, and that is deprecated in numpy
    nat = np.array("NaT", dtype=unit)[()]
    a = np.array([30, 10, 20], dtype=np.int64).view(unit)
    a[1] = nat
    # the comparison of numba: the NaT is incomparable, both ways False
    assert not nb_lt(a[1], a[0])
    assert not nb_lt(a[0], a[1])
    # the sort makes no move - the input comes back
    np.testing.assert_array_equal(nb_sort(a).view(np.int64), a.view(np.int64))
    assert list(nb_argsort(a)) == [0, 1, 2]
    # and numpy sorts the NaT to the end, where the values come first
    np.testing.assert_array_equal(np.sort(a).view(np.int64), np.array([20, 30, np.iinfo(np.int64).min], dtype=np.int64))
    # the refusal of the kernel says what aggregate() does instead - the
    # degeneration of the sort is reported upstream as numba/numba#10981,
    # and when it is fixed, this test is the one that notices
    group_idx = np.array([0, 0, 0])
    kind = "datetime64" if unit.startswith("M8") else "timedelta64"
    with pytest.raises((TypeError, nb.TypingError), match="takes no " + kind):
        aggregate_numba.sort(group_idx, a, 1)


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
    if impl_name(impl) == "numba":
        # the sequential kernel of the numba backend is the yardstick
        from numpy_groupies import aggregate_numba

        sequential = aggregate_numba.ops._tick_ops["mean"](group_idx, ticks, groups)
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
    plain = aggregate_numba.ops._tick_ops["mean"](group_idx, ticks, 2)
    np.testing.assert_array_equal(plain, np.array([np.iinfo(np.int64).min, 4], dtype=np.int64))
    nan = aggregate_numba.ops._tick_ops["nanmean"](group_idx, ticks, 2)
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
    if impl_name(impl) == "numba":
        # the sequential kernel of the numba backend is the yardstick - the
        # plain one averages the NaT tick like any other, so its groups that
        # hold a NaT are compared as the kernel answers them, and the rest
        # as the plain answer of the aggregation (the NaT of the wrapper)
        from numpy_groupies import aggregate_numba

        nat_groups = np.zeros(groups, dtype=bool)
        np.maximum.at(nat_groups, group_idx[a.view(np.int64) == np.iinfo(np.int64).min], True)
        sequential = aggregate_numba.ops._tick_ops["median"](group_idx, ticks, groups)
        sequential_nan = aggregate_numba.ops._tick_ops["nanmedian"](group_idx, ticks, groups)
        np.testing.assert_array_equal(
            np.asarray(parallel).view(np.int64)[~nat_groups], np.asarray(sequential)[~nat_groups]
        )
        # the groups that hold a NaT answer the NaT of the wrapper
        np.testing.assert_array_equal(
            np.asarray(parallel).view(np.int64)[nat_groups], np.full(int(nat_groups.sum()), np.iinfo(np.int64).min)
        )
        np.testing.assert_array_equal(np.asarray(parallel_nan).view(np.int64), sequential_nan)


def test_the_parallel_path_of_the_quantile_answers_the_same(impl):
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
    for func, plain in (("quantile", "quantile"), ("nanquantile", "quantile")):
        res = impl.aggregate(group_idx, a, func=func, q=0.5, size=groups)
        # the nan variant answers the same as the plain one over the values
        # that are left - and the plain one answers NaT for the groups that
        # hold a NaT
        good = ~np.isnat(a)
        ref = impl.aggregate(group_idx[good], a[good], func=plain, q=0.5, size=groups)
        nat_groups = np.zeros(groups, dtype=bool)
        np.maximum.at(nat_groups, group_idx[~good], True)
        got = np.asarray(res).view(np.int64)
        if func == "quantile":
            assert np.isnat(np.asarray(res).view("m8[ns]")[nat_groups]).all()
            assert np.array_equal(got[~nat_groups], np.asarray(ref).view(np.int64)[~nat_groups])
        else:
            assert np.array_equal(got, np.asarray(ref).view(np.int64))


def test_the_bare_tick_median_poisons_the_nat_of_a_group():
    # the plain kernel takes the group over, the way the plain kernel of the
    # numbers takes the group of a nan over - the answer of a group that
    # holds a NaT is the NaT of the unit; the nan kernel leaves the NaT out
    # on its own and answers the middle of the ticks that remain
    from numpy_groupies import aggregate_numba

    group_idx = np.array([0, 0, 1, 1])
    ticks = np.array([np.iinfo(np.int64).min, 10, 3, 5], dtype=np.int64)
    plain = aggregate_numba.ops._tick_ops["median"](group_idx, ticks, 2)
    # the NaT of the unit
    assert int(plain[0]) == np.iinfo(np.int64).min
    assert int(plain[1]) == 4
    nan = aggregate_numba.ops._tick_ops["nanmedian"](group_idx, ticks, 2)
    np.testing.assert_array_equal(nan, np.array([10, 4], dtype=np.int64))
    # an even group whose middles sum beyond the window of int64 - the exact
    # average of (2**62, 2**62 + 1) is a half above 2**62, and half to even
    # keeps the even neighbour
    edge = aggregate_numba.ops._tick_ops["nanmedian"](np.array([0, 0]), np.array([2**62, 2**62 + 1], dtype=np.int64), 1)
    assert int(edge[0]) == 2**62
