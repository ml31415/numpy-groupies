"""The exact arithmetic of the temporal values.

A temporal value never reaches a kernel as a timestamp: it is taken apart
into its int64 ticks - the number of units since the epoch - and the answer
is put together again.  The ticks are numbers, and their sum leaves the
window float64 represents exactly, so the mean of a group is computed on the
deviations from its first value - the kernel of the numba backend does that
in its own pass, and the numpy backend falls back to the exact limb sums when
the values of a group reach too far apart for their sum to stay inside
int64.

The python integers are the yardstick here: the exact quotient of the exact
total, rounded half to even.
"""

from fractions import Fraction

import numpy as np
import pytest

from numpy_groupies import npg_numba
from numpy_groupies.npg_numba.specials import _tick_pair_average
from numpy_groupies.tests.helpers import impl_name

pytestmark = pytest.mark.usefixtures("skip_purepy")

# ---------------------------------------------------------------------------
# the exact mean


def _round_half_even(fr):
    q, r = divmod(fr.numerator, fr.denominator)
    if 2 * r > fr.denominator or (2 * r == fr.denominator and q % 2 == 1):
        q += 1
    return q


def test_average_of_two_ticks_is_exact_over_whole_int64_range():
    # the mean of the two middles of an even group must not take the sum of
    # the two: pairs at the ends of the int64 range sum past it, and the
    # wraparound answers a tick of no value of the group.  The average is
    # built from the bits the two share and the bits they differ in, and it
    # rounds the half remainder to the even neighbour - here it is checked
    # against the python integers, over the corners of the range, the ties
    # of half to even and twenty thousand random pairs.

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


def test_mean_default_dtype_is_float_of_exact_quotient(impl):
    # an explicitly requested float64 holds the exact quotient, not the
    # rounded tick
    ticks = np.array([2**62, 2**62, 2**62 + 1], dtype=np.int64)
    a = ticks.view("M8[ns]")
    res = impl.aggregate(np.array([0, 0, 0]), a, func="mean", size=1, dtype=np.float64)
    assert res[0] == float(Fraction(int(ticks.astype(object).sum()), 3))


def test_mean_float_dtype_poisons_nat_groups(impl):
    # the float answer of a group a NaT poisons is no quotient at all - nan
    ticks = np.array([2**62, 2**62 + 2, 0, 2**62], dtype=np.int64)
    a = ticks.view("M8[ns]")
    a[1] = np.datetime64("NaT", "ns")
    group_idx = np.array([0, 0, 1, 1])
    res = impl.aggregate(group_idx, a, func="mean", size=2, dtype=np.float64)
    assert res[1] == float(Fraction(int(ticks[2]) + int(ticks[3]), 2))
    assert res[0] != res[0]  # nan


def test_nanmean_float_dtype_leaves_nat_out(impl):
    # the nan variant of the float answer averages the values it has
    ticks = np.array([2**62, 2**62 + 4, 2**62 + 8, 2**62], dtype=np.int64)
    a = ticks.view("M8[ns]")
    a[1] = np.datetime64("NaT", "ns")
    group_idx = np.array([0, 0, 0, 1])
    res = impl.aggregate(group_idx, a, func="nanmean", size=2, dtype=np.float64)
    assert res[0] == float(Fraction(int(ticks[0]) + int(ticks[2]), 2))
    assert res[1] == float(ticks[3])


def test_mean_of_spanning_group_with_nat_answers_nat(impl):
    # a NaT and a value of this epoch: the span of the ticks alone would
    # overflow the deviations of the kernel, and the limb sums take over -
    # the poison of the NaT survives the fallback
    group_idx = np.array([0, 0, 1])
    a = np.array(["1970-01-01", "NaT", "2011-01-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="mean", size=2)
    np.testing.assert_array_equal(res, np.array(["NaT", "2011-01-01"], dtype=a.dtype))


# ---------------------------------------------------------------------------
# the mean of the ticks
#
# The mean of a temporal input is the mean of its ticks: the number of units
# since the epoch.  Their sum leaves the window float64 represents exactly,
# so the mean is computed on the deviations from the first value of the
# group - the kernel of the numba backend does that in its own pass, and the
# numpy backend falls back to the exact limb sums when the values of a group
# reach too far apart for their sum to stay inside int64.


def test_mean_of_ticks_is_exact_and_centered(impl):
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


def test_mean_of_ticks_keeps_unit(impl):
    group_idx = np.array([0, 0, 0])
    a = np.array([1, 2, 4], dtype="m8[ns]")
    res = impl.aggregate(group_idx, a, func="mean", size=1)
    assert res.dtype == a.dtype
    # 7 / 3 = 2.333... - the tick answer rounds half to even, so 2
    np.testing.assert_array_equal(res, np.array([2], dtype="m8[ns]"))


def test_parallel_path_of_mean_answers_same(impl):
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
        sequential = npg_numba.ops._tick_ops["mean"](group_idx, ticks, groups)
        np.testing.assert_array_equal(np.asarray(parallel).view(np.int64), sequential)
    # and the answer is the exact mean of the ticks, NaT and all
    for g in range(groups):
        mask = group_idx == g
        values = ticks[mask]
        if not np.isnat(a[mask]).any():
            want = _round_half_even(Fraction(int(values.astype(object).sum()), int(mask.sum())))
            assert int(np.asarray(parallel).view(np.int64)[g]) == want


def test_bare_tick_mean_poisons_nat_of_group():
    # the kernel sees the ticks alone - the tick of the NaT is the smallest
    # one, and the plain kernel takes its group over with it, while the nan
    # kernel leaves it out
    group_idx = np.array([0, 0, 1, 1])
    ticks = np.array([10, np.iinfo(np.int64).min, 3, 5], dtype=np.int64)
    plain = npg_numba.ops._tick_ops["mean"](group_idx, ticks, 2)
    np.testing.assert_array_equal(plain, np.array([np.iinfo(np.int64).min, 4], dtype=np.int64))
    nan = npg_numba.ops._tick_ops["nanmean"](group_idx, ticks, 2)
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


def test_parallel_path_of_median_answers_same(impl):
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
        nat_groups = np.zeros(groups, dtype=bool)
        np.maximum.at(nat_groups, group_idx[a.view(np.int64) == np.iinfo(np.int64).min], True)
        sequential = npg_numba.ops._tick_ops["median"](group_idx, ticks, groups)
        sequential_nan = npg_numba.ops._tick_ops["nanmedian"](group_idx, ticks, groups)
        np.testing.assert_array_equal(
            np.asarray(parallel).view(np.int64)[~nat_groups], np.asarray(sequential)[~nat_groups]
        )
        # the groups that hold a NaT answer the NaT of the wrapper
        np.testing.assert_array_equal(
            np.asarray(parallel).view(np.int64)[nat_groups], np.full(int(nat_groups.sum()), np.iinfo(np.int64).min)
        )
        np.testing.assert_array_equal(np.asarray(parallel_nan).view(np.int64), sequential_nan)


def test_parallel_path_of_quantile_answers_same(impl):
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
    for func in ("quantile", "nanquantile"):
        res = impl.aggregate(group_idx, a, func=func, q=0.5, size=groups)
        # the nan variant answers the same as the plain one over the values
        # that are left - and the plain one answers NaT for the groups that
        # hold a NaT
        good = ~np.isnat(a)
        ref = impl.aggregate(group_idx[good], a[good], func="quantile", q=0.5, size=groups)
        nat_groups = np.zeros(groups, dtype=bool)
        np.maximum.at(nat_groups, group_idx[~good], True)
        got = np.asarray(res).view(np.int64)
        if func == "quantile":
            assert np.isnat(np.asarray(res).view("m8[ns]")[nat_groups]).all()
            assert np.array_equal(got[~nat_groups], np.asarray(ref).view(np.int64)[~nat_groups])
        else:
            assert np.array_equal(got, np.asarray(ref).view(np.int64))


def test_bare_tick_median_poisons_nat_of_group():
    # the plain kernel takes the group over, the way the plain kernel of the
    # numbers takes the group of a nan over - the answer of a group that
    # holds a NaT is the NaT of the unit; the nan kernel leaves the NaT out
    # on its own and answers the middle of the ticks that remain
    group_idx = np.array([0, 0, 1, 1])
    ticks = np.array([np.iinfo(np.int64).min, 10, 3, 5], dtype=np.int64)
    plain = npg_numba.ops._tick_ops["median"](group_idx, ticks, 2)
    # the NaT of the unit
    assert int(plain[0]) == np.iinfo(np.int64).min
    assert int(plain[1]) == 4
    nan = npg_numba.ops._tick_ops["nanmedian"](group_idx, ticks, 2)
    np.testing.assert_array_equal(nan, np.array([10, 4], dtype=np.int64))
    # an even group whose middles sum beyond the window of int64 - the exact
    # average of (2**62, 2**62 + 1) is a half above 2**62, and half to even
    # keeps the even neighbour
    edge = npg_numba.ops._tick_ops["nanmedian"](np.array([0, 0]), np.array([2**62, 2**62 + 1], dtype=np.int64), 1)
    assert int(edge[0]) == 2**62
