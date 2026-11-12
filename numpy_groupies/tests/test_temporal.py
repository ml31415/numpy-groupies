"""The datetime64 and timedelta64 support of the numpy and numba backends.

Temporal input never reaches the kernels as datetime values: the wrapper of
``temporal`` takes it apart into its int64 ticks and turns the results back.
The tests pin the semantics that emerge - the NaT propagation of the plain
reductions, the skipping of the nan variants, the order statistics of the
durations, the unit of the answer, and the refusals numpy shares - and they
run them over the aggregate entry points of both numpy based backends.  The
pure python backend knows no temporal types: without numpy there are none, so
its parameterizations are skipped here.

The exact arithmetic of the ticks has a module of its own
(:mod:`.test_temporal_ticks`), and the bare kernels of the numba backend
another (:mod:`.test_temporal_kernels`).
"""

import numpy as np
import pytest

from numpy_groupies.tests.helpers import impl_name
from numpy_groupies.tests.reference import aggregate_ref

pytestmark = pytest.mark.usefixtures("skip_purepy")

# the datetime64 cases run on a coarse unit and on nanoseconds - the ns ticks
# are the ones float64 cannot represent exactly
DTYPES = ["M8[s]", "M8[ns]", "m8[s]", "m8[ns]"]


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
def test_mode_skips_nat(impl, unit):
    """The mode has no nan variant, and the NaT is in no selection: it is
    never the most frequent value of a group, and a group of nothing but NaT
    has no mode at all - it keeps the fill value, which is the NaT of the
    unit for a value function like this one."""
    group_idx = np.array([0, 0, 0, 0, 1, 1])
    a = np.array(["2011-01-01", "NaT", "2011-01-01", "2011-01-03", "NaT", "NaT"], dtype=f"M8[{unit}]")
    res = impl.aggregate(group_idx, a, func="mode", size=2)
    assert res.dtype == a.dtype
    np.testing.assert_array_equal(res, np.array(["2011-01-01", "NaT"], dtype=a.dtype))


def test_mode_of_temporal_input_can_be_asked_for_in_ticks(impl):
    """The ticks the caller asked for are the ticks the mode counted."""
    group_idx = np.array([0, 0, 0, 1])
    a = np.array(["2011-01-01", "2011-01-01", "2011-01-03", "2011-01-05"], dtype="M8[s]")
    ticks = impl.aggregate(group_idx, a, func="mode", size=2, dtype=np.int64)
    np.testing.assert_array_equal(ticks, np.array([1293840000, 1294185600], dtype=np.int64))


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_argmin_and_argmax_name_nat_position(impl, unit):
    # numpy answers the position of the first NaT of a group for both
    # argreductions - as the smallest tick for argmin, and through the plan
    # for argmax - and the nan variants skip it
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2011-01-03", "NaT", "2011-01-01", "NaT", "2011-02-01"], dtype=f"M8[{unit}]")
    assert list(impl.aggregate(group_idx, a, func="argmin", size=2)) == [1, 3]
    assert list(impl.aggregate(group_idx, a, func="argmax", size=2)) == [1, 3]
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [2, 4]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [0, 4]


def test_plain_argreductions_of_temporal_group_follow_numpy(impl):
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


def test_nan_argreductions_of_temporal_group_skip_nat(impl):
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


def test_nan_argreductions_of_temporal_group_differ_from_numpy(impl):
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
def test_nan_argreductions_of_temporal_group_match_reference(impl, trial):
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


def test_nan_argreductions_of_temporal_group_answer_fill_of_empty_one(impl):
    # a group that holds nothing but NaT has nothing to name, and answers the
    # fill value - like the plain forms of a group without any item
    group_idx = np.array([0, 0, 1, 1])
    a = np.array(["NaT", "NaT", "2020-01-01", "2020-02-01"], dtype="datetime64[s]")
    # the group 0 holds nothing but NaT and answers the fill value - the
    # group 1 holds 2020-01-01 at the position 2 and 2020-02-01 at the
    # position 3, so it names the one of its extremes
    assert list(impl.aggregate(group_idx, a, func="nanargmin", size=2)) == [-1, 2]
    assert list(impl.aggregate(group_idx, a, func="nanargmax", size=2)) == [-1, 3]


def test_nan_argreductions_of_temporal_group_keep_positions_of_input(impl, request):
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
def test_temporal_quantile_answers_like_numpy(impl, method, q):
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
def test_temporal_percentile_answers_like_numpy(impl, method):
    # the percentile is the quantile of the fraction of a hundred
    group_idx = np.array([0, 0, 0, 1, 1])
    a = np.array(["2020-01-01", "2020-05-01", "2020-03-01", "2020-02-01", "2020-06-01"], dtype="M8[s]")
    for q in (0.0, 25.0, 50.0, 75.0, 100.0):
        res = impl.aggregate(group_idx, a, func="percentile", q=q, method=method, size=2)
        for g in range(2):
            mask = group_idx == g
            want = np.percentile(a[mask], q, method=method)
            assert res[g] == want, (method, q, g)


def test_datetime_quantile_propagates_nat(impl):
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


def test_temporal_quantile_refuses_unknown_method(impl):
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


def test_array_collects_values(impl):
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


def test_user_fill_value_dates_empties(impl):
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-02-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="min", size=2, fill_value=np.datetime64("2000-01-01", "s"))
    assert res[1] == np.datetime64("2000-01-01", "s")


def test_user_fill_value_int_is_tick(impl):
    group_idx = np.array([0, 0])
    a = np.array(["2011-01-01", "2011-02-01"], dtype="M8[s]")
    res = impl.aggregate(group_idx, a, func="min", size=2, fill_value=0)
    assert res[1] == np.datetime64("1970-01-01", "s")


def test_int64_dtype_answers_ticks(impl):
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


def test_custom_callable_sees_values(impl):
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
