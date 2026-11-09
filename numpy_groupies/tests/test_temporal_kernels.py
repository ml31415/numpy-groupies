"""The bare kernels of the numba backend over temporal values.

The kernels are the low-level entry of the numba backend: they take flat
one-dimensional input and answer in compiled code.  The temporal dtypes
reach them as they are - the comparisons of compiled code order datetime64
and timedelta64 values, and ``val != val`` detects NaT like a nan - so the
selection and the reduction kernels aggregate them directly.  The ones that
cannot (the arithmetic of the moments, the partitions, the truthiness) say
so by name.
"""

import numba as nb
import numpy as np
import pytest

from numpy_groupies.aggregate_numba.jit import njit_stable

# ---------------------------------------------------------------------------
# the bare numba kernels
#
# The kernels are called here and not through ``aggregate``: the tests are
# about what the kernel answers, not about the dispatch that reaches it.


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

    @njit_stable
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

    @njit_stable
    def nb_sort(x):
        return np.sort(x)

    @njit_stable
    def nb_argsort(x):
        return np.argsort(x)

    @njit_stable
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
