"""Operations with fully custom pass structures.

``median`` is an order statistic: values are placed group by group via a
counting sort (O(n)) and the middle of each group is then *selected* via
partition (O(n) on average) instead of sorting within the groups.
``trapezoid`` integrates consecutive sample pairs within each group in a
single pass - it composes like the streaming reductions, but carries the
sample spacing ``dx`` in the ``ddof`` slot.
"""

import typing

import numpy as np
import numpy.typing as npt

from ..aggregate_numpy import _aggregate_base
from .compose import custom_kernel, kernel, rule_float
from .jit import njit_stable


def _median_fallback(group_idx, a, func, size, fill_value, order, dtype, axis):
    """numba cannot partition complex values - the numpy implementation
    selects the middle of the lexicographically ordered values."""
    return _aggregate_base(group_idx, a, func, size=size, fill_value=fill_value, order=order, dtype=dtype, axis=axis)


def _median_count(group_idx, size):
    """bounds-checked per-group element counts"""
    counts = np.zeros(size, np.int64)
    for i in range(len(group_idx)):
        g = group_idx[i]
        if g < 0:
            raise ValueError("negative indices not supported")
        if g >= size:
            raise ValueError("one or more indices in group_idx are too large")
        counts[g] += 1
    return counts


def _median_starts(counts):
    """first index of each group-block"""
    return np.cumsum(counts) - counts


def _median_place(group_idx, a, cursor, a_sorted):
    """scatter the values into their group-blocks"""
    for i in range(len(group_idx)):
        g = group_idx[i]
        a_sorted[cursor[g]] = a[i]
        cursor[g] += 1


def _median_valid(g, a_sorted, counts, starts, buf, nans):
    """copy group g into buf, returning the number of usable values;
    groups with nans yield 0 for plain median (nan is contagious),
    while nanmedian keeps the group with the nans filtered out"""
    begin, size_g = starts[g], counts[g]
    if not nans:
        buf[:size_g] = a_sorted[begin : begin + size_g]
        if np.sum(buf[:size_g] != buf[:size_g]) > 0:
            return 0
        return size_g
    m = 0
    for i in range(size_g):
        v = a_sorted[begin + i]
        if v == v:
            buf[m] = v
            m += 1
    return m


def _median_middle(values):
    """select the middle value(s) of a group - no sorting needed"""
    m = len(values)
    part = np.partition(values, m // 2)
    if m % 2 == 1:
        return part[m // 2]
    # the lower middle is the maximum of the unsorted left half; the 1.0
    # promotes integral input to float64, just like np.median promotes
    # before averaging - summing the two middles in their own dtype
    # overflows (int64 2**62 + 2**62) and adds bool as a logical or
    return (np.max(part[: m // 2]) * 1.0 + part[m // 2]) / 2


def _median_driver_factory(nans):
    _count = njit_stable(_median_count)
    _starts = njit_stable(_median_starts)
    _place = njit_stable(_median_place)
    _valid = njit_stable(_median_valid)
    _middle = njit_stable(_median_middle)

    @njit_stable
    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        # median is order-insensitive and takes no ddof - both slots unused
        size = np.max(group_idx) + 1
        if size_arg >= 0:
            if size > size_arg:
                raise ValueError("one or more indices in group_idx are too large")
            size = size_arg
        out = np.full(size, fill_value, out_dtype)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the median ignores)
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        buf = np.empty(group_idx.size, a.dtype)
        for g in range(size):
            if counts[g] == 0:
                continue  # untouched group, keeps the fill value
            m = _valid(g, a_sorted, counts, starts, buf, nans)
            if m == 0:
                # all-nan group for nanmedian, or a nan-poisoned group
                # for plain median (where any nan is contagious)
                out[g] = fill_value if nans else np.nan
            else:
                out[g] = _middle(buf[:m])
        return out

    return driver


median = custom_kernel(
    "median",
    driver_factory=_median_driver_factory,
    rule=rule_float,
    fill_default=np.nan,
    fallback=_median_fallback,
    doc="Median of the values of every group",
)


class TrapezoidState(typing.NamedTuple):
    """``out`` accumulates the integral and doubles as the result, ``prev``
    keeps the previous sample of each group, ``pending`` marks the groups
    whose first sample is still due and ``untouched`` the groups without
    values."""

    out: npt.NDArray[np.inexact]
    prev: npt.NDArray[np.generic]
    pending: npt.NDArray[np.bool_]
    untouched: npt.NDArray[np.bool_]


def _trapezoid_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value):
    return TrapezoidState(
        np.zeros(size, out_dtype),
        np.empty(size, in_dtype),
        np.ones(size, np.bool_),
        np.ones(size, np.bool_),
    )


def _trapezoid_update(st, ri, i, val, fill_value, ddof):
    st.untouched[ri] = False
    if st.pending[ri]:
        # the first sample of a group has no predecessor to integrate with
        st.pending[ri] = False
    else:
        # the sample spacing dx rides in the ddof slot
        st.out[ri] += 0.5 * ddof * (st.prev[ri] + val)
    st.prev[ri] = val


def _trapezoid_untouched(st):
    return st.untouched


trapezoid = kernel(
    "trapezoid",
    init=_trapezoid_init,
    update=_trapezoid_update,
    untouched=_trapezoid_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_float,
    ddof_default=1.0,
    doc="Trapezoidal integration of the samples of every group",
)
