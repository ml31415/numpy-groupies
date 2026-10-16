"""Operations with fully custom pass structures.

``median`` is an order statistic: values are placed group by group via a
counting sort (O(n)) and the middle of each group is then *selected* via
partition (O(n) on average) instead of sorting within the groups.
``sort`` uses the same counting-sort placement, but orders every group-block -
by insertion sort while the block is small, by ``np.sort`` beyond that - and
scatters the blocks back to the positions their group occupies in the input.
``trapezoid`` integrates consecutive sample pairs within each group in a
single pass - it composes like the streaming reductions, but carries the
sample spacing ``dx`` in the ``ddof`` slot.
"""

import typing

import numpy as np
import numpy.typing as npt

from ..aggregate_numpy import _aggregate_base
from .compose import build_custom_kernel, build_kernel, output_size, rule_float, rule_same
from .jit import njit_stable


def _numpy_fallback(group_idx, a, func, size, fill_value, order, dtype, axis):
    """Hand an input dtype the compiled drivers cannot handle to numpy.

    Reached for complex input of ``median``: numba can neither partition
    complex values nor select a middle one of them.  ``sort`` does not need
    this - ``np.sort`` of numba orders complex like numpy does.
    """
    return _aggregate_base(group_idx, a, func, size=size, fill_value=fill_value, order=order, dtype=dtype, axis=axis)


def _group_count(group_idx, size):
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


def _group_starts(counts):
    """first index of each group-block"""
    return np.cumsum(counts) - counts


def _group_place(group_idx, a, cursor, a_sorted):
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
    _count = njit_stable(_group_count)
    _starts = njit_stable(_group_starts)
    _place = njit_stable(_group_place)
    _valid = njit_stable(_median_valid)
    _middle = njit_stable(_median_middle)

    @njit_stable
    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        # median is order-insensitive and takes no ddof - both slots unused
        size = output_size(group_idx, size_arg)
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


median = build_custom_kernel(
    "median",
    driver_factory=_median_driver_factory,
    rule=rule_float,
    fill_default=np.nan,
    fallback=_numpy_fallback,
    doc="Median of the values of every group",
)


# Blocks this small are ordered by insertion sort in the flat array, the larger
# ones by ``np.sort``: the array view that ``arr.sort()`` has to build per group
# costs more than the ordering does once the groups are tiny.  Measured with
# numba 0.68 on 500,000 float64 values ordered group by group, insertion sort
# is eleven times faster at one value per group, five times at five, and still a
# tenth ahead at fifty; from a hundred values upwards the library sort wins, by
# half at a hundred and by six times at a thousand.  The limit stays well below
# that crossover, so groups of very different sizes cannot push many of them
# across it.
_INSERTION_SORT_LIMIT = 32


def _sort_blocks(a_sorted, counts, starts, reverse):
    """order the values inside every group-block.

    ``np.sort`` leaves the nan at the end of a block, like the ``lexsort`` of
    the numpy twin does with its keys - so the reverse pass flips the ordered
    prefix only and lets the nan tail stay last.  Flipping that prefix rather
    than ordering on a negated key is what keeps the descending order exact:
    negation is not order reversing for the integral dtypes (whose minimum
    survives it as itself, and whose zero of an unsigned dtype stays the
    smallest key), which is why the python twins invert bitwise or flip the
    group's run instead.
    """
    for g in range(len(counts)):
        begin, size_g = starts[g], counts[g]
        if size_g > 1:
            # the insertion pass needs a comparison that decides every pair, and
            # complex values have none: numpy compares them partially, leaving
            # a value with a nan in either part neither smaller nor greater,
            # which an insertion sort would simply leave where it sits.  The
            # library sort of the other branch does put those at the end, like
            # numpy does, so complex blocks go there
            if size_g <= _INSERTION_SORT_LIMIT and not isinstance(a_sorted[begin], (complex, np.complex64)):
                for i in range(1, size_g):
                    v = a_sorted[begin + i]
                    j = i - 1
                    while j >= 0:
                        w = a_sorted[begin + j]
                        # nan is the largest value of the order, so every other
                        # value shifts it right - in front of another nan the
                        # loop stops, which keeps the nans in input order
                        if v != v:
                            before = False
                        elif w != w:
                            before = True
                        else:
                            before = v < w
                        if not before:
                            break
                        a_sorted[begin + j + 1] = a_sorted[begin + j]
                        j -= 1
                    a_sorted[begin + j + 1] = v
            else:
                a_sorted[begin : begin + size_g].sort()
            if reverse:
                m = size_g
                while m > 0 and a_sorted[begin + m - 1] != a_sorted[begin + m - 1]:
                    m -= 1
                lo, hi = begin, begin + m - 1
                while lo < hi:
                    v = a_sorted[lo]
                    a_sorted[lo] = a_sorted[hi]
                    a_sorted[hi] = v
                    lo += 1
                    hi -= 1


def _sort_scatter(group_idx, a_sorted, starts, out):
    """write the ordered blocks back to the positions their group occupies
    in the input - the j-th value of a group goes to its j-th appearance"""
    cursor = starts.copy()
    for i in range(len(group_idx)):
        g = group_idx[i]
        out[i] = a_sorted[cursor[g]]
        cursor[g] += 1


def _sort_driver_factory(nans):
    _count = njit_stable(_group_count)
    _starts = njit_stable(_group_starts)
    _place = njit_stable(_group_place)
    _blocks = njit_stable(_sort_blocks)
    _scatter = njit_stable(_sort_scatter)

    @njit_stable
    def driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof):
        # sort emits one value per input item, so neither the group count nor
        # the fill value is of any use here; the reverse flag arrives in the
        # slot trapezoid uses for its sample spacing
        size = output_size(group_idx, size_arg)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        _blocks(a_sorted, counts, starts, ddof != 0.0)
        out = np.empty(group_idx.size, out_dtype)
        _scatter(group_idx, a_sorted, starts, out)
        return out

    return driver


sort = build_custom_kernel(
    "sort",
    driver_factory=_sort_driver_factory,
    rule=rule_same,
    nans=False,
    nton=True,
    doc="Values of every group in ascending, or with reverse=True in descending order",
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
    track: bool


def _trapezoid_init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value, forced_fill):
    track = fill_value != forced_fill
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return TrapezoidState(
        np.zeros(size, out_dtype),
        np.empty(size, in_dtype),
        np.ones(size, np.bool_),
        untouched,
        track,
    )


def _trapezoid_update(st, ri, i, val, fill_value, ddof):
    if st.track:
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


trapezoid = build_kernel(
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
