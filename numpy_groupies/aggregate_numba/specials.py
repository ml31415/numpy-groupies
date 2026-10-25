"""Operations with fully custom pass structures.

``median`` is an order statistic: values are placed group by group via a
counting sort (O(n)) and the middle of each group is then *selected* (O(n) on
average) instead of sorting within the groups.
``sort`` uses the same counting-sort placement, but orders every group-block -
by insertion sort while the block is small, by ``np.sort`` beyond that - and
scatters the blocks back to the positions their group occupies in the input.
``trapezoid`` integrates consecutive sample pairs within each group in a
single pass - it composes like the streaming reductions and takes the sample
spacing ``dx`` as the argument of its own.
"""

import math
from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.compose import (
    build_custom_kernel,
    build_kernel,
    complex_less,
    output_size,
    rule_float,
    rule_int64,
    rule_same,
)
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numba.mt import group_chunk_count, should_use_group_mt
from numpy_groupies.aggregate_numba.steps import group_count, group_place, group_ranges, group_starts


def _group_valid_plain(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
) -> int:
    """the number of usable values of group g, 0 if a nan poisons the group.

    The values stay where they are: the block of the group is contiguous
    already, and the selection that follows copies it anyway - copying it
    here as well would write every value twice (measured 1.6ms per 500k
    values).  The nans are looked for in a loop instead of with a temporary
    array, which would have to be allocated and filled for every group.
    """
    begin: int = starts[g]
    size_g: int = counts[g]
    for i in range(size_g):
        v = a_sorted[begin + i]
        if v != v:
            return 0
    return size_g


def _group_valid_nan(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
) -> int:
    """copy the usable values of group g into buf, returning their number -
    the nans are filtered out, so the group keeps its usable values"""
    begin: int = starts[g]
    size_g: int = counts[g]
    m = 0
    for i in range(size_g):
        v = a_sorted[begin + i]
        if v == v:
            buf[m] = v
            m += 1
    return m


# The number of partition passes a selection may make per power of two of the
# length of the block it works on, before it orders the block instead: a
# quickselect is O(n) on average, and the fallback keeps the worst case at
# O(n log n).
_SELECT_PASSES = 8


def _select_kth(
    arr: npt.NDArray[np.number],
    lo: int,
    hi: int,
    k: int,
) -> Any:
    """The k-th smallest of ``arr[lo:hi]``, the block reordered in place.

    A quickselect: the block is partitioned around a pivot until the position
    of the wanted element is settled.  The pivot is the median of the first,
    the middle and the last element of the range, which keeps the number of
    passes logarithmic for the blocks a group is made of - sorted, reversed
    and constant blocks alike.  A block that needs more passes than the limit
    allows is ordered instead, so that the worst case stays at O(n log n).

    ``k`` is an index into ``arr``, not an offset into the range.

    Numba's ``np.partition`` would do the same job, but it copies every block
    it is given and hands the copy back (measured 9.2ms against 1.2ms for
    this, over 500k values in 1000 groups) - and the copy is not even needed,
    since the block is a temporary that the median does not care about.

    The values are compared with ``complex_less``, the ordering every other
    kernel of this package compares with: numpy orders complex values
    lexicographically, and the median of a complex group is the middle of that
    order.  For the real dtypes the call is the plain comparison - the branch
    on the dtype is settled when the kernel is compiled.
    """
    limit = _SELECT_PASSES
    span = hi - lo
    remaining = span
    while remaining > 1:
        remaining >>= 1
        limit += 4
    while span > 1:
        limit -= 1
        if limit <= 0:
            # the fallback: order the range and take the element from there
            arr[lo:hi].sort()
            return arr[k]
        mid = lo + span // 2
        # the pivot: the median of the first, the middle and the last value of
        # the range, with the ordering of the package - for the real dtypes
        # that is the plain comparison, for complex values it is the
        # lexicographic order numpy uses
        x = arr[lo]
        y = arr[mid]
        z = arr[hi - 1]
        if complex_less(x, y):
            small, middle = x, y
        else:
            small, middle = y, x
        if complex_less(middle, z):
            pivot = middle
        elif complex_less(small, z):
            pivot = z
        else:
            pivot = small
        i = lo - 1
        j = hi
        while True:
            i += 1
            while complex_less(arr[i], pivot):
                i += 1
            j -= 1
            while complex_less(pivot, arr[j]):
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
        # everything in arr[lo..j] is at most the pivot and everything in
        # arr[j+1..hi-1] is at least it, so the wanted element is in one of
        # the two halves
        if k <= j:
            hi = j + 1
        else:
            lo = j + 1
        span = hi - lo
    return arr[k]


# ---------------------------------------------------------------------------
# the work one group of the placed values needs, at the module level
#
# The sequential driver of an order statistic and its parallel driver call the
# ONE compiled body below: what a group costs is written once, and the parallel
# path runs the very code the sequential one runs.  A jitted function of its
# own, disk cached and shared by its callers is what ``complex_less`` is for the
# comparisons of the reductions - and the note there says why it has to be
# cached: numba cannot resolve an uncached callee from cached caller code.

# the same helpers serve the median and the quantile - the same source, the same
# signature, the same specializations: one compiled artifact serves both
_valid_plain = njit_stable(_group_valid_plain)
_valid_nan = njit_stable(_group_valid_nan)
# the selection: the same block for every operation that selects, so one
# compiled artifact serves the median and the quantile and both of their variants
_select = njit_stable(_select_kth)


def _middle(a_sorted: npt.NDArray[np.number], begin: int, m: int) -> Any:
    """the middle value(s) of the block at ``a_sorted[begin:begin+m]``.

    No sorting needed: the wanted element is selected in place, and the block is
    left in the order the selection put it in - the median does not care about
    the order of the values it looks at.
    """
    k = m // 2
    upper = _select(a_sorted, begin, begin + m, begin + k)
    if m % 2 == 1:
        return upper
    # the lower middle is the largest of the left half, which the selection left
    # in place; the 1.0 promotes integral input to float64, just like np.median
    # promotes before averaging - summing the two middles in their own dtype
    # overflows (int64 2**62 + 2**62) and adds bool as a logical or
    lower = a_sorted[begin]
    for i in range(begin + 1, begin + k):
        if complex_less(lower, a_sorted[i]):
            lower = a_sorted[i]
    return (lower * 1.0 + upper) / 2


_middle = njit_stable(_middle, name="median_middle")


def _median_group_plain(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    out: npt.NDArray[np.number],
    fill_value: float,
) -> None:
    """the median of one group of the placed values

    ``buf`` and ``fill_value`` are the two arguments the nan variant of the very
    same step needs and this one does not: an empty buffer is handed over, and
    the poisoned cell is shaped out of the output itself.  The two bodies share
    the signature so that the drivers of the two variants are the same loop.
    """
    if counts[g] == 0:
        return  # untouched group, keeps the fill value
    m = _valid_plain(g, a_sorted, counts, starts)
    if m == 0:
        # a nan-poisoned group for plain median (where any nan is contagious) -
        # the poison is shaped like the output, since a plain nan written into a
        # complex column leaves its imaginary part at a zero the group never
        # claimed to have.  The cell still holds the fill value, which is of the
        # output dtype, and a nan times a value of that dtype is a nan in every
        # part it has (for a real dtype: the plain nan)
        out[g] = out[g] * np.nan
    else:
        out[g] = _middle(a_sorted, starts[g], m)


def _median_group_nan(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    out: npt.NDArray[np.number],
    fill_value: float,
) -> None:
    """the median of one group, the nan values of the group left out

    ``buf`` is the scratch of the range this group belongs to: a thread writes
    into the buffer of its own range and into no other one, which is what makes
    the loop over the ranges free of a lock.
    """
    if counts[g] == 0:
        return  # untouched group, keeps the fill value
    m = _valid_nan(g, a_sorted, counts, starts, buf)
    if m == 0:
        # all-nan group: it keeps the fill value
        out[g] = fill_value
    else:
        out[g] = _middle(buf, 0, m)


_median_group_plain = njit_stable(_median_group_plain, name="median_group")
_median_group_nan = njit_stable(_median_group_nan, name="nanmedian_group")


def _quantile_at(a_sorted: npt.NDArray[np.number], begin: int, m: int, q: float) -> Any:
    """the quantile q of the block at ``a_sorted[begin:begin+m]``.

    The quantile names a position between two of the values of the group: the
    virtual index q * (m - 1) says how far between the smallest and the largest
    value it lies, and the two values it falls between are interpolated the way
    numpy interpolates them.
    """
    pos = q * (m - 1)
    lower_index = int(np.floor(pos))
    upper_index = int(np.ceil(pos))
    upper = _select(a_sorted, begin, begin + m, begin + upper_index)
    if lower_index == upper_index:
        # the quantile is one of the values of the group
        return upper
    # the value below: the largest of the elements the selection left in
    # front of the one it picked - they are the ones below it
    lower = a_sorted[begin]
    for i in range(begin + 1, begin + lower_index + 1):
        if complex_less(lower, a_sorted[i]):
            lower = a_sorted[i]
    # the 1.0 promotes integral input to float64, just like np.quantile
    # promotes before interpolating - the subtraction in its own dtype
    # overflows (int8 100 - (-100)) and bool addition is a logical or
    return lower * 1.0 + (upper * 1.0 - lower * 1.0) * (pos - lower_index)


_quantile_at = njit_stable(_quantile_at, name="quantile_quantile_at")


def _quantile_group_plain(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    out: npt.NDArray[np.number],
    fill_value: float,
    q: float,
) -> None:
    """the quantile of one group of the placed values

    The arguments of the nan variant it does not use are there for the reason
    stated at ``_median_group_plain``.
    """
    if counts[g] == 0:
        return  # untouched group, keeps the fill value
    m = _valid_plain(g, a_sorted, counts, starts)
    if m == 0:
        # a nan-poisoned group (where any nan is contagious) - the poison is
        # shaped like the output, since a plain nan written into a complex
        # column leaves its imaginary part at a zero the group never claimed to
        # have.  The cell still holds the fill value, which is of the output
        # dtype, and a nan times a value of that dtype is a nan in every part it
        # has (for a real dtype: the plain nan)
        out[g] = out[g] * np.nan
    else:
        out[g] = _quantile_at(a_sorted, starts[g], m, q)


def _quantile_group_nan(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    out: npt.NDArray[np.number],
    fill_value: float,
    q: float,
) -> None:
    """the quantile of one group, the nan values of the group left out"""
    if counts[g] == 0:
        return  # untouched group, keeps the fill value
    m = _valid_nan(g, a_sorted, counts, starts, buf)
    if m == 0:
        # all-nan group: it keeps the fill value
        out[g] = fill_value
    else:
        out[g] = _quantile_at(buf, 0, m, q)


_quantile_group_plain = njit_stable(_quantile_group_plain, name="quantile_group")
_quantile_group_nan = njit_stable(_quantile_group_nan, name="nanquantile_group")


# ---------------------------------------------------------------------------
# the counting sort and the range cut, at the module level
#
# The four blocks below have no closure of their own - the same source, the
# same signature, the same specializations for every operation that uses them
# - so ONE compiled artifact serves them all.  Per-operation copies buy
# nothing here: the identity of ``njit_stable`` covers the source and the
# closure, and neither differs from operation to operation (the note at
# ``compose.build_kernel`` is about the blocks whose closures do).

_count = njit_stable(group_count)
_starts = njit_stable(group_starts)
_place = njit_stable(group_place)
_ranges = njit_stable(group_ranges)

# what the shared setup of the group parallel passes hands back: the counts,
# the group starts, the placed values, the boundaries of the ranges and the
# largest block of every range, in that order - or None, when the cost model
# keeps the call on the sequential path
_GroupMTSetup: TypeAlias = (
    tuple[
        npt.NDArray[np.int64],  # the number of the values of every group
        npt.NDArray[np.int64],  # where the block of every group starts
        npt.NDArray[np.number],  # the values, placed group by group
        npt.NDArray[np.intp],  # the boundaries of the ranges
        npt.NDArray[np.int64],  # the largest block of every range
    ]
    | None
)


def _group_mt_setup(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size_arg: int,
    ns_per_value: float,
    n_threads: int,
) -> _GroupMTSetup:
    """the shared setup of the parallel passes that cut the GROUPS into ranges

    The gate of the cost model, the counting sort that places the values and
    the cut of the ranges are the same for every operation of this shape, so
    one compiled body does all three: it returns the counts, the group starts,
    the placed values and the range cut - or None when the model keeps the
    call on the sequential path (the caller then runs the sequential driver,
    with its result bit for bit).

    Jitted and disk cached like everything a compiled caller reaches into:
    numba cannot resolve an uncached callee from cached caller code (see the
    note at the top of the module).
    """
    size = output_size(group_idx, size_arg)
    n_values = group_idx.size
    counts = _count(group_idx, size)
    starts = _starts(counts)
    if not should_use_group_mt(size, n_threads, n_values, ns_per_value):
        return None
    a_sorted = np.empty(n_values, a.dtype)
    _place(group_idx, a, starts.copy(), a_sorted)
    n_ranges = group_chunk_count(n_threads, size)
    edges, largest = _ranges(counts, n_ranges)
    return counts, starts, a_sorted, edges, largest


_group_mt_setup = njit_stable(_group_mt_setup)


# what one value of the operation costs in the sequential kernel, measured on
# the machine of ``docs/benchmarks.md``, the counting sort of the placement
# included.  The select is the cheapest of the operations that cut their
# groups into ranges and the sort the most expensive; all of them cost orders
# of magnitude more per value than the sum the cost model was fitted on, so
# the fixed cost of a parallel region is a far smaller share of their time.
_NS_PER_VALUE_SELECT = 18.0
_NS_PER_VALUE_NUNIQUE = 19.0
_NS_PER_VALUE_SORT = 68.0


def _median_driver_factory(nans: bool) -> Any:
    # the block of the group is what the two drivers work on - the plain one
    # and the parallel one reach the ONE compiled body of it (see the note
    # above ``_median_group_plain``); the counting sort is the shared one of
    # the module (see the note at ``_count``)
    group = _median_group_nan if nans else _median_group_plain

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> npt.NDArray[np.number]:
        # median is order-insensitive and takes nothing of its own
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the median ignores)
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        # the nan variant filters the nans out into a buffer of its own, the
        # plain one leaves the values where they are and needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}median_driver")


def _median_mt_factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Any:
    """The parallel passes of the median: the groups cut into ranges.

    The gate, the counting sort that places the values and the cut of the
    ranges are the shared setup of ``_group_mt_setup``.  Every group belongs
    to exactly one range, so no two threads ever touch the same block, the
    buffer of a range is written by the thread that works on it and nothing
    has to be folded afterwards - the result is the one of the sequential
    driver bit for bit, whatever the number of the threads and of the ranges.
    """
    group = _median_group_nan if nans else _median_group_plain

    def mt_passes(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        extra: Any,
        n_threads: int,
    ) -> npt.NDArray[np.number]:
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_SELECT, n_threads)
        if setup is None:
            # not worth a parallel region: the sequential driver, with its
            # result bit for bit
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
        counts, starts, a_sorted, edges, largest = setup
        # the groups that receive no values keep this seeding - the same one
        # the sequential driver gives them
        out = np.full(counts.size, fill_value, out_dtype)
        # the buffer of a range is sized after the largest block of that range:
        # summed over the ranges that is at most the number of the values, so
        # the parallel path holds no more scratch than the sequential one
        bufs = [np.empty(k, a.dtype) for k in largest] if nans else [np.empty(0, a.dtype)] * largest.size
        for c in nb.prange(largest.size):
            buf = bufs[c]
            for g in range(edges[c], edges[c + 1]):
                group(g, a_sorted, counts, starts, buf, out, fill_value)
        return out

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}median_mt_passes", parallel=True)


median = build_custom_kernel(
    "median",
    driver_factory=_median_driver_factory,
    mt_driver_factory=_median_mt_factory,
    rule=rule_float,
    fill_default=np.nan,
    doc="Median of the values of every group",
)


def _quantile_driver_factory(nans: bool) -> Any:
    """The driver of the quantile - the median is the quantile 0.5.

    The quantile is an order statistic, so it groups the data the same way the
    median does: a counting sort places the values of every group in a
    contiguous block, and the wanted quantile is then selected out of the block
    (O(n) on average) instead of sorting it.
    """
    op_name = "nanquantile" if nans else "quantile"
    group = _quantile_group_nan if nans else _quantile_group_plain

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        q: float,
    ) -> npt.NDArray[np.number]:
        # the quantile is order-insensitive and takes its quantile
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the quantile ignores)
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        # the nan variant filters the nans out into a buffer of its own, the
        # plain one leaves the values where they are and needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value, q)
        return out

    return njit_stable(driver, name=f"{op_name}_driver")


def _quantile_mt_factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Any:
    """The parallel passes of the quantile, the groups cut into ranges.

    The same shape as ``_median_mt_factory``, for the same reasons.
    """
    group = _quantile_group_nan if nans else _quantile_group_plain

    def mt_passes(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        q: float,
        n_threads: int,
    ) -> npt.NDArray[np.number]:
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_SELECT, n_threads)
        if setup is None:
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, q)
        counts, starts, a_sorted, edges, largest = setup
        out = np.full(counts.size, fill_value, out_dtype)
        bufs = [np.empty(k, a.dtype) for k in largest] if nans else [np.empty(0, a.dtype)] * largest.size
        for c in nb.prange(largest.size):
            buf = bufs[c]
            for g in range(edges[c], edges[c + 1]):
                group(g, a_sorted, counts, starts, buf, out, fill_value, q)
        return out

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}quantile_mt_passes", parallel=True)


quantile = build_custom_kernel(
    "quantile",
    driver_factory=_quantile_driver_factory,
    mt_driver_factory=_quantile_mt_factory,
    rule=rule_float,
    fill_default=np.nan,
    variant="q",
    doc="Quantile of the values of every group",
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


@njit_stable
def _order_block(a_sorted: npt.NDArray[np.number], begin: int, size_g: int, reverse: bool) -> npt.NDArray[np.number]:
    """order one group-block of the flat array, descending if ``reverse``.

    ``size_g`` is at least two, the caller leaves a block of a single value
    alone.  While a block is small its values walk into place where they are,
    beyond that ``np.sort`` of a view is the cheaper of the two - see the note at
    ``_INSERTION_SORT_LIMIT``.

    Both passes order ascending and leave a nan last: the library sort because
    that is what numpy does, the insertion pass because it shifts a nan right by
    every other value and stops only in front of another nan, which keeps the
    nans in the order of the input.  Complex values never take the insertion
    pass, since numpy compares them partially - a value with a nan in either
    part is neither smaller nor greater, and a pass that has to decide every
    pair cannot order that.

    ``reverse`` therefore cannot turn the whole block around, the nan tail has to
    stay last, so only the ordered prefix is flipped.  Flipping it rather than
    ordering on a negated key is what keeps the descending order exact: negation
    is not order reversing for the integral dtypes (whose minimum survives it as
    itself, and whose zero of an unsigned dtype stays the smallest key), which is
    why the python twins invert bitwise or flip the group's run instead.
    """
    if size_g <= _INSERTION_SORT_LIMIT and not isinstance(a_sorted[begin], (complex, np.complex64)):
        for i in range(1, size_g):
            v = a_sorted[begin + i]
            j = i - 1
            while j >= 0:
                w = a_sorted[begin + j]
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
    return a_sorted


def _sort_blocks(
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    reverse: bool,
) -> None:
    """order the values inside every group-block, the singletons left alone

    ``_order_block`` of one group does not look at the block of another one, so
    the parallel driver of the sort walks the ranges of ``group_ranges`` over
    this instead of walking the groups, and needs no fold and no lock - the
    blocks it writes are disjoint by construction.
    """
    for g in range(len(counts)):
        if counts[g] > 1:
            _order_block(a_sorted, starts[g], counts[g], reverse)


def _sort_scatter(
    group_idx: npt.NDArray[np.integer],
    a_sorted: npt.NDArray[np.number],
    starts: npt.NDArray[np.int64],
    out: npt.NDArray[np.number],
) -> None:
    """write the ordered blocks back to the positions their group occupies
    in the input - the j-th value of a group goes to its j-th appearance"""
    cursor = starts.copy()
    for i in range(len(group_idx)):
        g = group_idx[i]
        out[i] = a_sorted[cursor[g]]
        cursor[g] += 1


def _sort_driver_factory(nans: bool) -> Any:
    # the ordering and the scatter are the sort's own blocks - the counting
    # sort is the shared one of the module (see the note at ``_count``)
    _blocks = njit_stable(_sort_blocks)
    _scatter = njit_stable(_sort_scatter)

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        reverse: bool,
    ) -> npt.NDArray[np.number]:
        # sort emits one value per input item, so neither the group count nor
        # the fill value is of any use here
        size = output_size(group_idx, size_arg)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        _blocks(a_sorted, counts, starts, reverse)
        out = np.empty(group_idx.size, out_dtype)
        _scatter(group_idx, a_sorted, starts, out)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}sort_driver")


def _sort_mt_factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Any:
    """The parallel passes of the sort: the blocks cut into ranges.

    The ordering of the blocks is what the sort spends most of its time on,
    and the block of one group is nowhere near the block of the next one, so
    the ranges need neither a fold nor a lock.  The scatter that writes the
    ordered blocks back to the positions their group occupies in the input
    stays single threaded: it walks the input in order and moves a cursor per
    group, which two threads cannot share without a per-range cursor per
    group.
    """
    _scatter = njit_stable(_sort_scatter)

    def mt_passes(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        reverse: bool,
        n_threads: int,
    ) -> npt.NDArray[np.number]:
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_SORT, n_threads)
        if setup is None:
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, reverse)
        counts, starts, a_sorted, edges, _ = setup
        for c in nb.prange(edges.size - 1):
            for g in range(edges[c], edges[c + 1]):
                # the same step the sequential loop runs, on the block of a
                # group no other thread has touched
                if counts[g] > 1:
                    _order_block(a_sorted, starts[g], counts[g], reverse)
        out = np.empty(group_idx.size, out_dtype)
        _scatter(group_idx, a_sorted, starts, out)
        return out

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}sort_mt_passes", parallel=True)


sort = build_custom_kernel(
    "sort",
    driver_factory=_sort_driver_factory,
    mt_driver_factory=_sort_mt_factory,
    rule=rule_same,
    nans=False,
    nton=True,
    variant="reverse",
    doc="Values of every group in ascending, or with reverse=True in descending order",
)


@njit_stable
def _mix_hash(x: np.uint64) -> Any:
    """spread the bits of a hash over the whole word (splitmix64).

    The result is an unsigned integer - numba's stubs do not know the
    arithmetic of them, so the type checker is told nothing here.
    """
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit_stable
def _hash_float(v: float) -> Any:
    """the hash of a floating point value.

    The mantissa of a double is exact in 53 bits, so the mantissa and the
    exponent name the value - and no two different values have the same pair.
    A zero of either sign has the same mantissa, so the two have the same
    hash, as they are the same value.
    """
    m, e = math.frexp(v)
    return _mix_hash(
        np.uint64(int(m * 9007199254740992.0)) ^ (np.uint64(e & 0xFFFFFFFF) * np.uint64(0x9E3779B97F4A7C15))
    )


@njit_stable
def _hash_value(v: Any) -> Any:
    """the hash of a value - the values that are equal have the same hash.

    A complex value mixes the hashes of its two parts, a floating point one
    has a hash of its own, an integer its own value.
    """
    if isinstance(v, (complex, np.complex64, np.complex128)):
        return _mix_hash(_hash_float(v.real) ^ (_hash_float(v.imag) * np.uint64(0x9E3779B97F4A7C15)))
    if isinstance(v, (float, np.float32, np.float64)):
        return _hash_float(v)
    return _mix_hash(np.uint64(v))


@njit_stable
def _count_distinct(
    values: npt.NDArray[np.number],
    begin: int,
    n: int,
    slots: npt.NDArray[np.number],
    gens: npt.NDArray[np.int32],
    generation: int,
    mask: int,
) -> tuple[int, bool]:
    """the number of distinct values of the block, and whether it holds a nan.

    The values are counted in a hash table with open addressing: a slot that
    holds the value already is a value seen before, a free one makes it a new
    one.  The generation of the group says which of the slots are in use, so
    that the table needs no clearing between the groups.

    The nans are not counted at all - they are one value among themselves,
    which the caller adds.
    """
    distinct = 0
    has_nan = False
    for i in range(begin, begin + n):
        v = values[i]
        if v != v:
            # val != val - the nan of a complex value in either part
            has_nan = True
            continue
        # the index is an int64: numba would widen a uint64 against the
        # int64 of the mask to a float, which is no index
        h = np.int64(_hash_value(v) & np.uint64(mask))
        while True:
            if gens[h] != generation:
                # a free slot: the value is a new one
                gens[h] = generation
                slots[h] = v
                distinct += 1
                break
            if slots[h] == v:
                break  # a value seen before
            h = (h + 1) & mask
    return distinct, has_nan


@njit_stable
def _nunique_group(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    out: npt.NDArray[np.integer],
    slots: npt.NDArray[np.number],
    gens: npt.NDArray[np.int32],
    count_nans: bool,
) -> None:
    """the number of the distinct values of one group of the placed values

    ``slots`` and ``gens`` are the hash table of the caller - the table of the
    range of the parallel driver, of which one thread writes into its own and
    none of another.  The generation of a group is its index, so a table needs no
    clearing between the groups of its range.
    """
    n = counts[g]
    if n == 0:
        return  # untouched group, keeps the fill value
    # the table of this group holds only as many slots as the group needs - the
    # slots of the other groups are not touched
    mask = 1
    while mask < 2 * n:
        mask *= 2
    mask -= 1
    distinct, has_nan = _count_distinct(a_sorted, starts[g], n, slots, gens, g, mask)
    if has_nan and count_nans:
        # the plain variant counts the nans of the group as one value
        distinct += 1
    out[g] = distinct


def _table_capacity(n: int) -> int:
    """the slots of the table of a block of ``n`` values: two for every value, so
    that the table is never more than half full and the probing stays short.
    """
    capacity = 1
    while capacity < 2 * n:
        capacity *= 2
    return capacity


_table_capacity = njit_stable(_table_capacity)


def _nunique_driver_factory(nans: bool) -> Any:
    """The driver of the number of distinct values of every group.

    The values are placed group by group with a counting sort, and the
    distinct ones of every group are then counted in a hash table - which
    needs no ordering of the values and so is linear in the number of them.
    The nans are one value among themselves: the plain variant counts them
    as one, the nan variant leaves them out.
    """
    op_name = "nannunique" if nans else "nunique"
    capacity_of = _table_capacity

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> npt.NDArray[np.number]:
        # the number of the distinct values takes nothing of its own
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        # counting-sort placement into contiguous group-blocks
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        max_count = 0
        for g in range(size):
            max_count = max(max_count, counts[g])
        capacity = capacity_of(max_count)
        slots = np.empty(capacity, a.dtype)
        # the generation of a group says which of the slots it uses
        gens = np.full(capacity, -1, np.int32)
        for g in range(size):
            _nunique_group(g, a_sorted, counts, starts, out, slots, gens, not nans)
        return out

    return njit_stable(driver, name=f"{op_name}_driver")


def _nunique_mt_factory(nans: bool, driver: Callable[..., npt.NDArray[np.number]]) -> Any:
    """The parallel passes of the distinct count: the groups cut into ranges.

    The one thing the shape needs that the median does not is a hash table of
    its own per range - the table is scratch that every group of the range
    clears by its generation, which two ranges cannot do in one table.  Sized
    after the largest block of the range and not after the largest group of
    the whole call: summed over the ranges that is at most twice the number
    of the values, where a table per range after the global maximum would be
    twice the number of the values per thread (the largest group of a
    lopsided call can be most of the call).
    """
    capacity_of = _table_capacity

    def mt_passes(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        extra: Any,
        n_threads: int,
    ) -> npt.NDArray[np.number]:
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_NUNIQUE, n_threads)
        if setup is None:
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
        counts, starts, a_sorted, edges, largest = setup
        out = np.full(counts.size, fill_value, out_dtype)
        tables = [(np.empty(capacity_of(k), a.dtype), np.full(capacity_of(k), -1, np.int32)) for k in largest]
        for c in nb.prange(largest.size):
            slots, gens = tables[c]
            for g in range(edges[c], edges[c + 1]):
                _nunique_group(g, a_sorted, counts, starts, out, slots, gens, not nans)
        return out

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}nunique_mt_passes", parallel=True)


nunique = build_custom_kernel(
    "nunique",
    driver_factory=_nunique_driver_factory,
    mt_driver_factory=_nunique_mt_factory,
    rule=rule_int64,
    fill_default=0,
    doc="Number of distinct values of every group",
)


class TrapezoidState(NamedTuple):
    """``out`` accumulates the integral and doubles as the result, ``prev``
    keeps the previous sample of each group, ``pending`` marks the groups
    whose first sample is still due and ``untouched`` the groups without
    values."""

    out: npt.NDArray[np.inexact]
    prev: npt.NDArray[np.number]
    pending: npt.NDArray[np.bool_]
    untouched: npt.NDArray[np.bool_]
    track: bool


def _trapezoid_init(
    size: int,
    n_input: int,
    out_dtype: np.dtype,
    aux_dtype: np.dtype,
    in_dtype: np.dtype,
    fill_value: float,
    forced_fill: float,
) -> TrapezoidState:
    track = fill_value != forced_fill
    untouched = np.ones(size, np.bool_) if track else np.empty(0, np.bool_)
    return TrapezoidState(
        np.zeros(size, out_dtype),
        np.empty(size, in_dtype),
        np.ones(size, np.bool_),
        untouched,
        track,
    )


def _trapezoid_update(st: TrapezoidState, ri: int, i: int, val: float, fill_value: float, dx: float) -> None:
    if st.track:
        st.untouched[ri] = False
    if st.pending[ri]:
        # the first sample of a group has no predecessor to integrate with
        st.pending[ri] = False
    else:
        st.out[ri] += 0.5 * dx * (st.prev[ri] + val)
    st.prev[ri] = val


def _trapezoid_untouched(st: TrapezoidState) -> npt.NDArray[np.bool_]:
    return st.untouched


trapezoid = build_kernel(
    "trapezoid",
    init=_trapezoid_init,
    update=_trapezoid_update,
    untouched=_trapezoid_untouched,
    forced_fill=0,
    fill_default=0,
    rule=rule_float,
    # the sample spacing is the argument of the operation, and the default of
    # the kernel is the integer 1: a float in that slot compiles the kernel of
    # the trapezoid a few percent slower than the integer it used to carry -
    # the nan variant of the trapezoid measured ~3% slower with it
    variant="dx",
    doc="Trapezoidal integration of the samples of every group",
)
