"""Operations with fully custom pass structures.

``median`` is an order statistic: values are placed group by group via a
counting sort (O(n)) and the middle of each group is then *selected* (O(n) on
average) instead of sorting within the groups.
``sort`` uses the same counting-sort placement, but orders every group-block -
by insertion sort while the block is small, by ``np.sort`` beyond that - and
scatters the blocks back to the positions their group occupies in the input.
``trapezoid`` integrates consecutive sample pairs within each group in a
single pass - it composes like the streaming reductions, but carries the
sample spacing ``dx`` in the ``ddof`` slot.
"""

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.compose import (
    build_custom_kernel,
    build_kernel,
    complex_less,
    output_size,
    rule_float,
    rule_same,
)
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numba.steps import group_count, group_place, group_starts


def _median_valid_plain(
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


def _median_valid_nan(
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


def _median_driver_factory(nans: bool) -> Any:
    # every operation builds its own copies of the shared helpers - the
    # compiled specialization of one must not reach into the other (see
    # compose.build_kernel)
    op_name = "nanmedian" if nans else "median"
    _count = njit_stable(group_count)
    _starts = njit_stable(group_starts)
    _place = njit_stable(group_place)
    _valid_plain = njit_stable(_median_valid_plain)
    _valid_nan = njit_stable(_median_valid_nan)
    # the selection and the middle that uses it: the middle closes over the
    # selection, so that the compiled block belongs to the operation that uses
    # it (see njit_stable)
    _select = njit_stable(_select_kth, name=f"{op_name}_select")

    def _middle(a_sorted: npt.NDArray[np.number], begin: int, m: int) -> Any:
        """the middle value(s) of the block at ``a_sorted[begin:begin+m]``.

        No sorting needed: the wanted element is selected in place, and the
        block is left in the order the selection put it in - the median does
        not care about the order of the values it looks at.
        """
        k = m // 2
        upper = _select(a_sorted, begin, begin + m, begin + k)
        if m % 2 == 1:
            return upper
        # the lower middle is the largest of the left half, which the
        # selection left in place; the 1.0 promotes integral input to float64,
        # just like np.median promotes before averaging - summing the two
        # middles in their own dtype overflows (int64 2**62 + 2**62) and adds
        # bool as a logical or
        lower = a_sorted[begin]
        for i in range(begin + 1, begin + k):
            if complex_less(lower, a_sorted[i]):
                lower = a_sorted[i]
        return (lower * 1.0 + upper) / 2

    _middle = njit_stable(_middle, name=f"{op_name}_middle")

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        ddof: int,
    ) -> npt.NDArray[np.number]:
        # median is order-insensitive and takes no ddof - both slots unused
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        counts = _count(group_idx, size)
        starts = _starts(counts)
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the median ignores)
        a_sorted = np.empty(group_idx.size, a.dtype)
        _place(group_idx, a, starts.copy(), a_sorted)
        # the nan variant filters the nans out into a buffer of its own, the
        # plain one leaves the values where they are
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            if counts[g] == 0:
                continue  # untouched group, keeps the fill value
            if nans:
                m = _valid_nan(g, a_sorted, counts, starts, buf)
                if m == 0:
                    # all-nan group: it keeps the fill value
                    out[g] = fill_value
                else:
                    out[g] = _middle(buf, 0, m)
            else:
                m = _valid_plain(g, a_sorted, counts, starts)
                if m == 0:
                    # a nan-poisoned group for plain median (where any nan is
                    # contagious) - the poison is shaped like the output, since
                    # a plain nan written into a complex column leaves its
                    # imaginary part at a zero the group never claimed to have.
                    # The cell still holds the fill value, which is of the
                    # output dtype, and a nan times a value of that dtype is a
                    # nan in every part it has (for a real dtype: the plain nan)
                    out[g] = out[g] * np.nan
                else:
                    out[g] = _middle(a_sorted, starts[g], m)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}median_driver")


median = build_custom_kernel(
    "median",
    driver_factory=_median_driver_factory,
    rule=rule_float,
    fill_default=np.nan,
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
    """order the values inside every group-block, the singletons left alone"""
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
    # every operation builds its own copies of the shared helpers - the
    # compiled specialization of one must not reach into the other (see
    # compose.build_kernel)
    _count = njit_stable(group_count)
    _starts = njit_stable(group_starts)
    _place = njit_stable(group_place)
    _blocks = njit_stable(_sort_blocks)
    _scatter = njit_stable(_sort_scatter)

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        ddof: int,
    ) -> npt.NDArray[np.number]:
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

    return njit_stable(driver, name=f"{'nan' if nans else ''}sort_driver")


sort = build_custom_kernel(
    "sort",
    driver_factory=_sort_driver_factory,
    rule=rule_same,
    nans=False,
    nton=True,
    doc="Values of every group in ascending, or with reverse=True in descending order",
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


def _trapezoid_update(st: TrapezoidState, ri: int, i: int, val: float, fill_value: float, ddof: int) -> None:
    if st.track:
        st.untouched[ri] = False
    if st.pending[ri]:
        # the first sample of a group has no predecessor to integrate with
        st.pending[ri] = False
    else:
        # the sample spacing dx rides in the ddof slot
        st.out[ri] += 0.5 * ddof * (st.prev[ri] + val)
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
    ddof_default=1.0,
    doc="Trapezoidal integration of the samples of every group",
)
