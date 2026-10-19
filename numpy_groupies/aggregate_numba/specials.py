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

from numpy_groupies.aggregate_numba.compose import (
    build_custom_kernel,
    build_kernel,
    output_size,
    rule_float,
    rule_same,
)
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numpy import _aggregate_base


def _numpy_fallback(
    group_idx: npt.ArrayLike,
    a: npt.ArrayLike,
    func: str,
    size: int | None,
    fill_value: float,
    order: str,
    dtype: npt.DTypeLike | None,
    axis: int | None,
) -> npt.NDArray[np.number]:
    """Hand an input dtype the compiled drivers cannot handle to numpy.

    Reached for complex input of ``median``: numba can neither partition
    complex values nor select a middle one of them.  ``sort`` does not need
    this - ``np.sort`` of numba orders complex like numpy does.
    """
    return _aggregate_base(group_idx, a, func, size=size, fill_value=fill_value, order=order, dtype=dtype, axis=axis)


def _group_count(group_idx: npt.NDArray[np.signedinteger], size: int) -> npt.NDArray[np.int64]:
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


def _group_starts(counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """first index of each group-block"""
    return np.cumsum(counts) - counts


def _group_place(
    group_idx: npt.NDArray[np.signedinteger],
    a: npt.NDArray[np.number],
    cursor: npt.NDArray[np.int64],
    a_sorted: npt.NDArray[np.number],
) -> None:
    """scatter the values into their group-blocks"""
    for i in range(len(group_idx)):
        g = group_idx[i]
        a_sorted[cursor[g]] = a[i]
        cursor[g] += 1


def _median_valid(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    nans: bool,
) -> int:
    """copy group g into buf, returning the number of usable values;
    groups with nans yield 0 for plain median (nan is contagious),
    while nanmedian keeps the group with the nans filtered out"""
    begin: int = starts[g]
    size_g: int = counts[g]
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


def _median_middle(values: npt.NDArray[np.number]) -> typing.Any:
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


def _median_driver_factory(nans: bool) -> typing.Any:
    _count = njit_stable(_group_count)
    _starts = njit_stable(_group_starts)
    _place = njit_stable(_group_place)
    _valid = njit_stable(_median_valid)
    _middle = njit_stable(_median_middle)

    def driver(
        group_idx: npt.NDArray[np.signedinteger],
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
        buf = np.empty(group_idx.size, a.dtype)
        for g in range(size):
            if counts[g] == 0:
                continue  # untouched group, keeps the fill value
            m = _valid(g, a_sorted, counts, starts, buf, nans)
            if m == 0:
                # all-nan group for nanmedian, or a nan-poisoned group
                # for plain median (where any nan is contagious) - the poison is
                # shaped like the output, since a plain nan written into a complex
                # column leaves its imaginary part at a zero the group never
                # claimed to have.  The cell still holds the fill value, which is
                # of the output dtype, and a nan times a value of that dtype is a
                # nan in every part it has (for a real dtype: the plain nan)
                out[g] = fill_value if nans else out[g] * np.nan
            else:
                out[g] = _middle(buf[:m])
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}median_driver")


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
    group_idx: npt.NDArray[np.signedinteger],
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


def _sort_driver_factory(nans: bool) -> typing.Any:
    _count = njit_stable(_group_count)
    _starts = njit_stable(_group_starts)
    _place = njit_stable(_group_place)
    _blocks = njit_stable(_sort_blocks)
    _scatter = njit_stable(_sort_scatter)

    def driver(
        group_idx: npt.NDArray[np.signedinteger],
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


class TrapezoidState(typing.NamedTuple):
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
    forced_fill: typing.Any,
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
