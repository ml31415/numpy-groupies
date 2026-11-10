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

The types of the answers are the ones of the input - a number of the dtype of
the values, an integer tick - and the type checker is told so where the stubs
of numpy know no type: the element of an array is ``Any`` there, and
``typing.cast`` is not compiled in nopython mode, so the few places that read
such an element state the type of the local variable instead.
"""

import functools
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.compose import (
    TickKernel,
    _fill_differs,
    build_custom_kernel,
    build_kernel,
    complex_less,
    make_tick_call,
    make_tick_call_q,
    output_size,
    rule_float,
    rule_int64,
    rule_same,
)
from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.aggregate_numba.mt import group_chunk_count, should_use_group_mt
from numpy_groupies.aggregate_numba.steps import group_count, group_place, group_ranges, group_starts
from numpy_groupies.functions import nan
from numpy_groupies.temporal import NAT_TICK


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


@functools.lru_cache(maxsize=2)
def _valid_nan_scan(ticks: bool) -> Callable[..., int]:
    """copy the usable values of group g into buf, returning their number -
    the missing values are filtered out, so the group keeps its usable values.

    The missing value of the numbers is the nan - the value that is not equal
    to itself - and the missing value of a temporal input is the NaT tick
    ``-2**63``, which is an ordinary int64 among the other ticks.  The flag
    is a constant of the closure, so numba compiles the two branches apart
    and the one that does not belong to the call is not even typed.
    """

    @njit_stable(name=f"{'tick_' if ticks else ''}valid_nan")
    def valid_nan(
        g: int,
        a_sorted: npt.NDArray[np.number],
        counts: npt.NDArray[np.int64],
        starts: npt.NDArray[np.int64],
        buf: npt.NDArray[np.number],
    ) -> int:
        begin: int = starts[g]
        size_g: int = counts[g]
        m = 0
        for i in range(size_g):
            v = a_sorted[begin + i]
            if ticks:
                if v != NAT_TICK:
                    buf[m] = v
                    m += 1
            else:
                if v == v:
                    buf[m] = v
                    m += 1
        return m

    return valid_nan


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
) -> np.number:
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
            # the value of the block
            ordered: np.number = arr[k]
            return ordered
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
    # the value of the block
    element: np.number = arr[k]
    return element


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


# the nan scan of the numbers - the temporal one is the same scan with the
# NaT tick as the missing value (see ``_valid_nan_scan``)
@njit_stable(name="tick_holds_nat")
def _holds_nat_tick(a_sorted: npt.NDArray[np.number], begin: int, m: int) -> bool:
    """Whether the block of the ticks holds the NaT tick.

    The NaT tick is the smallest of them all, so it is the first one of the
    block - but the block is not ordered, and the scan is the price of the
    answer.
    """
    for i in range(begin, begin + m):
        if a_sorted[i] == NAT_TICK:
            return True
    return False


_valid_nan = _valid_nan_scan(False)
# the selection: the same block for every operation that selects, so one
# compiled artifact serves the median and the quantile and both of their variants
_select = njit_stable(_select_kth)


@functools.lru_cache(maxsize=2)
def _middle_of(ticks: bool) -> Callable[..., np.number | int]:
    """the middle value(s) of the block at ``a_sorted[begin:begin+m]``.

    No sorting needed: the wanted element is selected in place, and the block is
    left in the order the selection put it in - the median does not care about
    the order of the values it looks at.

    The two flavours differ in the mean of the two middles of an even group:
    the numbers promote it to float64, just like np.median promotes before
    averaging - summing the two middles in their own dtype overflows (int64
    2**62 + 2**62) and adds bool as a logical or - while the ticks keep it an
    exact int64, where a pair around the epoch carries more bits than the 53
    of a float64 mantissa holds and the promotion would round the answer off
    the tick grid.  The flag is a constant of the closure, so numba compiles
    the two branches apart and the one that does not belong to the call is
    not even typed.
    """

    @njit_stable(name=f"{'tick_' if ticks else ''}median_middle")
    def middle(a_sorted: npt.NDArray[np.number], begin: int, m: int) -> np.number | int:
        k = m // 2
        upper = _select(a_sorted, begin, begin + m, begin + k)
        if m % 2 == 1:
            return upper
        # the lower middle is the largest of the left half, which the selection left
        # in place
        lower = a_sorted[begin]
        for i in range(begin + 1, begin + k):
            if complex_less(lower, a_sorted[i]):
                lower = a_sorted[i]
        if ticks:
            # the ticks are integers - the exact mean of two of them is one
            return _tick_pair_average(int(lower), int(upper))
        # the mean of the two middles, in float64
        average: np.number = (lower * 1.0 + upper) / 2
        return average

    return middle


_middle = _middle_of(False)


def _tick_pair_average(lower: int, upper: int) -> int:
    """The exact mean of two ticks, rounded half to even.

    The sum of the two is not taken: two ticks of one end of the int64
    range sum past it, and the wraparound would answer a tick that belongs
    to no value of the group.  The mean is taken from the bits instead -

    - ``a & b`` are the bits the two share: they are in the mean once,
    whatever the other one carries there,
    - ``a ^ b`` are the bits they differ in: the mean of those is half of
    them, and the shift of a negative number floors - which is the floor
    the mean asks for,

    so ``q = (a & b) + ((a ^ b) >> 1)`` is ``floor((a + b) / 2)`` for the
    whole int64 range.  An odd sum leaves a remainder of a half, and half
    to even hands the answer to the even of the two neighbours one apart:
    the low bit of ``a ^ b`` is the parity of the sum, the low bit of ``q``
    the one of the floor - when both are set, the floor is the odd one and
    the neighbour above is taken.

    Measured against a python ``Fraction`` over the pairs of the corners
    of the int64 range, and the ties at zero and the smallest tick
    (test_temporal, the bare kernel of the tick median) - the wraparound
    of the plain sum and the rounding of the float64 promotion both stay
    out of it.
    """
    q = (lower & upper) + ((lower ^ upper) >> 1)
    if (lower ^ upper) & 1 and q & 1:
        q += 1
    return q


_tick_pair_average = njit_stable(_tick_pair_average, name="tick_pair_average")


@functools.lru_cache(maxsize=4)
def _median_group(ticks: bool, nans: bool) -> Callable[..., None]:
    """the median of one group of the placed values.

    The two flavours of the operation - the numbers and the ticks of a
    temporal input - and the two of the variants - the plain one, where a
    missing value poisons the group, and the nan one, where it is left out -
    are one body with two flags.  The flags are constants of the closure, so
    numba compiles the four combinations apart and the branches that do not
    belong to the call are not even typed.

    The difference between the flavours is the middle of the block (the exact
    mean of the two middles for the ticks, the float64 promotion for the
    numbers) and the missing value (the NaT tick for the temporal input, the
    nan for the numbers).  The difference between the variants is what a
    missing value does to the group: it takes it over, or it is left out.

    ``buf`` and ``fill_value`` are the two arguments the nan variant of the
    very same step needs and the plain one does not: an empty buffer is
    handed over, and the poisoned cell is shaped out of the output itself.
    The two bodies share the signature so that the drivers of the two
    variants are the same loop.
    """
    middle = _middle_of(ticks)
    valid_nan = _valid_nan_scan(ticks)

    @njit_stable(name=f"{'nan' if nans else ''}{'tick_' if ticks else ''}median_group")
    def group(
        g: int,
        a_sorted: npt.NDArray[np.number],
        counts: npt.NDArray[np.int64],
        starts: npt.NDArray[np.int64],
        buf: npt.NDArray[np.number],
        out: npt.NDArray[np.number],
        fill_value: Any,
    ) -> None:
        if counts[g] == 0:
            return  # untouched group, keeps the fill value
        if nans:
            # the nan variant: the missing values are left out, and the group
            # that holds nothing but them keeps the fill value
            m = valid_nan(g, a_sorted, counts, starts, buf)
            if m == 0:
                out[g] = fill_value
            else:
                out[g] = middle(buf, 0, m)
        elif ticks:
            # the plain variant of the ticks: a NaT tick takes the group
            # over, the way a nan takes the group of the numbers over - the
            # answer of such a group is the NaT of the unit
            if _holds_nat_tick(a_sorted, starts[g], counts[g]):
                out[g] = NAT_TICK
            else:
                out[g] = middle(a_sorted, starts[g], counts[g])
        else:
            # the plain variant of the numbers: a nan poisons the group (where
            # any nan is contagious) - the poison is shaped like the output,
            # since a plain nan written into a complex column leaves its
            # imaginary part at a zero the group never claimed to have.  The
            # cell still holds the fill value, which is of the output dtype,
            # and a nan times a value of that dtype is a nan in every part it
            # has (for a real dtype: the plain nan)
            m = _valid_plain(g, a_sorted, counts, starts)
            if m == 0:
                out[g] = out[g] * np.nan
            else:
                out[g] = middle(a_sorted, starts[g], m)

    return group


_median_group_plain = _median_group(False, False)
_median_group_nan = _median_group(False, True)
_tick_median_group_plain = _median_group(True, False)
_tick_median_group_nan = _median_group(True, True)


def _quantile_at(a_sorted: npt.NDArray[np.number], begin: int, m: int, q: float, method: str) -> np.number:
    """the quantile q of the block at ``a_sorted[begin:begin+m]``.

    The quantile names a position between two of the values of the group: the
    virtual index q * (m - 1) says how far between the smallest and the largest
    value it lies, and the two values it falls between are interpolated the way
    numpy interpolates them.  All five methods of numpy run here - the branch
    on the method happens once per group, and a group costs far more than the
    branch.
    """
    pos = q * (m - 1)
    if method == "lower":
        return _select(a_sorted, begin, begin + m, begin + int(np.floor(pos)))
    if method == "higher":
        return _select(a_sorted, begin, begin + m, begin + min(int(np.ceil(pos)), m - 1))
    if method == "nearest":
        # numpy rounds the position to the nearest element, halves to even
        return _select(a_sorted, begin, begin + m, begin + min(int(np.round(pos)), m - 1))
    lower_index = int(np.floor(pos))
    upper_index = min(int(np.ceil(pos)), m - 1)
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
    # the two branches of numpy's own ``_lerp``: below a half the answer
    # grows from the lower value, from a half on it shrinks from the upper
    # one.  The 1.0 promotes integral input to float64, just like np.quantile
    # promotes before interpolating - the subtraction in its own dtype
    # overflows (int8 100 - (-100)) and bool addition is a logical or
    diff = upper * 1.0 - lower * 1.0
    if method == "midpoint":
        gamma = 0.5
    else:
        gamma = pos - lower_index
    if gamma < 0.5:
        # the quantile, interpolated from the lower neighbour
        interpolated: np.number = lower * 1.0 + diff * gamma
        return interpolated
    # the quantile, interpolated from the upper one
    extrapolated: np.number = upper - diff * (1 - gamma)
    return extrapolated


_quantile_at = njit_stable(_quantile_at, name="quantile_quantile_at")


def _quantile_at_ticks(a_sorted: npt.NDArray[np.int64], begin: int, m: int, q: float, method: str) -> int:
    """The quantile q of the ticks of the block at ``a_sorted[begin:begin+m]``.

    The rule is numpy's own one for a temporal value: the quantile names a
    position between two of the ticks of the group, the difference of the two
    is a duration, and its product with the fraction of the way is truncated
    towards zero - the answer is a tick again.  The two branches are the ones
    of numpy's interpolation: below a half the answer grows from the lower
    neighbour, from a half on it shrinks from the upper one.

    The product is taken in float64, like numpy takes it - the difference of
    the two ticks leaves the window a float64 holds exactly only for the
    small ones, and numpy rounds it there.  The truncation is towards zero,
    like the truncation of numpy's multiplication of a duration and a float.

    The other methods of numpy name the neighbours of the quantile in their
    own way: ``lower`` takes the one below the position, ``higher`` the one
    above it, ``nearest`` the one nearest to it (numpy rounds the position
    to the nearest element, halves to even) - and the ``midpoint`` takes the
    mean of the two neighbours, which is the rule of the interpolation with a
    fraction of a half.
    """
    pos = q * (m - 1)
    if method == "lower":
        return int(_select(a_sorted, begin, begin + m, begin + int(np.floor(pos))))
    if method == "higher":
        return int(_select(a_sorted, begin, begin + m, begin + min(int(np.ceil(pos)), m - 1)))
    if method == "nearest":
        return int(_select(a_sorted, begin, begin + m, begin + min(int(np.round(pos)), m - 1)))
    lower_index = int(np.floor(pos))
    upper_index = int(np.ceil(pos))
    upper = int(_select(a_sorted, begin, begin + m, begin + upper_index))
    if lower_index == upper_index:
        # the quantile is one of the ticks of the group
        return upper
    # the tick below: the largest of the elements the selection left in
    # front of the one it picked - they are the ones below it
    lower = int(a_sorted[begin])
    for i in range(begin + 1, begin + lower_index + 1):
        lower = max(lower, int(a_sorted[i]))
    difference = upper - lower
    gamma = 0.5 if method == "midpoint" else pos - lower_index
    if gamma < 0.5:
        return lower + int(np.float64(difference) * np.float64(gamma))
    return upper - int(np.float64(difference) * np.float64(1 - gamma))


_quantile_at_ticks = njit_stable(_quantile_at_ticks, name="tick_quantile_at")


def _quantile_group_plain(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    out: npt.NDArray[np.number],
    fill_value: float,
    q: float,
    method: str,
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
        out[g] = _quantile_at(a_sorted, starts[g], m, q, method)


def _quantile_group_nan(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    buf: npt.NDArray[np.number],
    out: npt.NDArray[np.number],
    fill_value: float,
    q: float,
    method: str,
) -> None:
    """the quantile of one group, the nan values of the group left out"""
    if counts[g] == 0:
        return  # untouched group, keeps the fill value
    m = _valid_nan(g, a_sorted, counts, starts, buf)
    if m == 0:
        # all-nan group: it keeps the fill value
        out[g] = fill_value
    else:
        out[g] = _quantile_at(buf, 0, m, q, method)


_quantile_group_plain = njit_stable(_quantile_group_plain, name="quantile_group")
_quantile_group_nan = njit_stable(_quantile_group_nan, name="nanquantile_group")


def _tick_quantile_group(nans: bool) -> Callable[..., None]:
    """The quantile of one group of the ticks, the way numpy computes it.

    The plain variant takes the group over, the way the plain variant of the
    numbers takes the group of a nan over: the answer of a group that holds
    a NaT is the NaT of the unit.  The nan variant leaves the NaT tick out,
    as every nan variant leaves the nan out.
    """
    valid_nan = _valid_nan_scan(True)

    @njit_stable(name=f"{'nan' if nans else ''}tick_quantile_group")
    def group(
        g: int,
        a_sorted: npt.NDArray[np.int64],
        counts: npt.NDArray[np.int64],
        starts: npt.NDArray[np.int64],
        buf: npt.NDArray[np.int64],
        out: npt.NDArray[np.int64],
        fill_value: int,
        q: float,
        method: str,
    ) -> None:
        if counts[g] == 0:
            return  # untouched group, keeps the fill value
        if nans:
            m = valid_nan(g, a_sorted, counts, starts, buf)
            if m == 0:
                out[g] = fill_value
            else:
                out[g] = _quantile_at_ticks(buf, 0, m, q, method)
        else:
            # the plain variant of the ticks: a NaT tick takes the group
            # over, the way a nan takes the group of the numbers over - the
            # answer of such a group is the NaT of the unit
            if _holds_nat_tick(a_sorted, starts[g], counts[g]):
                out[g] = NAT_TICK
            else:
                out[g] = _quantile_at_ticks(a_sorted, starts[g], counts[g], q, method)

    return group


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


@njit_stable
def _place_values(
    group_idx: npt.NDArray[np.integer], a: npt.NDArray[np.number], size: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.number]]:
    """The counting sort that places the values of every group in a block.

    It hands back the number of the values of every group, where the block
    of every group starts, and the values themselves in that order - the
    order they appear in the input (the same order a stable argsort would
    give them, at a fraction of the cost).

    The four blocks it is built of have no closure of their own, so ONE
    compiled artifact serves every operation that uses them - and so does
    this one, which is the only place where they are put together.
    """
    counts = _count(group_idx, size)
    starts = _starts(counts)
    a_sorted = np.empty(group_idx.size, a.dtype)
    _place(group_idx, a, starts.copy(), a_sorted)
    return counts, starts, a_sorted


class GroupMTSetup(NamedTuple):
    """what the shared setup of the parallel passes that cut the GROUPS into ranges

    Numba compiles a named tuple to the very native struct a plain tuple
    becomes, and the field access of it is resolved when the caller is
    compiled - the names cost nothing at run time.

    """

    counts: npt.NDArray[np.int64]  # the number of the values of every group
    starts: npt.NDArray[np.int64]  # where the block of every group starts
    a_sorted: npt.NDArray[np.number]  # the values, placed group by group
    edges: npt.NDArray[np.intp]  # the boundaries of the ranges
    largest: npt.NDArray[np.int64]  # the largest block of every range


def _group_mt_setup(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size_arg: int,
    ns_per_value: float,
    n_threads: int,
) -> GroupMTSetup | None:
    """the shared setup of the parallel passes that cut the GROUPS into ranges

    The gate of the cost model, the counting sort that places the values and
    the cut of the ranges are the same for every operation of this shape, so
    one compiled body does all three: it returns the ``GroupMTSetup`` of the
    call, or None when the model keeps the call on the sequential path.

    None is safe here, because the callers unpack the setup before their
    prange loop and use plain variables in it: numba casts an Optional of a
    named tuple to its payload with a runtime check that fires inside prange
    bodies even when the value is valid (the crash of the benchmarks) - the
    cast belongs outside of the loop.

    Jitted and disk cached like everything a compiled caller reaches into:
    numba cannot resolve an uncached callee from cached caller code (see the
    note at the top of the module).
    """
    size = output_size(group_idx, size_arg)
    n_values = group_idx.size
    if not should_use_group_mt(size, n_threads, n_values, ns_per_value):
        return None
    counts, starts, a_sorted = _place_values(group_idx, a, size)
    n_ranges = group_chunk_count(n_threads, size)
    edges, largest = _ranges(counts, n_ranges)
    return GroupMTSetup(counts, starts, a_sorted, edges, largest)


_group_mt_setup = njit_stable(_group_mt_setup)


# the cost of a value of the operation in the sequential kernel, measured on
# the machine of ``docs/benchmarks.md``.  The select is the cheapest of the
# operations that cut their groups into ranges and the sort the most
# expensive; all of them cost orders of magnitude more per value than the sum
# the cost model was fitted on, so the fixed cost of a parallel region is a
# far smaller share of their time.
_NS_PER_VALUE_SELECT = 18.0
# the distinct count and the mode count their values in a hash table, which
# needs no ordering of them - 12.3 and 14.0 ns a value, measured with the
# tool of the benchmarks
_NS_PER_VALUE_NUNIQUE = 13.0
_NS_PER_VALUE_SORT = 68.0
# the selection of the topk, with the gathering of the values that are
# better than the border - the work of the median and a sweep over the block
_NS_PER_VALUE_TOPK = 24.0
# the mode: the hash table of the distinct count, the counts kept and the
# best value tracked besides.  Measured with the tool of the benchmarks:
# 14.0 ns a value, against the 12.3 of the distinct count - the same table,
# with the counter and the tie-break besides.
_NS_PER_VALUE_MODE = 14.0


def _median_driver_factory(nans: bool, ticks: bool = False) -> Callable[..., npt.NDArray[np.number]]:
    # the block of the group is what the two drivers work on - the plain one
    # and the parallel one reach the ONE compiled body of it (see the note
    # above ``_median_group``); the counting sort is the shared one of
    # the module (see the note at ``_count``)
    group = _median_group(ticks, nans)

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
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the median ignores)
        counts, starts, a_sorted = _place_values(group_idx, a, size)
        # the nan variant filters the nans out into a buffer of its own, the
        # plain one leaves the values where they are and needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}{'tick_' if ticks else ''}median_driver")


def _median_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.number]], ticks: bool = False
) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the median: the groups cut into ranges.

    The gate, the counting sort that places the values and the cut of the
    ranges are the shared setup of ``_group_mt_setup``.  Every group belongs
    to exactly one range, so no two threads ever touch the same block, the
    buffer of a range is written by the thread that works on it and nothing
    has to be folded afterwards - the result is the one of the sequential
    driver bit for bit, whatever the number of the threads and of the ranges.
    """
    group = _median_group(ticks, nans)

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
        # the tuple is unpacked into locals before the prange - a tuple that
        # crosses the border of a parallel region trips the gufunc creation
        # of numba (parfor_lowering reads a .loc of it that is not there)
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

    return njit_stable(
        mt_passes, name=f"{'nan' if nans else ''}{'tick_' if ticks else ''}median_mt_passes", parallel=True
    )


def _tick_quantile_driver_factory(nans: bool) -> Callable[..., npt.NDArray[np.number]]:
    """The driver of the quantile of the ticks - the same counting sort and
    the same selection the median uses, and numpy's rule of the interpolation
    on the ticks of the values.
    """
    group = _tick_quantile_group(nans)

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.int64],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: int,
        q: float,
        method: str,
    ) -> npt.NDArray[np.int64]:
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the quantile ignores)
        counts, starts, a_sorted = _place_values(group_idx, a, size)
        # the nan variant filters the NaT ticks out into a buffer of its own,
        # the plain one leaves the values where they are and needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value, q, method)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}tick_quantile_driver")


def _tick_quantile_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.int64]]
) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the quantile of the ticks: the groups cut into
    ranges, every group belonging to exactly one of them - so the result is
    the one of the sequential driver bit for bit.
    """
    group = _tick_quantile_group(nans)

    def mt_passes(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.int64],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: int,
        extra: Any,
        n_threads: int,
    ) -> npt.NDArray[np.int64]:
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_SELECT, n_threads)
        if setup is None:
            # not worth a parallel region: the sequential driver, with its
            # result bit for bit
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, extra[0], extra[1])
        counts, starts, a_sorted, edges, largest = setup
        out = np.full(counts.size, fill_value, out_dtype)
        bufs = [np.empty(k, a.dtype) for k in largest] if nans else [np.empty(0, a.dtype)] * largest.size
        for c in nb.prange(largest.size):
            buf = bufs[c]
            for g in range(edges[c], edges[c + 1]):
                group(g, a_sorted, counts, starts, buf, out, fill_value, extra[0], extra[1])
        return out

    return njit_stable(mt_passes, name=f"{'nan' if nans else ''}tick_quantile_mt_passes", parallel=True)


tick_median = build_custom_kernel(
    "tick_median",
    # the same driver and the same parallel passes as the numeric median,
    # with the tick bodies: the exact int64 middle, and the NaT tick as the
    # missing value of the nan variant.  The answer stays on the tick grid -
    # an int64 output, where the numeric kernel promotes to float64.
    driver_factory=lambda nans: _median_driver_factory(nans, True),
    mt_driver_factory=lambda nans, driver: _median_mt_factory(nans, driver, True),
    rule=rule_int64,
    fill_default=0,
    doc="Median of the ticks of every group, the exact middle of the ticks - the tick of a NaT poisons its group, and the nan form leaves it out",
)


median = build_custom_kernel(
    "median",
    driver_factory=_median_driver_factory,
    mt_driver_factory=_median_mt_factory,
    rule=rule_float,
    fill_default=np.nan,
    doc="Median of the values of every group",
    # the median of the temporal input is the median of the ticks
    tick=TickKernel(
        plain=make_tick_call("tick_median_call", tick_median.plain),
        nan=make_tick_call("nantick_median_call", tick_median.nan),
    ),
)


def _quantile_driver_factory(
    nans: bool, scale: float = 1.0, name: str = "quantile"
) -> Callable[..., npt.NDArray[np.number]]:
    """The driver of the quantile - the median is the quantile 0.5.

    The quantile is an order statistic, so it groups the data the same way the
    median does: a counting sort places the values of every group in a
    contiguous block, and the wanted quantile is then selected out of the block
    (O(n) on average) instead of sorting it.

    The percentile is the quantile in percent: its driver is the one of the
    quantile with the argument scaled down before it reaches the selection.
    """
    op_name = ("nan" if nans else "") + name
    group = _quantile_group_nan if nans else _quantile_group_plain

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        q: float,
        method: str,
    ) -> npt.NDArray[np.number]:
        # the quantile is order-insensitive and takes its quantile
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        # counting-sort placement into contiguous group-blocks (the order
        # within a block is arbitrary, which the quantile ignores)
        counts, starts, a_sorted = _place_values(group_idx, a, size)
        # the nan variant filters the nans out into a buffer of its own, the
        # plain one leaves the values where they are and needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        # the percentile takes its argument in percent
        fraction = q * scale
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value, fraction, method)
        return out

    return njit_stable(driver, name=f"{op_name}_driver")


def _quantile_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.number]], scale: float = 1.0, name: str = "quantile"
) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the quantile, the groups cut into ranges.

    The same shape as ``_median_mt_factory``, for the same reasons.
    """
    op_name = ("nan" if nans else "") + name
    group = _quantile_group_nan if nans else _quantile_group_plain

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
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, extra[0] * scale, extra[1])
        # unpacked before the prange - see the note in _median_mt_factory
        counts, starts, a_sorted, edges, largest = setup
        out = np.full(counts.size, fill_value, out_dtype)
        bufs = [np.empty(k, a.dtype) for k in largest] if nans else [np.empty(0, a.dtype)] * largest.size
        # the percentile takes its argument in percent
        fraction = extra[0] * scale
        method = extra[1]
        for c in nb.prange(largest.size):
            buf = bufs[c]
            for g in range(edges[c], edges[c + 1]):
                group(g, a_sorted, counts, starts, buf, out, fill_value, fraction, method)
        return out

    return njit_stable(mt_passes, name=f"{op_name}_mt_passes", parallel=True)


tick_quantile = build_custom_kernel(
    "tick_quantile",
    # the same counting sort and the same selection as the median, with
    # numpy's rule of the interpolation on the ticks: the difference of the
    # two ticks it falls between is a duration, its product with the fraction
    # is truncated towards zero, and the answer is a tick again.  All five
    # methods of numpy run in the kernel - the branch on the method happens
    # once per group.
    driver_factory=_tick_quantile_driver_factory,
    mt_driver_factory=_tick_quantile_mt_factory,
    rule=rule_int64,
    fill_default=0,
    variant="q_method",
    doc="Quantile of the ticks of every group, the way numpy computes it - the tick of a NaT poisons its group, and the nan form leaves it out",
)


quantile = build_custom_kernel(
    "quantile",
    driver_factory=_quantile_driver_factory,
    mt_driver_factory=_quantile_mt_factory,
    rule=rule_float,
    fill_default=np.nan,
    variant="q_method",
    doc="Quantile of the values of every group",
    # the quantile of the temporal input is the quantile of the ticks
    tick=TickKernel(
        plain=make_tick_call_q("tick_quantile_call", tick_quantile.plain),
        nan=make_tick_call_q("nantick_quantile_call", tick_quantile.nan),
    ),
)


percentile = build_custom_kernel(
    "percentile",
    # the driver and the parallel passes of the quantile, with the argument
    # scaled down: the percentile is the quantile in percent.  The five
    # methods of numpy run in the kernel, the same way they run in the
    # kernel of the quantile.
    driver_factory=partial(_quantile_driver_factory, scale=0.01, name="percentile"),
    mt_driver_factory=partial(_quantile_mt_factory, scale=0.01, name="percentile"),
    rule=rule_float,
    fill_default=np.nan,
    variant="q_method",
    doc="Percentile of the values of every group - the quantile in percent",
    # the percentile of the temporal input is the percentile of the ticks
    tick=TickKernel(
        plain=make_tick_call_q("tick_percentile_call", tick_quantile.plain, scale=0.01),
        nan=make_tick_call_q("nantick_percentile_call", tick_quantile.nan, scale=0.01),
    ),
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


def _sort_driver_factory(nans: bool) -> Callable[..., npt.NDArray[np.number]]:
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
        counts, starts, a_sorted = _place_values(group_idx, a, size)
        _blocks(a_sorted, counts, starts, reverse)
        out = np.empty(group_idx.size, out_dtype)
        _scatter(group_idx, a_sorted, starts, out)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}sort_driver")


def _sort_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.number]]
) -> Callable[..., npt.NDArray[np.number]]:
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
        # unpacked before the prange - see the note in _median_mt_factory
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


# ---------------------------------------------------------------------------
# the k largest (or the k smallest) values of every group
#
# The values of a group are selected, not ordered: the k wanted ones are
# found with the quickselect of the module - the same one the median and the
# quantile use - and the values that are better than the border it names are
# gathered.  The order of the k is set afterwards, over the few values that
# are left.
#
# The values that are missing are in no selection: the nans of the numbers
# and the NaT of the ticks are neither the largest nor the smallest value of
# a group, and the sort of a group puts them last.  They are therefore
# pushed to the end of the block before the selection runs - the comparisons
# of a nan with a value answer neither way, and a selection that ran over
# them would name a border that is no border at all.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _compact_valid_of(ticks: bool) -> Callable[..., int]:
    """The values of the block that are no missing value, pushed to its front.

    The number of them is handed back; the block is shorter from there on.
    The write index never runs ahead of the read index, so no value that is
    not read yet is overwritten.

    The two kinds of values need two bodies: the missing value of the
    numbers is the nan, the only value that is not equal to itself, and the
    missing value of the ticks is the NaT, the smallest tick of all - a
    comparison of a tick with a number is no comparison numba has.  The flag
    is settled when the body is compiled, so the branch that does not belong
    to the call is not even typed.
    """

    if ticks:

        @njit_stable(name="compact_valid_ticks")
        def compact_valid(block: npt.NDArray[np.number], begin: int, m: int) -> int:
            good = 0
            for i in range(begin, begin + m):
                v = block[i]
                if v != NAT_TICK:
                    block[begin + good] = v
                    good += 1
            return good

    else:

        @njit_stable(name="compact_valid_numbers")
        def compact_valid(block: npt.NDArray[np.number], begin: int, m: int) -> int:
            good = 0
            for i in range(begin, begin + m):
                v = block[i]
                if v == v:
                    block[begin + good] = v
                    good += 1
            return good

    return compact_valid


@njit_stable
def _rank_order(values: npt.NDArray[np.number], g: int, k: int, largest: int) -> None:
    """The k values of the row in the order of their rank.

    The largest one comes first when ``largest`` is true, the smallest one
    otherwise - the order a reader of the row expects.  The few values are
    ordered in place; a row of a hundred values and more goes through the
    library sort, which is the cheaper of the two there.
    """
    if k < 2:
        return
    if k > _INSERTION_SORT_LIMIT:
        # the library sort of a row of a matrix is a sort of a copy - the
        # slice of a two dimensional array is a copy in numba - so the values
        # are copied into a buffer of their own and sorted there
        row = np.empty(k, values.dtype)
        row[:] = values[g, :k]
        row.sort()
        if largest != 0:
            lo, hi = 0, k - 1
            while lo < hi:
                v = row[lo]
                row[lo] = row[hi]
                row[hi] = v
                lo += 1
                hi -= 1
        values[g, :k] = row
        return
    for i in range(1, k):
        v = values[g, i]
        j = i - 1
        while j >= 0:
            w = values[g, j]
            # the order of the package: the larger value comes first
            before = complex_less(w, v) if largest != 0 else complex_less(v, w)
            if not before:
                break
            values[g, j + 1] = w
            j -= 1
        values[g, j + 1] = v


@njit_stable
def _topk_group(
    block: npt.NDArray[np.number],
    begin: int,
    m: int,
    out: npt.NDArray[np.number],
    g: int,
    k: int,
    largest: int,
) -> None:
    """The k largest (or smallest) values of the block, into the row ``g``.

    The border of the selection is the k-th value of the group in the order
    of the rank: the (m-k)-th one when the largest values are wanted, the
    (k-1)-th one when the smallest are.  The values that are better than the
    border are the ones the answer holds, and the places they leave are
    filled with the border itself - the group holds enough of them, since
    the border is the k-th of the group.
    """
    if m == 0:
        return
    if m <= k:
        # every value of the group fits: the row holds them all, ordered
        for i in range(m):
            out[g, i] = block[begin + i]
        _rank_order(out, g, m, largest)
        return
    if largest != 0:
        border_index = begin + m - k
    else:
        border_index = begin + k - 1
    border = _select(block, begin, begin + m, border_index)
    found = 0
    for i in range(begin, begin + m):
        v = block[i]
        if largest != 0:
            better = complex_less(border, v)
        else:
            better = complex_less(v, border)
        if better:
            out[g, found] = v
            found += 1
    # the places the better values leave: the border itself, which the group
    # holds often enough
    i = found
    while i < k:
        out[g, i] = border
        i += 1
    _rank_order(out, g, k, largest)


def _topk_driver_factory(ticks: bool) -> Callable[..., npt.NDArray[np.number]]:
    """The compiled driver of the topk.

    The values of every group are placed in their blocks, the missing values
    of a block are pushed to its end, and the k wanted values of every group
    are selected out of what is left.
    """

    compact = _compact_valid_of(ticks)

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        extra: Any,
    ) -> npt.NDArray[np.number]:
        k, largest = extra
        size = output_size(group_idx, size_arg)
        counts, starts, a_sorted = _place_values(group_idx, a, size)
        out = np.full((size, k), fill_value, out_dtype)
        if k == 0:
            # no value is wanted: the answer is a matrix of no columns
            return out
        for g in range(size):
            if counts[g] > 0:
                m = compact(a_sorted, starts[g], counts[g])
                if m > 0:
                    _topk_group(a_sorted, starts[g], m, out, g, k, largest)
        return out

    return njit_stable(driver, name=f"{'tick_' if ticks else ''}topk_driver")


def _topk_mt_factory(
    ticks: bool, driver: Callable[..., npt.NDArray[np.number]]
) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the topk: the blocks cut into ranges.

    The block of a group belongs to that group alone, and the threads work
    on the groups of a range each - so the ranges need neither a fold nor a
    lock, and the answer is the one of the sequential path bit for bit.
    """

    compact = _compact_valid_of(ticks)

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
        k, largest = extra
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_TOPK, n_threads)
        if setup is None:
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, extra)
        # unpacked before the prange - see the note in _median_mt_factory
        counts, starts, a_sorted, edges, _ = setup
        out = np.full((output_size(group_idx, size_arg), k), fill_value, out_dtype)
        if k == 0:
            return out
        for c in nb.prange(edges.size - 1):
            for g in range(edges[c], edges[c + 1]):
                # the same step the sequential loop runs, on the block of a
                # group no other thread has touched
                if counts[g] > 0:
                    m = compact(a_sorted, starts[g], counts[g])
                    if m > 0:
                        _topk_group(a_sorted, starts[g], m, out, g, k, largest)
        return out

    return njit_stable(mt_passes, name=f"{'tick_' if ticks else ''}topk_mt_passes", parallel=True)


topk = build_custom_kernel(
    "topk",
    driver_factory=lambda nans: _topk_driver_factory(False),
    mt_driver_factory=lambda nans, driver: _topk_mt_factory(False, driver),
    rule=rule_same,
    fill_default=nan,
    nans=False,
    variant="topk",
    doc="The k largest, or with largest=False the k smallest, values of every group",
)

# the topk of the ticks of a temporal input - the same selection on the
# int64 ticks, where the NaT is the smallest tick of all
tick_topk = build_custom_kernel(
    "tick_topk",
    driver_factory=lambda nans: _topk_driver_factory(True),
    mt_driver_factory=lambda nans, driver: _topk_mt_factory(True, driver),
    rule=rule_int64,
    fill_default=0,
    nans=False,
    variant="topk",
    doc="The k largest, or with largest=False the k smallest, ticks of every group",
)


@njit_stable
def _mix_bits(x: np.uint64) -> np.uint64:
    """the finalizer of splitmix64: a bijection of the 64 bits.

    Every bit of the input reaches every bit of the output, and two
    different inputs have two different outputs - the property a hash
    table needs of its hash, and the reason the counts of the distinct
    count and of the mode are right.

    The arithmetic of the unsigned integers wraps around, which is what a
    hash wants: the bits of the word are the whole state.
    """
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB61D108B)
    return x ^ (x >> np.uint64(31))


# the bits of -0.0 - the two zeros are one value, so they have one hash
_ZERO_BITS = np.int64(-9223372036854775808)


@njit_stable
def _hash_float(v: float | np.floating[Any]) -> np.uint64:
    """the hash of a floating point value - over its bits.

    The bits of a double name the value: no two different values have the
    same bits, and the two zeros - the same value - are the one exception,
    so they are put on the same hash here.

    The bits are cheaper to get than the mantissa and the exponent of the
    value: the frexp of it is a call into the floating point library, and
    the two of them together cost more than the whole rest of the hash -
    measured 2.0ms against 0.3ms per 500k values, which is more than
    half of the time the mode spends on its data.
    """
    bits = np.float64(v).view(np.int64)
    if bits == _ZERO_BITS:
        # -0.0 and 0.0 are the same value
        bits = np.int64(0)
    return _mix_bits(np.uint64(bits))


@njit_stable
def _hash_value(v: Any) -> np.uint64:
    """the hash of a value - the values that are equal have the same hash.

    A complex value mixes the hashes of its two parts, a floating point one
    has a hash of its own, an integer its own value.
    """
    if isinstance(v, (complex, np.complex64, np.complex128)):
        return _mix_bits(_hash_float(v.real) ^ (_hash_float(v.imag) * np.uint64(0x9E3779B97F4A7C15)))
    if isinstance(v, (float, np.float32, np.float64)):
        return _hash_float(v)
    return _mix_bits(np.uint64(v))


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


def _nunique_driver_factory(nans: bool) -> Callable[..., npt.NDArray[np.number]]:
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
        # counting-sort placement into contiguous group-blocks
        counts, starts, a_sorted = _place_values(group_idx, a, size)
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


def _nunique_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.number]]
) -> Callable[..., npt.NDArray[np.number]]:
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
        # unpacked before the prange - see the note in _median_mt_factory
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


# ---------------------------------------------------------------------------
# the most frequent value of every group
#
# The values of a group are counted, not ordered: a hash table with open
# addressing holds one slot per distinct value of the group, and the counts
# are read off the slots - the same table the distinct count runs, with the
# counts kept and the best value tracked besides.  A tie hands the answer to
# the smallest of the values that share the highest count, the way
# ``scipy.stats.mode`` breaks one: the value is taken over only when its
# count is strictly higher, or it compares smaller than the one that is held.
#
# The values that are missing are in no selection, like the ones of the
# topk: a nan is never the most frequent value of a group - the check of the
# hash table would refuse it anyway, a nan being not equal to itself - and a
# group that holds nothing but them has no mode at all, so it keeps the fill
# value.  A temporal value reaches the kernel as it is, the NaT tick included:
# ``val != val`` detects it like a nan, and the temporal input is what the
# kernel was given, so the answer is a value of the input.
# ---------------------------------------------------------------------------


@njit_stable
def _mode_group(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    out: npt.NDArray[np.number],
    slots: npt.NDArray[np.number],
    gens: npt.NDArray[np.int32],
    slot_counts: npt.NDArray[np.int64],
) -> None:
    """The most frequent value of one group of the placed values, into row ``g``.

    ``slots``, ``gens`` and ``slot_counts`` are the hash table of the caller -
    the table of the range of the parallel driver, of which one thread writes
    into its own and none of another.  The generation of a group is its index,
    so a table needs no clearing between the groups of its range: the slots
    whose generation is behind hold counts of a group that is gone, and the
    first touch of a value in this generation overwrites them.
    """
    n = counts[g]
    if n == 0:
        return  # untouched group, keeps the fill value
    # the table of this group holds only as many slots as the group needs -
    # the slots of the other groups are not touched
    mask = 1
    while mask < 2 * n:
        mask *= 2
    mask -= 1
    best_count = 0
    best_value = a_sorted[starts[g]]
    for i in range(starts[g], starts[g] + n):
        v = a_sorted[i]
        if v != v:
            # the missing value of the numbers - in either part of a complex
            # value - and the NaT of the temporal ones: in no selection
            continue
        # the index is an int64: numba would widen a uint64 against the
        # int64 of the mask to a float, which is no index
        h = np.int64(_hash_value(v) & np.uint64(mask))
        while True:
            if gens[h] != g:
                # a free slot: the value is a new one
                gens[h] = g
                slots[h] = v
                slot_counts[h] = 1
                seen = 1
                break
            if slots[h] == v:
                slot_counts[h] += 1
                seen = slot_counts[h]
                break
            h = (h + 1) & mask
        # the value takes the answer over when it is more frequent than the
        # one that holds it, and a count that is even with a smaller value
        # takes it as well - the smallest of the most frequent ones wins.
        # A value that is equal to the one that holds the answer leaves it
        # alone: the answer is the first of the equal values, so a zero
        # keeps the sign it came with - the numpy backend orders the values
        # of the group and takes the first of the longest run, and the two
        # have to agree on the sign of a zero as well.
        if seen > best_count:
            best_count = seen
            if v != best_value:
                best_value = v
        elif seen == best_count and complex_less(v, best_value):
            best_value = v
    if best_count > 0:
        out[g] = best_value


def _mode_driver_factory() -> Callable[..., npt.NDArray[np.number]]:
    """The driver of the mode: the values are placed group by group with the
    counting sort, and the most frequent value of every block is counted in a
    hash table - which needs no ordering of the values and so is linear in
    their number, like the distinct count is."""
    capacity_of = _table_capacity

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> npt.NDArray[np.number]:
        # the mode takes nothing of its own
        size = output_size(group_idx, size_arg)
        out = np.full(size, fill_value, out_dtype)
        # counting-sort placement into contiguous group-blocks
        counts, starts, a_sorted = _place_values(group_idx, a, size)
        max_count = 0
        for g in range(size):
            max_count = max(max_count, counts[g])
        capacity = capacity_of(max_count)
        slots = np.empty(capacity, a.dtype)
        gens = np.full(capacity, -1, np.int32)
        slot_counts = np.empty(capacity, np.int64)
        for g in range(size):
            _mode_group(g, a_sorted, counts, starts, out, slots, gens, slot_counts)
        return out

    return njit_stable(driver, name="mode_driver")


def _mode_mt_factory(driver: Callable[..., npt.NDArray[np.number]]) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the mode: the groups cut into ranges.

    The one thing the shape needs that the median does not is a hash table of
    its own per range - the table is scratch that every group of the range
    clears by its generation, which two ranges cannot do in one table.  Sized
    after the largest block of the range and not after the largest group of
    the whole call, like the table of the distinct count is.
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
        setup = _group_mt_setup(group_idx, a, size_arg, _NS_PER_VALUE_MODE, n_threads)
        if setup is None:
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
        # unpacked before the prange - see the note in _median_mt_factory
        counts, starts, a_sorted, edges, largest = setup
        out = np.full(counts.size, fill_value, out_dtype)
        tables = [
            (
                np.empty(capacity_of(k), a.dtype),
                np.full(capacity_of(k), -1, np.int32),
                np.empty(capacity_of(k), np.int64),
            )
            for k in largest
        ]
        for c in nb.prange(largest.size):
            slots, gens, slot_counts = tables[c]
            for g in range(edges[c], edges[c + 1]):
                _mode_group(g, a_sorted, counts, starts, out, slots, gens, slot_counts)
        return out

    return njit_stable(mt_passes, name="mode_mt_passes", parallel=True)


mode = build_custom_kernel(
    "mode",
    # the mode has no nan variant: the missing values are in no selection
    driver_factory=lambda nans: _mode_driver_factory(),
    mt_driver_factory=lambda nans, driver: _mode_mt_factory(driver),
    rule=rule_same,
    fill_default=nan,
    nans=False,
    doc="The most frequent value of every group - the smallest of the values that share the highest count",
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
    track = _fill_differs(fill_value, forced_fill)
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
