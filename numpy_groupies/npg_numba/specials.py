"""Operations with fully custom pass structures.

``median`` places the values group by group via a counting sort (O(n)) and
then *selects* the middle of each group (O(n) on average) instead of sorting
within the groups.  ``sort`` uses the same placement, orders every group-block
- by insertion sort while the block is small, by ``np.sort`` beyond that - and
scatters the blocks back to the positions their group occupies in the input.
``trapezoid`` integrates consecutive sample pairs within each group in a
single pass and takes the sample spacing ``dx`` as the argument of its own.

The answers have the dtype of the input - a number of the values' dtype, an
integer tick - and where the stubs of numpy know no element type (the element
is ``Any`` there, and ``typing.cast`` is not compiled in nopython mode) the
few places that read such an element state the type of the local instead.
"""

import functools
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.functions import nan
from numpy_groupies.npg_numba.compose import (
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
from numpy_groupies.npg_numba.jit import njit_stable
from numpy_groupies.npg_numba.mt import group_chunk_count, should_use_group_mt
from numpy_groupies.npg_numba.steps import group_count, group_place, group_ranges, group_starts
from numpy_groupies.temporal import NAT_TICK


def _group_valid_plain(
    g: int,
    a_sorted: npt.NDArray[np.number],
    counts: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
) -> int:
    """the number of usable values of group g, 0 if a nan poisons the group.

    The values stay where they are - the block is contiguous already and the
    selection copies it anyway (copying here as well would write every value
    twice, 1.6ms per 500k); the nans are looked for in a loop, not with a
    temporary array per group.
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
    """copy the usable values of group g into buf, returning their number.

    The missing value of the numbers is the nan (not equal to itself), of a
    temporal input the NaT tick -2**63, an ordinary int64 among the others.
    The flag is a constant of the closure, so numba compiles the two branches
    apart and the one that does not belong to the call is not even typed.
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


# partition passes per power of two of the block length, before the block is
# ordered instead - the quickselect is O(n) on average, the fallback keeps
# the worst case at O(n log n)
_SELECT_PASSES = 8


def _select_kth(
    arr: npt.NDArray[np.number],
    lo: int,
    hi: int,
    k: int,
) -> np.number:
    """The k-th smallest of ``arr[lo:hi]``, the block reordered in place.

    A quickselect: partitioned around a pivot until the position of the wanted
    element settles; the pivot is the median of the first, the middle and the
    last element, which keeps the passes logarithmic for sorted, reversed and
    constant blocks alike.  A block past the pass limit is ordered instead, so
    the worst case stays at O(n log n).  ``k`` is an index into ``arr``.

    Numba's ``np.partition`` would copy every block it is given and hand the
    copy back (9.2ms against 1.2ms over 500k values in 1000 groups) - the
    copy is not needed here, the block is a temporary.

    The values compare with ``complex_less``, the lexicographic order numpy
    uses for complex values; for the real dtypes the call is the plain
    comparison - the branch settles when the kernel is compiled.
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
            # the fallback: order the range, take the element from there
            arr[lo:hi].sort()
            # the value of the block
            ordered: np.number = arr[k]
            return ordered
        mid = lo + span // 2
        # the pivot: the median of the first, the middle and the last value,
        # with the ordering of the package
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
        # arr[lo..j] is at most the pivot, arr[j+1..hi-1] at least it - the
        # wanted element is in one of the two halves
        if k <= j:
            hi = j + 1
        else:
            lo = j + 1
        span = hi - lo
    # the value of the block
    element: np.number = arr[k]
    return element


# ---------------------------------------------------------------------------
# the work one group of the placed values needs, at the module level - the
# sequential driver of an order statistic and its parallel driver call the
# ONE compiled body below.  Jitted, disk cached and shared by its callers,
# like ``complex_less``: numba cannot resolve an uncached callee from cached
# caller code.

# the same helpers serve the median and the quantile - one compiled artifact
# for both
_valid_plain = njit_stable(_group_valid_plain)


# the nan scan of the numbers - the temporal one is the same scan with the
# NaT tick as the missing value (see ``_valid_nan_scan``)
@njit_stable(name="tick_holds_nat")
def _holds_nat_tick(a_sorted: npt.NDArray[np.number], begin: int, m: int) -> bool:
    """Whether the block holds the NaT tick - the block is not ordered, so it is scanned."""
    for i in range(begin, begin + m):
        if a_sorted[i] == NAT_TICK:
            return True
    return False


_valid_nan = _valid_nan_scan(False)
# the selection: one compiled artifact for the median, the quantile and both
# of their variants
_select = njit_stable(_select_kth)


@functools.lru_cache(maxsize=2)
def _middle_of(ticks: bool) -> Callable[..., np.number | int]:
    """the middle value(s) of the block at ``a_sorted[begin:begin+m]``.

    No sorting: the wanted element is selected in place, the median does not
    care about the order the selection leaves.

    An even group takes the mean of the two middles, and the two flavours
    differ in it: the numbers promote to float64, like np.median promotes
    before averaging (summing in their own dtype overflows - int64
    2**62 + 2**62 - and adds bool as a logical or), the ticks keep an exact
    int64 mean, where a pair around the epoch carries more bits than the 53
    of a float64 mantissa.  The flag is a constant of the closure, so the
    branch that does not belong to the call is not even typed.
    """

    @njit_stable(name=f"{'tick_' if ticks else ''}median_middle")
    def middle(a_sorted: npt.NDArray[np.number], begin: int, m: int) -> np.number | int:
        k = m // 2
        upper = _select(a_sorted, begin, begin + m, begin + k)
        if m % 2 == 1:
            return upper
        # the lower middle is the largest of the left half, which the
        # selection left in place
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

    No plain sum: two ticks of one end of the int64 range sum past it, the
    wraparound would answer a tick of no value of the group.  From the bits
    instead - ``a & b`` are the shared bits (in the mean once), ``a ^ b`` the
    differing ones (halved, and the shift of a negative number floors like
    the mean asks) - so ``q = (a & b) + ((a ^ b) >> 1)`` is
    ``floor((a + b) / 2)`` over the whole int64 range.  An odd sum leaves a
    half, and half to even takes the even neighbour: the low bit of ``a ^ b``
    is the sum's parity, the low bit of ``q`` the floor's - when both are
    set, the floor is odd and the neighbour above is taken.  Measured against
    a python ``Fraction`` over the corners of int64 and the ties
    (test_temporal, the bare kernel of the tick median).
    """
    q = (lower & upper) + ((lower ^ upper) >> 1)
    if (lower ^ upper) & 1 and q & 1:
        q += 1
    return q


_tick_pair_average = njit_stable(_tick_pair_average, name="tick_pair_average")


@functools.lru_cache(maxsize=4)
def _median_group(ticks: bool, nans: bool) -> Callable[..., None]:
    """the median of one group of the placed values.

    One body with two closure flags - the flavours (numbers/ticks) and the
    variants (plain/nan) - compiled into the four combinations, the branches
    that do not belong to the call not even typed.  The flavours differ in
    the middle of the block and in the missing value, the variants in what a
    missing value does to the group.  ``buf`` and ``fill_value`` are the two
    arguments the nan variant needs and the plain one does not; the shared
    signature keeps the drivers of the two variants the same loop.
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
            # the nan variant: the missing values are left out, a group of
            # nothing but them keeps the fill value
            m = valid_nan(g, a_sorted, counts, starts, buf)
            if m == 0:
                out[g] = fill_value
            else:
                out[g] = middle(buf, 0, m)
        elif ticks:
            # the plain variant of the ticks: a NaT tick takes the group over,
            # the answer of such a group is the NaT of the unit
            if _holds_nat_tick(a_sorted, starts[g], counts[g]):
                out[g] = NAT_TICK
            else:
                out[g] = middle(a_sorted, starts[g], counts[g])
        else:
            # the plain variant of the numbers: a nan poisons the group - the
            # poison is shaped like the output (a plain nan in a complex
            # column would leave its imaginary part at zero), so the cell's
            # fill value of the output dtype is multiplied by nan
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

    The virtual index q * (m - 1) names the position between two of the
    values, the two around it are interpolated the way numpy does.  All five
    methods of numpy run here - the branch on the method happens once per
    group, and a group costs far more than the branch.
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
    # front of the one it picked - the ones below it
    lower = a_sorted[begin]
    for i in range(begin + 1, begin + lower_index + 1):
        if complex_less(lower, a_sorted[i]):
            lower = a_sorted[i]
    # numpy's own ``_lerp`` branches: below a half the answer grows from the
    # lower value, from a half on it shrinks from the upper one.  The 1.0
    # promotes integral input to float64, like np.quantile promotes - the
    # subtraction in its own dtype overflows (int8 100 - (-100))
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

    numpy's own rule for a temporal value: the difference of the two ticks
    around the position is a duration, its product with the fraction of the
    way is truncated towards zero, the answer a tick again - below a half it
    grows from the lower neighbour, from a half on it shrinks from the upper
    one.  The product is taken in float64, like numpy takes it (the
    difference of the two ticks leaves the exactly-representable float64
    window early, and numpy rounds there).  ``lower``, ``higher``,
    ``nearest`` (halves to even) and ``midpoint`` name the neighbours in
    numpy's way.
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
    # front of the one it picked
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
    """the quantile of one group of the placed values.

    The arguments of the nan variant it does not use are there for the reason
    stated at ``_median_group``.
    """
    if counts[g] == 0:
        return  # untouched group, keeps the fill value
    m = _valid_plain(g, a_sorted, counts, starts)
    if m == 0:
        # a nan-poisoned group - the poison is shaped like the output: the
        # cell's fill value of the output dtype, times nan (a plain nan in a
        # complex column would leave its imaginary part at zero)
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

    The plain variant takes the group over like the plain variant of the
    numbers takes the group of a nan over; the nan variant leaves the NaT
    tick out, as every nan variant leaves the nan out.
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
            # plain ticks: a NaT tick takes the group over, the answer of
            # such a group is the NaT of the unit
            if _holds_nat_tick(a_sorted, starts[g], counts[g]):
                out[g] = NAT_TICK
            else:
                out[g] = _quantile_at_ticks(a_sorted, starts[g], counts[g], q, method)

    return group


# ---------------------------------------------------------------------------
# the counting sort and the range cut, at the module level - the four blocks
# below have no closure of their own (same source, same signature, same
# specializations), so ONE compiled artifact serves every operation that uses
# them; per-operation copies buy nothing (``njit_stable`` keys on source and
# closure, and neither differs here - the note at ``compose.build_kernel`` is
# about the blocks whose closures do)

_count = njit_stable(group_count)
_starts = njit_stable(group_starts)
_place = njit_stable(group_place)
_ranges = njit_stable(group_ranges)


@njit_stable
def _place_values(
    group_idx: npt.NDArray[np.integer], a: npt.NDArray[np.number], size: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.number]]:
    """The counting sort that places the values of every group in a block: the
    counts, the block starts, and the values in input order (a stable
    argsort's order, at a fraction of the cost).  The blocks it is built of
    have no closure of their own, so one compiled artifact serves every
    operation that uses them - and this one, the only place where they are
    put together.
    """
    counts = _count(group_idx, size)
    starts = _starts(counts)
    a_sorted = np.empty(group_idx.size, a.dtype)
    _place(group_idx, a, starts.copy(), a_sorted)
    return counts, starts, a_sorted


class GroupMTSetup(NamedTuple):
    """The shared setup of the parallel passes that cut the GROUPS into ranges -
    numba compiles the named tuple to the native struct a plain tuple becomes,
    the field access resolves when the caller is compiled."""

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
    """the shared setup of the parallel passes that cut the GROUPS into ranges.

    Cost-model gate, counting sort and range cut in one body for every
    operation of this shape; None keeps the call on the sequential path.
    None is safe because the callers unpack the setup before their prange
    loop and use plain variables in it: numba casts an Optional of a named
    tuple to its payload with a runtime check that fires inside prange bodies
    even when the value is valid - the cast belongs outside of the loop.
    Jitted and disk cached like everything a compiled caller reaches into:
    numba cannot resolve an uncached callee from cached caller code.
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


# the cost per value of the sequential kernels, measured on the machine of
# ``docs/benchmarks.md`` - all of them cost orders of magnitude more than the
# sum the cost model was fitted on, so the fixed cost of a parallel region is
# a far smaller share of their time
_NS_PER_VALUE_SELECT = 18.0
# the distinct count and the mode count their values in a hash table, no
# ordering needed
_NS_PER_VALUE_NUNIQUE = 13.0
_NS_PER_VALUE_SORT = 68.0
# the topk's selection, with the gathering of the values better than the
# border
_NS_PER_VALUE_TOPK = 24.0
# the distinct count's table, with the counter and the tie-break besides
_NS_PER_VALUE_MODE = 14.0


def _median_driver_factory(nans: bool, ticks: bool = False) -> Callable[..., npt.NDArray[np.number]]:
    # both drivers work on the ONE compiled body of the block of a group (see
    # ``_median_group``); the counting sort is the shared one of the module
    # (see the note at ``_count``)
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
        # plain one needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}{'tick_' if ticks else ''}median_driver")


def _median_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.number]], ticks: bool = False
) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the median: the groups cut into ranges.

    Gate, counting sort and range cut are the shared setup of
    ``_group_mt_setup``; every group belongs to exactly one range, so no two
    threads touch one block, nothing is folded afterwards and the result is
    the sequential driver's bit for bit.
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
            # not worth a parallel region: the sequential driver, bit for bit
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
        # unpacked before the prange - a tuple that crosses the parallel
        # border trips numba's gufunc creation (parfor_lowering reads a .loc
        # of it that is not there)
        counts, starts, a_sorted, edges, largest = setup
        # the same seeding of the empty groups as the sequential driver gives
        # them
        out = np.full(counts.size, fill_value, out_dtype)
        # the buffer of a range is sized after the largest block of that
        # range - summed over the ranges at most the number of the values, so
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
    selection the median uses, and numpy's rule of the interpolation on the
    ticks of the values."""
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
        # the plain one needs none
        buf = np.empty(group_idx.size, a.dtype) if nans else np.empty(0, a.dtype)
        for g in range(size):
            group(g, a_sorted, counts, starts, buf, out, fill_value, q, method)
        return out

    return njit_stable(driver, name=f"{'nan' if nans else ''}tick_quantile_driver")


def _tick_quantile_mt_factory(
    nans: bool, driver: Callable[..., npt.NDArray[np.int64]]
) -> Callable[..., npt.NDArray[np.number]]:
    """The parallel passes of the quantile of the ticks: the groups cut into
    ranges, every group in exactly one of them - the result of the sequential
    driver bit for bit."""
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
            # not worth a parallel region: the sequential driver, bit for bit
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
    # the numeric median's driver and parallel passes with the tick bodies:
    # the exact int64 middle, and the NaT tick as the missing value - the
    # answer stays on the tick grid
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

    It groups the data the way the median does: a counting sort places the
    values of every group in a contiguous block, the wanted quantile is
    selected out of it.  The percentile's driver scales its percent argument
    down before it reaches the selection.
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
        # plain one needs none
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
    """The parallel passes of the quantile, the groups cut into ranges - the
    same shape as ``_median_mt_factory``, for the same reasons."""
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
    # the median's counting sort and selection, with numpy's rule of the
    # interpolation on the ticks - the branch on the method happens once per
    # group
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
    # scaled down: the percentile is the quantile in percent
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


# blocks this small are ordered by insertion sort in the flat array, larger
# ones by ``np.sort``: the array view ``arr.sort()`` builds per group costs
# more than the ordering does once the groups are tiny.  Measured with numba
# 0.68 on 500k float64 values ordered group by group: insertion sort is
# eleven times faster at one value per group, five times at five, still a
# tenth ahead at fifty; from a hundred values upwards the library sort wins,
# by half at a hundred, six times at a thousand - the limit stays well below
# that crossover
_INSERTION_SORT_LIMIT = 32


@njit_stable
def _order_block(a_sorted: npt.NDArray[np.number], begin: int, size_g: int, reverse: bool) -> npt.NDArray[np.number]:
    """order one group-block of the flat array, descending if ``reverse``.

    ``size_g`` is at least two.  Small blocks walk into place where they are,
    beyond that ``np.sort`` of a view is the cheaper of the two (see
    ``_INSERTION_SORT_LIMIT``).  Both passes order ascending and leave a nan
    last: the library sort because that is what numpy does, the insertion pass
    because it shifts a nan right and stops only in front of another nan,
    keeping the nans in input order.  Complex values never take the insertion
    pass - numpy compares them partially, a pass that has to decide every
    pair cannot order that.

    ``reverse`` therefore cannot turn the whole block around - only the
    ordered prefix is flipped, which keeps the descending order exact:
    negation is not order reversing for the integral dtypes, which is why the
    python twins invert bitwise or flip the group's run instead.
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
    """order the values inside every group-block, the singletons left alone.

    ``_order_block`` of one group does not look at the block of another one,
    so the parallel driver of the sort walks the ranges over this - no fold,
    no lock, the blocks it writes are disjoint by construction.
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
    """write the ordered blocks back to the positions their group occupies in
    the input - the j-th value of a group goes to its j-th appearance"""
    cursor = starts.copy()
    for i in range(len(group_idx)):
        g = group_idx[i]
        out[i] = a_sorted[cursor[g]]
        cursor[g] += 1


def _sort_driver_factory(nans: bool) -> Callable[..., npt.NDArray[np.number]]:
    # the ordering and the scatter are the sort's own blocks - the counting
    # sort is the shared one of the module
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
# Selected, not ordered: the quickselect of the module names a border, the
# values better than it are gathered, the order of the few that are left is
# set afterwards.  The missing values are in no selection - the nan and the
# NaT tick compare neither way, so they are pushed to the end of the block
# before the selection runs; a selection over them would name a border that
# is no border at all.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _compact_valid_of(ticks: bool) -> Callable[..., int]:
    """The values of the block that are no missing value, pushed to its front;
    their number handed back.  The write index never runs ahead of the read
    index, so no unread value is overwritten.

    Numbers and ticks need two bodies: a comparison of a tick with a number
    is no comparison numba has; the closure flag is a compile-time constant,
    so the branch that does not belong to the call is not even typed.
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
    """The k values of the row in the order of their rank - largest first when
    ``largest`` is true, smallest otherwise.  The few values are ordered in
    place; a row of a hundred values and more goes through the library sort.
    """
    if k < 2:
        return
    if k > _INSERTION_SORT_LIMIT:
        # the library sort of a matrix row is a sort of a copy - the slice of
        # a two dimensional array is a copy in numba - so the values are
        # sorted in a buffer of their own
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

    The border of the selection is the k-th value of the group in rank order:
    the (m-k)-th one when the largest are wanted, the (k-1)-th when the
    smallest are.  The values better than the border are the answer, the
    places they leave are filled with the border itself - the group holds
    enough, the border being its k-th value.
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
    # the places the better values leave: the border itself
    i = found
    while i < k:
        out[g, i] = border
        i += 1
    _rank_order(out, g, k, largest)


def _topk_driver_factory(ticks: bool) -> Callable[..., npt.NDArray[np.number]]:
    """The compiled driver of the topk: the values placed in their blocks, the
    missing values pushed to the block's end, the k wanted ones selected out
    of what is left."""

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
    """The parallel passes of the topk: the blocks cut into ranges, one group
    per range - no fold, no lock, the answer of the sequential path bit for
    bit."""

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
                # the sequential loop's step, on the block of a group no
                # other thread has touched
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
    """the finalizer of splitmix64: a bijection of the 64 bits - every bit of
    the input reaches every bit of the output, two different inputs have two
    different outputs, which is why the counts come out right.  The
    wraparound of the unsigned arithmetic is what a hash wants: the bits of
    the word are the whole state.
    """
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB61D108B)
    return x ^ (x >> np.uint64(31))


# the bits of -0.0 - the two zeros are one value, so they have one hash
_ZERO_BITS = np.int64(-9223372036854775808)


@njit_stable
def _hash_float(v: float | np.floating[Any]) -> np.uint64:
    """the hash of a floating point value - over its bits, which name the
    value; the two zeros are the one exception, the same value on one hash.
    The bits beat the frexp of the mantissa and the exponent: 2.0ms against
    0.3ms per 500k values, more than half of the time the mode spends on its
    data.
    """
    bits = np.float64(v).view(np.int64)
    if bits == _ZERO_BITS:
        # -0.0 and 0.0 are the same value
        bits = np.int64(0)
    return _mix_bits(np.uint64(bits))


@njit_stable
def _hash_value(v: Any) -> np.uint64:
    """the hash of a value - the values that are equal have the same hash; a
    complex value mixes the hashes of its two parts."""
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

    Counted in a hash table with open addressing: a slot that holds the value
    already is a value seen before; the group's generation marks the slots in
    use, so the table needs no clearing between groups.  The nans are not
    counted - one value among themselves, added by the caller.
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
    """the number of the distinct values of one group of the placed values.

    ``slots``/``gens`` are the caller's hash table, one per parallel range -
    one thread writes into its own, none of another; the group index is the
    generation, so no clearing between groups.
    """
    n = counts[g]
    if n == 0:
        return  # untouched group, keeps the fill value
    # this group's table holds only as many slots as it needs - the slots of
    # the other groups are not touched
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
    """the slots for a block of ``n`` values - two per value, so the table is
    never more than half full and the probing stays short."""
    capacity = 1
    while capacity < 2 * n:
        capacity *= 2
    return capacity


_table_capacity = njit_stable(_table_capacity)


def _nunique_driver_factory(nans: bool) -> Callable[..., npt.NDArray[np.number]]:
    """The driver of the distinct count: counting-sort placement, then a hash
    table per group - no ordering of the values needed.  The plain variant
    counts the nans as one value, the nan variant leaves them out.
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

    A hash table of its own per range - scratch that every group clears by
    its generation, which two ranges cannot do in one table.  Sized after the
    range's largest block: summed over the ranges at most twice the number of
    the values, where a table per the global maximum would be that per thread.
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
# the most frequent value of every group - counted, not ordered: the
# distinct count's hash table with the counts kept and the best value tracked
# besides.  A tie hands the answer to the smallest of the highest-count
# values, the way ``scipy.stats.mode`` breaks one.  The missing values (nan,
# NaT, detected by ``val != val``) are in no selection; a group of nothing
# but them has no mode and keeps the fill value, and a temporal answer is a
# value of the input as given.
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

    ``slots``/``gens``/``slot_counts`` are the caller's table, one per
    parallel range; the group index is the generation, so no clearing between
    groups - a stale slot is overwritten at the first touch.
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
        # a higher count takes over, an even count only for the smaller value
        # - the smallest of the most frequent ones wins.  An equal value
        # leaves the answer alone, so it stays the first of the equal ones
        # and a zero keeps its sign - the numpy backend's first of the
        # longest run agrees on that
        if seen > best_count:
            best_count = seen
            if v != best_value:
                best_value = v
        elif seen == best_count and complex_less(v, best_value):
            best_value = v
    if best_count > 0:
        out[g] = best_value


def _mode_driver_factory() -> Callable[..., npt.NDArray[np.number]]:
    """The driver of the mode: counting-sort placement, then the hash table
    count of every block - no ordering of the values needed, like the
    distinct count."""
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

    A hash table of its own per range, sized after the range's largest block,
    like the table of the distinct count is.
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
    # the kernel's default is the integer 1: a float in that slot compiles
    # the trapezoid kernel a few percent slower
    variant="dx",
    doc="Trapezoidal integration of the samples of every group",
)
