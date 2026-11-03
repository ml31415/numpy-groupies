"""The operations of the numpy implementation that work on ordered values.

The median, the quantile, the percentile, the number of distinct values and
the sort - the operations that need the values of a group in their order, and
so gather them group by group first.
"""

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.steps import group_order, sort_key
from numpy_groupies.common import nan_of
from numpy_groupies.nptools import QUANTILE_METHODS
from numpy_groupies.temporal import NAT_TICK
from numpy_groupies.typing import AResult, AScalar

# the ways a quantile interpolates between the two values it falls between -
# the names numpy uses for them
_QUANTILE_METHODS = QUANTILE_METHODS


def _median(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.inexact]:
    """
    Aggregate operation of the median within each group.

    The median is an order statistic, so unlike the streaming reductions it
    works on group-ordered data: the values are gathered group by group and
    the middle of each group is then *selected* via partition (O(n) on
    average) instead of sorting the values within the groups.

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _median(group_idx, a, np.max(group_idx) + 1)
    >>> array([0. , 8. , 4.5, 3. , 0. , 0. , 0. , 7.5, 0. ])
    """
    if group_idx.size == 0:
        return np.full(size, fill_value, dtype=dtype or np.result_type(a, np.float64))
    # the values of every group in a contiguous block - any argsort kind
    # works, the median is insensitive to the order of equal group labels
    order = group_order(group_idx, a)
    group_idx_srt, a_srt = order.group_idx, order.values
    starts, counts = order.starts, order.counts
    mids = counts // 2
    odd = counts % 2 == 1

    # partition places the kth element at its sorted position while leaving
    # the rest unsorted - one kth is enough: the lower middle of an even group
    # is the largest of the elements before it, which the partition leaves
    # there (np.partition orders complex values lexicographically, so the
    # dtype is kept).  Two kths would make numpy select twice, nested into
    # each other - measured 10.1ms against 3.6ms for the same 1000 groups
    vals = np.empty(starts.size, dtype=np.result_type(a_srt, np.float64))
    # the mean of the two middles is taken in float64 for integral input -
    # adding them in their own dtype overflows (int8 100 + 100) and bool
    # addition is a logical or (True + True), while np.median promotes first
    promote = np.issubdtype(a_srt.dtype, np.integer) or np.issubdtype(a_srt.dtype, np.bool_)
    for grp in range(starts.size):
        start, mid = starts[grp], mids[grp]
        part = np.partition(a_srt[start : start + counts[grp]], mid)
        if odd[grp]:
            vals[grp] = part[mid]
        elif promote:
            vals[grp] = (part[:mid].max().astype(np.float64) + part[mid]) / 2
        else:
            vals[grp] = (part[:mid].max() + part[mid]) / 2
    if np.issubdtype(a_srt.dtype, np.inexact):
        # like np.median, any nan poisons its whole group - partition does
        # not order nans, so they are detected with a per-group reduction, and
        # the poison is shaped like the result rather than written in as a plain
        # nan, which for a complex group would leave its other part at a zero
        nan_groups = np.logical_or.reduceat(np.isnan(a_srt), starts)
        vals[nan_groups] = nan_of(vals.dtype)

    ret = np.full(size, fill_value, dtype=dtype or vals.dtype)
    ret[group_idx_srt[starts]] = vals
    return ret


def _quantile(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
    q: float = 0.5,
    method: str = "linear",
) -> npt.NDArray[np.inexact]:
    """
    The quantile of the values within each group.

    The quantile is an order statistic, so it works on group-ordered data like
    the median: the values are gathered group by group and the wanted
    quantile is then *selected* via partition (O(n) on average) instead of
    sorting the values within the groups.  The two values the quantile falls
    between are interpolated as numpy does it.

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _quantile(group_idx, a, np.max(group_idx) + 1, q=0.5)
    >>> array([0. , 8. , 4.5, 3. , 0. , 0. , 0. , 7.5, 0. ])
    """
    if method not in _QUANTILE_METHODS:
        raise ValueError(f"unknown interpolation method '{method}' - use one of {sorted(_QUANTILE_METHODS)}")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantiles must be in the range [0, 1]")
    if group_idx.size == 0:
        return np.full(size, fill_value, dtype=dtype or np.result_type(a, np.float64))
    # the values of every group in a contiguous block - any argsort kind
    # works, the quantile is insensitive to the order of equal group labels
    order = group_order(group_idx, a)
    group_idx_srt, a_srt = order.group_idx, order.values
    starts, counts = order.starts, order.counts

    vals = np.empty(starts.size, dtype=np.result_type(a_srt, np.float64))
    # the mean of the two middles is taken in float64 for integral input -
    # adding them in their own dtype overflows (int8 100 + 100) and bool
    # addition is a logical or (True + True), while np.quantile promotes first
    promote = np.issubdtype(a_srt.dtype, np.integer) or np.issubdtype(a_srt.dtype, np.bool_)
    for grp in range(starts.size):
        start, count = starts[grp], counts[grp]
        # the virtual index of the quantile: the position it asks for in the
        # sorted group, counted in fractions of the way from 0 to n - 1
        pos = q * (count - 1)
        if method == "nearest":
            # numpy rounds the virtual index to the nearest element, halves to
            # even (np.round), and takes that element as it is
            index = min(int(np.round(pos)), count - 1)
            part = np.partition(a_srt[start : start + count], index)
            vals[grp] = part[index]
            continue
        lower_index = int(np.floor(pos))
        upper_index = int(np.ceil(pos))
        # the partition places the element at upper_index at its sorted
        # position and leaves the elements before it in no particular order -
        # their largest is the one below (np.partition orders complex values
        # lexicographically, so the dtype is kept)
        part = np.partition(a_srt[start : start + count], upper_index)
        upper = part[upper_index]
        if lower_index == upper_index:
            # the quantile is one of the values of the group
            lower = upper
        elif promote:
            lower = part[:upper_index].max().astype(np.float64)
        else:
            lower = part[:upper_index].max()
        if method == "lower":
            vals[grp] = lower
        elif method == "higher":
            vals[grp] = upper
        elif method == "midpoint":
            vals[grp] = (lower + upper) / 2
        else:
            # the linear interpolation between the two neighbours
            vals[grp] = lower + (upper - lower) * (pos - lower_index)
    if np.issubdtype(a_srt.dtype, np.inexact):
        # like np.quantile, any nan poisons its whole group - partition does
        # not order nans, so they are detected with a per-group reduction, and
        # the poison is shaped like the result rather than written in as a plain
        # nan, which for a complex group would leave its other part at a zero
        nan_groups = np.logical_or.reduceat(np.isnan(a_srt), starts)
        vals[nan_groups] = nan_of(vals.dtype)

    ret = np.full(size, fill_value, dtype=dtype or vals.dtype)
    ret[group_idx_srt[starts]] = vals
    return ret


def _percentile(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
    q: float = 50.0,
    method: str = "linear",
) -> npt.NDArray[np.inexact]:
    """The percentile of the values within each group - the quantile in percent."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("Percentiles must be in the range [0, 100]")
    return _quantile(group_idx, a, size, fill_value, dtype=dtype, q=q / 100.0, method=method)


def _nunique(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.int64]:
    """The number of distinct values of every group.

    The values of a group are gathered and ordered, and the values that
    differ from their predecessor are the distinct ones.  The nans are one
    value among themselves - a nan in either part of a complex value makes
    it one - so they are left out of the ordering and counted apart: the
    plain function counts them, the nan variant never sees them.
    """
    if group_idx.size == 0:
        return np.full(size, fill_value, dtype=dtype or np.int64)
    if a.dtype.kind in "fc":
        nans = np.isnan(a)
    else:
        nans = np.zeros(a.size, dtype=bool)
    good = ~nans
    ret = np.full(size, fill_value, dtype=dtype or np.int64)
    if good.any():
        # the values of every group in one row, the nans left out
        order = group_order(group_idx, a, good)
        group_idx_srt, a_srt = order.group_idx, order.values
        starts, counts = order.starts, order.counts
        # the distinct values of every group: the values that differ from
        # their predecessor in the ordered group
        vals = np.empty(starts.size, dtype=np.int64)
        for grp in range(starts.size):
            start, count = starts[grp], counts[grp]
            block = np.sort(a_srt[start : start + count])
            vals[grp] = 1 + np.count_nonzero(block[1:] != block[:-1])
        ret[group_idx_srt[starts]] = vals
    if nans.any():
        # the nans are one value of their own, in every group that holds one
        ret += np.bincount(group_idx[nans], minlength=size) > 0
    return ret


def _sort(
    group_idx: npt.NDArray[np.integer],
    a: Any,
    size: int | tuple[int, ...] | None = None,
    fill_value: AScalar | None = None,
    dtype: npt.DTypeLike | None = None,
    reverse: bool = False,
) -> AResult:
    if reverse:
        # the descending key.  Bitwise inversion reverses the order of the
        # integral and boolean dtypes exactly, where negation does not: the
        # smallest value of a signed dtype survives negation as itself, and a
        # zero of an unsigned one stays the smallest key of all.  The inexact
        # dtypes have no inversion and negate cleanly - a nan stays last
        # either way, and -0.0 compares equal to 0.0.
        key = cast("npt.NDArray[np.integer]", ~a) if a.dtype.kind in "iub" else cast("npt.NDArray[np.floating]", -a)
    else:
        key = a
    sort_group_idx = sort_key(group_idx)
    sortidx = np.lexsort((key, sort_group_idx))
    # Unsort back into original order, but preserving the groupwise value
    # sorting: scattering through the group-stable argsort is exactly the
    # inverse permutation, avoiding the classic argsort-of-argsort.
    gsort = np.argsort(sort_group_idx, kind="stable")
    ret = cast(AResult, np.empty_like(a))
    ret[gsort] = a[sortidx]
    return ret


# ---------------------------------------------------------------------------
# the order statistics of the ticks of a temporal input
#
# The median and the quantile of a temporal input are computed on the int64
# ticks of its values, and exactly: the middle of an even group averages its
# two middles as python integers, where a float64 would round the answer off
# the tick grid, and the quantile interpolates between two ticks, which
# scales their difference by a fraction - a product no int64 arithmetic
# holds.


def _round_half_even(total: int, count: int) -> int:
    """The exact integer division of ``total`` by ``count``, half to even."""
    q, r = divmod(total, count)
    if 2 * r > count or (2 * r == count and q % 2 == 1):
        q += 1
    return q


def _temporal_group_quantile(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_tick: int,
    q: float,
    method: str = "linear",
) -> npt.NDArray[np.int64]:
    """The quantile of the ticks of every group, the way numpy computes it.

    numpy computes the quantile of a temporal value by interpolating between
    the two ticks it falls between - the difference of the two scaled by the
    fraction of the way - and it does that in the arithmetic of the ticks:
    the difference is a duration, the product of it and the fraction is
    truncated towards zero, and the answer is a tick again.  The two
    neighbours are taken from the sorted group, and the branch of the
    interpolation depends on the fraction: below a half the answer grows
    from the lower neighbour, from a half on it shrinks from the upper one.

    The two branches are the ones of numpy's own ``_lerp``, and the
    truncation is the one of its multiplication of a duration and a float -
    so the answers are the answers of numpy, tick for tick.
    """
    order = np.argsort(group_idx, kind="stable")
    values = ticks[order]
    counts = np.bincount(group_idx, minlength=size)
    starts = np.zeros(size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    ret = np.full(size, NAT_TICK, dtype=np.int64)
    if method not in _QUANTILE_METHODS:
        raise ValueError(f"unknown interpolation method '{method}' - use one of {sorted(_QUANTILE_METHODS)}")
    for grp in np.flatnonzero(counts):
        count = int(counts[grp])
        block = values[starts[grp] : starts[grp] + count]
        position = q * (count - 1)
        low = min(int(np.floor(position)), count - 1)
        high = min(int(np.ceil(position)), count - 1)
        if method == "nearest":
            # numpy rounds the position to the nearest element, halves to
            # even, and takes that element as it is
            index = min(int(np.round(position)), count - 1)
            ret[grp] = np.partition(block, index)[index]
            continue
        if method == "lower":
            ret[grp] = np.partition(block, low)[low]
            continue
        if method == "higher":
            ret[grp] = np.partition(block, high)[high]
            continue
        # the two neighbours it falls between: the midpoint takes their mean
        # - numpy interpolates it with the fraction of a half
        gamma = 0.5 if method == "midpoint" else position - low
        if low == high:
            ret[grp] = np.partition(block, high)[high]
            continue
        # the partition places the element at the upper index at its sorted
        # position and leaves the ones before it in no particular order -
        # their largest is the one below
        part = np.partition(block, high)
        difference = int(part[high]) - int(part[:high].max())
        if gamma < 0.5:
            # the product of the difference and the fraction is truncated
            # towards zero, like numpy's multiplication of a duration and a
            # float truncates it
            ret[grp] = int(part[:high].max()) + int(np.float64(difference) * np.float64(gamma))
        else:
            ret[grp] = int(part[high]) - int(np.float64(difference) * np.float64(1 - gamma))
    ret[counts == 0] = fill_tick
    return ret


def _temporal_group_median(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_tick: int,
) -> npt.NDArray[np.int64]:
    """The median of the ticks of every group, exact.

    The values of a group are brought together by a stable argsort, a
    partition finds the middle of the block, and the two middles of an even
    group average to their exact mean rounded half to even - the two add up
    as python integers, so even ticks at the edges of int64 cannot wrap the
    sum.
    """
    order = np.argsort(group_idx, kind="stable")
    values = ticks[order]
    counts = np.bincount(group_idx, minlength=size)
    starts = np.zeros(size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    ret = np.full(size, NAT_TICK, dtype=np.int64)
    for grp in np.flatnonzero(counts):
        count = int(counts[grp])
        part = np.partition(values[starts[grp] : starts[grp] + count], count // 2)
        if count % 2 == 1:
            ret[grp] = part[count // 2]
        else:
            ret[grp] = _round_half_even(int(part[: count // 2].max()) + int(part[count // 2]), 2)
    ret[counts == 0] = fill_tick
    return ret


def _temporal_median(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_value: int | None = None,
    dtype: npt.DTypeLike | None = None,
    nan: bool = False,
    **kwargs: object,
) -> npt.NDArray[np.int64]:
    """The exact median of the ticks, behind the signature the temporal
    wrapper calls the tick median of a backend with.

    The nan form drops the rows of the NaT ticks before the selection - the
    same work the nan median of the numba backend does inside its kernel.
    The dtype of the caller and the arguments of no other function reach
    the selection: the answer is a tick, and the median takes none.
    """
    fill_tick = NAT_TICK if fill_value is None else int(fill_value)
    if nan:
        good = ticks != NAT_TICK
        if not good.all():
            group_idx, ticks = group_idx[good], ticks[good]
    return _temporal_group_median(group_idx, ticks, size, fill_tick)


def _temporal_quantile(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_tick: int,
    fraction: float,
    method: str = "linear",
) -> npt.NDArray[np.int64]:
    """The quantile of the ticks, behind the signature the temporal
    wrapper calls the tick quantile of a backend with.

    The quantile interpolates between the two ticks it falls between, and
    the difference of the two is scaled by the fraction - a product that no
    int64 arithmetic holds, so the interpolation runs on the ticks the way
    numpy runs it on the durations of the values.
    """
    return _temporal_group_quantile(group_idx, ticks, size, fill_tick, fraction, method)
