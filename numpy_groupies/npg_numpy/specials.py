"""The order statistics of the numpy implementation: the median, the quantile,
the percentile, the number of distinct values, the sort - the operations that
need the values of a group in their order, gathered group by group first."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.fills import nan_of
from numpy_groupies.functions import QUANTILE_METHODS
from numpy_groupies.npg_numpy.steps import group_order, sort_key
from numpy_groupies.temporal import NAT_TICK
from numpy_groupies.typing import AResult, AScalar

# the quantile interpolations between the two values it falls between, under
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

    An order statistic: the values are gathered group by group and the middle
    of each group is *selected* via partition (O(n) on average) instead of
    sorting the values within the groups.

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _median(group_idx, a, np.max(group_idx) + 1)
    >>> array([0. , 8. , 4.5, 3. , 0. , 0. , 0. , 7.5, 0. ])
    """
    if group_idx.size == 0:
        return np.full(size, fill_value, dtype=dtype or np.result_type(a, np.float64))
    # every group's values in one contiguous block - any argsort kind works,
    # the median is insensitive to the order of equal group labels
    order = group_order(group_idx, a)
    group_idx_srt, a_srt = order.group_idx, order.values
    starts, counts = order.starts, order.counts
    mids = counts // 2
    odd = counts % 2 == 1

    # one kth is enough: partition places the kth element at its sorted
    # position and leaves the rest unsorted, so the lower middle of an even
    # group is the max of the part before it (np.partition orders complex
    # values lexicographically, so the dtype is kept); two kths nest the
    # selections, 10.1ms against 3.6ms for 1000 groups
    vals = np.empty(starts.size, dtype=np.result_type(a_srt, np.float64))
    # the two middles add in float64 for integral input: their own dtype
    # overflows (int8 100 + 100) and bool adds as or - np.median promotes first
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
        # like np.median, any nan poisons its whole group - partition does not
        # order nans, so a per-group reduction finds them, and the poison is
        # shaped like the result (a plain nan would leave a complex part at 0)
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

    An order statistic like the median: the values are gathered group by group
    and the wanted quantile is *selected* via partition (O(n) on average)
    instead of sorting the values within the groups, interpolated between its
    two neighbours as numpy does it.

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
    # every group's values in one contiguous block - any argsort kind works,
    # the quantile is insensitive to the order of equal group labels
    order = group_order(group_idx, a)
    group_idx_srt, a_srt = order.group_idx, order.values
    starts, counts = order.starts, order.counts

    vals = np.empty(starts.size, dtype=np.result_type(a_srt, np.float64))
    # the two middles add in float64 for integral input: their own dtype
    # overflows (int8 100 + 100) and bool adds as or - np.quantile promotes
    # first
    promote = np.issubdtype(a_srt.dtype, np.integer) or np.issubdtype(a_srt.dtype, np.bool_)
    for grp in range(starts.size):
        start, count = starts[grp], counts[grp]
        # the virtual index of the quantile, in fractions of the way from
        # 0 to n - 1 of the sorted group
        pos = q * (count - 1)
        if method == "nearest":
            # numpy rounds the index to the nearest element, halves to even
            # (np.round), and takes that element as it is
            index = min(int(np.round(pos)), count - 1)
            part = np.partition(a_srt[start : start + count], index)
            vals[grp] = part[index]
            continue
        lower_index = int(np.floor(pos))
        upper_index = int(np.ceil(pos))
        # the element at upper_index lands sorted, the ones before it stay
        # unsorted - their max is the one below (np.partition orders complex
        # values lexicographically, so the dtype is kept)
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
        # not order nans, so a per-group reduction finds them, and the poison
        # is shaped like the result (a plain nan would leave a complex part at 0)
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


def _topk(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
    k: int = 5,
    largest: bool = True,
) -> npt.NDArray[np.generic]:
    """The k largest (or the k smallest) values of every group, one row per
    group and k columns in the order of their rank - the largest first if
    ``largest`` is true, the smallest first otherwise; a group with fewer
    than k values fills the empty places with the fill value.

    The missing values - the nans of the inexact dtypes, the NaT of the
    temporal ones - are in no selection, and a group of nothing but them
    answers the fill value in every column.  One numpy partition per group
    leaves the k wanted ones at the end of the block.
    """
    out_dtype = np.dtype(dtype) if dtype is not None else a.dtype
    out = np.full((size, k), fill_value, dtype=out_dtype)
    if k == 0 or a.size == 0:
        return out
    # the missing values are left out before the groups are gathered
    good: npt.NDArray[np.bool_] | None = None
    if a.dtype.kind in "fc":
        good = ~np.isnan(a)
    elif a.dtype.kind in "mM":
        good = ~np.isnat(a)
    order = group_order(group_idx, a, good)
    starts, counts, values = order.starts, order.counts, order.values
    labels = order.group_idx
    for grp in range(starts.size):
        count = int(counts[grp])
        if count == 0:
            continue
        block = values[starts[grp] : starts[grp] + count]
        if count <= k:
            chosen = np.sort(block)
        elif largest:
            # the k wanted ones sit behind the k-th, in no order of their own
            chosen = np.sort(np.partition(block, count - k)[count - k :])
        else:
            # the k smallest sit in front of the k-th, which lands sorted
            chosen = np.sort(np.partition(block, k - 1)[:k])
        if largest:
            chosen = chosen[::-1]
        # a group with no value has no block and keeps the fill value
        out[labels[starts[grp]], : min(count, k)] = chosen[:k]
    return out


def _nunique(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.int64]:
    """The number of distinct values of every group: the ordered values that
    differ from their predecessor.  The nans are one value among themselves
    (a nan in either part of a complex value makes it one), counted apart -
    the plain function counts them, the nan variant never sees them.
    """
    if group_idx.size == 0:
        return np.full(size, fill_value, dtype=dtype or np.intp)
    if a.dtype.kind in "fc":
        nans = np.isnan(a)
    else:
        nans = np.zeros(a.size, dtype=bool)
    good = ~nans
    ret = np.full(size, fill_value, dtype=dtype or np.intp)
    if good.any():
        # the values of every group in one row, the nans left out
        order = group_order(group_idx, a, good)
        group_idx_srt, a_srt = order.group_idx, order.values
        starts, counts = order.starts, order.counts
        # distinct values: those that differ from their predecessor in the
        # ordered group
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


def _mode(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.generic]:
    """The most frequent value of every group, the smallest on a tie - the
    way ``scipy.stats.mode`` breaks one: in the ordered group the first run
    of maximal length wins.  Like the topk, the mode has no nan variant: the
    missing values are in no selection, and a group of nothing but them
    keeps the fill value, as does a group the labels never name.
    """
    out_dtype = np.dtype(dtype) if dtype is not None else a.dtype
    ret = np.full(size, fill_value, dtype=out_dtype)
    if group_idx.size == 0:
        return ret
    # the nans of the inexact dtypes are left out - a temporal input never
    # reaches here, the wrapper hands the int64 ticks over without them
    good: npt.NDArray[np.bool_] | None = None
    if a.dtype.kind in "fc":
        good = ~np.isnan(a)
        if not good.any():
            return ret
    order = group_order(group_idx, a, good)
    values, starts, counts = order.values, order.starts, order.counts
    labels = order.group_idx
    vals = np.empty(starts.size, dtype=out_dtype)
    for grp in range(starts.size):
        begin, count = starts[grp], counts[grp]
        block = np.sort(values[begin : begin + count])
        # runs of equal values are neighbours; the first maximal run is the
        # smallest of the values that share the count
        bounds = np.flatnonzero(np.concatenate(([True], block[1:] != block[:-1])))
        lengths = np.diff(np.append(bounds, block.size))
        vals[grp] = block[bounds[int(np.argmax(lengths))]]
    ret[labels[starts]] = vals
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
        # the descending key: bitwise inversion reverses the order of the
        # integral and boolean dtypes exactly, where negation does not (the
        # smallest value of a signed dtype survives negation as itself, a zero
        # of an unsigned one stays the smallest key) - the inexact dtypes
        # negate cleanly, a nan stays last either way
        key = cast("npt.NDArray[np.integer]", ~a) if a.dtype.kind in "iub" else cast("npt.NDArray[np.floating]", -a)
    else:
        key = a
    sort_group_idx = sort_key(group_idx)
    sortidx = np.lexsort((key, sort_group_idx))
    # unsort back into the original order, keeping the groupwise value sort:
    # scattering through the group-stable argsort is exactly the inverse
    # permutation, avoiding the classic argsort-of-argsort
    gsort = np.argsort(sort_group_idx, kind="stable")
    ret = cast(AResult, np.empty_like(a))
    ret[gsort] = a[sortidx]
    return ret


# ---------------------------------------------------------------------------
# the order statistics of the ticks of a temporal input
#
# computed on the int64 ticks, and exactly: the middle of an even group
# averages its two middles as python integers, where a float64 would round
# the answer off the tick grid, and the quantile scales a tick difference by
# a fraction - a product no int64 arithmetic holds


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
    """The quantile of the ticks of every group, tick for tick like numpy:
    the difference of the two neighbours scaled by the fraction, the product
    truncated towards zero like numpy's multiplication of a duration and a
    float (the two branches of its ``_lerp``) - below a half the answer grows
    from the lower neighbour, from a half on it shrinks from the upper one.
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
            # numpy rounds the position half to even and takes that element
            index = min(int(np.round(position)), count - 1)
            ret[grp] = np.partition(block, index)[index]
            continue
        if method == "lower":
            ret[grp] = np.partition(block, low)[low]
            continue
        if method == "higher":
            ret[grp] = np.partition(block, high)[high]
            continue
        # the two neighbours it falls between: the midpoint takes their mean,
        # the fraction of a half in numpy's interpolation
        gamma = 0.5 if method == "midpoint" else position - low
        if low == high:
            ret[grp] = np.partition(block, high)[high]
            continue
        # the element at the upper index lands sorted, the ones before it
        # stay unsorted - their largest is the one below
        part = np.partition(block, high)
        difference = int(part[high]) - int(part[:high].max())
        if gamma < 0.5:
            # the product truncates towards zero, like numpy's duration
            # times float
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
    """The median of the ticks of every group, exact: the two middles of an
    even group add up as python integers and round half to even, so even
    ticks at the edges of int64 cannot wrap the sum.
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
    wrapper calls with.  The nan form drops the rows of the NaT ticks before
    the selection, like the nan median of the numba backend does inside its
    kernel; the answer is a tick, and the median takes none.
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
    """The quantile of the ticks, behind the signature the temporal wrapper
    calls with - the interpolation scales a tick difference by a fraction, a
    product no int64 arithmetic holds, the way numpy runs it on the durations
    of the values.
    """
    return _temporal_group_quantile(group_idx, ticks, size, fill_tick, fraction, method)
