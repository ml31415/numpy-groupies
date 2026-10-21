"""The cumulative operations of the numpy implementation.

The cumulative sum of every group, and its nan-skipping variant.
"""

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.steps import group_bounds, group_order
from numpy_groupies.utils import AScalar


def _group_cumsum_sorted(
    group_idx_srt: npt.NDArray[np.integer], a_srt: npt.NDArray[np.number], dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.number]:
    """
    Cumsum within each contiguous group, with independent accumulators.

    Subtracting offsets from a global cumsum loses small increments after
    large groups, and nonfinite values contaminate subsequent groups.
    """
    starts, counts = group_bounds(group_idx_srt)
    # Match cumsum's default integer promotion, including for singleton groups.
    result_dtype = np.cumsum(a_srt[:0], dtype=dtype).dtype
    a_srt_cumsum = a_srt.astype(result_dtype, copy=True)
    # Batch equal-length groups into rows, keeping every accumulator separate
    # without a Python call for each short group. Singleton groups are done.
    for count in np.unique(counts[counts > 1]):
        selected = starts[counts == count]
        if selected.size == starts.size:
            np.cumsum(a_srt.reshape(-1, count), axis=1, dtype=dtype, out=a_srt_cumsum.reshape(-1, count))
        elif selected.size == 1:
            start = selected[0]
            np.cumsum(a_srt[start : start + count], dtype=dtype, out=a_srt_cumsum[start : start + count])
        else:
            positions = selected[:, None] + np.arange(count)
            a_srt_cumsum[positions] = np.cumsum(a_srt[positions], axis=1, dtype=dtype)
    return a_srt_cumsum


def _cumsum(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar | None = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    """
    N to N aggregate operation of cumsum. Perform cumulative sum for each group.

    NaNs propagate within their own group only - entries of other groups as
    well as entries preceding the first NaN of the same group are unaffected
    (issue #91).

    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    _cumsum(group_idx, a, np.max(group_idx) + 1)
    >>> array([ 3,  4,  5,  6, 15,  9, 15, 22,  7,  0, 15, 17,  6, 14, 31, 39])
    """
    order = group_order(group_idx, a)

    a_srt_cumsum = _group_cumsum_sorted(order.group_idx, order.values, dtype=dtype)

    ret = np.empty_like(a_srt_cumsum)
    ret[order.sortidx] = a_srt_cumsum
    return ret


def _nancumsum(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar | None = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    a_nonans = np.where(np.isnan(a), 0, a)
    group_idx_nonans = np.where(np.isnan(group_idx), np.nanmax(group_idx) + 1, group_idx)
    return _cumsum(group_idx_nonans, a_nonans, size, fill_value=fill_value, dtype=dtype)
