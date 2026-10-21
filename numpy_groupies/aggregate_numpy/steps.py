"""The group boundaries of the numpy implementation, and the steps the
operations on group-ordered data share.

The first two functions are the public ``step_count`` and ``step_indices``.
The ones below them put the values of every group into a contiguous block -
the operations that need it (the median, the quantile, the sort, the
cumulative sums and the collecting function) all start there.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from numpy_groupies.utils_np import AScalar


def step_count(group_idx: npt.NDArray[np.integer]) -> int:
    """Return the number of contiguous runs in ``group_idx``.

    A strictly increasing array of length ``n`` has ``n`` runs, an empty
    array has 0, and an array of length 1 has 1. The return value matches
    ``aggregate_numba.step_count`` exactly, so the two backends can be used
    interchangeably.
    """
    group_idx = np.asarray(group_idx)
    if group_idx.size < 1:
        return 0
    return 1 + int(np.count_nonzero(np.diff(group_idx) != 0))


def step_indices(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """Return the run-edge indices in ``group_idx``.

    The first element is always 0, the last is always ``group_idx.size``,
    and the interior elements are the start indices of each new run. An
    empty input returns ``[0]`` to match the numba implementation.
    """
    group_idx = np.asarray(group_idx)
    n = group_idx.size
    if n < 1:
        return np.array([0], dtype=np.int64)
    starts = np.flatnonzero(np.diff(group_idx) != 0) + 1
    return np.concatenate(([0], starts, [n])).astype(np.int64, copy=False)


class GroupOrder(NamedTuple):
    """The values of the groups, gathered group by group."""

    sortidx: npt.NDArray[np.intp]
    group_idx: npt.NDArray[np.integer]
    values: npt.NDArray[np.number]
    starts: npt.NDArray[np.intp]
    counts: npt.NDArray[np.intp]


def sort_key(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """The group indices in the dtype the sorts work fastest with.

    The operations that work on group-ordered data sort the group indices,
    and numpy's radix sort works per byte of its key: the int64 indices cost
    four times the int16 ones (33ms against 8ms per 500k values).  A cast to
    the narrower dtype pays off whenever the group indices fit into it - the
    order of the values is the same, only their representation is narrower.
    """
    if group_idx.size == 0:
        return group_idx
    if group_idx.min() < 0 or group_idx.max() > np.iinfo(np.int16).max:
        return group_idx
    return group_idx.astype(np.int16)


def group_bounds(group_idx_srt: npt.NDArray[np.integer]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """The start index and the length of every group in the sorted indices.

    The group indices have to be sorted: the groups are the runs of equal
    indices, and their starts are where a run begins.
    """
    if group_idx_srt.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    new_group = np.empty(group_idx_srt.size, dtype=bool)
    new_group[0] = True
    np.not_equal(group_idx_srt[1:], group_idx_srt[:-1], out=new_group[1:])
    starts = np.flatnonzero(new_group)
    counts = np.diff(np.append(starts, group_idx_srt.size))
    return starts, counts


def group_order(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    good: npt.NDArray[np.bool_] | None = None,
) -> GroupOrder:
    """The values of every group in a contiguous block, in the order they appear.

    The operations that need the values of a group in their order start here:
    they get the values of every group in one row, and the bounds of the
    groups within that row.

    ``good`` leaves the values out that the mask holds back (the nans of the
    plain nunique, for instance).  The mask is applied to the group indices
    and to the values one after another, and the copy of the group indices
    is released before the sort runs: the sort needs a buffer of its own,
    and the allocator does not hand it the memory of a copy that is still in
    use (measured 6ms of the 20 for 500k values).
    """
    if good is None:
        sortidx = np.argsort(sort_key(group_idx), kind="stable")
        group_idx_srt = group_idx[sortidx]
        a_srt = a[sortidx]
    else:
        sortidx = np.argsort(sort_key(group_idx[good]), kind="stable")
        group_idx_srt = group_idx[good][sortidx]
        a_srt = a[good][sortidx]
    starts, counts = group_bounds(group_idx_srt)
    return GroupOrder(sortidx, group_idx_srt, a_srt, starts, counts)


def fill_untouched(idx: npt.NDArray[np.integer], ret: npt.NDArray[np.generic], fill_value: AScalar) -> None:
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value
