"""The group boundaries of the numpy implementation, and the steps the
operations on group-ordered data share: ``step_count``, ``step_indices`` and
the gathering of every group's values into one contiguous block."""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


def step_count(group_idx: npt.NDArray[np.integer]) -> int:
    """The number of contiguous runs in ``group_idx``, matching
    ``npg_numba.step_count`` exactly."""
    group_idx = np.asarray(group_idx)
    if group_idx.size < 1:
        return 0
    return 1 + int(np.count_nonzero(np.diff(group_idx) != 0))


def step_indices(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """The run-edge indices of ``group_idx``: 0, the start of each new run and
    ``group_idx.size`` - an empty input answers ``[0]``, like the numba one."""
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
    """The group indices in the dtype the sorts work fastest with: numpy's
    radix sort works per byte of its key, int16 costs 8ms against 33ms per
    500k values of int64 - the order is the same, only narrower."""
    if group_idx.size == 0:
        return group_idx
    if group_idx.min() < 0 or group_idx.max() > np.iinfo(np.int16).max:
        return group_idx
    return group_idx.astype(np.int16)


def group_bounds(group_idx_srt: npt.NDArray[np.integer]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """The start index and length of every group in the sorted indices - the
    groups are the runs of equal indices."""
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
    """The values of every group in one contiguous block, in input order, with
    the bounds of the groups within it.

    ``good`` holds values back (the nans of the plain nunique, say) - the mask
    is applied to indices and values one after another, and the index copy is
    released before the sort: the sort needs a buffer of its own, and the
    allocator does not hand it the memory of a live copy (6ms of 20 for 500k
    values).
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
