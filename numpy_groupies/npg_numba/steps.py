"""Jitted group-boundary utilities, shared with the other implementations.

``step_count`` and ``step_indices`` are public; the counting-sort helpers below
place each group's values in a contiguous block, compiled per operation so the
artifacts of one never reach into another.
"""

import numpy as np
import numpy.typing as npt

from numpy_groupies.npg_numba.jit import njit_stable
from numpy_groupies.validation import NEGATIVE_INDICES, TOO_LARGE


# the annotations of the jitted functions below are documentation only - numba
# compiles from the argument types, the type checker needs them anyway
@njit_stable
def step_count(group_idx: npt.NDArray[np.integer]) -> int:
    """Return the amount of index changes within group_idx."""
    cmp_pos = 0
    steps = 1
    if len(group_idx) < 1:
        return 0
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            steps += 1
    return steps


@njit_stable
def step_indices(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """Return the edges of areas within group_idx, which are filled with the same value."""
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, np.intp)
    indices[0] = 0
    indices[-1] = group_idx.size
    cmp_pos = 0
    ri = 1
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            indices[ri] = i
            ri += 1
    return indices


# the counting sort: every group's values in a contiguous block, in input
# order, cut into ranges of about equal value count for the parallel path.
# Plain functions - every operation compiles its own named copies (see the
# note in ``compose.build_kernel``).


def group_count(group_idx: npt.NDArray[np.integer], size: int) -> npt.NDArray[np.int64]:
    """bounds-checked per-group element counts"""
    counts = np.zeros(size, np.int64)
    for i in range(len(group_idx)):
        g = group_idx[i]
        if g < 0:
            raise ValueError(NEGATIVE_INDICES)
        if g >= size:
            raise ValueError(TOO_LARGE)
        counts[g] += 1
    return counts


def group_starts(counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """first index of each group-block"""
    return np.cumsum(counts) - counts


def group_place(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    cursor: npt.NDArray[np.int64],
    a_sorted: npt.NDArray[np.number],
) -> None:
    """scatter the values into their group-blocks"""
    for i in range(len(group_idx)):
        g = group_idx[i]
        a_sorted[cursor[g]] = a[i]
        cursor[g] += 1


def group_ranges(
    counts: npt.NDArray[np.int64],
    n_ranges: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """cut the groups into ``n_ranges`` consecutive ranges of about equal *value* count

    Returns the ``n_ranges + 1`` monotone group boundaries (first at 0, last at
    the group count) and the largest block of every range.  Equal in values,
    not in groups: the pool hands the iterations out once, at the start, so
    ranges of equal group count leave the thread that drew the heavy half
    waiting - on a zipf-like distribution the call takes a multiple of the
    sequential time.  A group bigger than the target cannot be split - it stays
    the single group of its range, which a block that selects or orders within
    the group requires anyway.  The largest-block counts size the per-range
    scratch (the nan variants' filter buffer, the distinct-count hash table);
    summed they stay within the values, so the parallel path holds no more
    scratch than the sequential one.
    """
    size = counts.size
    edges = np.zeros(n_ranges + 1, np.intp)
    largest = np.zeros(n_ranges, np.int64)
    total = int(counts.sum())
    target = max(total // n_ranges, 1)
    seen = 0
    part = 1
    for g in range(size):
        seen += counts[g]
        while part < n_ranges and seen > part * target:
            # the group that passed the target still belongs to the range before
            edges[part] = g + 1
            part += 1
    # ranges the walk never reached - one big group cannot fill every target -
    # hold no groups at all
    for part in range(1, n_ranges):
        if edges[part] == 0:
            edges[part] = size
    edges[n_ranges] = size
    for part in range(n_ranges):
        for g in range(edges[part], edges[part + 1]):
            largest[part] = max(largest[part], counts[g])
    return edges, largest
