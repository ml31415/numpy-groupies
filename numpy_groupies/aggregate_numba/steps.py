"""Jitted group-boundary utilities, shared with the other implementations.

The first two functions are the public ``step_count`` and ``step_indices``;
the three below are the building blocks of the counting sort that places the
values of a group in a contiguous block - the operations that need it (median,
sort and the collecting function) each compile their own copies of them, so
that the compiled artifacts of one operation never reach into another.
"""

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.validation import NEGATIVE_INDICES, TOO_LARGE


# the annotations of the jitted functions below are documentation only - numba
# compiles from the argument types and ignores them, the type checker needs them
@nb.njit(cache=True)
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


@nb.njit(cache=True)
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


# The four functions below are the counting sort: they place the values of
# every group in a contiguous block, in the order they appear in the input, and
# they cut that block-sequence into ranges of about equal value count for the
# parallel path.  They are plain functions - every operation compiles its own
# copies of them, with a name of its own (see the note in ``compose.build_kernel``).


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

    Returns the ``n_ranges + 1`` group boundaries (the first at 0, the last at
    the number of groups, monotone, so every group belongs to exactly one
    range) and the number of the largest block of every range.

    The ranges are equal in values and not in groups: a loop over the groups of
    a range is work proportional to the values of those groups, and the
    schedule of the pool hands the iterations out once, at the start.  Cutting
    ``size`` iterations in as many pieces as there are threads therefore leaves
    the thread that drew the heavy half of the groups waiting for the others -
    on a zipf-like distribution the call then takes a multiple of the
    sequential time, where the ranges of equal value count below stay close
    to it.

    The walk cuts a new range whenever the values it has seen pass the next
    multiple of ``total / n_ranges``.  A group bigger than the target cannot be
    split - it stays the single group of its range and that range is the heavy
    one - and it cannot be avoided either, since every work split of a block
    that selects or orders within the group has to keep the block whole.

    The number of the largest block is what the scratch of a range is sized
    after (the buffer the nan variants filter into, the hash table of the
    distinct count).  Summed over the ranges it is at most the number of the
    values - every range takes its scratch out of its own values - so the
    parallel path does not hold more scratch than the sequential one, whatever
    the number of the ranges.
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
    # the ranges the walk never reached - the values of one big group cannot
    # fill every target - hold no groups at all
    for part in range(1, n_ranges):
        if edges[part] == 0:
            edges[part] = size
    edges[n_ranges] = size
    for part in range(n_ranges):
        for g in range(edges[part], edges[part + 1]):
            largest[part] = max(largest[part], counts[g])
    return edges, largest
