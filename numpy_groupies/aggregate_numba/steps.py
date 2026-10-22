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


# The three functions below are the counting sort: they place the values of
# every group in a contiguous block, in the order they appear in the input.
# They are plain functions - every operation compiles its own copies of them,
# with a name of its own (see the note in ``compose.build_kernel``).


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
