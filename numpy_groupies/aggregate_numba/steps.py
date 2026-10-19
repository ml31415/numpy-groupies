"""Jitted group-boundary utilities, shared with the other implementations."""

import numba as nb
import numpy as np
import numpy.typing as npt


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
def step_indices(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    """Return the edges of areas within group_idx, which are filled with the same value."""
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, np.int64)
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
