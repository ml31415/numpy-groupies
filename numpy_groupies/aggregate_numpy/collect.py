"""The collecting operation of the numpy implementation.

The members of every group, as an array of sequences - and the callable of
the caller, which sees the members of a group at a time.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.steps import sort_key
from numpy_groupies.nptools import blocks_to_object_array
from numpy_groupies.typing import AScalar


def _array(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: Sequence[AScalar],
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.object_]:
    """groups a into separate arrays, keeping the order intact."""
    order_group_idx = np.argsort(sort_key(group_idx), kind="stable")
    counts = np.bincount(group_idx, minlength=size)
    blocks = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    return blocks_to_object_array(blocks, group_idx, fill_value)


def _generic_callable(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar | None,
    dtype: npt.DTypeLike | None = None,
    func: Callable[..., AScalar] = lambda g: g,
    **kwargs: Any,
) -> npt.NDArray[np.number]:
    """groups a by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(group_idx, a, size, ())
    ret = np.full(size, fill_value, dtype=dtype or np.float64)

    for i, grp in enumerate(groups):
        if np.ndim(grp) == 1 and len(grp) > 0:
            ret[i] = func(grp)
    return ret
