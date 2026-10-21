"""The collecting operation of the numpy implementation.

The members of every group, as an array of sequences - and the callable of
the caller, which sees the members of a group at a time.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.steps import fill_untouched, sort_key
from numpy_groupies.utils_np import AScalar


def blocks_to_object_array(
    blocks: Iterable[npt.NDArray[np.generic]],
    group_idx: npt.NDArray[np.integer],
    fill_value: AScalar | Sequence[AScalar] | None,
) -> npt.NDArray[np.object_]:
    """The blocks of the groups as an object array, one sequence per group.

    A group without values takes the fill value - an empty sequence leaves the
    empty block of the group alone.  The array is built element by element:
    ``np.asanyarray`` of a list of blocks collapses it into a regular array
    whenever the groups happen to be of the same length - and the result would
    lose the sequences it is meant to hold (and the size the caller asked
    for).
    """
    if fill_value is not None and not (
        np.isscalar(fill_value) or (isinstance(fill_value, Sequence) and len(fill_value) == 0)
    ):
        raise ValueError("fill_value must be None, a scalar or an empty sequence")
    blocks = list(blocks)
    ret = np.empty(len(blocks), dtype=object)
    for g, block in enumerate(blocks):
        ret[g] = block
    if fill_value is None or np.isscalar(fill_value):
        fill_untouched(group_idx, ret, cast("AScalar", fill_value))
    return ret


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
