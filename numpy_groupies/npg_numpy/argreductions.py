"""The arg-reductions of the numpy implementation: the index of the largest and
of the smallest value of every group."""

from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import (
    maxval,
    minval,
    object_nan_mask,
)
from numpy_groupies.functions import nan
from numpy_groupies.npg_numpy.reductions import _max, _min, _object_extreme
from numpy_groupies.typing import AScalar


def _object_arg_extreme(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[Any],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None,
    is_max: bool,
    skip_nans: bool,
) -> npt.NDArray[np.int64]:
    """The position of the extreme of every group of an object column.

    The first value of the group that is the extreme wins, and a group that
    holds a nan has no extreme to point at: it keeps the fill value.  The nan
    variants point at the extreme of the other values of the group.
    """
    nans = object_nan_mask(a)
    values, groups = a, group_idx
    if skip_nans:
        keep = ~nans
        values, groups = a[keep], group_idx[keep]
    extreme = _object_extreme(groups, values, size, nan, is_min=not is_max)
    is_extreme = (a == extreme[group_idx]) & ~nans
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_extreme = group_idx[is_extreme]
    (positions,) = is_extreme.nonzero()
    ret[group_idx_extreme[::-1]] = positions[::-1]  # reverse to ensure first value for each group wins
    return ret


def _argmax(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = int,
    _nansqueeze: bool = False,
) -> npt.NDArray[np.int64]:
    if a.dtype.kind == "O":
        # the values of an object column have no dtype of their own: the seed
        # of the reduction would be a finfo of a dtype that is not there
        return _object_arg_extreme(group_idx, a, size, fill_value, dtype, is_max=True, skip_nans=_nansqueeze)
    # the mask value is the smallest value of numpy's order, -inf-infj for a
    # complex dtype (see minval/maxval); integer input has no nans, and
    # replacing there could promote uint64 to float64
    a_ = np.where(np.isnan(a), minval(fill_value, a.dtype), a) if _nansqueeze and a.dtype.kind in "fc" else a
    group_max = _max(group_idx, a_, size, np.nan)
    # nan should never be maximum, so use a and not a_
    is_max = a == group_max[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_max = group_idx[is_max]
    (argmax,) = is_max.nonzero()
    ret[group_idx_max[::-1]] = argmax[::-1]  # reverse to ensure first value for each group wins
    return ret


def _argmin(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = int,
    _nansqueeze: bool = False,
) -> npt.NDArray[np.int64]:
    if a.dtype.kind == "O":
        # the values of an object column have no dtype of their own: the seed
        # of the reduction would be a finfo of a dtype that is not there
        return _object_arg_extreme(group_idx, a, size, fill_value, dtype, is_max=False, skip_nans=_nansqueeze)
    # the mask value is the largest value of numpy's order, inf+infj for a
    # complex dtype (see minval/maxval); integer input has no nans, and
    # replacing there could promote uint64 to float64
    a_ = np.where(np.isnan(a), maxval(fill_value, a.dtype), a) if _nansqueeze and a.dtype.kind in "fc" else a
    group_min = _min(group_idx, a_, size, np.nan)
    # nan should never be minimum, so use a and not a_
    is_min = a == group_min[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_min = group_idx[is_min]
    (argmin,) = is_min.nonzero()
    ret[group_idx_min[::-1]] = argmin[::-1]  # reverse to ensure first value for each group wins
    return ret
