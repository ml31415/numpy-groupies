"""The arg-reductions of the numpy implementation: the index of the largest and
of the smallest value of every group."""

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import (
    maxval,
    minval,
)
from numpy_groupies.npg_numpy.reductions import _max, _min
from numpy_groupies.typing import AScalar


def _argmax(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = int,
    _nansqueeze: bool = False,
) -> npt.NDArray[np.int64]:
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
