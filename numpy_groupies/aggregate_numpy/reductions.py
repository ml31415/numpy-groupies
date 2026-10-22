"""The streaming reductions of the numpy implementation.

Every function of this module reduces the values of a group to a single
number, without needing the values of the group in any particular order.
"""

from typing import cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.steps import fill_untouched
from numpy_groupies.utils_np import (
    AScalar,
    iscomplexobj,
    maxval,
    minimum_dtype,
    minimum_dtype_scalar,
    minval,
)
from numpy_groupies.validation import check_boolean, is_scalar_input


def _sum(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype_scalar(fill_value, dtype, a)

    if is_scalar_input(a):
        ret = np.bincount(group_idx, minlength=size).astype(dtype, copy=False)
        if a != 1:
            ret *= a
    else:
        exact_integer = False
        if a.dtype.kind in "iu" and np.dtype(dtype).kind in "iu":
            # the kind test above established an integer dtype
            limits = np.iinfo(cast("np.dtype[np.integer]", a.dtype))
            bound = max(abs(int(limits.min)), int(limits.max))
            if a.size * bound > 2**53:
                bound = max(abs(int(a.min(initial=0))), abs(int(a.max(initial=0))))
                exact_integer = a.size * bound > 2**53
        if iscomplexobj(a):
            ret = np.empty(size, dtype=dtype)
            ret.real = np.bincount(group_idx, weights=a.real, minlength=size)
            ret.imag = np.bincount(group_idx, weights=a.imag, minlength=size)
        elif exact_integer:
            # bincount converts weights to float64. Keep integer accumulation
            # when its partial sums are not guaranteed exactly representable.
            ret = np.zeros(size, dtype=dtype)
            np.add.at(ret, group_idx, a.astype(dtype, copy=False))
        else:
            # integer input, the kind test above guarantees it, the stubs want
            # the union of all bincount-legal dtypes
            ret = np.bincount(group_idx, weights=cast("npt.NDArray[np.integer]", a), minlength=size).astype(
                dtype, copy=False
            )

    if fill_value != 0:
        fill_untouched(group_idx, ret, fill_value)
    return ret


def _prod(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 1:
        ret[group_idx] = 1  # product starts from 1
    np.multiply.at(ret, group_idx, a)
    return ret


def _len(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.int64]:
    # _sum folds a scalar column of ones into its ndim-0 branch
    # _sum folds a scalar column of ones into its ndim-0 branch
    return cast(
        "npt.NDArray[np.int64]",
        _sum(group_idx, cast("npt.NDArray[np.number]", 1), size, fill_value, dtype=int),
    )


def _last(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    ret[group_idx] = a
    return ret


def _first(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[group_idx[::-1]] = a[::-1]  # same trick as _last, but in reverse
    return ret


def _all(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.bool_],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.bool_]:
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if not fill_value:
        ret[group_idx] = True
    ret[group_idx.compress(np.logical_not(a))] = False
    return ret


def _any(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.bool_],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.bool_]:
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if fill_value:
        ret[group_idx] = False
    # convert to bool explicitly - ndarray.compress on a float mask is slow
    mask = np.asarray(a, dtype=bool)
    # numpy quirk: compress wins for sparse masks, fancy indexing for dense
    # ones (up to ~3x either way) - so pick based on mask density
    if np.count_nonzero(mask) < mask.size / 2:
        ret[group_idx.compress(mask)] = True
    else:
        ret[group_idx[mask]] = True
    return ret


def _min(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmax = maxval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmax:
        ret[group_idx] = dmax  # min starts from maximum
    with np.errstate(invalid="ignore"):
        np.minimum.at(ret, group_idx, a)
    return ret


def _max(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.number]:
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmin = minval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmin:
        ret[group_idx] = dmin  # max starts from minimum
    with np.errstate(invalid="ignore"):
        np.maximum.at(ret, group_idx, a)
    return ret


def _allnan(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = bool,
) -> npt.NDArray[np.bool_]:
    return _all(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _anynan(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int,
    fill_value: AScalar,
    dtype: npt.DTypeLike | None = bool,
) -> npt.NDArray[np.bool_]:
    return _any(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)
