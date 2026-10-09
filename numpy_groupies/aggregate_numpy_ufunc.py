from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .aggregate_common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    build_dispatch,
    get_func,
)
from .aggregate_numpy import _aggregate_base
from .utils import (
    aliasing,
    check_boolean,
    maxval,
    minimum_dtype,
    minimum_dtype_scalar,
    minval,
)


def _anynan(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    return _any(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _allnan(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    return _all(group_idx, np.isnan(a), size, fill_value=fill_value, dtype=dtype)


def _any(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if fill_value:
        ret[group_idx] = False  # any-test should start from False
    np.logical_or.at(ret, group_idx, a)
    return ret


def _all(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if not fill_value:
        ret[group_idx] = True  # all-test should start from True
    np.logical_and.at(ret, group_idx, a)
    return ret


def _sum(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 0:
        ret[group_idx] = 0  # sums should start at 0
    np.add.at(ret, group_idx, a)
    return ret


def _len(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    return _sum(group_idx, 1, size, fill_value, dtype=int)


def _prod(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Same as aggregate_numpy.py"""
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 1:
        ret[group_idx] = 1  # product should start from 1
    np.multiply.at(ret, group_idx, a)
    return ret


def _min(
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Same as aggregate_numpy.py"""
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
    group_idx: np.ndarray,
    a: np.ndarray,
    size: int | tuple[int, ...] | None,
    fill_value: Any,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Same as aggregate_numpy.py"""
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmin = minval(fill_value, dtype)
    with np.errstate(invalid="ignore"):
        ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmin:
        ret[group_idx] = dmin  # max starts from minimum
    with np.errstate(invalid="ignore"):
        np.maximum.at(ret, group_idx, a)
    return ret


_impl_dict = {
    "min": _min,
    "max": _max,
    "sum": _sum,
    "prod": _prod,
    "all": _all,
    "any": _any,
    "allnan": _allnan,
    "anynan": _anynan,
    "len": _len,
}

_dispatch = build_dispatch(_impl_dict, aliasing)


def aggregate(
    group_idx: npt.ArrayLike,
    a: npt.ArrayLike,
    func: str | Callable[..., Any] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: Any = DEFAULT_FILL_VALUE,
    order: str = "C",
    dtype: npt.DTypeLike = None,
    axis: int | None = None,
    **kwargs: Any,
) -> Any:
    funcname: str | Callable[..., Any]
    try:
        funcname, _ = _dispatch[func]
    except (KeyError, TypeError):
        # a custom callable is not supported here - get_func raises for
        # unknown names, and any callable it returns is not a ufunc impl
        funcname = get_func(func, aliasing, _impl_dict)
        if not isinstance(funcname, str):
            raise NotImplementedError("No such ufunc available")
    return _aggregate_base(
        group_idx,
        a,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        func=funcname,
        axis=axis,
        _impl_dict=_impl_dict,
        _dispatch=_dispatch,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    Unlike ``aggregate_numpy``, which in most cases does some custom
    optimisations, this version simply uses ``numpy``'s ``ufunc.at``.

    With numpy 1.25 the performance of ``ufunc.at`` improved substantially,
    however this implementation remains incomplete and is intended to be
    used in testing and benchmarking only.
    """
    + aggregate_common_doc
)
