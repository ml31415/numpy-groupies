"""The registry of the numpy implementation.

Every operation the package knows is here, under the names it answers to -
and the dispatch that resolves a name (or a callable of the caller) to the
operation behind it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.argreductions import _argmax, _argmin
from numpy_groupies.aggregate_numpy.averaging import _mean, _std, _sum_of_squres, _trapezoid, _var
from numpy_groupies.aggregate_numpy.collect import _array, _generic_callable
from numpy_groupies.aggregate_numpy.cumulative import _cumsum, _nancumsum
from numpy_groupies.aggregate_numpy.reductions import (
    _all,
    _allnan,
    _any,
    _anynan,
    _first,
    _last,
    _len,
    _max,
    _min,
    _prod,
    _sum,
)
from numpy_groupies.aggregate_numpy.specials import _median, _nunique, _percentile, _quantile, _sort
from numpy_groupies.common import (
    Registry,
    Resolved,
    custom_callable,
    funcs_no_separate_nan,
)
from numpy_groupies.utils_np import AResult, AScalar, aliasing

_op_dict: dict[str, Callable[..., AResult]] = {
    "min": _min,
    "max": _max,
    "sum": _sum,
    "prod": _prod,
    "last": _last,
    "first": _first,
    "all": _all,
    "any": _any,
    "mean": _mean,
    "median": _median,
    "quantile": _quantile,
    "percentile": _percentile,
    "std": _std,
    "var": _var,
    "anynan": _anynan,
    "allnan": _allnan,
    "sort": _sort,
    "array": _array,
    "argmax": _argmax,
    "argmin": _argmin,
    "len": _len,
    "nunique": _nunique,
    "cumsum": _cumsum,
    "sumofsquares": _sum_of_squres,
    "trapezoid": _trapezoid,
}
_op_dict.update(("nan" + k, v) for k, v in list(_op_dict.items()) if k not in funcs_no_separate_nan)
_op_dict["nancumsum"] = _nancumsum


def _custom_operation(func: Callable[..., AScalar]) -> Resolved:
    """The operation of a callable of the caller's own.

    The callable is applied to the values of every group, which are collected
    first.  The operation is the same for every callable - only the callable
    differs - so the registry holds one of them per caller's callable.
    """
    callable_ = custom_callable(func)

    def op(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size: int,
        fill_value: AScalar | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Any,
    ) -> npt.NDArray[np.number]:
        return _generic_callable(group_idx, a, size, fill_value, func=callable_, dtype=dtype, **kwargs)

    return Resolved(callable_, op)


_dispatch = Registry(_op_dict, aliasing, _custom_operation)
