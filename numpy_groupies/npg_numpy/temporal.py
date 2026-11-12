"""The temporal operations of the numpy backend, on the int64 ticks of the
values: the mean, the median and the quantile have exact tick operations
here, whose numeric operations are not exact on the ticks (a float64 mean
rounds, an int64 median promotes to float64, the quantile interpolates in
float64); every other function is answered by its numeric operation, exact
on the ticks.  The temporal operation of every function is built here, out
of the policy of ``numpy_groupies.temporal`` and the operations of the
backend - one per function name, like the numeric ones.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.groups import exact_integer_group_mean
from numpy_groupies.npg_numpy.reductions import _all, _any
from numpy_groupies.npg_numpy.specials import _temporal_median, _temporal_quantile
from numpy_groupies.registry import function_names
from numpy_groupies.temporal import NAT_TICK, build_temporal_op
from numpy_groupies.typing import AFillValue, AResult, AValue, RavelOrder


def _call_by_keyword(
    op: Callable[..., AResult],
    group_idx: npt.NDArray[np.integer],
    values: AValue,
    size: int,
    fill_value: AFillValue,
    dtype: npt.DTypeLike | None,
    order: RavelOrder,
    axis: int | None,
    **kwargs: Any,
) -> AResult:
    """The call of an operation of this backend: the arguments behind the
    values are keyword arguments."""
    return op(group_idx, values, size, fill_value=fill_value, dtype=dtype, **kwargs)


def _tick_mean(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_value: int = 0,
    dtype: npt.DTypeLike | None = None,
    nan: bool = False,
    **kwargs: Any,
) -> AResult:
    """The mean of the ticks of every group, rounded half to even - the ticks
    are numbers around 1.6e18 for a nanosecond datetime64, their sums leave
    the exact float64 window, so they are exact (see
    ``numpy_groupies.exact_integer_group_mean``).
    """
    if nan:
        # the NaT tick is the smallest one - left out, or the sums count it
        good = ticks != NAT_TICK
        ticks, group_idx = ticks[good], group_idx[good]
    return cast(
        "AResult",
        exact_integer_group_mean(group_idx, ticks, size, int(fill_value), np.dtype(np.int64)),
    )


def _tick_median(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_value: int = 0,
    dtype: npt.DTypeLike | None = None,
    nan: bool = False,
    **kwargs: Any,
) -> AResult:
    """The exact median of the ticks of every group - the nan form drops the
    rows of the NaT ticks before the selection, like the nan median of the
    numba backend does inside its kernel.
    """
    return _temporal_median(group_idx, ticks, size, fill_value=fill_value, nan=nan)


def _tick_quantile(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_value: int = 0,
    dtype: npt.DTypeLike | None = None,
    q: float = 0.5,
    method: str = "linear",
    **kwargs: Any,
) -> AResult:
    """The quantile of the ticks of every group - the interpolation scales a
    tick difference by a fraction, a product no int64 arithmetic holds, the
    way numpy runs it on the durations of the values.
    """
    return _temporal_quantile(group_idx, ticks, size, fill_value, q, method)


def _nan_tick_op(op: Callable[..., AResult]) -> Callable[..., AResult]:
    """The nan variant of an operation on the ticks: the NaT tick is the
    smallest one, so it is left out.
    """

    def nan_op(*args: Any, **kwargs: Any) -> AResult:
        kwargs["nan"] = True
        return op(*args, **kwargs)

    return nan_op


# the tick operations by the function name they answer - the nan variants
# are the same operation with the flag of the missing value set
_TICK_OPS: dict[str, Callable[..., AResult]] = {
    "mean": _tick_mean,
    "nanmean": _nan_tick_op(_tick_mean),
    "median": _tick_median,
    "nanmedian": _nan_tick_op(_tick_median),
    "quantile": _tick_quantile,
    "nanquantile": _tick_quantile,
    "percentile": _tick_quantile,
    "nanpercentile": _tick_quantile,
}


def _resolve(name: str) -> Callable[..., AResult] | None:
    """The operation of the backend behind a name - the flags of the missing
    value are the plain any and all of the NaT mask."""
    if name == "any":
        return _any
    if name == "all":
        return _all
    return None


def build_temporal_ops(op_dict: Mapping[str, Callable[..., AResult]]) -> dict[str, Callable[..., tuple[AResult, Any]]]:
    """The temporal operation of every function of the backend, one per
    function name - the policy of the function and the operations of the
    backend, composed like the numeric ones.  The functions the backend does
    not implement get one as well: the temporal policy refuses them, and
    that refusal is the answer of numpy as well.
    """
    ops: dict[str, Callable[..., tuple[AResult, Any]]] = {}
    for name in function_names():
        ops[name] = build_temporal_op(
            name,
            op_dict.get(name),
            tick_op=_TICK_OPS.get(name),
            call=_call_by_keyword,
            resolve=_resolve,
        )
    return ops
