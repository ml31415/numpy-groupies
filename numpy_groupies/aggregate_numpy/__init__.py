"""The numpy implementation of aggregate: the vectorized operations, one per
function, behind one dispatch."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.base import _aggregate_base
from numpy_groupies.aggregate_numpy.collect import blocks_to_object_array
from numpy_groupies.aggregate_numpy.gather import unpack
from numpy_groupies.aggregate_numpy.ops import _dispatch
from numpy_groupies.aggregate_numpy.specials import _temporal_median, _temporal_quantile
from numpy_groupies.aggregate_numpy.steps import step_count, step_indices
from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
)
from numpy_groupies.nptools import exact_integer_group_mean
from numpy_groupies.registry import (
    FuncName,
    bind_functions,
    function_names,
    missing_impl_error,
)
from numpy_groupies.temporal import NAT_TICK, _temporal_aggregation
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder

__all__ = ["aggregate", "blocks_to_object_array", "step_count", "step_indices", "unpack"]


def _tick_mean(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_value: int = 0,
    dtype: npt.DTypeLike | None = None,
    nan: bool = False,
    **kwargs: Any,
) -> AResult:
    """The mean of the ticks of every group, rounded half to even.

    The ticks of a temporal input are the number of units since the epoch -
    for a datetime64 in nanoseconds that is a number around 1.6e18, and the
    sum of the ticks of a group leaves the window float64 represents
    exactly.  The sums are therefore exact (see
    ``nptools.exact_integer_group_mean``), and the remainder of the
    division rounds half to even.
    """
    if nan:
        # the NaT of a group is left out - its tick is the smallest one, and
        # the sums of the ticks would count it
        good = ticks != NAT_TICK
        ticks, group_idx = ticks[good], group_idx[good]
    return cast(
        "AResult",
        exact_integer_group_mean(group_idx, ticks, size, int(fill_value), np.dtype(np.int64)),
    )


def _temporal_call(
    funcname: str,
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    size: int,
    fill_value: AFillValue,
    dtype: npt.DTypeLike | None,
    kwargs: dict[str, Any],
) -> tuple[AResult, AFillValue]:
    """The answer of an aggregation of temporal values.

    The wrapper of the temporal policy takes the input apart, refuses what
    numpy refuses, and answers with a temporal value.  The three answers
    that no operation of the package computes on its own - the mean, the
    median and the quantile of the ticks - are the ones of this backend:
    there is no compiled loop here, so they run on the python integers,
    where the sum of the ticks of a group and the mean of the two middles
    of an even group stay exact.
    """

    def resolve(name: str) -> Callable[..., AResult]:
        op = _dispatch[name].op
        if op is None:
            raise missing_impl_error(name)
        return op

    return _temporal_aggregation(
        resolve,
        _tick_mean,
        _temporal_median,
        _temporal_quantile,
        funcname,
        group_idx,
        a,
        size,
        fill_value,
        dtype,
        kwargs,
    )


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: AResult | None = None,
    **kwargs: Any,
) -> AResult:
    # the argument the numba implementation honours and this one does not:
    # there is no compiled loop here to write into an array of the caller -
    # the argument is taken, so that the signature of every backend is the
    # same, and it says what it cannot do
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")
    return _aggregate_base(
        group_idx,
        a,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        func=func,
        axis=axis,
        temporal=_temporal_call,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    This is the pure numpy implementation of aggregate.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the functions of the backend, by their name
#
# One entry point per function of the package, calling the implementation
# above with the name of its function - the same thing the numba module
# offers with its kernels, and the same thing the package itself offers with
# the functions of the best implementation available.

_functions = bind_functions(aggregate, function_names())


def __getattr__(name: str) -> Any:
    """The function of the backend, by its name.

    The names of the functions are not set one by one - they are looked up
    in the table of the entry points, so that they cannot drift from it.
    """
    try:
        return _functions[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(set(globals()) | set(_functions))
