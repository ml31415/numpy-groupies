"""The numba implementation of aggregate: compiled per-dtype kernels behind
one dispatch, with ``out=`` support and caching for custom callables."""

import typing
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_common import (
    DEFAULT_FILL_VALUE,
    FuncName,
    aggregate_common_doc,
    funcs_collecting,
    missing_impl_error,
)
from numpy_groupies.aggregate_numba.gather import unpack_into
from numpy_groupies.aggregate_numba.ops import _dispatch
from numpy_groupies.aggregate_numba.steps import step_count, step_indices
from numpy_groupies.utils import AFillValue, AResult, AValue, RavelOrder

__all__ = ["aggregate", "step_count", "step_indices", "unpack_into"]


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName | Callable[..., typing.Any] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: AResult | None = None,
    cache: bool = True,
    **kwargs: typing.Any,
) -> AResult:
    try:
        funcname, op = _dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all: it names no known
        # function either, so the registry answers it as a callable of the
        # caller's own
        funcname, op = _dispatch.custom(func)

    # numba has no float16: the values survive the widening cast to float32
    # exactly, so the kernels run on float32 and an inexact result is rounded
    # back - like the numpy twin does, whose bincount accumulates in float64
    # and rounds once at the end
    # the collecting functions are the exception: they hand out the values
    # themselves, so they keep the dtype they came in with
    is16 = getattr(a, "dtype", None) == np.float16
    if is16 and funcname not in funcs_collecting:
        # a float16 input is always an array - the getattr gate does not narrow the union
        a = typing.cast("npt.NDArray[np.number]", a).astype(np.float32)

    if op is None:
        # a function that is known but not implemented here
        raise missing_impl_error(funcname)
    if isinstance(funcname, str):
        ret = typing.cast(AResult, op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs))
    else:
        # a callable of the caller's own: it has no output array of its own,
        # and its compiled loop is memoized where the cache policy says
        if out is not None:
            raise NotImplementedError("out= is only supported for named functions") from None
        ret = typing.cast(
            "AResult",
            op(group_idx, a, size, fill_value, order, dtype, axis, cache=cache, **kwargs),
        )
    if is16 and dtype is None and ret.dtype.kind == "f":
        ret = ret.astype(np.float16)
    if out is not None:
        if out.shape != ret.shape or out.dtype != ret.dtype:
            raise TypeError(
                f"out must have shape {ret.shape} and dtype {ret.dtype}, got shape {out.shape} and dtype {out.dtype}"
            )
        out[...] = ret
        return out
    return ret


aggregate.__doc__ = (
    """
    This is the numba implementation of aggregate.

    This implementation accepts two additional keyword arguments:

    out: default=None
        an array the packed result is copied into (matching shape and dtype
        are required).  The array is returned, so calls can be chained.
    cache: default=True
        when aggregating with a custom callable ``func``, the compiled
        implementation is cached so that subsequent calls with the same
        function are fast.  Set to ``False`` to disable caching, or pass
        your own dictionary to control the caching manually.  With caching
        enabled (the default), callables are kept in a bounded LRU cache of
        128 entries.
    """
    + aggregate_common_doc
)
