"""The numba implementation of aggregate: compiled per-dtype kernels behind
one dispatch, with ``out=`` support and caching for custom callables."""

import functools
import typing
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ..aggregate_common import DEFAULT_FILL_VALUE, aggregate_common_doc, get_func
from ..utils import aliasing
from .gather import unpack_into
from .ops import _dispatch, _impl_dict, create_generic_aggregate
from .steps import step_count, step_indices

__all__ = ["aggregate", "step_count", "step_indices", "unpack_into"]


def aggregate(
    group_idx: npt.ArrayLike,
    a: npt.ArrayLike,
    func: str | Callable[..., typing.Any] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: typing.Any = DEFAULT_FILL_VALUE,
    order: str = "C",
    dtype: npt.DTypeLike = None,
    axis: int | None = None,
    out: npt.NDArray[np.generic] | None = None,
    cache: bool = True,
    **kwargs: typing.Any,
) -> typing.Any:
    try:
        aggregate_op = _dispatch[func][1]
    except (KeyError, TypeError):
        # a custom callable, or an unknown name that get_func complains about
        if out is not None:
            raise NotImplementedError("out= is only supported for named functions") from None
        return _aggregate_custom(group_idx, a, func, size, fill_value, order, dtype, axis, cache, **kwargs)
    ret = aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    if out is not None:
        if out.shape != ret.shape or out.dtype != ret.dtype:
            raise TypeError(
                f"out must have shape {ret.shape} and dtype {ret.dtype}, got shape {out.shape} and dtype {out.dtype}"
            )
        out[...] = ret
        return out
    return ret


def _aggregate_custom(
    group_idx,
    a,
    func,
    size,
    fill_value,
    order,
    dtype,
    axis,
    cache,
    **kwargs,
):
    """Slow dispatch path for custom callables and unknown function names."""
    func = get_func(func, aliasing, _impl_dict)
    if not isinstance(func, str):
        if cache in (None, False):
            # Keep None and False in order to accept empty dictionaries
            aggregate_op = create_generic_aggregate(func)
        elif cache is True:
            aggregate_op = _get_cached_generic(func)
        else:
            aggregate_op = cache.get(func)
            if aggregate_op is None:
                aggregate_op = create_generic_aggregate(func)
                cache[func] = aggregate_op
        return aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    else:
        func = _impl_dict[func]
        return func(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)


@functools.lru_cache(maxsize=128)
def _get_cached_generic(func):
    """LRU-cache generic aggregate loops per user callable (bounded)."""
    return create_generic_aggregate(func)


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
