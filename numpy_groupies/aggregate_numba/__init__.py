"""The numba implementation of aggregate: compiled per-dtype kernels behind
one dispatch, with ``out=`` support and caching for custom callables."""

import functools
import typing
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    can_name_a_function,
    custom_callable,
    missing_impl_error,
)
from numpy_groupies.aggregate_numba.gather import unpack_into
from numpy_groupies.aggregate_numba.ops import _dispatch, build_generic_aggregate
from numpy_groupies.aggregate_numba.steps import step_count, step_indices

__all__ = ["aggregate", "step_count", "step_indices", "unpack_into"]


def aggregate(
    group_idx: npt.ArrayLike,
    a: npt.ArrayLike,
    func: str | Callable[..., typing.Any] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: typing.Any = DEFAULT_FILL_VALUE,
    order: str = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: npt.NDArray[np.generic] | None = None,
    cache: bool = True,
    **kwargs: typing.Any,
) -> npt.NDArray[np.generic]:
    # numba has no float16: the values survive the widening cast to float32
    # exactly, so the kernels run on float32 and an inexact result is rounded
    # back - like the numpy twin does, whose bincount accumulates in float64
    # and rounds once at the end
    is16 = getattr(a, "dtype", None) == np.float16
    if is16:
        # a float16 input is always an array - the getattr gate does not narrow the union
        a = typing.cast("npt.NDArray[np.generic]", a).astype(np.float32)
    try:
        funcname, aggregate_op = _dispatch[func]
    except (KeyError, TypeError):
        # no key of the table, either because no function goes by that name or
        # because func could not be one of its keys (a dict asks its keys for a
        # hash): both leave the caller's own func to ``_aggregate_custom``, which
        # tells a func that named nothing from a func that can be run
        if out is not None:
            raise NotImplementedError("out= is only supported for named functions") from None
        ret = _aggregate_custom(group_idx, a, func, size, fill_value, order, dtype, axis, cache, **kwargs)
    else:
        if aggregate_op is None:
            # a function that is known but not implemented here
            raise missing_impl_error(funcname)
        ret = aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
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


def _aggregate_custom(
    group_idx: npt.ArrayLike,
    a: npt.ArrayLike,
    func: str | Callable[..., typing.Any],
    size: int | Sequence[int] | None,
    fill_value: typing.Any,
    order: str,
    dtype: npt.DTypeLike | None,
    axis: int | None,
    cache: bool | None | typing.MutableMapping[typing.Any, typing.Any],
    **kwargs: typing.Any,
) -> npt.NDArray[np.generic]:
    """Slow dispatch path for a func that is no key of the dispatch table.

    An object of one's own is what numba cannot compile - it compiles functions,
    and a *hashable* object of one's own is refused by numba itself, in its own
    words.  The one that cannot be a key of a dict at all is answered here and in
    front of the cache: that cache is a dict too and would complain about the
    missing hash of the func rather than about what it is asked to do with it.
    """
    # what is not callable named nothing, and says so in the same words the
    # other implementations choose, before numba is mentioned at all
    func = custom_callable(func)
    if not can_name_a_function(func):
        raise TypeError(
            "numba can only compile a function as func, a callable object is not one of those (the numpy implementation runs any callable)"
        ) from None
    aggregate_op: typing.Callable[..., typing.Any] | None
    if cache in (None, False):
        # Keep None and False in order to accept empty dictionaries
        aggregate_op = build_generic_aggregate(func)
    elif cache is True:
        aggregate_op = _get_cached_generic(func)
    else:
        cache_map = typing.cast("typing.MutableMapping[typing.Any, typing.Any]", cache)
        aggregate_op = cache_map.get(func)
        if aggregate_op is None:
            aggregate_op = build_generic_aggregate(func)
            cache_map[func] = aggregate_op
    return typing.cast(
        "npt.NDArray[np.generic]", aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    )


@functools.lru_cache(maxsize=128)
def _get_cached_generic(func: Callable[..., typing.Any]) -> typing.Any:
    """LRU-cache generic aggregate loops per user callable (bounded)."""
    return build_generic_aggregate(func)


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
