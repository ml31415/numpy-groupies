"""The pure python implementation of aggregate: the operations of the package
computed with python numbers, behind one dispatch.

The entry point of the implementation is here - the operations themselves live
in the module beside this one, and the composition of the groups in the other.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from numpy_groupies.fills import (
    DEFAULT_FILL_VALUE,
    resolve_fill_value,
)
from numpy_groupies.functions import (
    aggregate_common_doc,
    bind_functions,
    forced_float_types,
    forced_real_float_types,
)
from numpy_groupies.npg_purepy.compose import answer_groups
from numpy_groupies.npg_purepy.ops import (
    _ONE_PER_ITEM,
    _dispatch,
    _mean,
    _std,
    _topk,
    _var,
)
from numpy_groupies.registry import (
    FuncName,
    function_names,
    missing_impl_error,
    resolve_name,
)
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    GROUP_IDX_TOO_MANY_DIMS,
    LENGTH_MISMATCH,
    NEGATIVE_INDICES,
    NON_INTEGER_GROUP_IDX,
    OUTPUT_SIZE_NOT_SCALAR,
    TOO_LARGE_FOR_SIZE,
    TOPK_NEEDS_FLAT_GROUP_IDX,
    check_complex_dtype,
    check_fixed_fill_value,
    check_integer_variance_span_overflow,
    check_topk_k,
    count_dimensions,
    is_integer_dtype,
    is_scalar_input,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder

__all__ = ["aggregate", "unpack"]


def unpack(group_idx: npt.NDArray[np.integer] | Sequence[Any], ret: AResult | list[Any]) -> list[Any]:
    """The gather of a result: the result of the group of every value."""
    return [ret[int(ix)] for ix in group_idx]


def _a_dtype(a: Sequence[Any]) -> type:
    # the element types decide what a numpy array of these values would have
    # made of them; numpy scalars are recognised by their dtype, since e.g.
    # complex64 is no subclass of complex
    dtype: type = int
    for v in a:
        kind = getattr(getattr(v, "dtype", None), "kind", None)
        if isinstance(v, complex) or kind == "c":
            return complex
        if isinstance(v, float) or kind == "f":
            dtype = float
    return dtype


def _check_integer_precision(a: Sequence[Any], func: Callable[..., AScalar]) -> None:
    """The backend-uniform overflow gates for integer var/std; the mean needs none."""
    if _a_dtype(a) is not int or len(a) == 0:
        return
    values = [int(val) for val in a]
    low = min(values)
    high = max(values)
    if func is not _mean:
        check_integer_variance_span_overflow(low, high)


def _is_flat(values: Sequence[Any]) -> bool:
    """whether the values are a flat sequence.

    A mapping, a set, a generator and anything with rows of its own is no
    sequence of values - refused like every other path refuses it.
    """
    if getattr(values, "ndim", 0) > 1:
        return False
    if isinstance(values, (dict, set, frozenset)):
        return False
    try:
        length = len(values)
    except TypeError:
        return False
    if length == 0:
        return True
    first = values[0]
    return not isinstance(first, (list, tuple)) and getattr(first, "ndim", 0) == 0


def aggregate(
    group_idx: npt.NDArray[np.integer] | Sequence[Any],
    a: AValue | Sequence[Any],
    func: FuncName | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: AResult | None = None,
    **kwargs: Any,
) -> list[Any]:
    # both arguments are read purely as sized iterables of values - an array
    # or a plain sequence of them alike; the scalar that the signature of
    # ``a`` also admits is refused below
    idx: Sequence[Any] = cast(Sequence[Any], group_idx)
    vals: Sequence[Any] = cast(Sequence[Any], a)
    if axis is not None:
        raise NotImplementedError("axis arg not supported in purepy implementation.")
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")
    if len(idx) == 0:
        raise ValueError(EMPTY_GROUP_IDX)
    if not is_scalar_input(vals) and not _is_flat(vals):
        raise ValueError("a must be scalar or 1 dimensional, use .ravel to flatten. Alternatively specify axis.")
    # the shape of the labels is asked before what they are made of, as in every
    # other path: labels of more than two dimensions name no group in any
    # implementation, so this is the one complaint every backend says in the very
    # same words, while the two-dimensional ones below are a gap of this
    # implementation alone
    if count_dimensions(idx) > 2:
        raise ValueError(GROUP_IDX_TOO_MANY_DIMS)
    # a label that is no index is refused in the words of the package
    labels_dtype = getattr(idx, "dtype", None)
    if labels_dtype is not None and not is_integer_dtype(labels_dtype):
        raise TypeError(NON_INTEGER_GROUP_IDX)

    # the func is resolved before anything is measured of the labels or the
    # sizes, so a refused call speaks in the words of the package, not of
    # what this implementation happened to measure first
    try:
        funcname, op, _temporal_op = _dispatch[func]
    except TypeError:
        # no key of the table: run as the caller's own callable
        funcname, op, _temporal_op = _dispatch.custom(func)
    if op is None:
        raise missing_impl_error(funcname)

    # Check for 2d group_idx
    if size is None:
        try:
            size = 1 + int(max(idx))
        except (TypeError, ValueError):
            if op is _topk:
                # the matrix of the topk is a rule of every backend
                raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX) from None
            raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")
    else:
        if isinstance(size, Sequence):
            if op is _topk:
                if getattr(idx, "ndim", 1) <= 1 and not hasattr(idx[0], "__len__"):
                    raise ValueError(OUTPUT_SIZE_NOT_SCALAR)
                raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX)
            # a sequence of sizes is a multidimensional result, which this
            # implementation does not do
            raise NotImplementedError("pure python implementation doesn't accept a sequence of sizes.")
        # a size is a number of groups - anything that counts is one
        size = operator.index(size)

    k = 0
    largest = True
    if op is _topk:
        k = check_topk_k(kwargs.get("k", 5))
        largest = kwargs.get("largest", True)

    for value in idx:
        try:
            i = int(value)
        except (TypeError, ValueError):
            if op is _topk:
                raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX) from None
            if isinstance(value, (list, tuple)):
                raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")
            else:
                try:
                    len(value)
                except TypeError:
                    raise ValueError(f"invalid value found in group_idx: {value}")
                else:
                    raise NotImplementedError("pure python implementation doesn't accept ndim indexed input.")
        else:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                # a whole float names no group, it is not rounded
                raise TypeError(NON_INTEGER_GROUP_IDX)
            if i < 0:
                raise ValueError(NEGATIVE_INDICES)
            if i >= size:
                raise ValueError(TOO_LARGE_FOR_SIZE.format(size=size))

    if is_scalar_input(vals):
        if funcname not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        # [()] is the dtype preserving way to get the value out of a 0d array
        vals = [cast(Any, vals)[()] if hasattr(vals, "shape") else vals] * len(idx)
    elif len(idx) != len(vals):
        raise ValueError(LENGTH_MISMATCH)

    # nan is only a sensible default where the output datatype can hold it;
    # the averaging functions answer floats, the squared sums stay exact
    result_dtype = _a_dtype(vals)
    if (
        result_dtype is int
        and isinstance(funcname, str)
        and resolve_name(funcname) not in ("sumofsquares", "nansumofsquares")
        and (resolve_name(funcname) in forced_float_types or resolve_name(funcname) in forced_real_float_types)
    ):
        result_dtype = float
    if result_dtype is int and not isinstance(funcname, str) and dtype is None:
        # a callable of the caller: assume a floating result without a dtype
        result_dtype = float
    fill_value = resolve_fill_value(funcname, fill_value, result_dtype)
    check_fixed_fill_value(funcname, fill_value)
    if dtype is not None and isinstance(funcname, str):
        # the one place where this implementation has to reject a non-complex dtype
        check_complex_dtype(_a_dtype(vals), dtype, funcname)

    # the positions have to be taken before the nan filter shortens the input
    pos: Sequence[int] = range(len(vals))

    # the nan variants of the running values leave the nan in place, so they
    # are not shortened
    if isinstance(funcname, str) and funcname.startswith("nan") and op not in _ONE_PER_ITEM:
        kept = [(p, ix, val) for p, ix, val in zip(pos, idx, vals) if val == val]
        if not kept:
            # a group of nothing but nans takes the fill value
            return [fill_value] * size
        pos, idx, vals = zip(*kept)
    if op in (_mean, _var, _std):
        _check_integer_precision(vals, op)
    return answer_groups(pos, idx, vals, op, size, fill_value, kwargs, k, largest)


aggregate.__doc__ = (
    """
    This is the pure python implementation of aggregate. It is terribly slow.
    Using the numpy version is highly recommended.

    It computes with python numbers and needs no numpy.  Every function of
    the package is there except the temporal arithmetic and the
    multidimensional output: axis and a sequence of sizes are refused.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the functions of the backend, by their name
#
# one entry point per function of the package, calling the implementation
# above with the name of its function

_functions = bind_functions(aggregate, function_names())


def __getattr__(name: str) -> Any:
    """The function of the backend, by its name."""
    try:
        return _functions[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(set(globals()) | set(_functions))
