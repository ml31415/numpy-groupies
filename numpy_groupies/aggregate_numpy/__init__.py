"""The numpy implementation of aggregate: the vectorized operations, one per
function, behind one dispatch.

The module holds the driver of the implementation - the validation of the
input, the resolution of the operation and of the output dtype, and the
answer itself - and the entry point that calls it.  The operations
themselves live in the modules beside it, one per function family.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.collect import blocks_to_object_array
from numpy_groupies.aggregate_numpy.gather import unpack
from numpy_groupies.aggregate_numpy.ops import _dispatch
from numpy_groupies.aggregate_numpy.steps import step_count, step_indices
from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    resolve_fill_value,
)
from numpy_groupies.nptools import check_dtype, check_fill_value, check_nton_shape, input_validation
from numpy_groupies.registry import (
    FuncName,
    Registry,
    bind_functions,
    func_argument_default,
    function_names,
    missing_impl_error,
)
from numpy_groupies.temporal import is_temporal_input
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder
from numpy_groupies.validation import (
    check_topk_input,
    check_topk_shape,
    is_scalar_input,
)

__all__ = ["aggregate", "blocks_to_object_array", "step_count", "step_indices", "unpack"]


def _aggregate_base(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | npt.NDArray[np.integer] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    _dispatch: Registry = _dispatch,
    len_counts_values: bool = False,
    **kwargs: Any,
) -> AResult:
    """The driver of the numpy implementation: the validation of the input, the
    resolution of the operation and of the output dtype, and the answer itself.
    """
    iv = input_validation(group_idx, a, size=size, order=order, axis=axis, func=func)
    group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

    if group_idx.dtype == np.dtype("uint64"):
        # Force conversion to signed int, to avoid issues with bincount etc later
        group_idx = group_idx.astype(int)

    try:
        funcname, op, temporal_op = _dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all (a dict asks its keys
        # for a hash, which an object defining __eq__ without __hash__ does not
        # have): it names no known function either, so the registry answers it
        # as a callable of the caller's own
        funcname, op, temporal_op = _dispatch.custom(func)
    # datetime64 and timedelta64 run on their int64 ticks - the temporal
    # operation of the function takes the input apart, refuses what numpy
    # refuses, and answers with a temporal value.  Only the functions of the
    # package go through it: a callable of the caller's own defines its own
    # semantics and sees the values as they are.  The numpy functions that
    # mean the same thing are aliases of the functions of the package, so
    # np.mean is the mean of the package and gets its answer.  The temporal
    # operation is the other half of the operation of the function, and the
    # registry holds it beside the numeric one.
    if temporal_op is not None and is_temporal_input(a):
        ret, fill_value = temporal_op(
            group_idx,
            a,
            flat_size,
            fill_value,
            dtype,
            order=order,
            axis=axis,
            **kwargs,
        )
    else:
        if op is None:
            # a function that is known but not implemented here
            raise missing_impl_error(funcname)
        if isinstance(funcname, str):
            # a known function: the nan variants drop the nans of the input first
            if funcname.startswith("nan"):
                if is_scalar_input(a):
                    raise ValueError("nan-version not supported for scalar input.")
                if "nan" in funcname:
                    if "arg" in funcname:
                        kwargs["_nansqueeze"] = True
                    elif "cum" in funcname:
                        pass
                    else:
                        good = ~np.isnan(a)
                        if "len" not in funcname or len_counts_values:
                            # a is not needed for len, nanlen!
                            # a scalar a raised a branch above, only columns remain
                            a = cast("npt.NDArray[np.generic]", a)[good]
                        group_idx = group_idx[good]

            # the sum overflow guess needs the maximum number of values that can
            # be summed into a single group, i.e. the (nan-filtered) input length
            # - flat_size counts output groups and can be arbitrarily smaller
            dtype = check_dtype(dtype, funcname, a, len(group_idx))
            fill_value = resolve_fill_value(funcname, fill_value, dtype)
            check_fill_value(fill_value, dtype, func=funcname)
        else:
            # a callable of the caller's own: it sees the values as they are, and
            # its result is a number of its own making - the dtype it asked for is
            # the one it gets
            fill_value = resolve_fill_value(funcname, fill_value, dtype if dtype is not None else np.asarray(a).dtype)

        ret = op(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype, **kwargs)

    if funcname == "topk":
        # the topk hands out k values per group: its result is a matrix with
        # a row per group, and the unravelling of a multidimensional group
        # index has no place in that
        check_topk_input(ndim_idx, size)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        if unravel_shape is not None:
            # A negative fill_value cannot, and should not, be unraveled.
            mask = ret == fill_value
            ret[mask] = 0
            # the axis is not None whenever the shape to unravel to is known
            assert axis is not None
            ret = cast(
                "AResult",
                np.unravel_index(cast("npt.NDArray[np.integer]", ret), unravel_shape)[axis],
            )
            ret[mask] = cast("AScalar", fill_value)
        check_nton_shape(ret, size, funcname)
        ret = ret.reshape(size, order=order)
    elif funcname == "topk":
        # the topk gives k values per group, so its shape is (size, k) - the
        # check of the shape of a one-out-per-in function does not apply to it
        check_topk_shape(ret, flat_size, int(kwargs.get("k", func_argument_default("topk", "k"))))
    return cast(AResult, ret)


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
