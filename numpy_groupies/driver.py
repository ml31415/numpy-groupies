"""The driver of the numpy based backends, and the names of the functions of
numpy they resolve.

The names come first: the functions of numpy carry the names of the
functions of the package, and the table of the aliases is what a name
resolves against.

One call for all the numpy based backends: the validation of the input, the
resolution of the operation and of the output dtype, the answer of the
operation - and the shape of the answer.  What differs from one backend to
the next is the table of its operations and the meaning of the nan lengths,
and that is what the caller feeds in.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import check_dtype, is_temporal_input
from numpy_groupies.fills import (
    DEFAULT_FILL_VALUE,
    resolve_fill_value,
)
from numpy_groupies.functions import func_argument_default
from numpy_groupies.inputs import check_nton_shape, input_validation
from numpy_groupies.registry import (
    Registry,
    get_aliasing,
    missing_impl_error,
    resolve_name,
)
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder
from numpy_groupies.validation import (
    check_fill_value,
    check_topk_input,
    check_topk_shape,
    is_scalar_input,
)

_alias_numpy: dict[str | Callable[..., object], str] = {
    np.add: "sum",
    np.sum: "sum",
    np.any: "any",
    np.all: "all",
    np.multiply: "prod",
    np.prod: "prod",
    np.amin: "min",
    np.min: "min",
    np.minimum: "min",
    np.amax: "max",
    np.max: "max",
    np.maximum: "max",
    np.argmax: "argmax",
    np.argmin: "argmin",
    np.mean: "mean",
    np.median: "median",
    np.quantile: "quantile",
    np.percentile: "percentile",
    np.std: "std",
    np.var: "var",
    np.array: "array",
    np.asarray: "array",
    np.sort: "sort",
    np.cumsum: "cumsum",
    np.cumprod: "cumprod",
    np.nansum: "nansum",
    np.nanprod: "nanprod",
    np.nanmean: "nanmean",
    np.nanmedian: "nanmedian",
    np.nanquantile: "nanquantile",
    np.nanpercentile: "nanpercentile",
    np.nanvar: "nanvar",
    np.nanmax: "nanmax",
    np.nanmin: "nanmin",
    np.nanstd: "nanstd",
    np.nanargmax: "nanargmax",
    np.nanargmin: "nanargmin",
    np.nancumsum: "nancumsum",
}

if hasattr(np, "trapezoid"):
    # np.trapezoid replaced np.trapz in numpy 2.0
    _alias_numpy[np.trapezoid] = "trapezoid"

aliasing = get_aliasing(_alias_numpy)


def _drop_nans(
    funcname: str,
    a: AValue,
    group_idx: npt.NDArray[np.integer],
    kwargs: dict[str, Any],
    len_counts_values: bool,
) -> tuple[AValue, npt.NDArray[np.integer]]:
    """The input of a known nan function without its nan values.

    The cumulative nan variants keep every row, the rest drops the nan
    values - a is not needed for len, nanlen.
    """
    if is_scalar_input(a):
        raise ValueError("nan-version not supported for scalar input.")
    if "arg" in funcname:
        kwargs["_nansqueeze"] = True
        return a, group_idx
    if "cum" in funcname:
        return a, group_idx
    good = ~np.isnan(a)
    if "len" not in funcname or len_counts_values:
        # a scalar a raised a branch above, only columns remain
        a = cast("npt.NDArray[np.generic]", a)[good]
    return a, group_idx[good]


def _aggregate_wrap(
    dispatch: Registry,
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: str | Callable[..., AScalar],
    *,
    size: int | Sequence[int] | npt.NDArray[np.integer] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: AResult | None = None,
    len_counts_values: bool = False,
    **kwargs: Any,
) -> AResult:
    """The answer of one call of a numpy based backend, and the fill value it
    resolved."""
    # the argument the numba implementation honours and the numpy based ones
    # do not: there is no compiled loop here to write into an array of the
    # caller - the argument is taken, so that the signature of every backend
    # is the same, and it says what it cannot do
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")

    # the name the function of the call resolves to - the validation of the
    # input asks for it, and the table of the aliases is the one of numpy
    iv = input_validation(
        group_idx,
        a,
        size=size,
        order=order,
        axis=axis,
        func_name=resolve_name(func, aliasing),
    )
    group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

    if group_idx.dtype == np.dtype("uint64"):
        # Force conversion to signed int, to avoid issues with bincount etc later
        group_idx = group_idx.astype(int)

    try:
        funcname, op, temporal_op = dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all (a dict asks its keys
        # for a hash, which an object defining __eq__ without __hash__ does not
        # have): it names no known function either, so the registry answers it
        # as a callable of the caller's own
        funcname, op, temporal_op = dispatch.custom(func)

    if op is None:
        # a function that is known but not implemented here - one the backend
        # cannot do at all cannot refuse itself for a dtype either, so the
        # missing implementation is the answer before any temporal policy is
        # asked
        raise missing_impl_error(funcname)

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
    elif not isinstance(funcname, str):
        # a callable of the caller's own: it sees the values as they are, and
        # its result is a number of its own making - the dtype it asked for is
        # the one it gets
        fill_value = resolve_fill_value(funcname, fill_value, dtype if dtype is not None else np.asarray(a).dtype)
        ret = op(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype, **kwargs)
    else:
        # a known function: the nan variants drop the nans of the input first
        if funcname.startswith("nan"):
            a, group_idx = _drop_nans(funcname, a, group_idx, kwargs, len_counts_values)

        # the sum overflow guess needs the maximum number of values that can
        # be summed into a single group, i.e. the (nan-filtered) input length
        # - flat_size counts output groups and can be arbitrarily smaller
        dtype = check_dtype(dtype, funcname, a, len(group_idx))
        fill_value = resolve_fill_value(funcname, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=funcname)
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
        return cast(AResult, ret.reshape(size, order=order))
    if funcname == "topk":
        # the topk gives k values per group, so its shape is (size, k) - the
        # check of the shape of a one-out-per-in function does not apply to it
        check_topk_shape(ret, flat_size, int(kwargs.get("k", func_argument_default("topk", "k"))))
    return cast(AResult, ret)
