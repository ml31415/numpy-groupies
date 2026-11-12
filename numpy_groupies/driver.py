"""The driver of the numpy based backends, and the aliases resolving numpy
functions to the names of the package.

One call for all the numpy based backends: input validation, resolution of
the operation and output dtype, the answer and its shape. What differs per
backend is its table of operations and the nan lengths - which the caller
feeds in.
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
    check_topk_k,
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

# the aliases with the numpy functions in it: they name the functions of
# the package as well - np.sum is the sum
aliasing = get_aliasing(_alias_numpy)


def _drop_nans(
    funcname: str,
    a: AValue,
    group_idx: npt.NDArray[np.integer],
    kwargs: dict[str, Any],
    len_counts_values: bool,
) -> tuple[AValue, npt.NDArray[np.integer]]:
    """The input of a known nan function without its nan values.
    The cumulative variants keep every row, the rest drops the nans.
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
    """The answer of one call of a numpy based backend."""
    # taken only so every backend's signature is the same; there is no
    # compiled loop here to write into
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")

    # the name the function resolves to; the aliases are the ones of numpy
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
        # unhashable func: no known name, so answer it as a custom callable
        funcname, op, temporal_op = dispatch.custom(func)

    if op is None:
        # known but not implemented by this backend
        raise missing_impl_error(funcname)

    if funcname == "topk":
        # checked before the operation runs, so an unanswerable call says so
        # before other complaints, and the temporal path matches the numeric one
        kwargs["k"] = check_topk_k(kwargs.get("k", func_argument_default("topk", "k")))
        check_topk_input(ndim_idx)

    # temporal dtypes run on their int64 ticks; only the package's own
    # functions go through the temporal op, not a caller's callable
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
        # a caller's callable: the requested dtype is the one it gets; the
        # fill assumes a floating result without one, so a nan default does
        # not come out as a 0
        fill_value = resolve_fill_value(funcname, fill_value, dtype)
        ret = op(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype, **kwargs)
    else:
        # a known function: the nan variants drop the nans of the input first
        if funcname.startswith("nan"):
            a, group_idx = _drop_nans(funcname, a, group_idx, kwargs, len_counts_values)

        # the overflow guess needs the (nan-filtered) input length;
        # flat_size counts output groups and can be much smaller
        dtype = check_dtype(dtype, funcname, a, len(group_idx))
        fill_value = resolve_fill_value(funcname, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=funcname)
        ret = op(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype, **kwargs)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        if unravel_shape is not None:
            # a negative fill_value cannot, and should not, be unraveled
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
        # shape is (size, k), not one-out-per-in
        check_topk_shape(ret, flat_size, kwargs["k"])
    return cast(AResult, ret)
