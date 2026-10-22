"""The driver of the numpy implementation: the validation of the input, the
resolution of the operation and of the output dtype, and the answer itself.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numpy.ops import _dispatch
from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    resolve_fill_value,
)
from numpy_groupies.registry import (
    FuncName,
    Registry,
    missing_impl_error,
)
from numpy_groupies.utils_np import (
    AFillValue,
    AResult,
    AScalar,
    AValue,
    RavelOrder,
    check_dtype,
    check_fill_value,
    check_nton_shape,
    input_validation,
)
from numpy_groupies.validation import is_scalar_input


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
    iv = input_validation(group_idx, a, size=size, order=order, axis=axis, func=func)
    group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

    if group_idx.dtype == np.dtype("uint64"):
        # Force conversion to signed int, to avoid issues with bincount etc later
        group_idx = group_idx.astype(int)

    try:
        funcname, op = _dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all (a dict asks its keys
        # for a hash, which an object defining __eq__ without __hash__ does not
        # have): it names no known function either, so the registry answers it
        # as a callable of the caller's own
        funcname, op = _dispatch.custom(func)
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
                np.unravel_index(ret, unravel_shape)[axis],
            )
            ret[mask] = cast("AScalar", fill_value)
        check_nton_shape(ret, size, funcname)
        ret = ret.reshape(size, order=order)
    return cast(AResult, ret)
