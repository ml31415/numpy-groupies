"""Numpy specific functionality shared by the numpy-based aggregate implementations."""

from __future__ import annotations

import platform
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .aggregate_common import (
    DEFAULT_FILL_VALUE,
    _forced_float_types,
    _forced_real_float_types,
    check_complex_dtype,
    get_aliasing,
    resolve_fill_value,
)


def default_fill_value(func: str | Callable[..., Any], dtype: npt.DTypeLike = None) -> Any:
    """The value ``aggregate`` uses for groups missing from ``group_idx``.

    ``func`` may be a name, an alias or a callable.  Pass the ``dtype`` of your
    actual output to get the value that would really be used, ``None`` assumes
    a floating result (which is what the averaging functions give anyway).
    """
    return resolve_fill_value(func, DEFAULT_FILL_VALUE, np.dtype(dtype) if dtype is not None else np.float64)


def check_nton_shape(ret: np.ndarray, size, func) -> None:
    """Complain early when a one-out-per-in function has to fill a hole.

    ``sort`` and the ``cum``-functions emit exactly one value per input item,
    so a group without items leaves the output too short for the requested
    shape - which numpy would otherwise report as a plain reshape error.
    """
    if ret.size != int(np.prod(size)):
        name = getattr(func, "__name__", func)
        raise ValueError(
            f"'{name}' gives one value per input item and cannot fill the "
            f"{int(np.prod(size)) - ret.size} absent group entries of size {tuple(size)}"
        )


_alias_numpy = {
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


def check_boolean(x: Any) -> None:
    if x not in (0, 1):
        raise ValueError("Value not boolean")


_next_int_dtype = {
    "bool": np.int8,
    "uint8": np.int16,
    "int8": np.int16,
    "uint16": np.int32,
    "int16": np.int32,
    "uint32": np.int64,
    "int32": np.int64,
}

_next_float_dtype = {
    "float16": np.float32,
    "float32": np.float64,
    "float64": np.complex64,
    "complex64": np.complex128,
}


def minimum_dtype(x, dtype: np.dtype = np.bool_) -> np.dtype:
    """
    Returns the "most basic" dtype which represents `x` properly, which provides at least the same
    value range as the specified dtype.
    """

    def check_type(x, dtype):
        try:
            with np.errstate(invalid="ignore"):
                converted = np.array(x).astype(dtype)
        except (ValueError, OverflowError):
            return False
        # False if some overflow has happened
        return bool(converted == x) or (np.ndim(x) == 0 and np.isnan(x))

    def type_loop(x, dtype, dtype_dict, default=None):
        while True:
            try:
                dtype = np.dtype(dtype_dict[dtype.name])
                if check_type(x, dtype):
                    return np.dtype(dtype)
            except KeyError:
                if default is not None:
                    return np.dtype(default)
                raise ValueError(f"Can not determine dtype of {x!r}")

    dtype = np.dtype(dtype)
    if check_type(x, dtype):
        return dtype

    if np.issubdtype(dtype, np.inexact):
        return type_loop(x, dtype, _next_float_dtype)
    else:
        return type_loop(x, dtype, _next_int_dtype, default=np.float32)


def minimum_dtype_scalar(x, dtype: np.dtype | None, a) -> np.dtype:
    if dtype is None:
        dtype = np.dtype(type(a)) if isinstance(a, (int, float, complex)) else a.dtype
    return minimum_dtype(x, dtype)


_forced_types = {
    "array": object,
    "all": bool,
    "any": bool,
    "nanall": bool,
    "nanany": bool,
    "len": np.int64,
    "nanlen": np.int64,
    "allnan": bool,
    "anynan": bool,
    "argmax": np.int64,
    "argmin": np.int64,
    "nanargmin": np.int64,
    "nanargmax": np.int64,
}
if platform.architecture()[0] == "32bit":
    _forced_types = {
        "array": object,
        "all": bool,
        "any": bool,
        "nanall": bool,
        "nanany": bool,
        "len": np.int32,
        "nanlen": np.int32,
        "allnan": bool,
        "anynan": bool,
        "argmax": np.int32,
        "argmin": np.int32,
        "nanargmin": np.int32,
        "nanargmax": np.int32,
    }
_forced_same_type = {
    "min",
    "max",
    "first",
    "last",
    "nanmin",
    "nanmax",
    "nanfirst",
    "nanlast",
}


def check_dtype(dtype: npt.DTypeLike, func_str: str, a: np.ndarray, n: int | tuple[int, ...] | None) -> np.dtype:
    if np.isscalar(a) or not a.shape:
        if func_str not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        a_dtype = np.dtype(type(a))
    else:
        a_dtype = a.dtype

    return resolve_output_dtype(dtype, func_str, a_dtype, n)


def resolve_output_dtype(
    dtype: npt.DTypeLike, func_str: str, a_dtype: np.dtype, n: int | tuple[int, ...] | None
) -> np.dtype:
    """The dtype resolution of ``check_dtype`` given the input dtype directly.

    ``a_dtype`` is the dtype of the input data and ``n`` the number of values
    to be aggregated (only needed by the ``sum`` overflow guesses).
    """
    if dtype is not None:
        # dtype set by the user
        # Careful here: np.bool != np.bool_ !
        if np.issubdtype(dtype, np.bool_) and not ("all" in func_str or "any" in func_str):
            raise TypeError(f"function {func_str} requires a more complex datatype than bool")
        if not np.issubdtype(dtype, np.integer) and func_str in ("len", "nanlen"):
            raise TypeError(f"function {func_str} requires an integer datatype")
        check_complex_dtype(a_dtype, dtype, func_str)
        # TODO: Maybe have some more checks here
        return np.dtype(dtype)
    else:
        try:
            return np.dtype(_forced_types[func_str])
        except KeyError:
            if func_str in _forced_float_types:
                if np.issubdtype(a_dtype, np.inexact):
                    # floating input keeps its dtype, complex input stays complex
                    return a_dtype
                else:
                    return np.dtype(np.float64)
            elif func_str in _forced_real_float_types:
                if np.issubdtype(a_dtype, np.floating):
                    return a_dtype
                elif np.issubdtype(a_dtype, np.complexfloating):
                    # the real counterpart of the complex dtype (complex64 -> float32)
                    return np.dtype(a_dtype.char.lower())
                elif func_str in ("sumofsquares", "nansumofsquares"):
                    # exact squares of integers
                    return np.dtype(np.int64)
                else:
                    return np.dtype(np.float64)
            else:
                if func_str == "sum":
                    # Try to guess the minimally required int size
                    if np.issubdtype(a_dtype, np.int64):
                        # It's not getting bigger anymore
                        # TODO: strictly speaking it might need float
                        return np.dtype(np.int64)
                    elif np.issubdtype(a_dtype, np.integer):
                        maxval = np.iinfo(a_dtype).max * n
                        return minimum_dtype(maxval, a_dtype)
                    elif np.issubdtype(a_dtype, np.bool_):
                        return minimum_dtype(n, a_dtype)
                    else:
                        # floating, inexact, whatever
                        return a_dtype
                elif func_str in _forced_same_type:
                    return a_dtype
                else:
                    if isinstance(a_dtype, np.integer):
                        return np.dtype(np.int64)
                    else:
                        return a_dtype


def minval(fill_value, dtype: np.dtype | None) -> Any:
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.floating):
        return -np.inf
    if issubclass(dtype.type, np.complexfloating):
        # numpy orders complex values lexicographically, so -inf+0j is *larger*
        # than -inf-infj: the extreme of a single part is not the extreme value,
        # both parts have to sit at the same end of the order
        return complex(-np.inf, -np.inf)
    if issubclass(dtype.type, np.integer):
        return np.iinfo(dtype).min
    return np.finfo(dtype).min


def maxval(fill_value, dtype: np.dtype | None) -> Any:
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.floating):
        return np.inf
    if issubclass(dtype.type, np.complexfloating):
        # see minval: the seed of the reduction has to be the largest value of
        # numpy's order, which is inf+infj and not inf+0j
        return complex(np.inf, np.inf)
    if issubclass(dtype.type, np.integer):
        return np.iinfo(dtype).max
    return np.finfo(dtype).max


def check_fill_value(fill_value, dtype, func=None):
    if func in ("all", "any", "allnan", "anynan"):
        check_boolean(fill_value)
    else:
        try:
            return dtype.type(fill_value)
        except ValueError:
            raise ValueError(f"fill_value must be convertible into {dtype.type.__name__}")


def check_group_idx(group_idx: np.ndarray, a: np.ndarray | None = None, check_min: bool = True) -> None:
    if a is not None and group_idx.size != a.size:
        raise ValueError("The size of group_idx must be the same as a.size")
    if not issubclass(group_idx.dtype.type, np.integer):
        raise TypeError("group_idx must be of integer type")
    if check_min and np.min(group_idx) < 0:
        raise ValueError("group_idx contains negative indices")


def _ravel_group_idx(group_idx, a, axis, size, order, method="ravel"):
    ndim_a = a.ndim
    # Create the broadcast-ready multidimensional indexing.
    # Note the user could do this themselves, so this is
    # very much just a convenience.
    size_in = int(np.max(group_idx)) + 1 if size is None else size
    group_idx_in = group_idx
    group_idx = []
    size = []
    for ii, s in enumerate(a.shape):
        if method == "ravel":
            if ii == axis:
                ii_idx = group_idx_in
                if ii_idx.ndim == 1:
                    # 1d labels only carry the axis dimension - reshape so
                    # that ravel_multi_index broadcasts them against the
                    # arange-pieces of the other dimensions
                    ii_shape = [1] * ndim_a
                    ii_shape[ii] = s
                    ii_idx = ii_idx.reshape(ii_shape)
            else:
                ii_shape = [1] * ndim_a
                ii_shape[ii] = s
                ii_idx = np.arange(s).reshape(ii_shape)
            group_idx.append(ii_idx)
        size.append(size_in if ii == axis else s)
    # Use the indexing, and return. It's a bit simpler than
    # using trying to keep all the logic below happy
    if method == "ravel":
        group_idx = np.ravel_multi_index(group_idx, size, order=order, mode="raise")
    elif method == "offset":
        group_idx = offset_labels(group_idx_in, a.shape, axis, order, size_in)
    return group_idx, size


def offset_labels(group_idx, inshape, axis, order, size):
    """
    Offset group labels by dimension. This is used when we reduce over a subset of the dimensions of
    group_idx. It assumes that the reductions dimensions have been flattened in the last dimension
    Copied from
    https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """

    group_idx = np.broadcast_to(group_idx, inshape)
    if axis not in (-1, len(inshape) - 1):
        group_idx = np.moveaxis(group_idx, axis, -1)
    newshape = group_idx.shape[:-1] + (-1,)

    group_idx = group_idx + np.arange(np.prod(newshape[:-1]), dtype=int).reshape(newshape) * size
    if axis not in (-1, len(inshape) - 1):
        return np.moveaxis(group_idx, -1, axis)
    else:
        return group_idx


def input_validation(
    group_idx,
    a,
    size=None,
    order="C",
    axis=None,
    ravel_group_idx=True,
    check_bounds=True,
    func=None,
):
    """
    Do some fairly extensive checking of group_idx and a, trying to give the user as much help as
    possible with what is wrong. Also, convert ndim-indexing to 1d indexing.
    """
    if not isinstance(a, (int, float, complex)) and not is_duck_array(a):
        a = np.asanyarray(a)

    if len(group_idx) == 0:
        raise ValueError("group_idx must not be empty")
    if not is_duck_array(group_idx):
        group_idx = np.asanyarray(group_idx)

    # equivalent to np.issubdtype(group_idx.dtype, np.integer), just cheaper
    if group_idx.dtype.kind not in "iu":
        raise TypeError("group_idx must be of integer type")

    # This check works for multidimensional indexing as well
    if check_bounds and np.any(group_idx < 0):
        raise ValueError("negative indices not supported")

    ndim_idx = group_idx.ndim
    ndim_a = getattr(a, "ndim", 0)

    # Deal with the axis arg: if present, then turn 1d indexing into
    # multi-dimensional indexing along the specified axis.
    if axis is None:
        if ndim_a > 1:
            raise ValueError("a must be scalar or 1 dimensional, use .ravel to flatten. Alternatively specify axis.")
    elif axis >= ndim_a or axis < -ndim_a:
        raise ValueError("axis arg too large for np.ndim(a)")
    else:
        axis = axis if axis >= 0 else ndim_a + axis  # negative indexing
        if ndim_idx > 1:
            # multidimensional group labels - e.g. separate group labels for
            # each row - broadcast across the non-axis dimensions of a
            # (issue #74)
            if ndim_idx != ndim_a:
                raise ValueError("when using axis arg, group_idx must be 1d, or of the same dimensionality as a")
            try:
                group_idx = np.broadcast_to(group_idx, a.shape)
            except ValueError as err:
                raise ValueError(
                    f"group_idx with shape {group_idx.shape} cannot be broadcast to a with shape {a.shape}"
                ) from err
            ndim_idx = 1
        if group_idx.ndim == 1 and a.shape[axis] != group_idx.shape[0]:
            raise ValueError("a.shape[axis] doesn't match length of group_idx.")
        elif size is not None and not np.isscalar(size):
            raise NotImplementedError("when using axis arg, size must be None or scalar.")
        else:
            is_form_3 = ndim_a > 1
            orig_shape = a.shape if is_form_3 else group_idx.shape
            if isinstance(func, str) and "arg" in func:
                unravel_shape = orig_shape
            else:
                unravel_shape = None

            method = "offset" if axis == ndim_a - 1 else "ravel"
            group_idx, size = _ravel_group_idx(group_idx, a, axis, size, order, method=method)
            flat_size = np.prod(size)
            ndim_idx = ndim_a
            size = orig_shape if is_form_3 and not callable(func) and "cum" in func else size
            return (
                group_idx.ravel(),
                a.ravel(),
                flat_size,
                ndim_idx,
                size,
                unravel_shape,
            )

    if ndim_idx == 1:
        if size is None:
            size = int(np.max(group_idx)) + 1
        else:
            if not np.isscalar(size):
                raise ValueError("output size must be scalar or None")
            if check_bounds and np.any(group_idx > size - 1):
                raise ValueError(f"one or more indices are too large for size {size}")
        flat_size = size
    else:
        if size is None:
            size = np.max(group_idx, axis=1).astype(int) + 1
        elif np.isscalar(size):
            raise ValueError(f"output size must be of length {len(group_idx)}")
        elif len(size) != len(group_idx):
            raise ValueError(f"{len(size)} sizes given, but {len(group_idx)} output dimensions specified in index")
        if ravel_group_idx:
            group_idx = np.ravel_multi_index(group_idx, size, order=order, mode="raise")
        flat_size = np.prod(size)

    if not (ndim_a == 0 or len(a) == group_idx.size):
        raise ValueError("group_idx and a must be of the same length, or a can be scalar")

    return group_idx, a, flat_size, ndim_idx, size, None


# General tools


def unpack(group_idx: npt.ArrayLike, ret: np.ndarray) -> np.ndarray:
    """
    Take an aggregate packed array and uncompress it to the size of group_idx. This is equivalent to
    ret[group_idx].
    """
    return ret[group_idx]


def allnan(x):
    return np.all(np.isnan(x))


def anynan(x):
    return np.any(np.isnan(x))


def nanfirst(x):
    return x[~np.isnan(x)][0]


def nanlast(x):
    return x[~np.isnan(x)][-1]


def multi_arange(n: npt.ArrayLike) -> np.ndarray:
    """By example:

        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    That is it is equivalent to something like this :

        hstack((arange(n_i) for n_i in n))

    This version seems quite a bit faster, at least for some possible inputs, and at any rate it
    encapsulates a task in a function.
    """
    if n.ndim != 1:
        raise ValueError("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def label_contiguous_1d(X: np.ndarray) -> np.ndarray:
    """
    WARNING: API for this function is liable to change!!!

    By example:

        X =      [F T T F F T F F F T T T]
        result = [0 1 1 0 0 2 0 0 0 3 3 3]

    Or:
        X =      [0 3 3 0 0 5 5 5 1 1 0 2]
        result = [0 1 1 0 0 2 2 2 3 3 0 4]

    The ``0`` or ``False`` elements of ``X`` are labeled as ``0`` in the output. If ``X`` is a boolean
    array, each contiguous block of ``True`` is given an integer label, if ``X`` is not boolean, then
    each contiguous block of identical values is given an integer label. Integer labels are 1, 2, 3,
    ..... (i.e. start a 1 and increase by 1 for each block with no skipped numbers.)
    """

    if X.ndim != 1:
        raise ValueError("this is for 1d masks only.")

    is_start = np.empty(len(X), dtype=bool)
    is_start[0] = X[0]  # True if X[0] is True or non-zero

    if X.dtype.kind == "b":
        is_start[1:] = ~X[:-1] & X[1:]
        M = X
    else:
        M = X.astype(bool)
        is_start[1:] = X[:-1] != X[1:]
        is_start[~M] = False

    L = np.cumsum(is_start)
    L[~M] = 0
    return L


def relabel_groups_unique(group_idx: np.ndarray) -> np.ndarray:
    """
    See also ``relabel_groups_masked``.

    keep_group:  [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
    ret:         [0 3 3 3 0 2 4 2 0 1 1 0 3 4 4]

    Description of above: unique groups in input was ``1,2,3,5``, i.e.
    ``4`` was missing, so group 5 was relabeled to be ``4``.
    Relabeling maintains order, just "compressing" the higher numbers
    to fill gaps.
    """

    keep_group = np.zeros(np.max(group_idx) + 1, dtype=bool)
    keep_group[0] = True
    keep_group[group_idx] = True
    return relabel_groups_masked(group_idx, keep_group)


def relabel_groups_masked(group_idx: np.ndarray, keep_group: npt.ArrayLike) -> np.ndarray:
    """
    group_idx: [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]

                 0 1 2 3 4 5
    keep_group: [0 1 0 1 1 1]

    ret:       [0 2 2 2 0 0 4 0 0 1 1 0 2 4 4]

    Description of above in words: remove group 2, and relabel group 3,4, and 5 to be 2, 3 and 4
    respectively, in order to fill the gap.  Note that group 4 was never used in the input group_idx,
    but the user supplied mask said to keep group 4, so group 5 is only moved up by one place to fill
    the gap created by removing group 2.

    That is, the mask describes which groups to remove, the remaining groups are relabeled to remove the
    gaps created by the falsy elements in ``keep_group``. Note that ``keep_group[0]`` has no particular
    meaning because it refers to the zero group which cannot be "removed".

    ``keep_group`` should be bool and ``group_idx`` int. Values in ``group_idx`` can be any order.
    """

    keep_group = keep_group.astype(bool, copy=not keep_group[0])
    if not keep_group[0]:  # ensuring keep_group[0] is True makes life easier
        keep_group[0] = True

    relabel = np.zeros(keep_group.size, dtype=group_idx.dtype)
    relabel[keep_group] = np.arange(np.count_nonzero(keep_group))
    return relabel[group_idx]


def is_duck_array(value: Any) -> bool:
    """This function was copied from xarray/core/utils.py under the terms of Xarray's Apache-2 license."""

    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "__array_function__")
        and hasattr(value, "__array_ufunc__")
    )


def iscomplexobj(x: Any) -> bool:
    """Copied from np.iscomplexobj so that we place fewer requirements on duck array types."""

    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = np.asarray(x).dtype.type
    return issubclass(type_, np.complexfloating)
