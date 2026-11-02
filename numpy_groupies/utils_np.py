"""Numpy specific functionality shared by the numpy-based aggregate implementations."""

import platform
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, SupportsFloat, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.registry import (
    DEFAULT_FILL_VALUE,
    DefaultFillValue,
    forced_float_types,
    forced_real_float_types,
    get_aliasing,
    resolve_name,
)
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    FILL_NOT_CONVERTIBLE,
    LENGTH_MISMATCH_OR_SCALAR,
    NEGATIVE_INDICES,
    NON_INTEGER_GROUP_IDX,
    TOO_LARGE_FOR_SIZE,
    check_boolean,
    check_complex_dtype,
    is_integer_dtype,
    is_scalar_input,
)

# the values as the dispatchers carry them on: an array of any dtype, or
# the one python scalar that stands for that value repeated over every label
AValue: TypeAlias = npt.NDArray[np.generic] | int | float | complex | np.bool_

# the result of an aggregation: numbers for most functions, bools for
# 'all' and 'any', objects for 'array' - np.generic covers all three
AResult: TypeAlias = npt.NDArray[np.generic]

# the numbers and flags that can land in an array - a bool is an int, a
# np.bool_ is neither an int nor an np.number
AScalar: TypeAlias = float | int | complex | np.number | np.bool_

# what a fill value can be: a number, the sentinel that asks for the
# default of the function, a None or - for the object result - a sequence
AFillValue: TypeAlias = AScalar | None | Sequence[AScalar] | DefaultFillValue

# the memory layout the output is written in, as every signature of this
# package spells it: the docstring offers 'F' for fortran-style and the code
# only ever ravels and reshapes with these two
RavelOrder: TypeAlias = Literal["C", "F"]

# what input_validation hands back: (raveled integer indices, the values -
# possibly still a python scalar -, the flat output size, the input index
# dimensionality, the (possibly shaped) output size and the shape an arg
# reduction has to unravel its positions with, in that order
ValidatedInput: TypeAlias = tuple[
    npt.NDArray[np.integer],
    AValue,
    int,
    int,
    int | tuple[int, ...] | npt.NDArray[np.integer],
    tuple[int, ...] | None,
]


def check_nton_shape(
    ret: npt.NDArray[np.generic],
    size: int | tuple[int, ...] | npt.NDArray[np.integer],
    func: str | Callable[..., AScalar],
) -> None:
    """Complain early when a one-out-per-in function has to fill a hole.

    ``sort`` and the ``cum``-functions emit exactly one value per input item,
    so a group without items leaves the output too short for the requested
    shape - which numpy would otherwise report as a plain reshape error.
    """
    # np.prod takes an int as it is, a tuple or an array of ints
    prod = int(np.prod(size))
    if ret.size != prod:
        name = getattr(func, "__name__", func)
        raise ValueError(
            f"'{name}' gives one value per input item and cannot fill the "
            f"{prod - ret.size} absent group entries of size {_size_tuple(size)}"
        )


def _size_tuple(size: int | tuple[int, ...] | npt.NDArray[np.integer]) -> tuple[int, ...]:
    """The size as a tuple of its dimensions - an int is one dimension."""
    if isinstance(size, tuple):
        return size
    if isinstance(size, int):
        return (size,)
    return tuple(int(s) for s in size)


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


def minimum_dtype(x: AFillValue, dtype: npt.DTypeLike | None = np.bool_) -> np.dtype:
    """
    Returns the "most basic" dtype which represents `x` properly, which provides at least the same
    value range as the specified dtype.
    """

    def check_type(x: AFillValue, dtype: np.dtype) -> bool:
        try:
            with np.errstate(invalid="ignore"):
                converted = np.array(x).astype(dtype)
        except (ValueError, OverflowError):
            return False
        # False if some overflow has happened
        return bool(converted == x) or (np.ndim(cast(np.generic, x)) == 0 and np.isnan(cast(np.generic, x)))

    def type_loop(
        x: AFillValue, dtype: np.dtype, dtype_dict: Mapping[str, type], default: type | None = None
    ) -> np.dtype:
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


def minimum_dtype_scalar(x: AFillValue, dtype: npt.DTypeLike | None, a: AValue) -> np.dtype:
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
    "nunique": np.int64,
    "nanlen": np.int64,
    "nannunique": np.int64,
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
        "nunique": np.int32,
        "nanlen": np.int32,
        "nannunique": np.int32,
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


def scalar_dtype(a: Any) -> np.dtype:
    """The dtype of scalar input data.

    A python number carries no dtype, so its type stands in for one, while a
    numpy scalar and a 0d array both carry the dtype of their value.  Asking
    ``type(a)`` is not an option here: for the numpy scalar and for the 0d
    array that is ``numpy.float32``/``numpy.ndarray``, and
    ``np.dtype(np.ndarray)`` is the object dtype.
    """
    dtype = getattr(a, "dtype", None)
    return np.dtype(type(a)) if dtype is None else np.dtype(dtype)


def check_dtype(
    dtype: npt.DTypeLike | None,
    func_str: str | Callable[..., AScalar],
    # an array of any dtype the operation may meet, or the one python scalar
    # that stands for it - is_scalar_input below is what tells the two apart
    a: Any,
    n: int | tuple[int, ...] | None,
) -> np.dtype:
    if is_scalar_input(a):
        if func_str not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        a_dtype = scalar_dtype(a)
    else:
        a_dtype = a.dtype

    return resolve_output_dtype(dtype, func_str, a_dtype, n)


def resolve_output_dtype(
    dtype: npt.DTypeLike | None,
    func_str: str | Callable[..., AScalar],
    a_dtype: np.dtype,
    n: int | tuple[int, ...] | None,
) -> np.dtype:
    """The dtype resolution of ``check_dtype`` given the input dtype directly.

    ``a_dtype`` is the dtype of the input data and ``n`` the number of values
    to be aggregated (only needed by the ``sum`` overflow guesses).
    """
    named = isinstance(func_str, str)
    if dtype is not None:
        # dtype set by the user
        # Careful here: np.bool != np.bool_ !
        if np.issubdtype(dtype, np.bool_) and not (
            isinstance(func_str, str) and ("all" in func_str or "any" in func_str)
        ):
            raise TypeError(f"function {func_str} requires a more complex datatype than bool")
        if not np.issubdtype(dtype, np.integer) and named and func_str in ("len", "nanlen", "nunique", "nannunique"):
            raise TypeError(f"function {func_str} requires an integer datatype")
        check_complex_dtype(a_dtype, dtype, func_str)
        # TODO: Maybe have some more checks here
        return np.dtype(dtype)
    else:
        try:
            # a callable of the caller's own names no function - the lookup
            # misses, like a name the table does not hold
            return np.dtype(_forced_types[func_str if isinstance(func_str, str) else ""])
        except KeyError:
            if func_str in forced_float_types:
                if np.issubdtype(a_dtype, np.inexact):
                    # floating input keeps its dtype, complex input stays complex
                    return a_dtype
                else:
                    return np.dtype(np.float64)
            elif func_str in forced_real_float_types:
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
                        # the summing path arrives with a flat count, the one-out
                        # per-in path never reaches this branch
                        maxval = np.iinfo(a_dtype).max * cast("int", n)
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


def minval(fill_value: AFillValue, dtype: np.dtype | None) -> AScalar:
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.bool_):
        # a bool has no nan to mask, but the mask value has to be one of the
        # dtype - np.finfo knows no bool
        return cast("AScalar", np.False_)
    if issubclass(dtype.type, np.floating):
        return -np.inf
    if issubclass(dtype.type, np.complexfloating):
        # numpy orders complex values lexicographically, so -inf+0j is *larger*
        # than -inf-infj: the extreme of a single part is not the extreme value,
        # both parts have to sit at the same end of the order
        return complex(-np.inf, -np.inf)
    if issubclass(dtype.type, np.integer):
        return np.iinfo(dtype).min
    return cast("AScalar", np.finfo(dtype).min)


def maxval(fill_value: AFillValue, dtype: np.dtype | None) -> AScalar:
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.bool_):
        # see minval: the mask value has to be one of the dtype
        return cast("AScalar", np.True_)
    if issubclass(dtype.type, np.floating):
        return cast("AScalar", np.inf)
    if issubclass(dtype.type, np.complexfloating):
        # see minval: the seed of the reduction has to be the largest value of
        # numpy's order, which is inf+infj and not inf+0j
        return cast("AScalar", complex(np.inf, np.inf))
    if issubclass(dtype.type, np.integer):
        return np.iinfo(dtype).max
    return cast("AScalar", np.finfo(dtype).max)


def check_fill_value_for_dtype(fill_value: AFillValue, dtype: np.dtype) -> AScalar:
    """The fill value as the output dtype holds it - the check of a kernel.

    The rules are the ones ``check_fill_value`` applies to the python-level
    call, without the function-specific part: a kernel knows its output dtype
    and nothing else, so the value has to fit that dtype - a nan does not fit
    an integer output, and a boolean output takes the two booleans only.
    """
    if issubclass(dtype.type, np.bool_) and not isinstance(fill_value, (list, tuple, np.ndarray)):
        # a sequence is the fill of a group that holds a sequence (the sorting
        # function) - it is not a boolean and does not have to be one
        check_boolean(fill_value)
        return np.bool_(fill_value)
    try:
        return cast("AScalar", dtype.type(fill_value))
    except (ValueError, TypeError, OverflowError) as err:
        raise ValueError(FILL_NOT_CONVERTIBLE.format(dtype=dtype.type.__name__)) from err


def check_fill_value(
    fill_value: AFillValue, dtype: np.dtype, func: str | Callable[..., AScalar] | None = None
) -> AFillValue | None:
    if func in ("all", "any", "allnan", "anynan"):
        check_boolean(fill_value)
        return None
    else:
        return check_fill_value_for_dtype(fill_value, dtype)


def check_group_idx(
    group_idx: npt.NDArray[np.integer], a: npt.NDArray[np.number] | None = None, check_min: bool = True
) -> None:
    if a is not None and group_idx.size != a.size:
        raise ValueError("The size of group_idx must be the same as a.size")
    if not is_integer_dtype(group_idx.dtype):
        raise TypeError(NON_INTEGER_GROUP_IDX)
    if check_min and np.min(group_idx) < 0:
        raise ValueError(NEGATIVE_INDICES)


def _ravel_group_idx(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    axis: int,
    size: Any,
    order: RavelOrder,
    method: str = "ravel",
) -> tuple[Any, Any]:
    ndim_a = a.ndim
    # Create the broadcast-ready multidimensional indexing.
    # Note the user could do this themselves, so this is
    # very much just a convenience.
    size_in = int(np.max(group_idx)) + 1 if size is None else size
    group_idx_in = group_idx
    # the labels are collected per input dimension before ravel_multi_index folds them
    labels: list[npt.NDArray[np.integer]] = []
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
            labels.append(ii_idx)
        size.append(size_in if ii == axis else s)
    # Use the indexing, and return. It's a bit simpler than
    # using trying to keep all the logic below happy
    if method == "ravel":
        group_idx = np.ravel_multi_index(labels, size, order=order, mode="raise")
    elif method == "offset":
        group_idx = offset_labels(group_idx_in, a.shape, axis, order, size_in)
    return group_idx, size


def offset_labels(
    group_idx: npt.NDArray[np.integer], inshape: tuple[int, ...], axis: int, order: RavelOrder, size: int
) -> npt.NDArray[np.integer]:
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
    # the unvalidated user input - this function is exactly what narrows it,
    # down to the integer array the signatures of the package ask for, and it
    # still admits what those do not name: lists, duck arrays and the python
    # scalar that stands for one value repeated over every label
    group_idx: Any,
    a: Any,
    size: Any = None,
    order: RavelOrder = "C",
    axis: int | None = None,
    ravel_group_idx: bool = True,
    check_bounds: bool = True,
    func: str | Callable[..., AScalar] | None = None,
) -> ValidatedInput:
    """
    Do some fairly extensive checking of group_idx and a, trying to give the user as much help as
    possible with what is wrong. Also, convert ndim-indexing to 1d indexing.

    ``func`` is what the caller was given: a name, an alias or a callable, of
    which only the name it resolves to is of interest here - whether it is an
    ``arg``-reduction, which names a position in the input, or one of the
    one-out-per-input-item answers (``cum`` and ``sort``).  A func that resolves
    to no name is neither, which is what a custom callable is.
    """
    if not isinstance(a, (int, float, complex)) and not is_duck_array(a):
        a = np.asanyarray(a)

    if len(group_idx) == 0:
        raise ValueError(EMPTY_GROUP_IDX)
    if not is_duck_array(group_idx):
        group_idx = np.asanyarray(group_idx)

    # equivalent to np.issubdtype(group_idx.dtype, np.integer), just cheaper
    if not is_integer_dtype(group_idx.dtype):
        raise TypeError(NON_INTEGER_GROUP_IDX)

    # This check works for multidimensional indexing as well
    if check_bounds and np.any(group_idx < 0):
        raise ValueError(NEGATIVE_INDICES)

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
            func_name = resolve_name(func, aliasing)
            if isinstance(func_name, str) and "arg" in func_name:
                unravel_shape = orig_shape
            else:
                unravel_shape = None

            method = "offset" if axis == ndim_a - 1 and order == "C" else "ravel"
            group_idx, size = _ravel_group_idx(group_idx, a, axis, size, order, method=method)
            flat_size = np.prod(size)
            ndim_idx = ndim_a
            one_to_one = is_form_3 and isinstance(func_name, str) and ("cum" in func_name or func_name == "sort")
            size = orig_shape if one_to_one else size
            ravel_order = order if one_to_one else "C"
            return (
                group_idx.ravel(order=ravel_order),
                a.ravel(order=ravel_order),
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
            # a scalar size, which the comparison below needs as a number
            scalar_size = cast("int | float | np.number", size)
            if check_bounds and np.any(group_idx > scalar_size - 1):
                raise ValueError(TOO_LARGE_FOR_SIZE.format(size=scalar_size))
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
        raise ValueError(LENGTH_MISMATCH_OR_SCALAR)

    return (group_idx, a, flat_size, ndim_idx, size, None)


# General tools


def allnan(x: npt.NDArray[np.inexact]) -> np.bool_:
    return np.all(np.isnan(x))


def anynan(x: npt.NDArray[np.inexact]) -> np.bool_:
    return np.any(np.isnan(x))


def multi_arange(n: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
    """By example:

        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    This is equivalent to something like this :

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
    return cast("npt.NDArray[np.intp]", np.cumsum(ret)[:-1])


def label_contiguous_1d(X: npt.NDArray[np.number]) -> npt.NDArray[np.intp]:
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
        # the kind test has established a boolean array, the signature says generic
        M = cast("npt.NDArray[np.bool_]", X)
        is_start[1:] = ~M[:-1] & M[1:]
    else:
        M = X.astype(bool)
        is_start[1:] = X[:-1] != X[1:]
        is_start[~M] = False

    L = np.cumsum(is_start)
    L[~M] = 0
    return cast("npt.NDArray[np.intp]", L)


def relabel_groups_unique(group_idx: npt.NDArray[np.integer]) -> npt.NDArray[np.intp]:
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


def relabel_groups_masked(
    group_idx: npt.NDArray[np.integer], keep_group: npt.NDArray[np.bool_]
) -> npt.NDArray[np.intp]:
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

    # astype(bool) is what normalizes an int mask that arrives anyway
    keep_group = keep_group.astype(bool, copy=False)
    if not keep_group[0]:  # ensuring keep_group[0] is True makes life easier
        # the copy belongs here, to the one write of its own, and not to a flag
        # that has to read the element it is about to change
        keep_group = keep_group.copy()
        keep_group[0] = True

    relabel = np.zeros(keep_group.size, dtype=group_idx.dtype)
    relabel[keep_group] = np.arange(np.count_nonzero(keep_group))
    return cast("npt.NDArray[np.intp]", relabel[group_idx])


def is_duck_array(value: object) -> bool:
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


# ---------------------------------------------------------------------------
# datetime64 and timedelta64 support
#
# Temporal arrays never reach the kernels as datetime values: the wrapper
# below takes them apart into their int64 ticks ahead of the kernels and
# turns the results back afterwards.  NaT is the int64 tick ``-2**63``, so
# every value a kernel computes on the ticks is a tick again - the plan
# carries what the wrapper has to do to a tick result to make it temporal.
#
# The policy follows numpy:
# - plain reductions propagate NaT (a group holding one answers NaT), the
#   nan-variants skip it,
# - what numpy refuses, we refuse: datetime64 has no sum, prod, cumsum,
#   cumprod, median, quantile, var, std, skew, kurt or trapezoid, and
#   timedelta64 has no prod, cumprod, var, std, skew, kurt or trapezoid - in
#   both cases numpy itself raises,
# - the unit of the input is the unit of the answer, and NaT is the fill
#   value.

TEMPORAL_KINDS = "Mm"
NAT_TICK: int = -(2**63)

# the functions whose result holds ticks of the input again - their answer is
# viewed back into the dtype of the input.  datetime64 has no sum and no
# median, but both raise before anything is viewed back.
TEMPORAL_VALUE_FUNCS = frozenset(
    {
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "quantile",
        "percentile",
        "first",
        "last",
        "sort",
        "cumsum",
        "cummax",
        "cummin",
    }
)
# the functions numpy refuses for datetime64 - the plain and the nan variant
# share their fate
TEMPORAL_DATETIME_REJECTED = frozenset(
    {
        "sum",
        "prod",
        "cumsum",
        "cumprod",
        "median",
        "quantile",
        "percentile",
        "var",
        "std",
        "skew",
        "kurt",
        "trapezoid",
        "sumofsquares",
    }
)
# and the ones numpy refuses for timedelta64 - sum, cumsum, mean, median and
# the quantiles work there like they do on integers
TEMPORAL_TIMEDELTA_REJECTED = frozenset({"prod", "cumprod", "var", "std", "skew", "kurt", "trapezoid", "sumofsquares"})
# plain variants of these poison their whole group with a single NaT - the
# group answers NaT rather than the value of its ticks.  The sum of the
# durations joins them: numpy adds NaT into a temporal sum and the answer is
# NaT, while the ticks would wrap around in the int64 accumulation.
TEMPORAL_NAT_POISON = frozenset({"sum", "mean", "max", "median", "quantile", "percentile"})
# the value functions whose default fill is nan - the temporal answer of an
# empty group is NaT for them, and 0 (the epoch, or no duration) for the
# functions that fill with zero, like the sum of the durations
TEMPORAL_NAT_FILL = frozenset({"min", "max", "mean", "median", "quantile", "percentile", "first", "last"})


class TemporalPlan:
    """What :func:`finalize_temporal` has to do with a tick result.

    ``computed`` holds an already finished int64 result for the functions
    whose kernel path cannot stay exact on ticks: the temporal mean adds up
    ticks beyond the window float64 represents exactly, so the wrapper
    computes it on the limb sums of the ticks itself, and the median and the
    quantiles of timedelta64 average their middles as python integers.
    """

    __slots__ = (
        "children",
        "computed",
        "computed_final",
        "nat_groups",
        "nat_positions",
        "result_dtype",
        "running_nat",
        "sort_nat_restore",
        "user_dtype",
        "value_result",
    )

    def __init__(self, result_dtype: np.dtype) -> None:
        self.result_dtype = result_dtype
        self.user_dtype: np.dtype | None = None
        self.computed: npt.NDArray[np.int64] | None = None
        self.computed_final = False
        self.nat_groups: npt.NDArray[np.integer] | None = None
        self.nat_positions: npt.NDArray[np.intp] | None = None
        self.running_nat: npt.NDArray[np.bool_] | None = None
        self.children = False
        self.value_result = False
        self.sort_nat_restore = False


def is_temporal_input(a: Any) -> bool:
    """Whether the input carries datetime64 or timedelta64 values - an array
    of them, or the one scalar standing for a value of every group."""
    dtype = getattr(a, "dtype", None)
    return dtype is not None and np.dtype(dtype).kind in TEMPORAL_KINDS


def _temporal_fill_tick(fill_value: Any, a_dtype: np.dtype, default_is_nat: bool) -> int:
    """The tick of the fill value - NaT where the default of the function is
    nan, zero where it fills with zero, unless the caller asks for a date."""
    if fill_value is None or fill_value is DEFAULT_FILL_VALUE:
        return NAT_TICK if default_is_nat else 0
    if isinstance(fill_value, float) and fill_value != fill_value:
        return NAT_TICK if default_is_nat else 0
    if isinstance(fill_value, (int, np.integer)):
        return int(fill_value)
    # a datetime64/timedelta64, or anything numpy reads as one: its tick in
    # the unit of the input
    return int(np.asarray(fill_value, dtype=a_dtype).view(np.int64))


def _groups_holding(group_idx: npt.NDArray[np.integer], mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.integer]:
    """The sorted labels of the groups where ``mask`` holds at least once."""
    return np.unique(group_idx[mask])


def _first_true_position_per_group(
    group_idx: npt.NDArray[np.integer], mask: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.intp]]:
    """The labels of the masked groups and the first masked position of each.

    numpy answers the position of the first NaT for both argmin and argmax of
    a group that holds one - the argreductions on the ticks deliver the first
    NaT for argmin on their own (NaT is the smallest tick), argmax is
    corrected with this.
    """
    positions = np.flatnonzero(mask)
    labels = group_idx[positions]
    # a stable sort keeps the positions of every group in ascending order, so
    # the first of each run is the first NaT of the group
    order = np.argsort(labels, kind="stable")
    labels, positions = labels[order], positions[order]
    starts = np.concatenate(([True], labels[1:] != labels[:-1]))
    return labels[starts], positions[starts]


def _round_half_even(total: int, count: int) -> int:
    """The exact integer division of ``total`` by ``count``, half to even."""
    q, r = divmod(total, count)
    if 2 * r > count or (2 * r == count and q % 2 == 1):
        q += 1
    return q


def exact_integer_group_mean(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.integer],
    size: int,
    fill_value: AScalar,
    dtype: np.dtype,
) -> npt.NDArray[np.floating] | npt.NDArray[np.integer]:
    """The mean of integer values per group, exact where float64 would round.

    The ticks of datetime input add up beyond the +/-2**53 window float64
    represents exactly, and so can the sums of plain integers - instead of
    refusing the mean, the values are split into four 16-bit limbs, every
    limb accumulates in an exact sum through ``bincount``, and the totals of
    the groups are combined as python integers.  The division is exact and
    the remainder rounds half to even.

    An integer mean of small magnitudes keeps coming back as float64 - the
    quotient is exact in float64 there, so its answer does not change - and a
    caller asking for an integer dtype gets the exact rounded quotient.
    """
    u = np.ascontiguousarray(a).view(np.uint64)
    limb_sums = [
        np.bincount(group_idx, weights=(u >> np.uint64(shift)) & np.uint64(0xFFFF), minlength=size).astype(np.int64)
        for shift in (0, 16, 32, 48)
    ]
    counts = np.bincount(group_idx, minlength=size)
    total: npt.NDArray[Any] = (
        (limb_sums[3].astype(object) << 48)
        + (limb_sums[2].astype(object) << 32)
        + (limb_sums[1].astype(object) << 16)
        + limb_sums[0].astype(object)
    )
    if a.dtype.kind == "i":
        # signed values sum to the two's-complement limb sum of their bit
        # pattern plus one 2**64 per negative value - the pattern sums are
        # corrected by the count of the negatives of every group
        negatives = np.bincount(group_idx, weights=u >= np.uint64(2**63), minlength=size)
        total -= negatives.astype(np.int64).astype(object) << 64
    present = counts > 0
    quotients = np.zeros(size, dtype=object)
    remainders = np.zeros(size, dtype=object)
    quotients[present], remainders[present] = (
        np.floor_divide(total[present], counts[present].astype(object)),
        np.remainder(total[present], counts[present].astype(object)),
    )
    # the fill of the empty groups: the default of the dtype unless the caller
    # asked for a number - the nan of a float result and the sentinel of the
    # call both mean "default"
    fill: int | float | None
    if isinstance(fill_value, (float, np.floating)) and fill_value != fill_value:
        fill = None  # a nan asks for the default
    elif isinstance(fill_value, (bool, np.bool_, int, np.integer)):
        fill = int(fill_value)
    elif fill_value is None or fill_value.__class__.__name__ == "DefaultFillValue":
        fill = None  # the sentinel of the call asks for the default
    else:
        fill = float(cast("SupportsFloat", fill_value))
    if np.issubdtype(dtype, np.integer):
        rounded = quotients.copy()
        bump = (2 * remainders > counts) | ((2 * remainders == counts) & (quotients % 2 == 1))
        rounded[bump] += 1
        ret = rounded.astype(np.int64)
    else:
        # the float64 answer: the exact quotient and remainder of the python
        # integers, so the total itself can never round - the 64-bit mantissa
        # of the long double holds every int64 quotient exactly, and the two
        # roundings of the assembly stay within half an ulp of the exact
        # quotient
        quotients = quotients.astype(np.longdouble)
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = np.true_divide(remainders.astype(np.longdouble), counts.astype(np.longdouble))
        ret = (quotients + fractions).astype(np.float64)
        if fill is not None:
            ret[~present] = fill
        return ret
    if fill is not None and fill != 0:
        ret[~present] = fill
    return ret


def _temporal_group_median(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_tick: int,
) -> npt.NDArray[np.int64]:
    """The median of the ticks of a timedelta64 input, exact.

    The middles of an even group average to their exact mean rounded half to
    even - the two add up as python integers, so even ticks at the edges of
    int64 cannot wrap the sum.
    """
    order = np.argsort(group_idx, kind="stable")
    values = ticks[order]
    counts = np.bincount(group_idx, minlength=size)
    starts = np.zeros(size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    ret = np.full(size, NAT_TICK, dtype=np.int64)
    for grp in np.flatnonzero(counts):
        count = int(counts[grp])
        part = np.partition(values[starts[grp] : starts[grp] + count], count // 2)
        if count % 2 == 1:
            ret[grp] = part[count // 2]
        else:
            ret[grp] = _round_half_even(int(part[: count // 2].max()) + int(part[count // 2]), 2)
    ret[counts == 0] = fill_tick
    return ret


def _temporal_group_quantile(
    group_idx: npt.NDArray[np.integer],
    ticks: npt.NDArray[np.int64],
    size: int,
    fill_tick: int,
    q: float,
) -> npt.NDArray[np.int64]:
    """The quantile of the ticks of a timedelta64 input.

    Like numpy's ``linear`` method, the two ticks around the quantile
    position interpolate linearly - their difference is exact as a python
    integer, and the rounded product keeps the answer on the tick grid.
    """
    order = np.argsort(group_idx, kind="stable")
    values = ticks[order]
    counts = np.bincount(group_idx, minlength=size)
    starts = np.zeros(size, dtype=np.intp)
    np.cumsum(counts[:-1], out=starts[1:])
    ret = np.full(size, NAT_TICK, dtype=np.int64)
    for grp in np.flatnonzero(counts):
        count = int(counts[grp])
        part = np.partition(values[starts[grp] : starts[grp] + count], [0, count - 1] if count > 1 else 0)
        position = q * (count - 1)
        low = min(int(np.floor(position)), count - 1)
        high = min(low + 1, count - 1)
        difference = int(part[high]) - int(part[low])
        ret[grp] = int(part[low]) + int(np.rint((position - low) * difference))
    ret[counts == 0] = fill_tick
    return ret


def prepare_temporal(
    funcname: str,
    group_idx: npt.NDArray[np.integer],
    a: Any,
    fill_value: AFillValue,
    dtype: npt.DTypeLike | None,
    size: int,
    kwargs: dict[str, Any],
) -> tuple[str, npt.NDArray[np.integer], AValue, AFillValue, np.dtype | None, TemporalPlan]:
    """Take temporal input apart into the tick input of the kernels.

    Answers the ``funcname``, the (possibly NaT-filtered) ``group_idx``, the
    int64 ticks standing in for the values, the fill value as a tick, the
    output dtype the kernels are asked for, and the plan that turns a tick
    result back into a temporal one.  The functions numpy refuses raise here,
    and the functions the wrapper computes on its own carry their finished
    int64 result in the plan.
    """
    base = funcname.removeprefix("nan")
    is_datetime = np.dtype(a.dtype).kind == "M"
    rejected = TEMPORAL_DATETIME_REJECTED if is_datetime else TEMPORAL_TIMEDELTA_REJECTED
    if base in rejected:
        raise TypeError(
            f"aggregate {funcname!r} is not supported for {np.dtype(a.dtype).name} input - "
            f"numpy refuses {base} of {'datetime64' if is_datetime else 'timedelta64'} values as well"
        )
    plan = TemporalPlan(np.dtype(a.dtype))
    nat: npt.NDArray[np.bool_] | None = None
    if isinstance(a, np.ndarray):
        nat = np.isnat(a)
        if funcname.startswith("nan") and base in ("argmin", "argmax"):
            # the nan-variants of the argreductions keep every row - the
            # positions they name are the ones of the input - so the NaT rows
            # stand in as the extreme tick the reduction skips.  A group of
            # nothing but NaT holds no tick of its own after the swap, and is
            # left to the fill value like its float counterpart
            skip = np.iinfo(np.int64).max if base == "argmin" else NAT_TICK + 1
            a = np.where(nat, np.int64(skip), np.asarray(a).view(np.int64)).view(a.dtype)
            nat = None
        elif funcname.startswith("nan") and base != "cumsum":
            # the nan-variants skip NaT like they skip nan - the plain ones
            # propagate it, and numpy propagates it in nancumsum as well, so
            # the cumulative functions keep their rows
            if nat.any():
                good = ~nat
                a = a[good]
                group_idx = group_idx[good]
                nat = None
    else:
        # a temporal scalar comes through as its tick - sum and len of a
        # scalar timedelta64 answer in int64, like a plain integer scalar
        return (
            funcname,
            group_idx,
            cast("AValue", np.int64(np.asarray(a).view(np.int64))),
            fill_value,
            None,
            plan,
        )
    if base in ("all", "any"):
        # the truth of a temporal value is its tick being non-zero and the
        # value not being NaT - a NaT is no value at all
        truth: npt.NDArray[np.bool_] = np.asarray(a).view(np.int64) != 0
        if nat is not None:
            truth = truth & ~nat
        return funcname, group_idx, truth, fill_value, None, plan
    if base in ("allnan", "anynan"):
        # the missing-value flags read the NaT mask directly, and the flags
        # are the plain any/all of that mask
        return ("any" if base == "anynan" else "all"), group_idx, np.isnat(a), fill_value, None, plan
    if base == "argmax" and nat is not None and nat.any():
        # numpy answers the position of the first NaT of the group
        plan.nat_groups, plan.nat_positions = _first_true_position_per_group(group_idx, nat)
    elif base in TEMPORAL_NAT_POISON and nat is not None and nat.any():
        # a single NaT poisons the mean, the maximum and - for the durations
        # - the middles of its whole group
        plan.nat_groups = _groups_holding(group_idx, nat)
    ticks = np.asarray(a).view(np.int64)
    if base == "sort" and nat is not None and nat.any():
        # numpy sorts NaT to the end of its group although it compares as the
        # smallest value everywhere else - the largest tick stands in for it
        # on the way through the sort, and the plan turns it back into NaT
        ticks = np.where(nat, np.iinfo(np.int64).max, ticks)
        plan.sort_nat_restore = True
    if base in ("cumsum", "cummax", "cummin") and nat is not None and nat.any():
        # the accumulation carries the NaT through to the end of its group,
        # nancumsum included - numpy propagates it for temporal input
        plan.running_nat = np.maximum.accumulate(nat)
    if base == "mean":
        fill_tick = _temporal_fill_tick(fill_value, plan.result_dtype, True)
        out_user = np.dtype(dtype) if dtype is not None else None
        if out_user is not None and out_user.kind == "f":
            # a float answer is the exact quotient itself, not the rounded tick
            plan.computed = cast(
                "npt.NDArray[np.int64]",
                exact_integer_group_mean(group_idx, ticks, size, fill_tick, out_user),
            )
            plan.computed_final = True
        else:
            plan.computed = cast(
                "npt.NDArray[np.int64]",
                exact_integer_group_mean(group_idx, ticks, size, fill_tick, np.dtype(np.int64)),
            )
            plan.value_result = True
    elif base in ("median", "quantile", "percentile"):
        # only timedelta64 reaches this point - datetime64 has no middles
        fill_tick = _temporal_fill_tick(fill_value, plan.result_dtype, True)
        if base == "median":
            plan.computed = _temporal_group_median(group_idx, ticks, size, fill_tick)
        else:
            fraction = cast("float", kwargs.get("q", 50.0 if base == "percentile" else 0.5))
            if base == "percentile":
                fraction = fraction / 100.0
            plan.computed = _temporal_group_quantile(group_idx, ticks, size, fill_tick, fraction)
        plan.value_result = True
    elif base in TEMPORAL_VALUE_FUNCS:
        plan.value_result = True
    if base == "array":
        plan.children = True
        return funcname, group_idx, ticks, fill_value, np.dtype(dtype) if dtype is not None else None, plan
    if plan.value_result:
        # the kernels answer in ticks, the fill is a tick, and the dtype they
        # are asked for is int64 - the plan carries the dtype of the answer
        plan.user_dtype = np.dtype(dtype) if dtype is not None else None
        is_nat_fill = base in TEMPORAL_NAT_FILL
        return (
            funcname,
            group_idx,
            ticks,
            _temporal_fill_tick(fill_value, plan.result_dtype, is_nat_fill),
            np.dtype(np.int64),
            plan,
        )
    # positions, counts and flags: the kernels answer in their own dtypes
    return funcname, group_idx, ticks, fill_value, np.dtype(dtype) if dtype is not None else None, plan


def finalize_temporal(ret: AResult | list[npt.NDArray[np.int64]], plan: TemporalPlan) -> AResult:
    """Turn the tick result of the kernels back into a temporal answer."""
    if plan.computed is not None:
        ret = plan.computed
        if plan.computed_final:
            return cast("AResult", ret)
    if plan.running_nat is not None:
        ret = np.asarray(ret)
        ret[plan.running_nat] = NAT_TICK
    if plan.sort_nat_restore:
        # the largest tick stood in for NaT while sorting - hand it back
        ret = np.asarray(ret)
        ret[ret == np.iinfo(np.int64).max] = NAT_TICK
    if plan.nat_groups is not None:
        ret = np.asarray(ret)
        ret[plan.nat_groups] = plan.nat_positions if plan.nat_positions is not None else NAT_TICK
    if plan.children:
        # the values of every group, back in the dtype they came in
        if isinstance(ret, np.ndarray):
            return cast(
                "AResult",
                np.array([np.asarray(one).view(plan.result_dtype) for one in ret], dtype=object),
            )
        return cast("AResult", type(ret)(np.asarray(one).view(plan.result_dtype) for one in ret))
    if plan.value_result:
        ret = np.asarray(ret)
        if plan.user_dtype is not None:
            if plan.user_dtype == np.dtype(np.int64):
                return cast("AResult", ret)  # the ticks the caller asked for
            if plan.user_dtype.kind in TEMPORAL_KINDS:
                # rescale from the unit of the input to the one asked for
                return cast("AResult", ret.view(plan.result_dtype).astype(plan.user_dtype))
            return cast("AResult", ret.view(np.int64).astype(plan.user_dtype))
        return cast("AResult", ret.view(plan.result_dtype))
    return cast("AResult", ret)
