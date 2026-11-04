"""Numpy specific functionality shared by the numpy-based aggregate implementations."""

import platform
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.registry import (
    forced_float_types,
    forced_real_float_types,
    get_aliasing,
    resolve_name,
)
from numpy_groupies.typing import (
    AFillValue,
    AScalar,
    AValue,
    RavelOrder,
    ValidatedInput,
)
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    EXACT_FLOAT64_INTEGER_LIMIT,
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
    # the sign of the input stays: a sum of unsigned values does not become
    # negative, and one of signed values may - numpy keeps it as well (it
    # sums into int64 or uint64), and the 16 bit types are no steps of the
    # ladder
    "bool": np.int8,
    "uint8": np.uint32,
    "int8": np.int32,
    "uint16": np.uint32,
    "int16": np.int32,
    "uint32": np.uint64,
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
                # the comparison of a numpy scalar with a python integer that
                # does not fit into it raises (numpy 2) - such a value does
                # not fit into the dtype either, and that is what we say
                return bool(converted == x) or (np.ndim(cast(np.generic, x)) == 0 and np.isnan(cast(np.generic, x)))
        except (ValueError, OverflowError):
            return False

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
        # the last step of the ladder: the value does not fit into any of the
        # integer types - float64, and never float32, whose mantissa of 24
        # bits is too coarse for the sums it would carry (numpy has no float
        # here at all: it sums into int64 or uint64 and wraps)
        return type_loop(x, dtype, _next_int_dtype, default=np.float64)


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


# the ways a quantile interpolates between the two values it falls between -
# the names numpy uses for them.  The backends share the list: the numpy one
# computes the quantile of the numbers and of the ticks with it, the numba one
# the quantile of the ticks.
QUANTILE_METHODS = frozenset({"linear", "lower", "higher", "midpoint", "nearest"})


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
                    if a_dtype.kind in "Mm":  # the kinds temporal holds as TEMPORAL_KINDS
                        # the sum of durations is a duration - numpy answers
                        # in the unit of the input, and there is no integer
                        # size to guess for
                        return a_dtype
                    # Try to guess the minimally required int size
                    if a_dtype in (np.dtype(np.int64), np.dtype(np.uint64)):
                        # It's not getting bigger anymore
                        # TODO: strictly speaking it might need float
                        return a_dtype
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
    refusing the mean, the values are split into limbs, every limb
    accumulates in an exact sum through ``bincount``, and the totals of the
    groups are combined as python integers.  The division is exact and the
    remainder rounds half to even.

    The limbs are as wide as they can be: a 32-bit limb holds its sum
    inside the window float64 represents exactly as long as there are fewer
    than 2**21 values, and a 16-bit limb serves more values than that.  The
    wider the limb, the fewer of them there are - and every limb is one pass
    over the data.

    An integer mean of small magnitudes keeps coming back as float64 - the
    quotient is exact in float64 there, so its answer does not change - and a
    caller asking for an integer dtype gets the exact rounded quotient.
    """
    u = np.ascontiguousarray(a).view(np.uint64)
    # the width of the limbs: the sum of a limb has to stay inside the
    # window float64 represents exactly
    width = 32 if len(group_idx) * (2**32 - 1) <= EXACT_FLOAT64_INTEGER_LIMIT else 16
    mask = np.uint64((1 << width) - 1)
    shifts = tuple(range(0, 64, width))
    limb_sums = [
        np.bincount(
            group_idx,
            # the cast to float64 is explicit: numpy would cast the weights
            # itself, and the cast of the masked limbs - small numbers - is
            # far quicker than the cast of the whole bit patterns
            weights=((u >> np.uint64(shift)) & mask).astype(np.float64),
            minlength=size,
        ).astype(np.int64)
        for shift in shifts
    ]
    counts = np.bincount(group_idx, minlength=size)
    total: npt.NDArray[Any] = limb_sums[-1].astype(object)
    for limb in reversed(limb_sums[:-1]):
        total = (total << width) + limb.astype(object)
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


def fill_untouched(idx: npt.NDArray[np.integer], ret: npt.NDArray[np.generic], fill_value: AScalar) -> None:
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value


def blocks_to_object_array(
    blocks: Iterable[npt.NDArray[np.generic]],
    group_idx: npt.NDArray[np.integer],
    fill_value: AScalar | Sequence[AScalar] | None,
) -> npt.NDArray[np.object_]:
    """The blocks of the groups as an object array, one sequence per group.

    A group without values takes the fill value - an empty sequence leaves the
    empty block of the group alone.  The array is built element by element:
    ``np.asanyarray`` of a list of blocks collapses it into a regular array
    whenever the groups happen to be of the same length - and the result would
    lose the sequences it is meant to hold (and the size the caller asked
    for).

    Both backends collect their groups into such an array, and the function
    lives here so that neither of them has to import the other.
    """
    if fill_value is not None and not (
        np.isscalar(fill_value) or (isinstance(fill_value, Sequence) and len(fill_value) == 0)
    ):
        raise ValueError("fill_value must be None, a scalar or an empty sequence")
    blocks = list(blocks)
    ret = np.empty(len(blocks), dtype=object)
    for g, block in enumerate(blocks):
        ret[g] = block
    if fill_value is None or np.isscalar(fill_value):
        fill_untouched(group_idx, ret, cast("AScalar", fill_value))
    return ret
