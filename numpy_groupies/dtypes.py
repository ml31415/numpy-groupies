"""The dtypes: the ladders of the sums, the rules of the output, the fill
values of the answer and the facts of the temporal ones.

The dtype of the answer of a group is a rule of the package, and the rule
depends on the dtype of the input, on the function and on the dtype the
caller asked for.  The ladders are the steps the sums of the integers walk,
the forced types are the dtypes of the functions whose answer has a dtype of
its own, and the kinds of the temporal dtypes are the facts the backends ask
before they work on the ticks.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.functions import forced_float_types, forced_real_float_types
from numpy_groupies.typing import AFillValue, AScalar, AValue
from numpy_groupies.validation import check_complex_dtype, is_scalar_input

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

TEMPORAL_KINDS = "Mm"


def is_temporal_input(a: Any) -> bool:
    """Whether the input carries datetime64 or timedelta64 values - an array
    of them, or the one scalar standing for a value of every group.

    The kind of the dtype is asked, not the dtype itself: the check runs on
    every call, and building a dtype out of a dtype is not free.
    """
    kind = getattr(getattr(a, "dtype", None), "kind", None)
    return kind is not None and kind in TEMPORAL_KINDS


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
    # the last step of the ladder: the value does not fit into any of the
    # integer types - float64, and never float32, whose mantissa of 24
    # bits is too coarse for the sums it would carry (numpy has no float
    # here at all: it sums into int64 or uint64 and wraps)
    return type_loop(x, dtype, _next_int_dtype, default=np.float64)


def minimum_dtype_scalar(x: AFillValue, dtype: npt.DTypeLike | None, a: AValue) -> np.dtype:
    if dtype is None:
        dtype = np.dtype(type(a)) if isinstance(a, (int, float, complex)) else a.dtype
    return minimum_dtype(x, dtype)


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

_forced_same_type = {
    "min",
    "max",
    "first",
    "last",
    "nanmin",
    "nanmax",
    "nanfirst",
    "nanlast",
    # the topk hands out the values of the group themselves, so their dtype
    # is the dtype of the result
    "topk",
}


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
    if dtype is not None:
        # dtype set by the user
        # Careful here: np.bool != np.bool_ !
        if np.issubdtype(dtype, np.bool_) and not (
            isinstance(func_str, str) and ("all" in func_str or "any" in func_str)
        ):
            raise TypeError(f"function {func_str} requires a more complex datatype than bool")
        if (
            not np.issubdtype(dtype, np.integer)
            and isinstance(func_str, str)
            and func_str in ("len", "nanlen", "nunique", "nannunique")
        ):
            raise TypeError(f"function {func_str} requires an integer datatype")
        check_complex_dtype(a_dtype, dtype, func_str)
        # TODO: Maybe have some more checks here
        return np.dtype(dtype)

    try:
        # a callable of the caller's own names no function - the lookup
        # misses, like a name the table does not hold
        return np.dtype(_forced_types[func_str if isinstance(func_str, str) else ""])
    except KeyError:
        pass

    if func_str in forced_float_types:
        if np.issubdtype(a_dtype, np.inexact):
            # floating input keeps its dtype, complex input stays complex
            return a_dtype
        return np.dtype(np.float64)

    if func_str in forced_real_float_types:
        if np.issubdtype(a_dtype, np.floating):
            return a_dtype
        if np.issubdtype(a_dtype, np.complexfloating):
            # the real counterpart of the complex dtype (complex64 -> float32)
            return np.dtype(a_dtype.char.lower())
        if func_str in ("sumofsquares", "nansumofsquares"):
            # exact squares of integers
            return np.dtype(np.int64)
        return np.dtype(np.float64)

    if func_str == "sum":
        if a_dtype.kind in TEMPORAL_KINDS:
            # the sum of durations is a duration - numpy answers
            # in the unit of the input, and there is no integer
            # size to guess for
            return a_dtype
        # Try to guess the minimally required int size
        if a_dtype in (np.dtype(np.int64), np.dtype(np.uint64)):
            # It's not getting bigger anymore
            # TODO: strictly speaking it might need float
            return a_dtype
        if np.issubdtype(a_dtype, np.integer):
            # the summing path arrives with a flat count, the one-out
            # per-in path never reaches this branch
            maxval = np.iinfo(a_dtype).max * cast("int", n)
            return minimum_dtype(maxval, a_dtype)
        if np.issubdtype(a_dtype, np.bool_):
            return minimum_dtype(n, a_dtype)
        # floating, inexact, whatever
        return a_dtype

    if func_str in _forced_same_type:
        return a_dtype

    if isinstance(a_dtype, np.integer):
        return np.dtype(np.int64)
    return a_dtype


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


def minval(fill_value: AFillValue, dtype: np.dtype | None) -> AScalar:
    """The smallest value the dtype of the answer holds.

    The dtype is widened to hold the fill value the caller asked for as well,
    and the smallest value of it is the seed of a reduction and the stand-in
    of a nan - a nan of a float is not the smallest value of it.
    """
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
    """The largest value the dtype of the answer holds - see ``minval``."""
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
