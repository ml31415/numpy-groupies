"""The dtypes: sum ladders, output dtype rules, result fill values, temporal facts.

The result dtype of a group depends on the input dtype, the function and the
requested dtype; the ladders are the steps integer sums climb.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.functions import forced_float_types, forced_real_float_types
from numpy_groupies.typing import AFillValue, AScalar, AValue
from numpy_groupies.validation import check_complex_dtype, is_scalar_input

_next_int_dtype = {
    # the sign of the input stays (numpy sums into int64/uint64 likewise); no
    # 16-bit rungs
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
    """Whether the input carries datetime64 or timedelta64 values.

    Asks the dtype kind, not the dtype itself: this runs on every call.
    """
    kind = getattr(getattr(a, "dtype", None), "kind", None)
    return kind is not None and kind in TEMPORAL_KINDS


def minimum_dtype(x: AFillValue, dtype: npt.DTypeLike | None = np.bool_) -> np.dtype:
    """Return the "most basic" dtype representing ``x`` properly, providing at
    least the value range of ``dtype``.
    """

    def check_type(x: AFillValue, dtype: np.dtype) -> bool:
        try:
            with np.errstate(invalid="ignore"):
                converted = np.array(x).astype(dtype)
                # a numpy scalar compared to a python integer that does not fit
                # raises (numpy 2); such a value does not fit the dtype either
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
    # last rung of the ladder: the value fits no integer type - float64, and
    # never float32 (24-bit mantissa is too coarse); numpy itself wraps here
    return type_loop(x, dtype, _next_int_dtype, default=np.float64)


def minimum_dtype_scalar(x: AFillValue, dtype: npt.DTypeLike | None, a: AValue) -> np.dtype:
    if dtype is None:
        dtype = np.dtype(type(a)) if isinstance(a, (int, float, complex)) else a.dtype
    return minimum_dtype(x, dtype)


def scalar_dtype(a: Any) -> np.dtype:
    """The dtype of scalar input data: a python number stands in for its type
    (``np.dtype(np.ndarray)`` would be object for the 0d-array case).
    """
    dtype = getattr(a, "dtype", None)
    return np.dtype(type(a)) if dtype is None else np.dtype(dtype)


# a count and a position are as wide as a pointer (np.intp), the width the
# index of pandas and of numpy's own arg reductions carry: on a 32 bit
# platform that is int32, and the answer must not be wider there
_count_dtype = np.intp

_forced_types = {
    "array": object,
    "all": bool,
    "any": bool,
    "nanall": bool,
    "nanany": bool,
    "len": _count_dtype,
    "nunique": _count_dtype,
    "nanlen": _count_dtype,
    "nannunique": _count_dtype,
    "allnan": bool,
    "anynan": bool,
    "argmax": _count_dtype,
    "argmin": _count_dtype,
    "nanargmin": _count_dtype,
    "nanargmax": _count_dtype,
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
    # the topk, mode, sort and running extrema hand out the values of the
    # group themselves: their dtype is the input dtype, not a widened one
    "topk",
    "mode",
    "sort",
    "cummin",
    "cummax",
    "nancummin",
    "nancummax",
}


def check_dtype(
    dtype: npt.DTypeLike | None,
    func_str: str | Callable[..., AScalar],
    a: Any,  # an array of any dtype, or the one python scalar standing for it
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
    """The dtype resolution of ``check_dtype`` given the input dtype directly
    (``n`` is the number of values, needed by the ``sum`` overflow guesses).
    """
    if dtype is not None:
        # dtype set by the user (careful: np.bool != np.bool_)
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
        # a caller callable names no function and misses the lookup
        return np.dtype(_forced_types[func_str if isinstance(func_str, str) else ""])
    except KeyError:
        pass

    if func_str in forced_float_types:
        if np.issubdtype(a_dtype, np.inexact):
            # floating input keeps its dtype, complex stays complex
            return a_dtype
        return np.dtype(np.float64)

    if func_str in forced_real_float_types:
        if np.issubdtype(a_dtype, np.floating):
            return a_dtype
        if np.issubdtype(a_dtype, np.complexfloating):
            # the real counterpart of the complex dtype (complex64 -> float32)
            return np.dtype(a_dtype.char.lower())
        if func_str in ("sumofsquares", "nansumofsquares"):
            # exact squares of integers; unsigned input keeps unsigned so its
            # squares stay unsigned (uint32 squares leave int64)
            return np.dtype(np.uint64) if a_dtype.kind == "u" else np.dtype(np.int64)
        return np.dtype(np.float64)

    if func_str in ("sum", "nansum", "cumsum", "nancumsum"):
        # running sums climb the same ladder (linear growth, count known)
        if a_dtype.kind in TEMPORAL_KINDS:
            # the sum of durations is a duration, in the input's unit
            return a_dtype
        # Try to guess the minimally required int size
        if a_dtype in (np.dtype(np.int64), np.dtype(np.uint64)):
            # already the widest (TODO: strictly speaking it might need float)
            return a_dtype
        if np.issubdtype(a_dtype, np.integer):
            # summing path arrives with a flat count, one-out-per-in never
            # reaches this branch
            maxval = np.iinfo(a_dtype).max * cast("int", n)
            return minimum_dtype(maxval, a_dtype)
        if np.issubdtype(a_dtype, np.bool_):
            return minimum_dtype(n, a_dtype)
        # floating, inexact, whatever
        return a_dtype

    if func_str in ("prod", "nanprod", "cumprod", "nancumprod"):
        # no exact rung exists (products grow exponentially); the wide
        # accumulator of the input's own sign is the floor (numpy widens
        # likewise), and a duration keeps its unit
        if a_dtype.kind in TEMPORAL_KINDS:
            return a_dtype
        if a_dtype.kind in "iub":
            return np.promote_types(a_dtype, np.uint if a_dtype.kind == "u" else np.int_)
        return a_dtype

    if func_str in _forced_same_type:
        return a_dtype

    # temporal values answer temporally - the wide integer below would take
    # the unit away
    if a_dtype.kind in TEMPORAL_KINDS:
        return a_dtype

    # any other integer answer widens to the accumulator of the input's own
    # sign: the wide accumulator is the floor (int64 is the widest signed,
    # and could not hold uint64's half of the range)
    if a_dtype != np.dtype(np.uint64) and np.issubdtype(a_dtype, np.integer):
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
    """The smallest value the dtype of the answer holds (widened to hold the
    fill value); the seed of a reduction and the stand-in of a nan.
    """
    dtype = minimum_dtype(fill_value, dtype)
    if issubclass(dtype.type, np.bool_):
        # a bool has no nan, but the mask value must be one of the dtype
        return cast("AScalar", np.False_)
    if issubclass(dtype.type, np.floating):
        return -np.inf
    if issubclass(dtype.type, np.complexfloating):
        # numpy orders complex values lexicographically, so both parts have
        # to sit at the same end of the order
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
        # see minval: both parts of the seed at the same end of the order
        return cast("AScalar", complex(np.inf, np.inf))
    if issubclass(dtype.type, np.integer):
        return np.iinfo(dtype).max
    return cast("AScalar", np.finfo(dtype).max)
