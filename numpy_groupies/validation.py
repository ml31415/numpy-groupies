"""The input checks of the package, in one place: shared messages and checks.

Checks needing numpy live in ``numpy_groupies``, those compiled live where
they are compiled - both use these messages.  This module must not import
numpy: the pure python implementation loads before numpy is known to exist.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, SupportsInt, cast

from numpy_groupies.functions import complex_real_result, fill_value_key, fixed_result_types

if TYPE_CHECKING:
    import numpy.typing as npt

    from numpy_groupies.typing import AFillValue, AScalar

# ---------------------------------------------------------------------------
# the words of the complaints - one spelling for every path of the package

# the labels of a call
EMPTY_GROUP_IDX = "group_idx must not be empty"
NON_INTEGER_GROUP_IDX = "group_idx must be of integer type"
NEGATIVE_INDICES = "negative indices not supported"
TOO_LARGE = "one or more indices are too large"
TOO_LARGE_FOR_SIZE = f"{TOO_LARGE} for size {{size}}"
# the same sentence for the compiled code, which can only concatenate and call ``str()``
TOO_LARGE_SIZE_PREFIX = f"{TOO_LARGE} for size "

# the values of a call
LENGTH_MISMATCH = "group_idx and a must be of the same length"
LENGTH_MISMATCH_OR_SCALAR = f"{LENGTH_MISMATCH}, or a can be scalar"

# the fill value of a call
FILL_NOT_BOOLEAN = "Value not boolean"
FILL_NOT_CONVERTIBLE = "fill_value must be convertible into {dtype}"
MOMENTS_NOT_FOR_COMPLEX = "the skewness and the kurtosis are not defined for complex values"
FILL_NOT_INTEGER = "fill_value must be convertible into an integer"
# a size that is no number of groups; only topk's flat rule refuses it otherwise
OUTPUT_SIZE_NOT_SCALAR = "output size must be scalar or None"
# topk answers a matrix with a row per group
TOPK_NEEDS_FLAT_GROUP_IDX = "the topk needs a flat group index, because its answer is a matrix with a row per group"
# the count of the values the topk keeps: a whole number, and none below zero
TOPK_K_NOT_WHOLE = "k must be a whole number of values, got {k!r}"
TOPK_K_NEGATIVE = "k must not be negative, got {k}"
FILL_NONE_NOT_HELD = (
    "fill_value=None fills an object result, which is what func='array' returns and no other function does "
    "(the default of a function is asked for with DEFAULT_FILL_VALUE)"
)

# the dtypes a call may have: the labels are integers, the values are numbers
INTEGER_KINDS = "iu"
NUMERIC_KINDS = "iufcb"

# the functions whose result holds a None: only the object array of ``array``
NONE_RESULT_FUNCTIONS = frozenset({"array"})

# bounds of exact reductions: integers beyond these magnitudes round silently
# in float64 (2**53 itself is still exact, hence the inclusive limit)
EXACT_FLOAT64_INTEGER_LIMIT = 2**53
# largest magnitude int64 deviations represent exactly (variance centers on
# each group's first value)
INT64_DEVIATION_LIMIT = 2**63 - 1
INT64_MIN = -(2**63)


# ---------------------------------------------------------------------------
# the predicates that go with the words
#
# every predicate answers its question for numpy dtypes, dtype names and python types


def _dtype_kind(dtype: Any) -> str | None:
    """The kind of a numpy dtype, or None for what is not one."""
    return getattr(dtype, "kind", None)


def is_integer_dtype(dtype: Any) -> bool:
    """Whether the labels are of an integer dtype (a bool is not: labels index groups)."""
    kind = _dtype_kind(dtype)
    if kind is not None:
        return kind in INTEGER_KINDS
    if isinstance(dtype, str):
        return dtype.startswith(("int", "uint"))
    return isinstance(dtype, type) and issubclass(dtype, int) and not issubclass(dtype, bool)


def is_numeric_dtype(dtype: Any) -> bool:
    """Whether the values are of a dtype the kernels can read (numba refuses
    datetime, text and structured dtypes - a harder error than this one).
    """
    kind = _dtype_kind(dtype)
    if kind is not None:
        return kind in NUMERIC_KINDS
    if isinstance(dtype, str):
        return dtype.startswith(("int", "uint", "float", "complex", "bool"))
    return isinstance(dtype, type) and issubclass(dtype, (int, float, complex))


def is_inexact_dtype(dtype: Any) -> bool:
    """True for everything that can hold nan - numpy dtypes by kind, python types by identity."""
    kind = _dtype_kind(dtype)
    if kind is not None:  # a numpy dtype
        return kind in "fc"
    if isinstance(dtype, str):
        return dtype.startswith(("float", "complex"))
    return isinstance(dtype, type) and issubclass(dtype, (float, complex))


def is_complex_dtype(dtype: Any) -> bool:
    """True for complex numpy dtypes, the complex builtin and complex dtype strings."""
    kind = _dtype_kind(dtype)
    if kind is not None:  # a numpy dtype
        return kind == "c"
    if isinstance(dtype, str):
        return dtype.startswith("complex")
    # NumPy's complex64 and clongdouble do not inherit from Python's complex.
    # Inspect the public scalar base without importing the optional dependency.
    return isinstance(dtype, type) and any(
        base is complex or (base.__module__ == "numpy" and base.__name__ == "complexfloating") for base in dtype.__mro__
    )


def is_scalar_input(a: Any) -> bool:
    """Whether the aggregated data is a single value instead of a sequence.

    NumPy free on purpose; ``isinstance(a, (int, float, complex))`` alone
    would miss non-python numpy scalars like ``np.float32``.
    """
    return getattr(a, "ndim", None) == 0 or isinstance(a, (int, float, complex))


# ---------------------------------------------------------------------------
# the checks of the fill value of a call
#
# A fill value is the one argument every function takes; these checks say what
# a result can hold, without knowing the result itself.


def check_boolean(x: AFillValue) -> None:
    """Refuse a fill value that is not one of the two booleans."""
    if x not in (0, 1):
        raise ValueError(FILL_NOT_BOOLEAN)


def check_filling_with_none(key: str) -> None:
    """Refuse ``None`` as fill where no result can hold it (``DEFAULT_FILL_VALUE``
    asks for a function's default; ``None`` is only the fill of ``func='array'``).

    ``key`` is the caller-resolved name in the fill value table.
    """
    if key not in NONE_RESULT_FUNCTIONS:
        raise TypeError(FILL_NONE_NOT_HELD)


def check_fill_value_for_type(fill_value: AFillValue, target: type) -> None:
    """Refuse a ``fill_value`` the result type cannot hold, told without numpy:
    booleans fill with 0 or 1, counts and positions with anything int()-able.
    """
    if target is bool:
        check_boolean(fill_value)
    elif target is int:
        try:
            int(cast("SupportsInt", fill_value))
        except ValueError:
            raise ValueError(FILL_NOT_INTEGER)


def check_fill_value_for_dtype(fill_value: AFillValue, dtype: npt.DTypeLike) -> AScalar:
    """The fill value as the output dtype holds it - a kernel's check, without
    the function-specific part (a nan does not fit an integer output).
    """
    # numpy imported here, not at the top: the module must be importable
    # without numpy, but a dtype is one of numpy's
    import numpy as np

    out_dtype: np.dtype = np.dtype(dtype)
    if issubclass(out_dtype.type, np.bool_) and not isinstance(fill_value, (list, tuple, np.ndarray)):
        # a sequence fill (the sorting function) is no boolean and need not be one
        check_boolean(fill_value)
        return np.bool_(fill_value)
    try:
        return cast("AScalar", out_dtype.type(fill_value))
    except (ValueError, TypeError, OverflowError) as err:
        raise ValueError(FILL_NOT_CONVERTIBLE.format(dtype=out_dtype.type.__name__)) from err


def check_fill_value(
    fill_value: AFillValue, dtype: npt.DTypeLike, func: str | Callable[..., AScalar] | None = None
) -> AFillValue | None:
    if func in ("all", "any", "allnan", "anynan"):
        check_boolean(fill_value)
        return None
    return check_fill_value_for_dtype(fill_value, dtype)


def check_fixed_fill_value(func: str | Callable[..., AScalar], fill_value: AFillValue) -> None:
    """Refuse a ``fill_value`` the fixed result type of ``func`` cannot hold."""
    target = fixed_result_types.get(fill_value_key(func))
    if target is not None:
        check_fill_value_for_type(fill_value, target)


# ---------------------------------------------------------------------------
# the checks of the exactness of the reductions
#
# The mean and variance of integers can be exact; these checks refuse the
# inputs where float64 or int64 would round silently.


def mean_overflow_message(count: int, bound: int) -> str:
    """The complaint about an integer mean float64 cannot compute exactly,
    built by concatenation so numba can build it too.
    """
    return (
        "cannot compute an exact mean of "
        + str(count)
        + " integer values with magnitude up to "
        + str(bound)
        + ": their sums can exceed +/-"
        + str(EXACT_FLOAT64_INTEGER_LIMIT)
        + ", the range float64 represents exactly - cast the input to a float dtype to accept "
        + "rounding, or use sum and len"
    )


def variance_span_overflow_message(low_text: str, high: int) -> str:
    """The complaint about an integer variance int64 cannot center exactly.

    ``low_text`` comes as text: numba's ``str()`` of the smallest int64 wraps
    and fails to size the string buffer.
    """
    return (
        "cannot compute an exact variance: the integer values span ["
        + low_text
        + ", "
        + str(high)
        + "], wider than +/-"
        + str(INT64_DEVIATION_LIMIT)
        + ", so centering each group on its first value would overflow the signed 64-bit "
        + "deviation - cast the input to a float dtype first"
    )


def check_integer_mean_overflow(count: int, bound: int) -> None:
    """Refuse integer moments float64 cannot compute exactly (no exact integer
    fallback exists; the mean is computed exactly instead).
    """
    if count * bound > EXACT_FLOAT64_INTEGER_LIMIT:
        raise OverflowError(mean_overflow_message(count, bound))


def check_integer_variance_span_overflow(low: int, high: int) -> None:
    """Refuse integer variances whose exact centering would overflow int64."""
    if high - low > INT64_DEVIATION_LIMIT:
        raise OverflowError(variance_span_overflow_message(str(low), high))


# ---------------------------------------------------------------------------
# the checks of the requested dtype of a call


def check_complex_dtype(
    a_dtype: npt.DTypeLike, dtype: npt.DTypeLike | None, func_str: str | Callable[..., AScalar]
) -> None:
    """Complain when a non-complex ``dtype`` is requested for complex input.

    The pure python backend needs this itself, since python numbers carry
    no dtype and would silently ignore the request.
    """
    if (
        dtype is not None
        and is_complex_dtype(a_dtype)
        and not is_complex_dtype(dtype)
        and func_str not in complex_real_result
    ):
        raise TypeError(
            f"function {func_str} cannot aggregate complex values into the non-complex dtype "
            f"{dtype} - pass a complex dtype to keep the imaginary part"
        )


# ---------------------------------------------------------------------------
# the rules of the topk
#
# The topk answers a row per group and needs a flat index; both backends run
# their own entry points, so the rules live in one place here.


def check_topk_input(ndim_idx: int) -> None:
    """Refuse a group index of more than one dimension: topk answers a matrix
    with a row per group, leaving no place for such an index.
    """
    if ndim_idx > 1:
        raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX)


def check_topk_k(k: Any) -> int:
    """``k`` must be a whole, non-negative count of values, asked for itself so
    all backends say the same thing before anything is computed.
    """
    if isinstance(k, bool):
        raise TypeError(TOPK_K_NOT_WHOLE.format(k=k))
    try:
        whole = operator.index(k)
    except TypeError:
        raise TypeError(TOPK_K_NOT_WHOLE.format(k=k)) from None
    if whole < 0:
        raise ValueError(TOPK_K_NEGATIVE.format(k=k))
    return whole


def check_topk_shape(ret: Any, size: int, k: int) -> None:
    """The shape the answer of the topk has: a row per group, k columns."""
    if ret.shape != (size, k):
        raise ValueError(f"the topk of {size} groups with k={k} has the shape {ret.shape}")
