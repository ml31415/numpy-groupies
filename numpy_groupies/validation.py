"""The input checks of the package, in one place.

Every path into a computation checks its input and says the same words about
what is wrong with it: the python-level ``aggregate()``, the pure python one
and the compiled kernels of the numba backend share the messages below and the
checks that go with them.  The checks that need numpy live in ``numpy_groupies``,
the ones that have to run inside compiled code are written where they are
compiled - but they speak the words of this module.

This module must not import numpy: the pure python implementation is loaded
before numpy is known to be available, and has to work without it.
"""

from __future__ import annotations

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
# the same sentence for the compiled code, which cannot format a string:
# numba knows no ``str.format``, but it concatenates and it knows ``str()``
TOO_LARGE_SIZE_PREFIX = f"{TOO_LARGE} for size "

# the values of a call
LENGTH_MISMATCH = "group_idx and a must be of the same length"
LENGTH_MISMATCH_OR_SCALAR = f"{LENGTH_MISMATCH}, or a can be scalar"

# the fill value of a call
FILL_NOT_BOOLEAN = "Value not boolean"
FILL_NOT_CONVERTIBLE = "fill_value must be convertible into {dtype}"
MOMENTS_NOT_FOR_COMPLEX = "the skewness and the kurtosis are not defined for complex values"
FILL_NOT_INTEGER = "fill_value must be convertible into an integer"
# the topk answers a matrix with a row per group, and a group index of more
# than one dimension has no place in it
TOPK_NEEDS_FLAT_GROUP_IDX = "the topk needs a flat group index and a size that is a number"
FILL_NONE_NOT_HELD = (
    "fill_value=None fills an object result, which is what func='array' returns and no other function does "
    "(the default of a function is asked for with DEFAULT_FILL_VALUE)"
)

# the dtypes a call may have: the labels are integers, the values are numbers
INTEGER_KINDS = "iu"
NUMERIC_KINDS = "iufcb"

# the functions whose result holds a None: the object array of ``array`` -
# every other result is a number, a flag, a count, a position or the values of
# a group, and none of them holds a None
NONE_RESULT_FUNCTIONS = frozenset({"array"})

# the bounds of the exactness of the reductions
#
# integers beyond this magnitude are not exactly representable in float64, so
# once an exact integer sum can leave this window, a float64-accumulated mean
# silently rounds (2**53 itself is still exact, hence the inclusive limit)
EXACT_FLOAT64_INTEGER_LIMIT = 2**53
# largest magnitude a two's-complement signed 64-bit deviation represents
# exactly; integer variance centers each group on its first value and needs
# every deviation to stay within this bound
INT64_DEVIATION_LIMIT = 2**63 - 1
INT64_MIN = -(2**63)


# ---------------------------------------------------------------------------
# the predicates that go with the words
#
# Every predicate of this module answers its question about a dtype in any of
# its spellings: a numpy dtype (the kind of it is a one letter string), the
# name of one or the python type behind it.


def _dtype_kind(dtype: Any) -> str | None:
    """The kind of a numpy dtype, or None for what is not one."""
    return getattr(dtype, "kind", None)


def is_integer_dtype(dtype: Any) -> bool:
    """Whether the labels of a call are of an integer dtype.

    A bool is not an integer dtype here: the labels of a call index the
    groups, and a bool is not an index.
    """
    kind = _dtype_kind(dtype)
    if kind is not None:
        return kind in INTEGER_KINDS
    if isinstance(dtype, str):
        return dtype.startswith(("int", "uint"))
    return isinstance(dtype, type) and issubclass(dtype, int) and not issubclass(dtype, bool)


def is_numeric_dtype(dtype: Any) -> bool:
    """Whether the values of a call are of a dtype the kernels can read.

    The numeric ones: the complex drivers of the reductions, the median and
    the quantile included - datetime, timedelta, text and structured dtypes
    make numba refuse the call, which is a harder error to read than this
    one.
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

    Kept numpy free on purpose, since the pure python implementation uses it
    too: a python number has no ``ndim`` at all, while a numpy scalar and a 0d
    array both report 0 - ``isinstance(a, (int, float, complex))`` alone misses
    every numpy scalar that is not a python number (``np.float32``, ``np.int64``,
    ``np.bool_``, ``np.complex64``).
    """
    return getattr(a, "ndim", None) == 0 or isinstance(a, (int, float, complex))


# ---------------------------------------------------------------------------
# the checks of the fill value of a call
#
# A fill value is the one argument every function takes and every function
# answers differently: the checks below say what the result of a function can
# hold, without knowing the result itself.


def check_boolean(x: AFillValue) -> None:
    """Refuse a fill value that is not one of the two booleans."""
    if x not in (0, 1):
        raise ValueError(FILL_NOT_BOOLEAN)


def check_filling_with_none(key: str) -> None:
    """Refuse to fill with ``None`` where no result can hold it.

    ``None`` is a value and not a request: the request for the default of a
    function is made with ``DEFAULT_FILL_VALUE``, and ``None`` is what an object
    result - the sequence of ``func='array'`` - is filled with, nothing else.  A
    number, a flag, a count, a position and the values a sort returns cannot hold
    a ``None``, and the three implementations answered the request three
    different ways: the numpy one complained about an ``isnan`` of an object, the
    numba one failed to type a kernel, and the pure python one wrote the ``None``
    into a column of floats, which turned the result into an object array.

    ``key`` is the name of the function in the table of the fill values -
    the caller resolves it, this module knows nothing of the table.
    """
    if key not in NONE_RESULT_FUNCTIONS:
        raise TypeError(FILL_NONE_NOT_HELD)


def check_fill_value_for_type(fill_value: AFillValue, target: type) -> None:
    """Refuse a ``fill_value`` that the result type of a function cannot hold.

    This is the rule ``numpy_groupies.check_fill_value`` tells against a numpy
    dtype, told without one: a boolean answer is filled with 0 or 1 - a nan
    would pass for ``bool``, which is why the two values are asked for
    themselves - and a count or a position with whatever turns into an integer,
    which a nan does not.
    """
    if target is bool:
        check_boolean(fill_value)
    elif target is int:
        try:
            int(cast("SupportsInt", fill_value))
        except ValueError:
            raise ValueError(FILL_NOT_INTEGER)


def check_fill_value_for_dtype(fill_value: AFillValue, dtype: npt.DTypeLike) -> AScalar:
    """The fill value as the output dtype holds it - the check of a kernel.

    The rules are the ones ``check_fill_value`` applies to the python-level
    call, without the function-specific part: a kernel knows its output dtype
    and nothing else, so the value has to fit that dtype - a nan does not fit
    an integer output, and a boolean output takes the two booleans only.
    """
    # numpy is imported here and not at the top of the module: the module has
    # to be importable without numpy - the pure python implementation is
    # loaded before numpy is known to be available - but the dtype of the
    # answer is one of numpy, so numpy is there when this is called
    import numpy as np

    out_dtype: np.dtype = np.dtype(dtype)
    if issubclass(out_dtype.type, np.bool_) and not isinstance(fill_value, (list, tuple, np.ndarray)):
        # a sequence is the fill of a group that holds a sequence (the sorting
        # function) - it is not a boolean and does not have to be one
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
    """Refuse a ``fill_value`` that the result type of ``func`` cannot hold, for
    those functions whose result type the function itself settles.

    This is the rule ``check_fill_value`` tells against a numpy dtype, told
    without one: a boolean answer is filled with 0 or 1 - a nan would pass for
    ``bool``, which is why the two values are asked for themselves - and a count
    or a position with whatever turns into an integer, which a nan does not.
    """
    target = fixed_result_types.get(fill_value_key(func))
    if target is not None:
        check_fill_value_for_type(fill_value, target)


# ---------------------------------------------------------------------------
# the checks of the exactness of the reductions
#
# The mean and the variance of integers can be exact, and the ways to compute
# them are not exact as well: the checks below refuse the inputs where a
# float64 or an int64 would round silently.


def mean_overflow_message(count: int, bound: int) -> str:
    """The complaint about an integer mean float64 cannot compute exactly.

    Numba cannot call a python function, so the compiled twin of the check
    below needs the text of the complaint as something it can build - and
    this is it: a concatenation of strings and numbers, which numba knows.
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

    The lower bound comes as a text: numba's ``str()`` of the smallest
    int64 wraps and fails to size the string buffer, so the compiled twin
    spells that one value out.
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
    """Refuse integer moments float64 cannot compute exactly.

    The sums behind the third and the fourth central moment of ``count``
    integers of magnitude up to ``bound`` are exact in float64 only while
    they stay inside the window float64 represents exactly - the moments
    have no exact integer fallback, so the inputs are refused.  The mean is
    not among them any more: it is computed exactly instead.
    """
    if count * bound > EXACT_FLOAT64_INTEGER_LIMIT:
        raise OverflowError(mean_overflow_message(count, bound))


def check_integer_variance_span_overflow(low: int, high: int) -> None:
    """Refuse integer variances whose exact centering would overflow int64.

    The deviations of the values of a group from its first value have to stay
    inside the signed 64-bit range for the variance to be exact.
    """
    if high - low > INT64_DEVIATION_LIMIT:
        raise OverflowError(variance_span_overflow_message(str(low), high))


# ---------------------------------------------------------------------------
# the checks of the requested dtype of a call


def check_complex_dtype(
    a_dtype: npt.DTypeLike, dtype: npt.DTypeLike | None, func_str: str | Callable[..., AScalar]
) -> None:
    """Complain when a non-complex ``dtype`` is requested for complex input.

    ``resolve_output_dtype`` applies this for the backends that honour ``dtype``;
    the pure python implementation needs it on its own, since python numbers
    carry no dtype and it would otherwise silently ignore the request.
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
# The topk is the only function of the package that answers a row per group
# instead of one value per group, and the only one that has no place for a
# group index of more than one dimension.  The two backends run their own
# entry points and never each other's, so the rules live here - one place
# for both of them, and one place to change them in.


def check_topk_input(ndim_idx: int, size: Any) -> None:
    """The input the topk cannot take: a group index of more than one
    dimension, and a size that is a sequence.

    A group index of more than one dimension names its groups in more than
    one way, and the answer of the topk is a matrix with a row per group -
    the unravelling of such an index has no place in it.
    """
    if ndim_idx > 1 or isinstance(size, tuple):
        raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX)


def check_topk_shape(ret: Any, size: int, k: int) -> None:
    """The shape the answer of the topk has: a row per group, k columns."""
    if ret.shape != (size, k):
        raise ValueError(f"the topk of {size} groups with k={k} has the shape {ret.shape}")
