"""The input checks of the package, in one place.

Every path into a computation checks its input and says the same words about
what is wrong with it: the python-level ``aggregate()`` and the compiled
kernels of the numba backend share the messages below and the predicates
that go with them.  The checks that need numpy live in ``utils_np``, the ones
that have to run inside compiled code are written where they are compiled -
but they speak the words of this module.

This module must not import numpy: the pure python implementation is loaded
before numpy is known to be available.
"""

from typing import Any

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

# the dtypes a call may have: the labels are integers, the values are numbers
INTEGER_KINDS = "iu"
NUMERIC_KINDS = "iufcb"


# ---------------------------------------------------------------------------
# the predicates that go with the words


def is_integer_dtype(dtype: Any) -> bool:
    """Whether the labels of a call are of an integer dtype.

    The argument is a numpy dtype: the one of the array of a python call, and
    the one the numba type of an array carries (``as_dtype`` of it).  Both
    worlds ask the same question and answer it the same way.
    """
    kind = getattr(dtype, "kind", None)
    return isinstance(kind, str) and kind in INTEGER_KINDS


def is_numeric_dtype(dtype: Any) -> bool:
    """Whether the values of a call are of a dtype the kernels can read.

    The numeric ones: the complex drivers of the reductions, the median and
    the quantile included - datetime, timedelta, text and structured dtypes
    make numba refuse the call, which is a harder error to read than this
    one.
    """
    kind = getattr(dtype, "kind", None)
    return isinstance(kind, str) and kind in NUMERIC_KINDS
