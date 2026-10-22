"""The numba implementation of aggregate: compiled per-dtype kernels behind
one dispatch, with ``out=`` support and caching for custom callables.

``aggregate`` is the entry point of the implementation, and the module holds
the kernels themselves beside it: one function per function of the package,
under the name of that function.  A kernel is the compiled driver of its
operation and nothing else, so that jitted code can call it::

    import numba as nb
    from numpy_groupies import aggregate_numba as aggregate

    @nb.njit
    def zscore(group_idx, values):
        m = aggregate.mean(group_idx, values)
        s = aggregate.std(group_idx, values)
        return (values - m[group_idx]) / s[group_idx]

and - with the same arguments and the same result - from python::

    aggregate.mean(group_idx, values)

Both ways reach the same compiled driver: the jitted one resolves the output
dtype from the numba type of the values at compile time, the python call from
the dtype of the array it is given - the same rule read from the two sides of
numba, so the two cannot answer with different dtypes.  Both of them refuse
what the driver cannot read: the shapes are asked at compile time, the dtypes
of the two arrays in the python call.  That refusal is all the checking a
kernel does - it is not ``aggregate()`` and holds none of its validation
machinery: ``group_idx`` has to be flat, one-dimensional and of an integer
dtype, the values flat, one-dimensional and numeric.

Every kernel carries the signature of its function: ``(group_idx, a,
size=-1, fill_value=<default>)``, and the argument the function has of its own
in front of them - the ``ddof`` of the variance, the ``q`` of the quantile, the
``dx`` of the trapezoid, the ``reverse`` of the sort (which sorts each group in
descending order).  ``size=-1`` detects the group count from the data (an
explicit size must not be smaller than the largest index); the ``fill_value``
default depends on the output dtype (nan for inexact results, 0 otherwise), and
``fill_value=None`` asks for it.

``array`` is the exception to the "one value per group" shape: it collects the
values of every group, so it hands out a *list* of the arrays of the groups -
numba cannot hold an object array, and a list of arrays it can.  Its
``fill_value`` is of no use: a group without values holds an empty array, which
is the default of the function - ``aggregate()`` is the one that applies a
custom fill value to the groups that got none.

Compared to ``aggregate()`` the kernels are restricted to flat 1d input in
forward order (no dtype, axis, order or out), and the sum overflow guesses for
narrow integers are approximated by widening to int64 - ``aggregate()`` keeps
the exact length-dependent dtype rules.  The float16 input is what numba
cannot do at all: ``aggregate()`` runs it on float32 and rounds the result
back, a kernel of its own has nothing to run it on.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba import argreductions, averaging, collect, cumulative, reductions, specials
from numpy_groupies.aggregate_numba.gather import unpack_into
from numpy_groupies.aggregate_numba.ops import _dispatch
from numpy_groupies.aggregate_numba.steps import step_count, step_indices
from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    FuncName,
    aggregate_common_doc,
    funcs_collecting,
    missing_impl_error,
)
from numpy_groupies.utils_np import AFillValue, AResult, AScalar, AValue, RavelOrder


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    func: FuncName | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
    out: AResult | None = None,
    cache: bool = True,
    **kwargs: Any,
) -> AResult:
    try:
        funcname, op = _dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all: it names no known
        # function either, so the registry answers it as a callable of the
        # caller's own
        funcname, op = _dispatch.custom(func)

    # numba has no float16: the values survive the widening cast to float32
    # exactly, so the kernels run on float32 and an inexact result is rounded
    # back - like the numpy twin does, whose bincount accumulates in float64
    # and rounds once at the end
    # the collecting functions are the exception: they hand out the values
    # themselves, so they keep the dtype they came in with
    is16 = getattr(a, "dtype", None) == np.float16
    if is16 and funcname not in funcs_collecting:
        # a float16 input is always an array - the getattr gate does not narrow the union
        a = cast("npt.NDArray[np.number]", a).astype(np.float32)

    if op is None:
        # a function that is known but not implemented here
        raise missing_impl_error(funcname)
    if isinstance(funcname, str):
        ret = cast(AResult, op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs))
    else:
        # a callable of the caller's own: it has no output array of its own,
        # and its compiled loop is memoized where the cache policy says
        if out is not None:
            raise NotImplementedError("out= is only supported for named functions") from None
        ret = cast(
            "AResult",
            op(group_idx, a, size, fill_value, order, dtype, axis, cache=cache, **kwargs),
        )
    if is16 and dtype is None and ret.dtype.kind == "f":
        ret = ret.astype(np.float16)
    if out is not None:
        if out.shape != ret.shape or out.dtype != ret.dtype:
            raise TypeError(
                f"out must have shape {ret.shape} and dtype {ret.dtype}, got shape {out.shape} and dtype {out.dtype}"
            )
        out[...] = ret
        return out
    return ret


aggregate.__doc__ = (
    """
    This is the numba implementation of aggregate.

    This implementation accepts two additional keyword arguments:

    out: default=None
        an array the packed result is copied into (matching shape and dtype
        are required).  The array is returned, so calls can be chained.
    cache: default=True
        when aggregating with a custom callable ``func``, the compiled
        implementation is cached so that subsequent calls with the same
        function are fast.  Set to ``False`` to disable caching, or pass
        your own dictionary to control the caching manually.  With caching
        enabled (the default), callables are kept in a bounded LRU cache of
        128 entries.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the kernels
#
# One function per function of the package, every one of them the stub of its
# operation: the python entry point of the compiled driver and the name
# jitted code calls.  The pairs are reached through their modules, and the
# names of the kernels shadow the builtins 'len', 'sum', 'min', 'all' and
# 'any' for the whole of this module - so nothing in here may call them: this
# module is the one place of the package where 'sum' is not the builtin.

sum = reductions.sum.plain.stub
nansum = reductions.sum.nan.stub
prod = reductions.prod.plain.stub
nanprod = reductions.prod.nan.stub
len = reductions.len.plain.stub
nanlen = reductions.len.nan.stub
all = reductions.all.plain.stub
nanall = reductions.all.nan.stub
any = reductions.any.plain.stub
nanany = reductions.any.nan.stub
last = reductions.last.plain.stub
nanlast = reductions.last.nan.stub
first = reductions.first.plain.stub
nanfirst = reductions.first.nan.stub
allnan = reductions.allnan.plain.stub
anynan = reductions.anynan.plain.stub
min = reductions.min.plain.stub
nanmin = reductions.min.nan.stub
max = reductions.max.plain.stub
nanmax = reductions.max.nan.stub
sumofsquares = reductions.sumofsquares.plain.stub
nansumofsquares = reductions.sumofsquares.nan.stub

argmax = argreductions.argmax.plain.stub
nanargmax = argreductions.argmax.nan.stub
argmin = argreductions.argmin.plain.stub
nanargmin = argreductions.argmin.nan.stub

mean = averaging.mean.plain.stub
nanmean = averaging.mean.nan.stub
std = averaging.std.plain.stub
nanstd = averaging.std.nan.stub
var = averaging.var.plain.stub
nanvar = averaging.var.nan.stub
skew = averaging.skew.plain.stub
nanskew = averaging.skew.nan.stub
kurt = averaging.kurt.plain.stub
nankurt = averaging.kurt.nan.stub

median = specials.median.plain.stub
nanmedian = specials.median.nan.stub
# the percentile is the quantile in percent - the python-level aggregate
# scales its argument down, so the kernel only knows the quantile
quantile = specials.quantile.plain.stub
nanquantile = specials.quantile.nan.stub
trapezoid = specials.trapezoid.plain.stub
nantrapezoid = specials.trapezoid.nan.stub
sort = specials.sort.plain.stub
nunique = specials.nunique.plain.stub
nannunique = specials.nunique.nan.stub

# the collecting function: its result is a sequence per group, so the kernel
# hands out a list of the arrays of the groups - numba cannot hold an object
# array, but a list of arrays it can
array = collect.array.plain.stub

cumsum = cumulative.cumsum.plain.stub
nancumsum = cumulative.cumsum.nan.stub
cumprod = cumulative.cumprod.plain.stub
nancumprod = cumulative.cumprod.nan.stub
cummax = cumulative.cummax.plain.stub
nancummax = cumulative.cummax.nan.stub
cummin = cumulative.cummin.plain.stub
nancummin = cumulative.cummin.nan.stub

__all__ = [
    "aggregate",
    "all",
    "allnan",
    "any",
    "anynan",
    "argmax",
    "argmin",
    "array",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "first",
    "kurt",
    "last",
    "len",
    "max",
    "mean",
    "median",
    "min",
    "nanall",
    "nanany",
    "nanargmax",
    "nanargmin",
    "nancummax",
    "nancummin",
    "nancumprod",
    "nancumsum",
    "nanfirst",
    "nankurt",
    "nanlast",
    "nanlen",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nannunique",
    "nanprod",
    "nanquantile",
    "nanskew",
    "nanstd",
    "nansum",
    "nansumofsquares",
    "nantrapezoid",
    "nanvar",
    "nunique",
    "prod",
    "quantile",
    "sort",
    "std",
    "step_count",
    "step_indices",
    "sum",
    "sumofsquares",
    "trapezoid",
    "unpack_into",
    "var",
]
