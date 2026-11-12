"""The numba implementation of aggregate: compiled per-dtype kernels behind
one dispatch, with ``out=`` support and caching for custom callables.

``aggregate`` is the entry point; the modules beside it hold one kernel per
function of the package, under that function's name.  A kernel is the compiled
driver of its operation and nothing else, callable from jitted code and from
python alike - the jitted call resolves the output dtype from the numba types
at compile time, the python one from the dtype of the array, so the two cannot
answer with different dtypes.  Shapes are refused at compile time, the array
dtypes and the fill value in the python call; there is no further validation:
``group_idx`` must be flat, 1d and of an integer dtype, the values flat, 1d
and numeric.

Every kernel takes ``(group_idx, a, size=-1, fill_value=<default>)`` plus the
argument of its own (``ddof``, ``q``, ``dx``, ``reverse``, which sorts each
group descending).  ``size=-1`` detects the group count from the data; the
``fill_value`` default is nan for inexact results, 0 otherwise, and None asks
for it.  ``array`` hands out a list of the arrays of the groups (numba cannot
hold an object array), an empty group an empty one - a custom fill value is
applied by ``aggregate()``.  ``topk`` answers a (size, k) matrix of the k
largest values, the missing values (nans, NaT) in no selection.

Against ``aggregate()`` the kernels are restricted to flat 1d input: sum
overflow guesses widen to int64, float16 is widened to float32 in the python
entry (the values survive that cast exactly, an inexact result is rounded
back once - from jitted code a float16 never arrives).  Datetime64 and
timedelta64 run on their int64 ticks: the ``aggregate()`` entry composes the
policy of ``numpy_groupies.temporal`` with the operations of this backend
(refusing what numpy refuses, carrying NaT, returning the unit), while the
kernels whose arithmetic can take temporal values aggregate them directly
(NaT is the smallest tick, ``val != val`` detects it) - the product and
cumulative product, the sum of squares, the dispersion, the higher moments,
the trapezoid, the truthiness and the sort refuse them by name, and the mean,
the median and the quantiles answer on the ticks, the mean exact, rounded
half to even.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.dtypes import is_temporal_input
from numpy_groupies.fills import DEFAULT_FILL_VALUE
from numpy_groupies.functions import aggregate_common_doc
from numpy_groupies.groups import exact_integer_group_mean
from numpy_groupies.npg_numba import argreductions, averaging, collect, cumulative, reductions, specials
from numpy_groupies.npg_numba.fallback import FALLBACK
from numpy_groupies.npg_numba.gather import unpack, unpack_into
from numpy_groupies.npg_numba.ops import _dispatch
from numpy_groupies.npg_numba.steps import step_count, step_indices

# the numpy implementation answers the three calls numba cannot: the median
# of a complex input, the quantile of a non-linear method, and the collecting
# function of a dtype numba has none for - the only dependency on it
from numpy_groupies.npg_numpy import aggregate as _numpy_aggregate
from numpy_groupies.registry import (
    FuncName,
    missing_impl_error,
)
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder
from numpy_groupies.validation import EMPTY_GROUP_IDX

FALLBACK.set_implementation(_numpy_aggregate)


def _into_out(ret: AResult, out: AResult | None) -> AResult:
    """The answer of the call, written into the caller's array where one was given."""
    if out is None:
        return ret
    if out.shape != ret.shape or out.dtype != ret.dtype:
        raise TypeError(
            f"out must have shape {ret.shape} and dtype {ret.dtype}, got shape {out.shape} and dtype {out.dtype}"
        )
    out[...] = ret
    return out


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
    **kwargs: Any,
) -> AResult:
    try:
        funcname, op, temporal_op = _dispatch[func]
    except TypeError:
        # a func that is no key of the table names no function either: the
        # registry answers it as a callable of the caller
        funcname, op, temporal_op = _dispatch.custom(func)

    if op is None:
        # this backend implements every function of the package, so the guard
        # is only the table's shape
        raise missing_impl_error(funcname)

    # datetime64/timedelta64 run on their int64 ticks: the temporal operation
    # takes them apart, refuses what numpy refuses and answers a temporal
    # value - only the functions of the package go through it, and their
    # numpy aliases are the same functions, so np.mean gets its answer
    if temporal_op is not None and is_temporal_input(a):
        if size is None:
            if not group_idx.size:
                # the kernels of the operations refuse an empty input as well
                raise ValueError(EMPTY_GROUP_IDX)
            # the NaT rows the nan variants drop would shrink the detected
            # group count - pin it before the filter
            size = int(group_idx.max()) + 1
        ret, _ = temporal_op(
            group_idx,
            a,
            cast("int", size),
            fill_value,
            dtype,
            order=order,
            axis=axis,
            **kwargs,
        )
        return _into_out(ret, out)

    if not isinstance(funcname, str):
        # a callable of the caller: no output array of its own, its compiled
        # loop memoized in the bounded LRU of this backend
        if out is not None:
            raise NotImplementedError("out= is only supported for named functions") from None
        return cast("AResult", op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs))

    if funcname in ("mean", "nanmean") and isinstance(a, np.ndarray) and a.dtype.kind in "iu" and a.size:
        # these integer sums can leave the window float64 represents exactly,
        # and the kernel refuses the mean then (it knows the bounds from its
        # own accumulation pass) - the exact answer comes from the limb sums,
        # like the numpy backend computes it
        try:
            ret = cast(AResult, op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs))
        except OverflowError:
            n = int(group_idx.max()) + 1 if size is None else cast("int", size)
            out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
            ret = cast(
                AResult,
                exact_integer_group_mean(
                    group_idx,
                    cast("npt.NDArray[np.integer]", a),
                    n,
                    cast("AScalar", fill_value),
                    out_dtype,
                ),
            )
        return _into_out(ret, out)

    return _into_out(cast(AResult, op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)), out)


aggregate.__doc__ = (
    """
    This is the numba implementation of aggregate.

    This implementation accepts one additional keyword argument:

    out: default=None
        an array the packed result is copied into (matching shape and dtype
        are required).  The array is returned, so calls can be chained.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the kernels - one stub per function of the package: the python entry of
# the compiled driver and the name jitted code calls.  Their names shadow
# the builtins 'len', 'sum', 'min', 'all' and 'any' for this whole module -
# the one place of the package where 'sum' is not the builtin.

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
quantile = specials.quantile.plain.stub
nanquantile = specials.quantile.nan.stub
percentile = specials.percentile.plain.stub
nanpercentile = specials.percentile.nan.stub
trapezoid = specials.trapezoid.plain.stub
nantrapezoid = specials.trapezoid.nan.stub
sort = specials.sort.plain.stub
nunique = specials.nunique.plain.stub
nannunique = specials.nunique.nan.stub
topk = specials.topk.plain.stub
mode = specials.mode.plain.stub

# the collecting function: a list of the arrays of the groups - numba cannot
# hold an object array, a list of arrays it can
array = collect.array.plain.stub

cumsum = cumulative.cumsum.plain.stub
nancumsum = cumulative.cumsum.nan.stub
cumprod = cumulative.cumprod.plain.stub
nancumprod = cumulative.cumprod.nan.stub
cummax = cumulative.cummax.plain.stub
nancummax = cumulative.cummax.nan.stub
cummin = cumulative.cummin.plain.stub
nancummin = cumulative.cummin.nan.stub


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(globals())


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
    "mode",
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
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanskew",
    "nanstd",
    "nansum",
    "nansumofsquares",
    "nantrapezoid",
    "nanvar",
    "nunique",
    "percentile",
    "prod",
    "quantile",
    "skew",
    "sort",
    "std",
    "step_count",
    "step_indices",
    "sum",
    "sumofsquares",
    "topk",
    "trapezoid",
    "unpack",
    "unpack_into",
    "var",
]
