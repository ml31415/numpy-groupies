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
of the two arrays in the python call, and the fill value is checked against
the output it fills - in the words of the python-level call, which checks it
the same way.  That refusal is all the checking a kernel does - it is not
``aggregate()`` and holds none of its validation
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

Datetime64 and timedelta64 input is aggregated on its int64 ticks: the
``aggregate()`` entry hands temporal values to the wrapper of the numpy
based backends (``temporal``), which carries the policy -
what numpy refuses, the NaT of the plain reductions and the nan variants,
the unit of the answer - and turns the tick results back.  The kernels
themselves take temporal values as they are, as far as their arithmetic
allows: the comparisons of compiled code order datetime64 and timedelta64
values (NaT is the smallest tick among them, and ``val != val`` detects it
like a nan), so the selection and the reduction kernels - min, max, first,
last, their nan variants, the argreductions, len, allnan, anynan, array,
and the sum and the cumulative sums of the durations - aggregate them
directly, as a python call and as a
jitted one.  The kernels that cannot take them (the arithmetic of the
moments, the partitions and the sorts, the truthiness, and the mean - whose
rounding numba's division of durations would truncate, where the package
rounds half to even) say so by name.  The mean of the ticks of a temporal
input has its own kernels - ``tick_mean`` and ``nantick_mean`` - which
answer it exactly, half to even, on the int64 ticks the caller hands them.
"""

import builtins
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba import argreductions, averaging, collect, cumulative, reductions, specials
from numpy_groupies.aggregate_numba.fallback import FALLBACK
from numpy_groupies.aggregate_numba.gather import unpack, unpack_into
from numpy_groupies.aggregate_numba.ops import _dispatch, _driver_call
from numpy_groupies.aggregate_numba.steps import step_count, step_indices

# The numpy implementation answers the three calls of this backend that
# numba cannot: the median of a complex input, the quantile of a method
# other than the linear one, and the collecting function of a dtype that
# numba has no dtype for.  The dependency of this backend on the other one
# is here and nowhere else - the modules take the box, and the box gets
# the implementation now.
from numpy_groupies.aggregate_numpy import aggregate as _numpy_aggregate
from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
)
from numpy_groupies.nptools import QUANTILE_METHODS, exact_integer_group_mean
from numpy_groupies.registry import (
    FuncName,
    funcs_collecting,
    missing_impl_error,
)
from numpy_groupies.temporal import (
    _temporal_aggregation,
    is_own_function,
    is_temporal_input,
)
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder
from numpy_groupies.validation import EMPTY_GROUP_IDX, EXACT_FLOAT64_INTEGER_LIMIT

FALLBACK.set_implementation(_numpy_aggregate)


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
        funcname, op = _dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all: it names no known
        # function either, so the registry answers it as a callable of the
        # caller's own
        funcname, op = _dispatch.custom(func)

    # datetime64 and timedelta64 run on their int64 ticks - the temporal
    # wrapper takes the input apart, refuses what numpy refuses, and answers
    # with a temporal value.  Only the functions of the package go through it:
    # a callable of the caller's own defines its own semantics and sees the
    # values as they are.  The numpy functions that mean the same thing are
    # aliases of the functions of the package, so np.mean is the mean of the
    # package and gets its answer.
    temporal = is_own_function(funcname) and is_temporal_input(a)
    if temporal:
        if size is None:
            if not group_idx.size:
                # the kernels of the operations refuse an empty input as well
                raise ValueError(EMPTY_GROUP_IDX)
            # the NaT rows the nan-variants drop would shrink the group count
            # the kernels detect from the labels - pin it before the filter
            size = int(group_idx.max()) + 1

        def temporal_op(name: str) -> Callable[..., AResult]:
            # the compiled operation of a name, behind the signature the
            # temporal wrapper calls it with
            _, kernel = _dispatch[name]
            if kernel is None:
                raise missing_impl_error(name)

            def call(
                group_idx: npt.NDArray[np.integer],
                values: AValue,
                size_: int,
                fill_value: AFillValue = None,
                dtype: npt.DTypeLike | None = None,
                **kw: Any,
            ) -> AResult:
                return cast(AResult, kernel(group_idx, values, size_, fill_value, order, dtype, axis, **kw))

            return call

        def tick_mean(
            group_idx: npt.NDArray[np.integer],
            values: AValue,
            size_: int,
            fill_value: AFillValue = None,
            dtype: npt.DTypeLike | None = None,
            nan: bool = False,
            **kw: Any,
        ) -> AResult:
            """the compiled mean of the ticks - the deviations from the first
            value of a group, so the sum of the ticks stays exact"""
            return cast(
                AResult,
                _driver_call(averaging.tick_mean.nan if nan else averaging.tick_mean.plain)(
                    group_idx,
                    values,
                    size_,
                    np.dtype(np.int64),
                    np.dtype(np.int64),
                    fill_value,
                    # the sample spacing and the quantile are no argument of
                    # the tick mean - the driver of the plain variant takes
                    # them and leaves them alone
                    0.0,
                    0.0,
                ),
            )

        def tick_median(
            group_idx: npt.NDArray[np.integer],
            values: AValue,
            size_: int,
            fill_value: AFillValue = None,
            dtype: npt.DTypeLike | None = None,
            nan: bool = False,
            **kw: Any,
        ) -> AResult:
            """the compiled median of the ticks - the exact middle of every
            group, where the numeric kernel promotes to float64"""
            return cast(
                AResult,
                _driver_call(specials.tick_median.nan if nan else specials.tick_median.plain)(
                    group_idx,
                    values,
                    size_,
                    np.dtype(np.int64),
                    np.dtype(np.int64),
                    fill_value,
                    # the sample spacing and the quantile are no argument of
                    # the tick median - the driver of the plain variant takes
                    # them and leaves them alone
                    0.0,
                    0.0,
                ),
            )

        def tick_quantile(
            group_idx: npt.NDArray[np.integer],
            values: AValue,
            size_: int,
            fill_tick: int,
            fraction: float,
            method: str = "linear",
        ) -> AResult:
            """the compiled quantile of the ticks - numpy's rule of the
            interpolation, on the ticks of the values

            All five methods of numpy run in the kernel: the branch on the
            method happens once per group, and a group costs far more than
            the branch.  The NaT rows are already gone in the nan variants -
            the wrapper drops them - so the plain kernel is the one to call.
            """
            if method not in QUANTILE_METHODS:
                raise ValueError(f"unknown interpolation method '{method}' - use one of {sorted(QUANTILE_METHODS)}")
            return cast(
                AResult,
                _driver_call(specials.tick_quantile.plain)(
                    group_idx,
                    values,
                    size_,
                    np.dtype(np.int64),
                    np.dtype(np.int64),
                    fill_tick,
                    # the sample spacing is no argument of the tick quantile -
                    # the driver of the plain variant takes it and leaves it
                    # alone
                    0.0,
                    fraction,
                    method,
                ),
            )

        ret, fill_value = _temporal_aggregation(
            temporal_op,
            tick_mean,
            tick_median,
            tick_quantile,
            cast("str", funcname),
            group_idx,
            cast("AValue", a),
            cast("int", size),
            fill_value,
            dtype,
            kwargs,
            # the compiled kernels take the temporal values as they are
            native=True,
        )

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
    if temporal:
        # the temporal wrapper has answered - the mean, the median and the
        # quantiles of temporal input compute on their own, exact
        pass
    else:
        if (
            isinstance(funcname, str)
            and funcname in ("mean", "nanmean")
            and isinstance(a, np.ndarray)
            and a.dtype.kind in "iu"
            and a.size
        ):
            # The sums of these integers can leave the window float64
            # represents exactly, and the compiled kernel refuses the mean
            # then - it knows the bounds of the values from its own
            # accumulation pass, so the check costs nothing extra.  The exact
            # answer comes from the limb sums of the values, which the
            # wrapper computes when the kernel refuses - like the numpy
            # backend does.
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
        elif isinstance(funcname, str):
            ret = cast(AResult, op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs))
        else:
            # a callable of the caller's own: it has no output array of its own,
            # and its compiled loop is memoized in the bounded LRU of this backend
            if out is not None:
                raise NotImplementedError("out= is only supported for named functions") from None
            ret = cast(
                "AResult",
                op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs),
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

    This implementation accepts one additional keyword argument:

    out: default=None
        an array the packed result is copied into (matching shape and dtype
        are required).  The array is returned, so calls can be chained.
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
# the mean of the ticks of a temporal input - the kernel measures the
# deviations from the first value of the group, so the sum of the ticks
# stays exact and the answer is a tick again
tick_mean = averaging.tick_mean.plain.stub
nantick_mean = averaging.tick_mean.nan.stub
# the median of the ticks of a temporal input - the exact middle of the
# ticks, where the numeric kernel would promote the mean of the two
# middles to float64 and round it off the tick grid
tick_median = specials.tick_median.plain.stub
nantick_median = specials.tick_median.nan.stub
# the quantile of the ticks of a temporal input - numpy's rule of the
# interpolation, on the ticks of the values
tick_quantile = specials.tick_quantile.plain.stub
nantick_quantile = specials.tick_quantile.nan.stub
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


# The two functions the kernels do not cover - the percentile is the quantile
# in percent, so it is the entry point of the backend and not a kernel of its
# own.


def percentile(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    q: float = 50.0,
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
) -> AResult:
    """The percentile of every group - the quantile of the group, in percent.

    The kernel of the quantile is the one that does the work; the percentile
    is the quantile with its argument in percent.
    """
    return aggregate(
        group_idx,
        a,
        func="percentile",
        q=q,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        axis=axis,
    )


def nanpercentile(
    group_idx: npt.NDArray[np.integer],
    a: AValue,
    q: float = 50.0,
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike | None = None,
    axis: int | None = None,
) -> AResult:
    """The percentile of every group, the nans of the group left out."""
    return aggregate(
        group_idx,
        a,
        func="nanpercentile",
        q=q,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        axis=axis,
    )


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
    "trapezoid",
    "unpack",
    "unpack_into",
    "var",
]
