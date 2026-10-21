"""Njit-callable aggregate kernels.

Every function of this subpackage is a compiled, self-contained kernel that
can be called directly from jitted code::

    import numba as nb
    import numpy_groupies as npg

    @nb.njit
    def zscore(group_idx, values):
        m = npg.aggregate.mean(group_idx, values)
        s = npg.aggregate.std(group_idx, values)
        return (values - m[group_idx]) / s[group_idx]

Every kernel carries the signature of its function: ``(group_idx, a,
size=-1, fill_value=<default>)``, and the argument the function has of its
own in front of them - the ``ddof`` of the variance, the ``q`` of the
quantile, the ``dx`` of the trapezoid, the ``reverse`` of the sort (which
sorts each group in descending order).  ``size=-1`` detects the group count
from the data (an explicit size must not be smaller than the largest index);
the ``fill_value`` default depends on the output dtype (nan for inexact
results, 0 otherwise).

``array`` is the exception to the "one value per group" shape: it
collects the values of every group, so it hands out a *list* of the arrays of
the groups - numba cannot hold an object array, and a list of arrays it can.
Its ``fill_value`` is of no use: a group without values holds an empty array,
which is the default of the function - the python-level ``aggregate()`` is
the one that applies a custom fill value to the groups that got none.

Compared to the python-level ``aggregate()`` the kernels are restricted to
flat 1d input in forward order (no axis/order/out support), and the sum
overflow guesses for narrow integers are approximated by widening to int64 -
``aggregate()`` keeps the exact length-dependent dtype rules.
"""

from numpy_groupies.aggregate_numba import argreductions, averaging, collect, cumulative, reductions, specials

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
    "sum",
    "sumofsquares",
    "trapezoid",
    "var",
]
