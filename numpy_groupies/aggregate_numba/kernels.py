"""Njit-callable aggregate kernels.

Every function of this subpackage is a compiled, self-contained kernel that
can be called directly from jitted code::

    import numba as nb
    from numpy_groupies.aggregate_numba import kernels

    @nb.njit
    def zscore(group_idx, values):
        m = kernels.aggregate_mean(group_idx, values)
        s = kernels.aggregate_std(group_idx, values)
        return (values - m[group_idx]) / s[group_idx]

The signatures are ``(group_idx, a, ddof=0, size=-1, fill_value=<default>)``
- the third argument is named ``dx`` for trapezoid and ``reverse`` for sort
(1.0 sorts each group in descending order).  ``size=-1`` detects the
group count from the data (an explicit size must not be smaller than the
largest index); the ``fill_value`` default depends on the output dtype (nan
for inexact results, 0 otherwise).

``aggregate_array`` is the exception to the "one value per group" shape: it
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

aggregate_sum = reductions.sum.plain.stub
aggregate_nansum = reductions.sum.nan.stub
aggregate_prod = reductions.prod.plain.stub
aggregate_nanprod = reductions.prod.nan.stub
aggregate_len = reductions.len.plain.stub
aggregate_nanlen = reductions.len.nan.stub
aggregate_all = reductions.all.plain.stub
aggregate_nanall = reductions.all.nan.stub
aggregate_any = reductions.any.plain.stub
aggregate_nanany = reductions.any.nan.stub
aggregate_last = reductions.last.plain.stub
aggregate_nanlast = reductions.last.nan.stub
aggregate_first = reductions.first.plain.stub
aggregate_nanfirst = reductions.first.nan.stub
aggregate_allnan = reductions.allnan.plain.stub
aggregate_anynan = reductions.anynan.plain.stub
aggregate_min = reductions.min.plain.stub
aggregate_nanmin = reductions.min.nan.stub
aggregate_max = reductions.max.plain.stub
aggregate_nanmax = reductions.max.nan.stub
aggregate_sumofsquares = reductions.sumofsquares.plain.stub
aggregate_nansumofsquares = reductions.sumofsquares.nan.stub

aggregate_argmax = argreductions.argmax.plain.stub
aggregate_nanargmax = argreductions.argmax.nan.stub
aggregate_argmin = argreductions.argmin.plain.stub
aggregate_nanargmin = argreductions.argmin.nan.stub

aggregate_mean = averaging.mean.plain.stub
aggregate_nanmean = averaging.mean.nan.stub
aggregate_std = averaging.std.plain.stub
aggregate_nanstd = averaging.std.nan.stub
aggregate_var = averaging.var.plain.stub
aggregate_nanvar = averaging.var.nan.stub
aggregate_skew = averaging.skew.plain.stub
aggregate_nanskew = averaging.skew.nan.stub
aggregate_kurt = averaging.kurt.plain.stub
aggregate_nankurt = averaging.kurt.nan.stub

aggregate_median = specials.median.plain.stub
aggregate_nanmedian = specials.median.nan.stub
# the percentile is the quantile in percent - the python-level aggregate
# scales its argument down, so the kernel only knows the quantile
aggregate_quantile = specials.quantile.plain.stub
aggregate_nanquantile = specials.quantile.nan.stub
aggregate_trapezoid = specials.trapezoid.plain.stub
aggregate_nantrapezoid = specials.trapezoid.nan.stub
aggregate_sort = specials.sort.plain.stub
aggregate_nunique = specials.nunique.plain.stub
aggregate_nannunique = specials.nunique.nan.stub

# the collecting function: its result is a sequence per group, so the kernel
# hands out a list of the arrays of the groups - numba cannot hold an object
# array, but a list of arrays it can
aggregate_array = collect.array.plain.stub

aggregate_cumsum = cumulative.cumsum.plain.stub
aggregate_nancumsum = cumulative.cumsum.nan.stub
aggregate_cumprod = cumulative.cumprod.plain.stub
aggregate_nancumprod = cumulative.cumprod.nan.stub
aggregate_cummax = cumulative.cummax.plain.stub
aggregate_nancummax = cumulative.cummax.nan.stub
aggregate_cummin = cumulative.cummin.plain.stub
aggregate_nancummin = cumulative.cummin.nan.stub

__all__ = [
    "aggregate_all",
    "aggregate_allnan",
    "aggregate_any",
    "aggregate_anynan",
    "aggregate_argmax",
    "aggregate_argmin",
    "aggregate_array",
    "aggregate_cummax",
    "aggregate_cummin",
    "aggregate_cumprod",
    "aggregate_cumsum",
    "aggregate_first",
    "aggregate_kurt",
    "aggregate_last",
    "aggregate_len",
    "aggregate_max",
    "aggregate_mean",
    "aggregate_median",
    "aggregate_min",
    "aggregate_nanall",
    "aggregate_nanany",
    "aggregate_nanargmax",
    "aggregate_nanargmin",
    "aggregate_nancummax",
    "aggregate_nancummin",
    "aggregate_nancumprod",
    "aggregate_nancumsum",
    "aggregate_nanfirst",
    "aggregate_nankurt",
    "aggregate_nanlast",
    "aggregate_nanlen",
    "aggregate_nanmax",
    "aggregate_nanmean",
    "aggregate_nanmedian",
    "aggregate_nanmin",
    "aggregate_nannunique",
    "aggregate_nanprod",
    "aggregate_nanquantile",
    "aggregate_nanskew",
    "aggregate_nanstd",
    "aggregate_nansum",
    "aggregate_nansumofsquares",
    "aggregate_nantrapezoid",
    "aggregate_nanvar",
    "aggregate_nunique",
    "aggregate_prod",
    "aggregate_quantile",
    "aggregate_sort",
    "aggregate_std",
    "aggregate_sum",
    "aggregate_sumofsquares",
    "aggregate_trapezoid",
    "aggregate_var",
]
