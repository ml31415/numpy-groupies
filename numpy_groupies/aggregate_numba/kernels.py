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

Compared to the python-level ``aggregate()`` the kernels are restricted to
flat 1d input in forward order (no axis/order/out support), and the sum
overflow guesses for narrow integers are approximated by widening to int64 -
``aggregate()`` keeps the exact length-dependent dtype rules.
"""

from numpy_groupies.aggregate_numba import argreductions, averaging, cumulative, reductions, specials
from numpy_groupies.aggregate_numba.compose import nan_handle

aggregate_sum = reductions.sum.plain.stub
aggregate_nansum = nan_handle(reductions.sum, "sum").stub
aggregate_prod = reductions.prod.plain.stub
aggregate_nanprod = nan_handle(reductions.prod, "prod").stub
aggregate_len = reductions.len.plain.stub
aggregate_nanlen = nan_handle(reductions.len, "len").stub
aggregate_all = reductions.all.plain.stub
aggregate_nanall = nan_handle(reductions.all, "all").stub
aggregate_any = reductions.any.plain.stub
aggregate_nanany = nan_handle(reductions.any, "any").stub
aggregate_last = reductions.last.plain.stub
aggregate_nanlast = nan_handle(reductions.last, "last").stub
aggregate_first = reductions.first.plain.stub
aggregate_nanfirst = nan_handle(reductions.first, "first").stub
aggregate_allnan = reductions.allnan.plain.stub
aggregate_anynan = reductions.anynan.plain.stub
aggregate_min = reductions.min.plain.stub
aggregate_nanmin = nan_handle(reductions.min, "min").stub
aggregate_max = reductions.max.plain.stub
aggregate_nanmax = nan_handle(reductions.max, "max").stub
aggregate_sumofsquares = reductions.sumofsquares.plain.stub
aggregate_nansumofsquares = nan_handle(reductions.sumofsquares, "sumofsquares").stub

aggregate_argmax = argreductions.argmax.plain.stub
aggregate_nanargmax = nan_handle(argreductions.argmax, "argmax").stub
aggregate_argmin = argreductions.argmin.plain.stub
aggregate_nanargmin = nan_handle(argreductions.argmin, "argmin").stub

aggregate_mean = averaging.mean.plain.stub
aggregate_nanmean = nan_handle(averaging.mean, "mean").stub
aggregate_std = averaging.std.plain.stub
aggregate_nanstd = nan_handle(averaging.std, "std").stub
aggregate_var = averaging.var.plain.stub
aggregate_nanvar = nan_handle(averaging.var, "var").stub

aggregate_median = specials.median.plain.stub
aggregate_nanmedian = nan_handle(specials.median, "median").stub
aggregate_trapezoid = specials.trapezoid.plain.stub
aggregate_nantrapezoid = nan_handle(specials.trapezoid, "trapezoid").stub
aggregate_sort = specials.sort.plain.stub

aggregate_cumsum = cumulative.cumsum.plain.stub
aggregate_nancumsum = nan_handle(cumulative.cumsum, "cumsum").stub
aggregate_cumprod = cumulative.cumprod.plain.stub
aggregate_nancumprod = nan_handle(cumulative.cumprod, "cumprod").stub
aggregate_cummax = cumulative.cummax.plain.stub
aggregate_nancummax = nan_handle(cumulative.cummax, "cummax").stub
aggregate_cummin = cumulative.cummin.plain.stub
aggregate_nancummin = nan_handle(cumulative.cummin, "cummin").stub

__all__ = [
    "aggregate_all",
    "aggregate_allnan",
    "aggregate_any",
    "aggregate_anynan",
    "aggregate_argmax",
    "aggregate_argmin",
    "aggregate_cummax",
    "aggregate_cummin",
    "aggregate_cumprod",
    "aggregate_cumsum",
    "aggregate_first",
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
    "aggregate_nanlast",
    "aggregate_nanlen",
    "aggregate_nanmax",
    "aggregate_nanmean",
    "aggregate_nanmedian",
    "aggregate_nanmin",
    "aggregate_nanprod",
    "aggregate_nanstd",
    "aggregate_nansum",
    "aggregate_nansumofsquares",
    "aggregate_nantrapezoid",
    "aggregate_nanvar",
    "aggregate_prod",
    "aggregate_sort",
    "aggregate_std",
    "aggregate_sum",
    "aggregate_sumofsquares",
    "aggregate_trapezoid",
    "aggregate_var",
]
