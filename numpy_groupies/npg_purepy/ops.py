"""The operations of the pure python implementation: one per function of the
package, computed with python numbers and sequences - no numpy in sight."""

from __future__ import annotations

import itertools
import math
import numbers
import operator
from collections.abc import Callable, Sequence
from typing import Any

from numpy_groupies.functions import QUANTILE_METHODS
from numpy_groupies.registry import (
    Registry,
    Resolved,
    custom_callable,
    funcs_no_separate_nan,
)
from numpy_groupies.registry import (
    aliasing_py as aliasing,
)


def _sum(x: Sequence[Any]) -> Any:
    # python ints do not wrap, numpy integer scalars do (bool is not Integral)
    if all(isinstance(val, numbers.Integral) for val in x):
        return sum(int(val) for val in x)
    return sum(x)


def _nunique(x: Sequence[Any]) -> int:
    """The number of distinct values of the group, the nans as one value."""
    distinct: set[Any] = set()
    nans = 0
    for val in x:
        if val != val:  # val != val - np.isnan does not cover ints and objects
            nans = 1
        else:
            distinct.add(val)
    return len(distinct) + nans


def _last(x: Sequence[Any]) -> Any:
    return x[-1]


def _first(x: Sequence[Any]) -> Any:
    return x[0]


def _array(x: Sequence[Any]) -> Sequence[Any]:
    return x


def _mean(x: Sequence[Any]) -> Any:
    # python ints stay exact, numpy integer scalars would wrap (NEP 50)
    if all(isinstance(val, numbers.Integral) for val in x):
        return sum(int(val) for val in x) / len(x)
    return sum(x) / len(x)


def _median(x: Sequence[Any]) -> Any:
    if any(v != v for v in x):
        # any nan poisons the whole group, shaped like the values (x[0] * nan
        # answers a complex group with a complex nan, as np.median does)
        return x[0] * math.nan
    srt = sorted(x)
    mid = len(srt) // 2
    if len(srt) % 2 == 1:
        return srt[mid]
    lo, hi = srt[mid - 1], srt[mid]
    if isinstance(lo, (bool, int)) or getattr(getattr(lo, "dtype", None), "kind", None) in ("b", "i", "u"):
        # np.median promotes to float before dividing; do the same
        return (float(lo) + float(hi)) / 2
    return (lo + hi) / 2


def _deviations(values: list[float]) -> list[float]:
    """The deviations of the values of one group from their mean.

    The group is centered on its first value first, the mean of those
    deviations is taken off as well - the raw mean would leave the moments
    of a group without spread as the noise of an ulp instead of zero.
    """
    origin = values[0]
    devs = [val - origin for val in values]
    mean = sum(devs) / len(devs)
    return [dev - mean for dev in devs]


def _skew(x: Sequence[Any]) -> Any:
    """The skewness of the values of one group, pandas-corrected for size."""
    if any(isinstance(val, complex) for val in x):
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    n = len(x)
    if n < 3:
        return float("nan")
    devs = _deviations([float(val) for val in x])
    s2 = sum(dev * dev for dev in devs)
    if s2 == 0:
        return 0.0
    s3 = sum(dev**3 for dev in devs)
    return n * math.sqrt(n - 1) / (n - 2) * s3 / s2**1.5


def _kurt(x: Sequence[Any]) -> Any:
    """The excess kurtosis of the values of one group, pandas-corrected."""
    if any(isinstance(val, complex) for val in x):
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    n = len(x)
    if n < 4:
        return float("nan")
    devs = _deviations([float(val) for val in x])
    s2 = sum(dev * dev for dev in devs)
    if s2 == 0:
        return 0.0
    s4 = sum(dev**4 for dev in devs)
    return n * (n + 1) * (n - 1) * s4 / ((n - 2) * (n - 3) * s2**2) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


def _trapezoid(x: Sequence[Any], dx: float = 1.0) -> float:
    # trapezoidal integration over the group, in array order
    s = 0.0
    for prev, cur in itertools.pairwise(x):
        s += 0.5 * (prev + cur)
    return s * dx


def _var(x: Sequence[Any], ddof: int = 0) -> Any:
    # squared magnitude, so complex values give a real result like np.var;
    # integer input is centered on its first value in python ints, which
    # keeps the deviations exact
    if len(x) - ddof <= 0:
        # no degrees of freedom - the group takes the fill value
        return None
    if all(isinstance(val, numbers.Integral) for val in x):
        origin = int(x[0])
        devs = [int(val) - origin for val in x]
        mean = sum(devs) / len(devs)
        return sum(abs(dev - mean) ** 2 for dev in devs) / (len(x) - ddof)
    mean = _mean(x)
    return sum(abs(xx - mean) ** 2 for xx in x) / (len(x) - ddof)


def _std(x: Sequence[Any], ddof: int = 0) -> Any:
    var = _var(x, ddof=ddof)
    if var is None:
        return None
    return math.sqrt(var)


def _sum_of_squares(x: Sequence[Any]) -> Any:
    """The sum of the squared values of one group, exact for python ints."""
    if all(isinstance(val, numbers.Integral) for val in x):
        return sum(int(val) * int(val) for val in x)
    return sum(abs(xx) ** 2 for xx in x)


def _prod(x: Sequence[Any]) -> Any:
    # the multiplicative counterpart of _sum: python ints do not wrap
    if all(isinstance(val, numbers.Integral) for val in x):
        r = 1
        for xx in x:
            r *= int(xx)
        return r
    r = x[0]
    for xx in x[1:]:
        r *= xx
    return r


def _anynan(x: Sequence[Any]) -> bool:
    return any(xx != xx for xx in x)


def _allnan(x: Sequence[Any]) -> bool:
    return all(xx != xx for xx in x)


def _argmax(x_and_idx: Sequence[tuple[Any, Any]]) -> Any:
    return max(x_and_idx, key=operator.itemgetter(1))[0]


def _argmin(x_and_idx: Sequence[tuple[Any, Any]]) -> Any:
    return min(x_and_idx, key=operator.itemgetter(1))[0]


# the two that name a position take the index of the value as their first
# argument - the aggregate asks for it
_TAKES_INDEX_FIRST = (_argmax, _argmin)


def _sort(group_idx: Sequence[int], a: Sequence[Any], reverse: bool = False) -> list[Any]:
    def _argsort(unordered: Sequence[Any]) -> list[int]:
        return sorted(range(len(unordered)), key=lambda k: unordered[k])

    # a descending key is not order reversing for the integral dtypes, and
    # plain lists cannot be negated at all - so sort ascending and flip
    # each group's run afterwards
    # a nan compares neither smaller nor greater, so sorted() would leave it
    # wherever it sits; asking for the nan before the value settles every
    # comparison and moves the nans behind the values of the group
    poisoned = [val != val for val in a]
    sortidx = _argsort(list(zip(group_idx, poisoned, a)))
    revidx = _argsort(_argsort(group_idx))
    a_srt = [a[si] for si in sortidx]
    if reverse:
        # flip each group's run of values (the labels are sorted, so every
        # group is one contiguous run)
        labels = [group_idx[si] for si in sortidx]
        trailing = [poisoned[si] for si in sortidx]
        begin = 0
        while begin < len(a_srt):
            end = begin
            while end + 1 < len(a_srt) and labels[end + 1] == labels[begin]:
                end += 1
            # only the ordered part of the run is flipped: the nan stays
            # where the ascending order put it
            last = end + 1
            while last > begin and trailing[last - 1]:
                last -= 1
            a_srt[begin:last] = reversed(a_srt[begin:last])
            begin = end + 1
    return [a_srt[ri] for ri in revidx]


def _topk(x: Sequence[Any], k: int, largest: bool, fill_value: Any) -> list[Any]:
    """The k largest (or the k smallest) values of the group.

    The nans are in no selection; a group with fewer than k values fills
    the places it has none with the fill value.
    """
    values = [val for val in x if val == val]
    ordered = sorted(values)
    if largest:
        ordered = ordered[::-1]
    if len(ordered) < k:
        ordered = ordered + [fill_value] * (k - len(ordered))
    return ordered[:k]


def _mode(x: Sequence[Any]) -> Any:
    """The most frequent value of the group - the smallest on a tie, nans aside.

    A group of nothing but nans has no mode: the None leaves it at the fill
    value.
    """
    counts: dict[Any, int] = {}
    for val in x:
        if val != val:  # val != val - np.isnan does not cover ints and objects
            continue
        counts[val] = counts.get(val, 0) + 1
    best_value = None
    best_count = 0
    for value, count in counts.items():
        if count > best_count or (count == best_count and value < best_value):  # smallest on a tie
            best_value, best_count = value, count
    return best_value


def _propagate_nan(reduction: Callable[[Sequence[Any]], Any]) -> Callable[[Sequence[Any]], Any]:
    """Let a nan of the group win over its other values, like np.min and np.max do.

    The builtin reductions only do that when the nan is the first value.
    """

    def reduction_with_nan(values: Sequence[Any]) -> Any:
        for val in values:
            if val != val:  # val != val - np.isnan does not cover ints and objects
                return val
        return reduction(values)

    return reduction_with_nan


def _sorted_values(x: Sequence[Any]) -> list[Any]:
    """the values of a group, ordered like numpy does - complexes lexicographically."""
    if any(isinstance(val, complex) or getattr(getattr(val, "dtype", None), "kind", None) == "c" for val in x):
        return sorted(x, key=lambda val: (val.real, val.imag))
    return sorted(x)


def _quantile(x: Sequence[Any], q: float = 0.5, method: str = "linear") -> Any:
    """The quantile of the values of one group, the way numpy computes it:

    sorted values, the position ``q * (n - 1)`` interpolated per method
    (linear, lower, higher, midpoint, nearest).  A nan poisons the whole
    group, like it poisons the median.
    """
    if method not in QUANTILE_METHODS:
        raise ValueError(f"unknown interpolation method '{method}' - use one of {sorted(QUANTILE_METHODS)}")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantiles must be in the range [0, 1]")
    if any(val != val for val in x):
        # any nan poisons the whole group, shaped like the values
        return x[0] * math.nan
    values = _sorted_values(x)
    count = len(values)
    # the virtual index of the quantile in the sorted group
    pos = q * (count - 1)
    lower_index = math.floor(pos)
    upper_index = math.ceil(pos)
    lower = values[lower_index]
    upper = values[upper_index]
    if method == "nearest":
        # numpy rounds to the nearest element, halves to even
        return values[min(round(pos), count - 1)]
    if method == "lower":
        return lower
    if method == "higher":
        return upper
    if method == "midpoint":
        return (lower + upper) / 2
    if lower_index == upper_index:
        return lower
    return lower + (upper - lower) * (pos - lower_index)


def _percentile(x: Sequence[Any], q: float = 50.0, method: str = "linear") -> Any:
    """The percentile is the quantile in percent."""
    return _quantile(x, q / 100.0, method=method)


def _widened(x: Sequence[Any]) -> list[Any]:
    """the values of a group, the integers as python numbers, which do not wrap."""
    if all(isinstance(val, numbers.Integral) for val in x):
        return [int(val) for val in x]
    return list(x)


def _cumsum(x: Sequence[Any], fill_value: Any = 0, **kwargs: Any) -> list[Any]:
    """The running sum of the values of one group; a nan poisons from there on."""
    out: list[Any] = []
    running: Any = 0
    for val in _widened(x):
        # the addition carries the poison: a nan in one part of a complex
        # value poisons that part alone, the way numpy's cumsum carries it
        running = running + val
        out.append(running)
    return out


def _nancumsum(x: Sequence[Any], fill_value: Any = 0, **kwargs: Any) -> list[Any]:
    """The running sum of the values of one group, the nans left out."""
    out: list[Any] = []
    running: Any = 0
    for val in _widened(x):
        if val == val:
            running = running + val
        out.append(running)
    return out


def _cumprod(x: Sequence[Any], fill_value: Any = 0, **kwargs: Any) -> list[Any]:
    """The running product of the values of one group."""
    out: list[Any] = []
    running: Any = 1
    for val in _widened(x):
        running = running * val
        out.append(running)
    return out


def _nancumprod(x: Sequence[Any], fill_value: Any = 0, **kwargs: Any) -> list[Any]:
    """The running product of the values of one group, the nans left out."""
    out: list[Any] = []
    running: Any = 1
    for val in _widened(x):
        if val == val:
            running = running * val
        out.append(running)
    return out


def _cum_extreme(kind: str, skip_nan: bool) -> Callable[..., list[Any]]:
    """The running extreme of the values of one group, starting at the fill value.

    The nan variant leaves the nan out: it neither wins nor clears the
    "seen" flag, so the places before the first value hold the fill value.
    """

    def one_group(x: Sequence[Any], fill_value: Any = 0, **kwargs: Any) -> list[Any]:
        out: list[Any] = []
        running: Any = fill_value
        seen = False
        for val in x:
            if val != val:
                if skip_nan:
                    out.append(running)
                    continue
                running = val
                seen = True
                out.append(running)
                continue
            if not seen or (val > running if kind == "max" else val < running):
                running = val
            seen = True
            out.append(running)
        return out

    one_group.__name__ = f"{'nan' if skip_nan else ''}cum{kind}"
    return one_group


_cummin = _cum_extreme("min", skip_nan=False)
_cummax = _cum_extreme("max", skip_nan=False)
_nancummin = _cum_extreme("min", skip_nan=True)
_nancummax = _cum_extreme("max", skip_nan=True)


# the reductions hand out a scalar, the sorting and the collecting function a
# sequence - hence Any
_op_dict: dict[str, Callable[..., Any]] = {
    "min": _propagate_nan(min),
    "max": _propagate_nan(max),
    "sum": _sum,
    "prod": _prod,
    "last": _last,
    "first": _first,
    "all": all,
    "any": any,
    "mean": _mean,
    "median": _median,
    "quantile": _quantile,
    "percentile": _percentile,
    "trapezoid": _trapezoid,
    "std": _std,
    "var": _var,
    "sumofsquares": _sum_of_squares,
    "cumsum": _cumsum,
    "nancumsum": _nancumsum,
    "cumprod": _cumprod,
    "nancumprod": _nancumprod,
    "cummin": _cummin,
    "nancummin": _nancummin,
    "cummax": _cummax,
    "nancummax": _nancummax,
    "skew": _propagate_nan(_skew),
    "kurt": _propagate_nan(_kurt),
    "anynan": _anynan,
    "allnan": _allnan,
    "sort": _sort,
    "topk": _topk,
    "array": _array,
    "argmax": _argmax,
    "argmin": _argmin,
    "len": len,
    "nunique": _nunique,
    "mode": _mode,
}
# nan variants are only generated for the functions without one of their own
_op_dict.update(
    ("nan" + k, v) for k, v in list(_op_dict.items()) if k not in funcs_no_separate_nan and ("nan" + k) not in _op_dict
)

# one value per input item, in the order of the input - not one per group
_ONE_PER_ITEM = (_cumsum, _nancumsum, _cumprod, _nancumprod, _cummin, _cummax, _nancummin, _nancummax)


def _custom_operation(func: Any) -> Resolved:
    """The operation of a func that missed the table: a callable, run as it is."""
    callable_ = custom_callable(func)
    return Resolved(callable_, callable_)


_dispatch = Registry(_op_dict, aliasing, _custom_operation)
