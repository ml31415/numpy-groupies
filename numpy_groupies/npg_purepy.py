from __future__ import annotations

import itertools
import math
import numbers
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from numpy_groupies.fills import (
    DEFAULT_FILL_VALUE,
    resolve_fill_value,
)
from numpy_groupies.functions import (
    QUANTILE_METHODS,
    aggregate_common_doc,
    bind_functions,
    forced_float_types,
    forced_real_float_types,
)
from numpy_groupies.registry import (
    FuncName,
    Registry,
    Resolved,
    custom_callable,
    funcs_no_separate_nan,
    function_names,
    missing_impl_error,
    resolve_name,
)
from numpy_groupies.registry import (
    aliasing_py as aliasing,
)
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    LENGTH_MISMATCH,
    NEGATIVE_INDICES,
    NON_INTEGER_GROUP_IDX,
    OUTPUT_SIZE_NOT_SCALAR,
    TOO_LARGE_FOR_SIZE,
    TOPK_NEEDS_FLAT_GROUP_IDX,
    check_complex_dtype,
    check_fixed_fill_value,
    check_integer_variance_span_overflow,
    check_topk_k,
    is_integer_dtype,
    is_scalar_input,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder


def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> list[Any]:
    """The gather of a result: the result of the group of every value."""
    return [ret[int(ix)] for ix in group_idx]


# min, max, all, any - builtin


__all__ = ["aggregate", "unpack"]


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


# the names the functions of the package answer to - the aliases of the
# registry, which carry the numpy functions by name as well
def _a_dtype(a: Sequence[Any]) -> type:
    # the element types decide what a numpy array of these values would have
    # made of them; numpy scalars are recognised by their dtype, since e.g.
    # complex64 is no subclass of complex
    dtype: type = int
    for v in a:
        kind = getattr(getattr(v, "dtype", None), "kind", None)
        if isinstance(v, complex) or kind == "c":
            return complex
        if isinstance(v, float) or kind == "f":
            dtype = float
    return dtype


def _check_integer_precision(a: Sequence[Any], func: Callable[..., AScalar]) -> None:
    """The backend-uniform overflow gates for integer var/std; the mean needs none."""
    if _a_dtype(a) is not int or len(a) == 0:
        return
    values = [int(val) for val in a]
    low = min(values)
    high = max(values)
    if func is not _mean:
        check_integer_variance_span_overflow(low, high)


def _is_flat(values: Sequence[Any]) -> bool:
    """whether the values are a flat sequence.

    A mapping, a set, a generator and anything with rows of its own is no
    sequence of values - refused like every other path refuses it.
    """
    if getattr(values, "ndim", 0) > 1:
        return False
    if isinstance(values, (dict, set, frozenset)):
        return False
    try:
        length = len(values)
    except TypeError:
        return False
    if length == 0:
        return True
    first = values[0]
    return not isinstance(first, (list, tuple)) and getattr(first, "ndim", 0) == 0


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
) -> list[Any]:
    # both arguments are read purely as sized iterables of values; the scalar
    # the signature of ``a`` also admits is refused below
    idx: Sequence[Any] = cast(Sequence[Any], group_idx)
    vals: Sequence[Any] = cast(Sequence[Any], a)
    if axis is not None:
        raise NotImplementedError("axis arg not supported in purepy implementation.")
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")
    if len(idx) == 0:
        raise ValueError(EMPTY_GROUP_IDX)
    if not is_scalar_input(vals) and not _is_flat(vals):
        raise ValueError("a must be scalar or 1 dimensional, use .ravel to flatten. Alternatively specify axis.")

    # the func is resolved before anything is measured of the labels or the
    # sizes, so a refused call speaks in the words of the package, not of
    # what this implementation happened to measure first
    try:
        funcname, op, _temporal_op = _dispatch[func]
    except TypeError:
        # no key of the table: run as the caller's own callable
        funcname, op, _temporal_op = _dispatch.custom(func)
    if op is None:
        raise missing_impl_error(funcname)

    # Check for 2d group_idx
    if size is None:
        try:
            size = 1 + int(max(idx))
        except (TypeError, ValueError):
            if op is _topk:
                # the matrix of the topk is a rule of every backend
                raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX) from None
            raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")
    else:
        if isinstance(size, Sequence):
            if op is _topk:
                if getattr(idx, "ndim", 1) <= 1 and not hasattr(idx[0], "__len__"):
                    raise ValueError(OUTPUT_SIZE_NOT_SCALAR)
                raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX)
            # a sequence of sizes is a multidimensional result, which this
            # implementation does not do
            raise NotImplementedError("pure python implementation doesn't accept a sequence of sizes.")
        # a size is a number of groups - anything that counts is one
        size = operator.index(size)

    if op is _topk:
        k = check_topk_k(kwargs.get("k", 5))
        largest = kwargs.get("largest", True)

    # a label that is no index is refused in the words of the package
    labels_dtype = getattr(idx, "dtype", None)
    if labels_dtype is not None and not is_integer_dtype(labels_dtype):
        raise TypeError(NON_INTEGER_GROUP_IDX)

    for value in idx:
        try:
            i = int(value)
        except (TypeError, ValueError):
            if op is _topk:
                raise ValueError(TOPK_NEEDS_FLAT_GROUP_IDX) from None
            if isinstance(value, (list, tuple)):
                raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")
            else:
                try:
                    len(value)
                except TypeError:
                    raise ValueError(f"invalid value found in group_idx: {value}")
                else:
                    raise NotImplementedError("pure python implementation doesn't accept ndim indexed input.")
        else:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                # a whole float names no group, it is not rounded
                raise TypeError(NON_INTEGER_GROUP_IDX)
            if i < 0:
                raise ValueError(NEGATIVE_INDICES)
            if i >= size:
                raise ValueError(TOO_LARGE_FOR_SIZE.format(size=size))

    if is_scalar_input(vals):
        if funcname not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        # [()] is the dtype preserving way to get the value out of a 0d array
        vals = [cast(Any, vals)[()] if hasattr(vals, "shape") else vals] * len(idx)
    elif len(idx) != len(vals):
        raise ValueError(LENGTH_MISMATCH)

    # nan is only a sensible default where the output datatype can hold it;
    # the averaging functions answer floats, the squared sums stay exact
    result_dtype = _a_dtype(vals)
    if (
        result_dtype is int
        and isinstance(funcname, str)
        and resolve_name(funcname) not in ("sumofsquares", "nansumofsquares")
        and (resolve_name(funcname) in forced_float_types or resolve_name(funcname) in forced_real_float_types)
    ):
        result_dtype = float
    if result_dtype is int and not isinstance(funcname, str) and dtype is None:
        # a callable of the caller: assume a floating result without a dtype
        result_dtype = float
    fill_value = resolve_fill_value(funcname, fill_value, result_dtype)
    check_fixed_fill_value(funcname, fill_value)
    if dtype is not None and isinstance(funcname, str):
        # the one place where this implementation has to reject a non-complex dtype
        check_complex_dtype(_a_dtype(vals), dtype, funcname)

    # the positions have to be taken before the nan filter shortens the input
    pos: Sequence[int] = range(len(vals))

    # the nan variants of the running values leave the nan in place, so they
    # are not shortened
    if isinstance(funcname, str) and funcname.startswith("nan") and op not in _ONE_PER_ITEM:
        kept = [(p, ix, val) for p, ix, val in zip(pos, idx, vals) if val == val]
        if not kept:
            # a group of nothing but nans takes the fill value
            return [fill_value] * size
        pos, idx, vals = zip(*kept)

    if op is _sort:
        return _sort(idx, vals, reverse=kwargs.get("reverse", False))
    if op in _ONE_PER_ITEM:
        ret = [fill_value] * len(pos)
        running_data = sorted(zip(pos, idx, vals), key=operator.itemgetter(1))
        for _ix, entries_of_group in itertools.groupby(running_data, key=operator.itemgetter(1)):
            group_entries: list[tuple[Any, Any, Any]] = list(entries_of_group)  # the groupby iterator is spent
            running = op([val for _, _, val in group_entries], fill_value=fill_value, **kwargs)
            for (p, _ix2, _val), value in zip(group_entries, running):
                ret[p] = value
        return ret
    if op is _topk:
        # k values per group, a list per group - there are no arrays here
        topk_ret: list[Any] = [[fill_value] * k for _ in range(size)]
        topk_data = sorted(zip(idx, vals), key=operator.itemgetter(0))
        for ix, group in itertools.groupby(topk_data, key=operator.itemgetter(0)):
            topk_ret[ix] = _topk([val for _, val in group], k, largest, fill_value)
        return topk_ret
    if op in (_mean, _var, _std):
        _check_integer_precision(vals, op)

    # sort data and evaluate function on groups
    ret = [fill_value] * size
    if op not in _TAKES_INDEX_FIRST:
        data = sorted(zip(idx, vals), key=operator.itemgetter(0))
        for ix, values_of_group in itertools.groupby(data, key=operator.itemgetter(0)):
            value = op([val for _, val in values_of_group], **kwargs)
            if value is not None:  # no answer: the group stays at the fill value
                ret[ix] = value
    else:
        # a group holding a nan has no extreme to point at, and so no position
        poisoned = {ix for ix, val in zip(idx, vals) if val != val}
        data_with_pos = sorted(zip(pos, idx, vals), key=operator.itemgetter(1))
        for ix, entries in itertools.groupby(data_with_pos, key=operator.itemgetter(1)):
            if ix in poisoned:
                continue  # stays at the fill value
            ret[ix] = op([(p, val) for p, _, val in entries], **kwargs)

    return ret


aggregate.__doc__ = (
    """
    This is the pure python implementation of aggregate. It is terribly slow.
    Using the numpy version is highly recommended.

    It computes with python numbers and needs no numpy.  Every function of
    the package is there except the temporal arithmetic and the
    multidimensional output: axis and a sequence of sizes are refused.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the functions of the backend, by their name
#
# one entry point per function of the package, calling the implementation
# above with the name of its function

_functions = bind_functions(aggregate, function_names())


def __getattr__(name: str) -> Any:
    """The function of the backend, by its name."""
    try:
        return _functions[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(set(globals()) | set(_functions))
