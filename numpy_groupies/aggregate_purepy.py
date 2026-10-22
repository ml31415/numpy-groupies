from __future__ import annotations

import itertools
import math
import numbers
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from numpy_groupies.common import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    resolve_fill_value,
)
from numpy_groupies.registry import (
    FuncName,
    Registry,
    Resolved,
    bind_functions,
    custom_callable,
    funcs_no_separate_nan,
    function_names,
    missing_impl_error,
)
from numpy_groupies.registry import (
    aliasing_py as aliasing,
)
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    LENGTH_MISMATCH,
    check_complex_dtype,
    check_fixed_fill_value,
    check_integer_mean_overflow,
    check_integer_variance_span_overflow,
    is_scalar_input,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from numpy_groupies.utils_np import AFillValue, AResult, AScalar, AValue, RavelOrder


def unpack(group_idx: npt.NDArray[np.integer], ret: AResult | list[Any]) -> list[Any]:
    """The gather of a result: the result of the group of every value.

    The numpy implementation of the gather is a fancy index of an array; this
    one is the plain python loop, and it needs no numpy - the result of the
    pure python implementation is a list, and this is the gather of a list.
    """
    return [ret[int(ix)] for ix in group_idx]


# min, max, all, any - builtin


def _sum(x: Sequence[Any]) -> Any:
    # builtin sum over numpy integers accumulates in the input dtype without
    # promotion (NEP 50), so large groups silently wrap - python ints keep
    # the per-group accumulation exact, everything else (floats, complex,
    # numpy bools, decimals) keeps the builtin behaviour; numbers is stdlib
    # and numpy registers its integer scalars as Integral
    if all(isinstance(val, numbers.Integral) for val in x):
        return sum(int(val) for val in x)
    return sum(x)


def _nunique(x: Sequence[Any]) -> int:
    """The number of distinct values of the group.

    The values that are equal count once, and the nans count once among
    themselves: a nan in either part of a complex value is a nan, and the
    nans of a group are one value of their own.
    """
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
    # like _sum, numpy integer scalars would accumulate in their own dtype
    # (NEP 50) and silently wrap for large groups - python ints stay exact
    if all(isinstance(val, numbers.Integral) for val in x):
        return sum(int(val) for val in x) / len(x)
    return sum(x) / len(x)


def _median(x: Sequence[Any]) -> Any:
    if any(v != v for v in x):
        # like np.median, any nan poisons the whole group - and the poison is
        # shaped like the values it poisons: a plain nan answers a complex group
        # with a float, whose imaginary part states a zero the group never
        # claimed, while the other two implementations answer a nan in both parts
        # (and for real values the product is the plain nan all three agree on)
        return x[0] * math.nan
    srt = sorted(x)
    mid = len(srt) // 2
    if len(srt) % 2 == 1:
        return srt[mid]
    lo, hi = srt[mid - 1], srt[mid]
    if isinstance(lo, (bool, int)) or getattr(getattr(lo, "dtype", None), "kind", None) in ("b", "i", "u"):
        # numpy scalar kinds b/i/u (i.e. np.bool_ and np.integer) add in their own dtype, which
        # overflows for narrow
        # integers and makes bool addition a logical or - np.median promotes
        # to float before dividing, so do the same here
        return (float(lo) + float(hi)) / 2
    return (lo + hi) / 2


def _skew(x: Sequence[Any]) -> Any:
    """The skewness of the values of one group.

    The third central moment over the second one to the three-halves power,
    corrected for the size of the group the way pandas corrects it: fewer
    than three values have no skewness, and a group whose values are all the
    same has none either - its spread is zero, and the answer is zero too.
    """
    if any(isinstance(val, complex) for val in x):
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    n = len(x)
    if n < 3:
        return float("nan")
    values = [float(val) for val in x]
    mean = sum(values) / n
    devs = [val - mean for val in values]
    s2 = sum(dev * dev for dev in devs)
    if s2 == 0:
        return 0.0
    s3 = sum(dev**3 for dev in devs)
    return n * math.sqrt(n - 1) / (n - 2) * s3 / s2**1.5


def _kurt(x: Sequence[Any]) -> Any:
    """The excess kurtosis of the values of one group.

    The fourth central moment over the square of the second one, minus the
    three of a normal distribution, corrected for the size of the group the
    way pandas corrects it: fewer than four values have no kurtosis, and a
    group whose values are all the same has none either.
    """
    if any(isinstance(val, complex) for val in x):
        raise TypeError("the skewness and the kurtosis are not defined for complex values")
    n = len(x)
    if n < 4:
        return float("nan")
    values = [float(val) for val in x]
    mean = sum(values) / n
    devs = [val - mean for val in values]
    s2 = sum(dev * dev for dev in devs)
    if s2 == 0:
        return 0.0
    s4 = sum(dev**4 for dev in devs)
    return n * (n + 1) * (n - 1) * s4 / ((n - 2) * (n - 3) * s2**2) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


def _trapezoid(x: Sequence[Any], dx: float = 1.0) -> float:
    # trapezoidal integration over the group, in array order; a group of a
    # single element (or none) integrates to zero
    s = 0.0
    for prev, cur in itertools.pairwise(x):
        s += 0.5 * (prev + cur)
    return s * dx


def _var(x: Sequence[Any], ddof: int = 0) -> Any:
    # the squared magnitude, so complex values give a real result like np.var;
    # integer input is centered on its first value first - python ints keep
    # the deviations exact, so a large common offset cannot erase the small
    # deviations of a group after the conversion to float
    if all(isinstance(val, numbers.Integral) for val in x):
        origin = int(x[0])
        devs = [int(val) - origin for val in x]
        mean = sum(devs) / len(devs)
        return sum(abs(dev - mean) ** 2 for dev in devs) / (len(x) - ddof)
    mean = _mean(x)
    return sum(abs(xx - mean) ** 2 for xx in x) / (len(x) - ddof)


def _std(x: Sequence[Any], ddof: int = 0) -> float:
    return math.sqrt(_var(x, ddof=ddof))


def _prod(x: Sequence[Any]) -> Any:
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


_argmax.x_and_idx = True  # type: ignore[attr-defined]  # tell aggregate what to use as first arg


def _argmin(x_and_idx: Sequence[tuple[Any, Any]]) -> Any:
    return min(x_and_idx, key=operator.itemgetter(1))[0]


_argmin.x_and_idx = True  # type: ignore[attr-defined]  # tell aggregate what to use as first arg


def _sort(group_idx: Sequence[int], a: Sequence[Any], reverse: bool = False) -> list[Any]:
    def _argsort(unordered: Sequence[Any]) -> list[int]:
        return sorted(range(len(unordered)), key=lambda k: unordered[k])

    # ascending in both directions: a descending key would have to negate the
    # values, which is not order reversing for the integral dtypes (the smallest
    # signed value survives negation as itself, and a zero of an unsigned one
    # stays the smallest key) - and plain python lists cannot be negated at all

    # a nan has no place in the order python sees: it compares neither smaller nor
    # greater, so sorted() leaves it wherever it happened to sit, while numpy orders
    # it as the largest value there is.  Asking for the nan before the value settles
    # every comparison of a nan with a value before the two of them come up, and
    # moves the nans of a group behind its values
    poisoned = [val != val for val in a]
    sortidx = _argsort(list(zip(group_idx, poisoned, a)))
    revidx = _argsort(_argsort(group_idx))
    a_srt = [a[si] for si in sortidx]
    if reverse:
        # flip each group's run of values instead - the labels are sorted, so
        # every group is one contiguous run of the result
        labels = [group_idx[si] for si in sortidx]
        trailing = [poisoned[si] for si in sortidx]
        begin = 0
        while begin < len(a_srt):
            end = begin
            while end + 1 < len(a_srt) and labels[end + 1] == labels[begin]:
                end += 1
            # only the ordered part of the run is flipped: a nan is not made the
            # smallest value by asking for the order the other way round, so it
            # stays where the ascending order put it
            last = end + 1
            while last > begin and trailing[last - 1]:
                last -= 1
            a_srt[begin:last] = reversed(a_srt[begin:last])
            begin = end + 1
    return [a_srt[ri] for ri in revidx]


def _propagate_nan(reduction: Callable[[Sequence[Any]], Any]) -> Callable[[Sequence[Any]], Any]:
    """Let a nan of the group win over its other values, like np.min and np.max do.

    The builtin reductions only do that when the nan happens to be the first
    value of the group: a nan compares neither smaller nor greater, so `min([3,
    nan, 4])` returns 3.  The nan variants of the functions are unaffected -
    `aggregate` filters the nans out before handing the group over.
    """

    def reduction_with_nan(values: Sequence[Any]) -> Any:
        for val in values:
            if val != val:  # val != val - np.isnan does not cover ints and objects
                return val
        return reduction(values)

    return reduction_with_nan


def _quantile(x: Sequence[Any], **kwargs: Any) -> Any:
    """The quantile is not implemented by the pure python backend.

    The quantile is an order statistic over the values of a group - the same
    kind of operation the median is, and the median is implemented here.  The
    quantile needs the same machinery, though: the two values it interpolates
    between have to be picked out of the sorted group.  The numpy and the
    numba implementations do that; this one does not, and says so instead of
    answering something else.
    """
    raise NotImplementedError("the quantile is not implemented by the pure python backend")


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
    "percentile": _quantile,
    "trapezoid": _trapezoid,
    "std": _std,
    "var": _var,
    "skew": _propagate_nan(_skew),
    "kurt": _propagate_nan(_kurt),
    "anynan": _anynan,
    "allnan": _allnan,
    "sort": _sort,
    "array": _array,
    "argmax": _argmax,
    "argmin": _argmin,
    "len": len,
    "nunique": _nunique,
}
_op_dict.update(("nan" + k, v) for k, v in list(_op_dict.items()) if k not in funcs_no_separate_nan)


def _custom_operation(func: Any) -> Resolved:
    """The operation of a callable of the caller's own.

    The pure python implementation runs any callable, so the callable is its
    own implementation.
    """
    callable_ = custom_callable(func)
    return Resolved(callable_, callable_)


_dispatch = Registry(_op_dict, aliasing, _custom_operation)


def _a_dtype(a: Sequence[Any]) -> type:
    # python numbers carry no dtype; the element types decide what a numpy
    # array of these values would have made of them - a mixed list of ints
    # and floats is a float array for numpy, complex makes it complex, and
    # everything else (including bool) counts as non-inexact.  numpy scalars
    # are recognised by their dtype, since e.g. complex64 is not a subclass
    # of the python complex type
    dtype: type = int
    for v in a:
        kind = getattr(getattr(v, "dtype", None), "kind", None)
        if isinstance(v, complex) or kind == "c":
            return complex
        if isinstance(v, float) or kind == "f":
            dtype = float
    return dtype


def _check_integer_precision(a: Sequence[Any], func: Callable[..., AScalar]) -> None:
    """The backend-uniform overflow gates for integer mean/var/std.

    Python ints could compute these exactly, but every backend has to refuse
    the same inputs, so results never depend on the chosen implementation.
    """
    if _a_dtype(a) is not int or len(a) == 0:
        return
    values = [int(val) for val in a]
    low = min(values)
    high = max(values)
    if func is _mean:
        check_integer_mean_overflow(len(values), max(high, -low))
    else:
        check_integer_variance_span_overflow(low, high)


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
) -> list[Any]:
    # the pure python path reads both arguments purely as sized iterables of the
    # values - a sequence of labels and a sequence of values, which is what these
    # casts say without rejecting anything; the scalar that the signature of
    # ``a`` also admits is refused below, as is anything not sized
    idx: Sequence[Any] = cast(Sequence[Any], group_idx)
    vals: Sequence[Any] = cast(Sequence[Any], a)
    if axis is not None:
        raise NotImplementedError("axis arg not supported in purepy implementation.")
    if out is not None:
        raise NotImplementedError("out= is only supported by the numba implementation")
    if len(idx) == 0:
        raise ValueError(EMPTY_GROUP_IDX)

    # Check for 2d group_idx
    if size is None:
        try:
            size = 1 + int(max(idx))
        except (TypeError, ValueError):
            raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")
    else:
        if isinstance(size, Sequence):
            # a sequence of sizes is a multidimensional result, which this
            # implementation does not do
            raise NotImplementedError("pure python implementation doesn't accept a sequence of sizes.")
        # a size is a number of groups - anything that counts is one
        size = operator.index(size)

    for i in idx:
        try:
            i = int(i)
        except (TypeError, ValueError):
            if isinstance(i, (list, tuple)):
                raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")
            else:
                try:
                    len(i)
                except TypeError:
                    raise ValueError(f"invalid value found in group_idx: {i}")
                else:
                    raise NotImplementedError("pure python implementation doesn't accept ndim indexed input.")
        else:
            if i < 0:
                raise ValueError("group_idx contains negative value")

    try:
        funcname, op = _dispatch[func]
    except TypeError:
        # a func that cannot be a key of the table at all: it names no known
        # function either, so the registry answers it as a callable of the
        # caller's own
        funcname, op = _dispatch.custom(func)
    if op is None:
        # a function that is known but not implemented here - the registry
        # answers that too, and says it before a single group is aggregated
        raise missing_impl_error(funcname)
    if is_scalar_input(vals):
        if funcname not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        # a 0d array stands in for the numpy scalar it holds and the list wants
        # values, not 0d arrays - [()] is the dtype preserving way to get one
        # out (item() would drop back to a python number), and python numbers
        # have no shape to index at all
        vals = [cast(Any, vals)[()] if hasattr(vals, "shape") else vals] * len(idx)
    elif len(idx) != len(vals):
        raise ValueError(LENGTH_MISMATCH)

    # the datatype rule of the numpy implementations: nan is only a sensible
    # default where the output datatype can hold it
    fill_value = resolve_fill_value(funcname, fill_value, _a_dtype(vals))
    # the numpy implementations look the fill value against the datatype of the
    # result, not of the values: neither a count nor a boolean answer holds a nan
    check_fixed_fill_value(funcname, fill_value)
    if dtype is not None and isinstance(funcname, str):
        # python numbers carry no dtype at all, so this is the one place where
        # the pure python implementation has to reject a non-complex dtype
        check_complex_dtype(_a_dtype(vals), dtype, funcname)

    # the position of every value in the input, which is the position the
    # reductions reporting one have to name.  Taken before the nan filter below
    # shortens the input, or those positions would count the removed values too
    pos: Sequence[int] = range(len(vals))

    if isinstance(funcname, str) and funcname.startswith("nan"):
        # remove nans (works for complex values, where math.isnan fails) -
        # a group of nothing but nans loses every one of its values and is
        # then left to the fill value, exactly like a group missing from
        # group_idx, which is what the numpy implementations make of such a
        # group too.  Without the case below there is nothing left to
        # unpack once every value was a nan
        kept = [(p, ix, val) for p, ix, val in zip(pos, idx, vals) if val == val]
        if not kept:
            return [fill_value] * size
        pos, idx, vals = zip(*kept)

    if op is _sort:
        return _sort(idx, vals, reverse=kwargs.get("reverse", False))
    if op in (_mean, _var, _std):
        _check_integer_precision(vals, op)

    # sort data and evaluate function on groups
    ret = [fill_value] * size
    if not getattr(op, "x_and_idx", False):
        data: list[tuple[Any, ...]] = sorted(zip(idx, vals), key=operator.itemgetter(0))
        for ix, group in itertools.groupby(data, key=operator.itemgetter(0)):
            ret[ix] = op([val for _, val in group], **kwargs)
    else:
        # a group holding a nan has no extreme to point at, and so no position to
        # name either: the vectorised kernels leave such a group to the fill
        # value, and the nan counterparts reach here with none of them left
        poisoned = {ix for ix, val in zip(idx, vals) if val != val}
        data = sorted(zip(pos, idx, vals), key=operator.itemgetter(1))
        for ix, group in itertools.groupby(data, key=operator.itemgetter(1)):
            if ix in poisoned:
                continue  # stays at the fill value
            ret[ix] = op([(p, val) for p, _, val in group], **kwargs)

    return ret


aggregate.__doc__ = (
    """
    This is the pure python implementation of aggregate. It is terribly slow.
    Using the numpy version is highly recommended.
    """
    + aggregate_common_doc
)


# ---------------------------------------------------------------------------
# the functions of the backend, by their name
#
# One entry point per function of the package, calling the implementation
# above with the name of its function - the same thing the numba module
# offers with its kernels, and the same thing the package itself offers with
# the functions of the best implementation available.

_functions = bind_functions(aggregate, function_names())


def __getattr__(name: str) -> Any:
    """The function of the backend, by its name.

    The names of the functions are not set one by one - they are looked up
    in the table of the entry points, so that they cannot drift from it.
    """
    try:
        return _functions[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    """The names of the functions of the backend, among the module's own."""
    return sorted(set(globals()) | set(_functions))
