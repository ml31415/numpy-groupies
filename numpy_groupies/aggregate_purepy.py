import itertools
import math
import operator

import numpy as np

from .utils import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    build_dispatch,
    funcs_no_separate_nan,
    get_func,
    resolve_fill_value,
)
from .utils import aliasing_py as aliasing

# min, max, sum, all, any - builtin


def _last(x):
    return x[-1]


def _first(x):
    return x[0]


def _array(x):
    return x


def _mean(x):
    return sum(x) / len(x)


def _median(x):
    if any(math.isnan(v) for v in x):
        # like np.median, any nan poisons the whole group
        return math.nan
    srt = sorted(x)
    mid = len(srt) // 2
    if len(srt) % 2 == 1:
        return srt[mid]
    lo, hi = srt[mid - 1], srt[mid]
    if isinstance(lo, (bool, np.bool_, np.integer)):
        # numpy scalars add in their own dtype, which overflows for narrow
        # integers and makes bool addition a logical or - np.median promotes
        # to float before dividing, so do the same here
        return (float(lo) + float(hi)) / 2
    return (lo + hi) / 2


def _trapezoid(x, dx=1.0):
    # trapezoidal integration over the group, in array order; a group of a
    # single element (or none) integrates to zero
    s = 0.0
    for prev, cur in itertools.pairwise(x):
        s += 0.5 * (prev + cur)
    return s * dx


def _var(x, ddof=0):
    mean = _mean(x)
    return sum((xx - mean) ** 2 for xx in x) / (len(x) - ddof)


def _std(x, ddof=0):
    return math.sqrt(_var(x, ddof=ddof))


def _prod(x):
    r = x[0]
    for xx in x[1:]:
        r *= xx
    return r


def _anynan(x):
    return any(math.isnan(xx) for xx in x)


def _allnan(x):
    return all(math.isnan(xx) for xx in x)


def _argmax(x_and_idx):
    return max(x_and_idx, key=operator.itemgetter(1))[0]


_argmax.x_and_idx = True  # tell aggregate what to use as first arg


def _argmin(x_and_idx):
    return min(x_and_idx, key=operator.itemgetter(1))[0]


_argmin.x_and_idx = True  # tell aggregate what to use as first arg


def _sort(group_idx, a, reverse=False):
    def _argsort(unordered):
        return sorted(range(len(unordered)), key=lambda k: unordered[k])

    sortidx = _argsort(list(zip(group_idx, -a if reverse else a)))
    revidx = _argsort(_argsort(group_idx))
    a_srt = [a[si] for si in sortidx]
    return [a_srt[ri] for ri in revidx]


def _propagate_nan(reduction):
    """Let a nan of the group win over its other values, like np.min and np.max do.

    The builtin reductions only do that when the nan happens to be the first
    value of the group: a nan compares neither smaller nor greater, so `min([3,
    nan, 4])` returns 3.  The nan variants of the functions are unaffected -
    `aggregate` filters the nans out before handing the group over.
    """

    def reduction_with_nan(values):
        for val in values:
            if val != val:  # val != val - np.isnan does not cover ints and objects
                return val
        return reduction(values)

    return reduction_with_nan


_impl_dict = {
    "min": _propagate_nan(min),
    "max": _propagate_nan(max),
    "sum": sum,
    "prod": _prod,
    "last": _last,
    "first": _first,
    "all": all,
    "any": any,
    "mean": _mean,
    "median": _median,
    "trapezoid": _trapezoid,
    "std": _std,
    "var": _var,
    "anynan": _anynan,
    "allnan": _allnan,
    "sort": _sort,
    "array": _array,
    "argmax": _argmax,
    "argmin": _argmin,
    "len": len,
}
_impl_dict.update(("nan" + k, v) for k, v in list(_impl_dict.items()) if k not in funcs_no_separate_nan)

_dispatch = build_dispatch(_impl_dict, aliasing)


def aggregate(
    group_idx,
    a,
    func="sum",
    size=None,
    fill_value=DEFAULT_FILL_VALUE,
    order="C",
    dtype=None,
    axis=None,
    **kwargs,
):
    if axis is not None:
        raise NotImplementedError("axis arg not supported in purepy implementation.")
    if len(group_idx) == 0:
        raise ValueError("group_idx must not be empty")

    # Check for 2d group_idx
    if size is None:
        try:
            size = 1 + int(max(group_idx))
        except (TypeError, ValueError):
            raise NotImplementedError("pure python implementation doesn't accept ndim idx input.")

    for i in group_idx:
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
        func = _dispatch[func][0]
    except (KeyError, TypeError):
        func = get_func(func, aliasing, _impl_dict)
    if isinstance(a, (int, float)):
        if func not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        a = [a] * len(group_idx)
    elif len(group_idx) != len(a):
        raise ValueError("group_idx and a must be of the same length")

    # the datatype rule of the numpy implementations: nan is only a sensible
    # default where the output datatype can hold it
    fill_value = resolve_fill_value(func, fill_value, np.asarray(a).dtype)

    if isinstance(func, str):
        if func.startswith("nan"):
            func = func[3:]
            # remove nans
            group_idx, a = zip(*((ix, val) for ix, val in zip(group_idx, a) if not math.isnan(val)))

        func = _impl_dict[func]
    if func is _sort:
        return _sort(group_idx, a, reverse=kwargs.get("reverse", False))

    # sort data and evaluate function on groups
    ret = [fill_value] * size
    if not getattr(func, "x_and_idx", False):
        data = sorted(zip(group_idx, a), key=operator.itemgetter(0))
        for ix, group in itertools.groupby(data, key=operator.itemgetter(0)):
            ret[ix] = func([val for _, val in group], **kwargs)
    else:
        data = sorted(zip(range(len(a)), group_idx, a), key=operator.itemgetter(1))
        for ix, group in itertools.groupby(data, key=operator.itemgetter(1)):
            ret[ix] = func([(val_idx, val) for val_idx, _, val in group], **kwargs)

    return ret


aggregate.__doc__ = (
    """
    This is the pure python implementation of aggregate. It is terribly slow.
    Using the numpy version is highly recommended.
    """
    + aggregate_common_doc
)
