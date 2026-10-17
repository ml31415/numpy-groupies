"""``aggregate`` spelled out as the slow, obvious group loop.

One computation per group, no bincount, no ufunc trick and no compiled kernel -
the code is the specification the backends are measured against.  It exists so
that a test can compare a backend against something it does not share code
with: comparing ``aggregate_numpy`` against ``aggregate_numpy`` (or against its
own group loop) proves nothing about the paths that differ between them.

The module deliberately does not import pytest, so the benchmarks can reach it
the same way the tests do.  Reference implementations carry the ``_ref``
suffix, which keeps them next to the function they mirror when sorted.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from numpy_groupies.utils import default_fill_value


def trapezoid_ref(*args: Any, **kwargs: Any) -> Any:
    """``np.trapezoid``, under whichever name this numpy provides it.

    ``np.trapz`` was renamed to ``np.trapezoid`` in numpy 2.0.
    """
    func = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return func(*args, **kwargs)


def _not_nan(values: np.ndarray) -> np.ndarray:
    """the values a nan function may see: for a complex one, a nan in either part is a nan"""
    return values[~np.isnan(values)]


def _as_numbers(values: np.ndarray) -> np.ndarray:
    """the values as numbers to integrate over: numpy adds booleans as a
    logical or, which would make the trapezoid of a bool group nonsense"""
    return values.astype(np.float64) if values.dtype.kind == "b" else values


def _sum_of_squares(values: np.ndarray, skip_nan: bool = False) -> Any:
    """the sum of the squared values of one group.

    For a complex group that is the sum of the squared magnitudes - the real
    one, which is why ``a * a`` would be the wrong product here.  Integers are
    squared wide, so the total cannot wrap in the dtype of the input.
    """
    if skip_nan:
        values = _not_nan(values)
    if values.dtype.kind in "iub":
        return np.sum(values.astype(np.int64) ** 2)
    return np.sum(np.abs(values) ** 2)


def _sort_ref(values: np.ndarray, reverse: bool = False) -> np.ndarray:
    """one group, ordered the way numpy orders it."""
    ordered = np.sort(values)
    if not reverse:
        return ordered
    # not the ascending order turned around: numpy reaches a value it cannot
    # compare with - a nan, or one in either part of a complex value - by no
    # comparison at all, and leaves it behind every value it is reached by,
    # in either direction
    if ordered.dtype.kind in "fc":
        incomparable = np.isnan(ordered)
        return np.concatenate((ordered[~incomparable][::-1], ordered[incomparable]))
    return ordered[::-1]


# the callables behind the string names of the aggregation functions, each
# taking the values of one group and the positions those values sit at in the
# input - the arg reductions report an input position, numpy reports one
# inside the group
_REF_ONE_PER_GROUP: dict[str, Callable[..., Any]] = {
    "sum": lambda v, pos: np.sum(v),
    "prod": lambda v, pos: np.prod(v),
    "min": lambda v, pos: np.min(v),
    "max": lambda v, pos: np.max(v),
    "all": lambda v, pos: np.all(v),
    "any": lambda v, pos: np.any(v),
    "mean": lambda v, pos: np.mean(v),
    "median": lambda v, pos: np.median(v),
    "std": lambda v, pos, ddof=0: np.std(v, ddof=ddof),
    "var": lambda v, pos, ddof=0: np.var(v, ddof=ddof),
    "trapezoid": lambda v, pos, dx=1.0: trapezoid_ref(_as_numbers(v), dx=dx),
    "sumofsquares": lambda v, pos: _sum_of_squares(v),
    "len": lambda v, pos: len(v),
    "first": lambda v, pos: v[0],
    "last": lambda v, pos: v[-1],
    "anynan": lambda v, pos: bool(np.isnan(v).any()),
    "allnan": lambda v, pos: bool(np.isnan(v).all()),
    # a nan poisons the plain arg reductions: no value can be named the
    # extreme one, so there is no index to report and the group takes the fill
    "argmin": lambda v, pos: pos[np.argmin(v)] if not np.isnan(v).any() else None,
    "argmax": lambda v, pos: pos[np.argmax(v)] if not np.isnan(v).any() else None,
    "nansum": lambda v, pos: np.nansum(v),
    "nanprod": lambda v, pos: np.nanprod(v),
    "nanmin": lambda v, pos: np.nanmin(v),
    "nanmax": lambda v, pos: np.nanmax(v),
    "nanall": lambda v, pos: np.all(_not_nan(v)),
    "nanany": lambda v, pos: np.any(_not_nan(v)),
    "nanmean": lambda v, pos: np.nanmean(v),
    "nanmedian": lambda v, pos: np.nanmedian(v),
    "nanstd": lambda v, pos, ddof=0: np.nanstd(v, ddof=ddof),
    "nanvar": lambda v, pos, ddof=0: np.nanvar(v, ddof=ddof),
    "nantrapezoid": lambda v, pos, dx=1.0: trapezoid_ref(_as_numbers(_not_nan(v)), dx=dx),
    "nansumofsquares": lambda v, pos: _sum_of_squares(v, skip_nan=True),
    "nanlen": lambda v, pos: int(np.count_nonzero(~np.isnan(v))),
    "nanfirst": lambda v, pos: _not_nan(v)[0],
    "nanlast": lambda v, pos: _not_nan(v)[-1],
    # the nan counterparts skip the nans, and an all-nan group has as little
    # an extreme as an empty one has
    "nanargmin": lambda v, pos: pos[np.nanargmin(v)] if np.count_nonzero(~np.isnan(v)) else None,
    "nanargmax": lambda v, pos: pos[np.nanargmax(v)] if np.count_nonzero(~np.isnan(v)) else None,
}

# the functions that hand back one value per input item instead of one per
# group, so they have nothing to fill an absent group with
_REF_ONE_PER_ITEM: dict[str, Callable[..., Any]] = {
    "cumsum": lambda v, pos: np.cumsum(v),
    "nancumsum": lambda v, pos: np.nancumsum(v),
    "cumprod": lambda v, pos: np.cumprod(v),
    "cummin": lambda v, pos: np.minimum.accumulate(v),
    "cummax": lambda v, pos: np.maximum.accumulate(v),
    "sort": lambda v, pos, reverse=False: _sort_ref(v, reverse=reverse),
}

_REF_FUNCS = {**_REF_ONE_PER_GROUP, **_REF_ONE_PER_ITEM}


def aggregate_ref(
    group_idx: np.ndarray,
    a: np.ndarray,
    *,
    func: str,
    size: int | None = None,
    fill_value: Any = None,
    dtype: np.dtype | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """``aggregate(group_idx, a, func=func, ...)`` computed one group at a time.

    Accepts the string name of a function, the one-dimensional labels and the
    keyword arguments that the reference can express at all; everything else
    raises instead of quietly returning something else.  A function that has no
    answer for a group - a nan in the plain arg reductions, an all-nan group for
    their nan counterparts, a group without any value - reports ``None`` and the
    group takes ``fill_value``, or the function default that ``numpy_groupies``
    itself uses.
    """
    group_idx = np.asarray(group_idx)
    a = np.asarray(a)
    if group_idx.ndim != 1:
        raise NotImplementedError("aggregate_ref takes one-dimensional labels only")
    if func not in _REF_FUNCS:
        raise NotImplementedError(f"aggregate_ref does not know {func!r}")
    if size is None:
        size = int(group_idx.max()) + 1 if group_idx.size else 0

    call = _REF_FUNCS[func]
    # one stable argsort gathers the groups, each of them in the order its
    # values had in the input - the arg reductions report an input position
    order = np.argsort(group_idx, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(group_idx, minlength=size))]
    positions = {}
    results = {}
    for group in range(size):
        pos = order[bounds[group] : bounds[group + 1]]
        if not pos.size:
            continue
        value = call(a[pos], pos, **kwargs)
        if value is None:
            # the function has no answer for this group, which is handled like
            # an empty one: it takes the fill value
            continue
        positions[group] = pos
        results[group] = value

    if func in _REF_ONE_PER_ITEM:
        # the result holds one slot per input item, so a group that is not in
        # the labels simply has no slot - there is nothing to fill and nothing
        # that could be left uninitialized
        # the cumulative functions follow numpy's own promotion, the sort keeps
        # the dtype of what it was given
        out_dtype = dtype or np.asarray(next(iter(results.values()))).dtype
        ret = np.empty(a.shape, dtype=out_dtype)
        for group, values in results.items():
            ret[positions[group]] = values
        return ret

    values = np.asarray(list(results.values()))
    ret = np.full(
        size,
        default_fill_value(func, values.dtype) if fill_value is None else fill_value,
        dtype=dtype or values.dtype,
    )
    for group, value in results.items():
        ret[group] = value
    return ret
