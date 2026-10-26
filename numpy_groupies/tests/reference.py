"""``aggregate`` spelled out as the slow, obvious group loop.

One computation per group, no bincount, no ufunc trick and no compiled kernel -
the code is the specification the backends are measured against.  It exists so
that a test can compare a backend against something it does not share code
with: comparing ``aggregate_numpy`` against ``aggregate_numpy`` (or against its
own group loop) proves nothing about the paths that differ between them.

What the reference does not do is guess dtypes: a group reduction hands its
values to the numpy function of the same name and takes the dtype that comes
out of it.  The length dependent widening the package applies is its own
policy, pinned by explicit expectations in :mod:`.test_dtypes`, and a reference
that copied it would no longer be independent of the implementation.  The
memory order of a multi-dimensional result is not part of it either - see
:mod:`.test_axis`.

The module deliberately does not import pytest, so the benchmarks can reach it
the same way the tests do.  Reference implementations carry the ``_ref``
suffix, which keeps them next to the function they mirror when sorted.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_groupies.common import DEFAULT_FILL_VALUE, default_fill_value
from numpy_groupies.utils_np import check_fill_value


def trapezoid_ref(*args: Any, **kwargs: Any) -> Any:
    """``np.trapezoid``, under whichever name this numpy provides it.

    ``np.trapz`` was renamed to ``np.trapezoid`` in numpy 2.0.
    """
    func = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return func(*args, **kwargs)


def _not_nan(values: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """the values a nan function may see: for a complex one, a nan in either part is a nan"""
    return values[~np.isnan(values)]


def _of(numpy_function: Callable[..., Any]) -> Callable[[np.ndarray, np.ndarray], Any]:
    """a function of the values of one group, ignoring where they sit"""

    def one_group(values: npt.NDArray[np.floating], pos: npt.NDArray[np.integer]) -> Any:
        return numpy_function(values)

    one_group.__name__ = getattr(numpy_function, "__name__", "one_group")
    return one_group


def _at_a_position(numpy_function: Callable[..., Any]) -> Callable[[np.ndarray, np.ndarray], Any]:
    """a function that reports which value won, as the position it sits at in
    the input - numpy reports the position inside the group instead"""

    def one_group(values: npt.NDArray[np.floating], pos: npt.NDArray[np.integer]) -> Any:
        return pos[numpy_function(values)]

    one_group.__name__ = f"at_{numpy_function.__name__}"
    return one_group


def _quantile_ref(values: npt.NDArray[np.number], q: float = 0.5, method: str = "linear") -> Any:
    """the quantile of one group, the way numpy computes it.

    numpy's own quantile takes real values only, so a group of complex values
    is ordered the way numpy orders complex values in its median - the real
    part first, the imaginary part breaking the tie - and the quantile
    interpolates between the two values it falls between as it does for the
    real ones.
    """
    if values.dtype.kind != "c":
        return np.quantile(values, q, method=method)
    if np.isnan(values).any():
        # a nan in either part makes the value incomparable, and the group
        # answers with a nan, like the median of a complex group does
        return values[0] * np.nan
    ordered = np.sort(values)
    pos = q * (len(ordered) - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    if method == "nearest":
        return ordered[min(int(np.round(pos)), len(ordered) - 1)]
    if lower == upper:
        return ordered[lower]
    low, high = ordered[lower], ordered[upper]
    if method == "lower":
        return low
    if method == "higher":
        return high
    if method == "midpoint":
        return (low + high) / 2
    return low + (high - low) * (pos - lower)


def _skip_nans(one_group: Callable[..., Any]) -> Callable[..., Any]:
    """turn a function of a group into its nan counterpart.

    The same computation over the values that are left, at the positions they
    sit at - and a group of nothing but nans has nothing left, which is handled
    like an empty group: it takes the fill value rather than whatever numpy
    reports for a reduction over nothing.
    """

    def nan_variant(values: npt.NDArray[np.floating], pos: npt.NDArray[np.integer], **kwargs: Any) -> Any:
        keep = ~np.isnan(values)
        if not keep.any():
            return None
        return one_group(values[keep], pos[keep], **kwargs)

    nan_variant.__name__ = f"nan_{one_group.__name__}"
    return nan_variant


def _poisoned_by_a_nan(one_group: Callable[..., Any]) -> Callable[..., Any]:
    """the plain counterpart of a nan function, for those that report a position
    rather than a value.

    A nan reaches the answer of a sum or a mean on its own, a position cannot be
    poisoned: a group holding one has no extreme to name and no index to report,
    so it is handled like an empty group and takes the fill value.
    """

    def one_group_or_nothing(values: npt.NDArray[np.floating], pos: npt.NDArray[np.integer], **kwargs: Any) -> Any:
        if np.isnan(values).any():
            return None
        return one_group(values, pos, **kwargs)

    one_group_or_nothing.__name__ = f"poisoned_{one_group.__name__}"
    return one_group_or_nothing


def _as_numbers(values: npt.NDArray[np.floating]) -> np.ndarray:
    """the values as numbers to integrate over: numpy adds booleans as a
    logical or, which would make the trapezoid of a bool group nonsense"""
    return values.astype(np.float64) if values.dtype.kind == "b" else values


def _trapezoid_of(values: npt.NDArray[np.floating], pos: npt.NDArray[np.integer], dx: float = 1.0) -> Any:
    return trapezoid_ref(_as_numbers(values), dx=dx)


def _sum_of_squares(values: npt.NDArray[np.floating], pos: npt.NDArray[np.integer], skip_nan: bool = False) -> Any:
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


def _nunique_ref(values: npt.NDArray[np.number], pos: npt.NDArray[np.integer]) -> int:
    """the number of distinct values of one group.

    The values that are equal count once - a zero and a minus zero are one
    value, an integer and the float of the same number are one value.  The
    nans count once among themselves: a nan in either part of a complex value
    is a nan, and the nans of a group are one value of their own (pandas
    counts them that way, and so does polars, which counts the nan as a
    value).
    """
    if values.dtype.kind in "fc":
        nans = np.isnan(values)
        usable = values[~nans]
    else:
        nans = None
        usable = values
    distinct = len(np.unique(usable))
    if nans is not None and nans.any():
        distinct += 1
    return distinct


def _skew_ref(values: npt.NDArray[np.number], pos: npt.NDArray[np.integer]) -> float:
    """the skewness of the values of one group.

    The third central moment over the second one to the three-halves power,
    with the correction for the size of the group that pandas applies: the
    factor sqrt(n(n-1))/(n-2) makes the estimator unbiased for a normal
    distribution.  A group of fewer than three values has no skewness, and a
    group whose values are all the same has none either - its spread is zero,
    and the answer is zero as well.
    """
    n = len(values)
    if n < 3:
        return float("nan")
    devs = values - values.mean()
    s2 = float((devs**2).sum())
    if s2 == 0:
        return 0.0
    s3 = float((devs**3).sum())
    return n * (n - 1) ** 0.5 / (n - 2) * s3 / s2**1.5


def _kurt_ref(values: npt.NDArray[np.number], pos: npt.NDArray[np.integer]) -> float:
    """the excess kurtosis of the values of one group.

    The fourth central moment over the square of the second one, minus the
    three of a normal distribution, with the correction for the size of the
    group that pandas applies.  A group of fewer than four values has no
    kurtosis, and a group whose values are all the same has none either.
    """
    n = len(values)
    if n < 4:
        return float("nan")
    devs = values - values.mean()
    s2 = float((devs**2).sum())
    if s2 == 0:
        return 0.0
    s4 = float((devs**4).sum())
    return n * (n + 1) * (n - 1) * s4 / ((n - 2) * (n - 3) * s2**2) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


def _sort_ref(values: npt.NDArray[np.number], reverse: bool = False) -> npt.NDArray[np.integer]:
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


def _result_dtype(call: Callable[..., Any], in_dtype: np.dtype) -> np.dtype:
    """the dtype the operation yields for values of ``in_dtype``, asked of numpy.

    One value is enough to ask, which keeps the reference free of a dtype policy
    of its own - the one thing it must not copy from the implementation it is
    measured against.  It is also the only way to name the dtype of a result
    whose groups have all come back without an answer.
    """
    probe = np.ones(1, dtype=in_dtype)
    return np.asarray(call(probe, np.zeros(1, dtype=np.intp))).dtype


# the callables behind the string names of the aggregation functions, each
# taking the values of one group and the positions those values sit at in the
# input; a callable that returns None has no answer for the group, which is then
# handled like an empty one - the plain arg reductions over a group with a nan,
# and the nan counterparts over a group of nothing but nans
_REF_ONE_PER_GROUP: dict[str, Callable[..., Any]] = {
    "sum": _of(np.sum),
    "prod": _of(np.prod),
    "min": _of(np.min),
    "max": _of(np.max),
    "all": _of(np.all),
    "any": _of(np.any),
    "mean": _of(np.mean),
    "median": _of(np.median),
    "quantile": lambda v, pos, q=0.5, method="linear": _quantile_ref(v, q, method),
    "percentile": lambda v, pos, q=50.0, method="linear": _quantile_ref(v, q / 100.0, method),
    "std": lambda v, pos, ddof=0: np.std(v, ddof=ddof),
    "var": lambda v, pos, ddof=0: np.var(v, ddof=ddof),
    "trapezoid": _trapezoid_of,
    "sumofsquares": _sum_of_squares,
    "len": lambda v, pos: len(v),
    "nunique": _nunique_ref,
    "skew": _skew_ref,
    "kurt": _kurt_ref,
    "first": lambda v, pos: v[0],
    "last": lambda v, pos: v[-1],
    "anynan": lambda v, pos: bool(np.isnan(v).any()),
    "allnan": lambda v, pos: bool(np.isnan(v).all()),
    "argmin": _poisoned_by_a_nan(_at_a_position(np.argmin)),
    "argmax": _poisoned_by_a_nan(_at_a_position(np.argmax)),
    "nansum": _skip_nans(_of(np.sum)),
    "nanprod": _skip_nans(_of(np.prod)),
    "nanmin": _skip_nans(_of(np.min)),
    "nanmax": _skip_nans(_of(np.max)),
    "nanall": _skip_nans(_of(np.all)),
    "nanany": _skip_nans(_of(np.any)),
    "nanmean": _skip_nans(_of(np.mean)),
    "nanmedian": _skip_nans(_of(np.median)),
    "nanquantile": _skip_nans(lambda v, pos, q=0.5, method="linear": _quantile_ref(v, q, method)),
    "nanpercentile": _skip_nans(lambda v, pos, q=50.0, method="linear": _quantile_ref(v, q / 100.0, method)),
    "nanstd": _skip_nans(lambda v, pos, ddof=0: np.std(v, ddof=ddof)),
    "nanvar": _skip_nans(lambda v, pos, ddof=0: np.var(v, ddof=ddof)),
    "nantrapezoid": _skip_nans(_trapezoid_of),
    "nansumofsquares": _skip_nans(_sum_of_squares),
    "nanlen": _skip_nans(lambda v, pos: len(v)),
    "nannunique": _skip_nans(_nunique_ref),
    "nanskew": _skip_nans(_skew_ref),
    "nankurt": _skip_nans(_kurt_ref),
    "nanfirst": _skip_nans(lambda v, pos: v[0]),
    "nanlast": _skip_nans(lambda v, pos: v[-1]),
    "nanargmin": _skip_nans(_at_a_position(np.argmin)),
    "nanargmax": _skip_nans(_at_a_position(np.argmax)),
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

# the functions that collect the values of a group instead of reducing them:
# their result is a sequence per group, and the object array of them is built
# element by element
_REF_COLLECTING: dict[str, Callable[..., Any]] = {
    "array": lambda v, pos: v,
}
_REF_FUNCS.update(_REF_COLLECTING)


def reference_names() -> list[str]:
    """The names of the functions the group loop computes.

    The yardstick of the suite knows a function per group and a function per
    input item - the sweeps over the functions of the package ask the ones it
    knows.
    """
    return sorted(_REF_FUNCS)


def aggregate_ref(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    *,
    func: str | Callable[..., Any],
    size: int | tuple[int, ...] | None = None,
    fill_value: Any = DEFAULT_FILL_VALUE,
    dtype: np.dtype | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.number]:
    """``aggregate(group_idx, a, func=func, ...)`` computed one group at a time.

    Accepts the name of a function or the function itself - a numpy function
    stands for the name it carries, which is how the callers of the package
    spell it - one- or multi-dimensional labels and the keyword arguments the
    reference can express at all; anything else raises instead of quietly
    returning something else.  An absent group takes ``fill_value``, or the
    function default that ``numpy_groupies`` itself uses, and a fill value that
    does not fit the result dtype is refused the same way the package refuses
    it.
    """
    group_idx = np.asarray(group_idx)
    a = np.asarray(a)
    name = func if isinstance(func, str) else getattr(func, "__name__", None)
    if name not in _REF_FUNCS:
        raise NotImplementedError(f"aggregate_ref does not know {func!r}")
    if group_idx.ndim > 1 and size is None:
        raise NotImplementedError("aggregate_ref needs a size for multi-dimensional labels")

    out_shape: tuple[int, ...]
    if size is None:
        n_groups = int(group_idx.max()) + 1 if group_idx.size else 0
        out_shape = (n_groups,)
    elif isinstance(size, (int, np.integer)):
        n_groups = int(size)
        out_shape = (n_groups,)
    else:
        out_shape = tuple(int(dim) for dim in size)
        n_groups = int(np.prod(out_shape))
    labels = group_idx if group_idx.ndim <= 1 else np.ravel_multi_index(group_idx, out_shape)
    if labels.size and int(labels.max()) >= n_groups:
        raise ValueError("a group index is too large for the requested size")

    call = _REF_FUNCS[name]
    # one stable argsort gathers the groups, each of them in the order its
    # values had in the input - the arg reductions report an input position
    order = np.argsort(labels, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(labels, minlength=n_groups))]
    positions = {}
    results = {}
    for group in range(n_groups):
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

    if name in _REF_COLLECTING:
        # the result is a sequence per group, so the object array of them is
        # built element by element - a group that received no value takes the
        # fill value, which is an empty sequence by default
        out_dtype = dtype or np.dtype(object)
        if fill_value is DEFAULT_FILL_VALUE:
            fill = default_fill_value(name, out_dtype)
        else:
            check_fill_value(fill_value, out_dtype, name)
            fill = fill_value
        ret = np.empty(out_shape, dtype=object)
        flat = ret.reshape(-1)
        for group in range(n_groups):
            flat[group] = fill
        for group, value in results.items():
            flat[group] = value
        return ret

    if name in _REF_ONE_PER_ITEM:
        # the result holds one slot per input item, so a group that is not in
        # the labels simply has no slot - nothing to fill, nothing that could
        # be left uninitialized.  The cumulative functions follow numpy's own
        # promotion, the sort keeps the dtype of what it was given.
        out_dtype = dtype or np.asarray(next(iter(results.values()))).dtype
        ret = np.empty(a.shape, dtype=out_dtype)
        for group, values in results.items():
            ret[positions[group]] = values
        return ret

    out_dtype = dtype or _result_dtype(call, a.dtype)
    if fill_value is DEFAULT_FILL_VALUE:
        fill = default_fill_value(name, out_dtype)
    else:
        check_fill_value(fill_value, out_dtype, name)
        fill = fill_value
    ret = np.full(out_shape, fill, dtype=out_dtype)
    for group, value in results.items():
        ret.flat[group] = value
    return ret
