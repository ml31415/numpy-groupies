"""The polars participant of the benchmark: its group_by machinery.

Like the pandas one, this is a reference implementation for the benchmark, not
a backend of the package: it exists to have something to compare the numbers
of the real implementations with.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from numpy_groupies.aggregate_common import (
    DEFAULT_FILL_VALUE,
    Registry,
    Resolved,
    aggregate_common_doc,
    custom_callable,
    funcs_no_separate_nan,
)
from numpy_groupies.aggregate_numpy import _aggregate_base
from numpy_groupies.aggregate_numpy import aggregate as _aggregate_numpy
from numpy_groupies.utils import (
    RavelOrder,
    aliasing,
    check_dtype,
    iscomplexobj,
)

# the expressions of the group_by machinery, per function name
_EXPRESSIONS: dict[str, Callable[[], pl.Expr]] = {
    "sum": lambda: pl.col("a").sum(),
    "prod": lambda: pl.col("a").product(),
    "min": lambda: pl.col("a").min(),
    "max": lambda: pl.col("a").max(),
    "mean": lambda: pl.col("a").mean(),
    "median": lambda: pl.col("a").median(),
    "var": lambda: pl.col("a").var(),
    "std": lambda: pl.col("a").std(),
    "first": lambda: pl.col("a").first(),
    "last": lambda: pl.col("a").last(),
    "cumsum": lambda: pl.col("a").cum_sum(),
    "cumprod": lambda: pl.col("a").cum_prod(),
    "cummax": lambda: pl.col("a").cum_max(),
    "cummin": lambda: pl.col("a").cum_min(),
    "all": lambda: pl.col("a").all(),
    "any": lambda: pl.col("a").any(),
    # the length of a group is the number of its rows
    "len": lambda: pl.col("a").len(),
    # polars has no nan-aware reductions: the nans are values there, so the
    # nan-functions ask the mask of them
    "anynan": lambda: pl.col("a").is_nan().any(),
    "allnan": lambda: pl.col("a").is_nan().all(),
    # the position of the extreme in the input, not in the group: the position
    # is a column of its own and the index of the extreme gathers from it
    # (the gather yields a list, of which the wrapper takes the one element)
    "argmax": lambda: pl.col("i").gather(pl.col("a").arg_max()).first(),
    "argmin": lambda: pl.col("i").gather(pl.col("a").arg_min()).first(),
}

# the functions whose result polars reports per group, and that the group
# holding a nan poisons: polars skips the nans in the extremes and in the
# median, while the package counts a nan as a value that makes the whole group
# answer with the nan
_POISONED_BY_NAN = frozenset(["min", "max", "median", "argmax", "argmin"])


def _expression(funcname: str, ddof: int) -> pl.Expr:
    """The expression of the group_by machinery for one function.

    The dispersion functions take the degrees of freedom, the others do not.
    """
    if funcname == "var":
        return pl.col("a").var(ddof=ddof)
    if funcname == "std":
        return pl.col("a").std(ddof=ddof)
    return _EXPRESSIONS[funcname]()


def _wrapper(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    size: int | tuple[int, ...] | None,
    fill_value: float,
    func: str | Callable[..., Any] = "sum",
    dtype: np.dtype | None = None,
    ddof: int = 0,
    **kwargs: Any,
) -> npt.NDArray[np.number]:
    if len(group_idx) == 0:
        raise ValueError("group_idx must not be empty")
    # scalar input needs broadcasting before anything group-based can run
    a = a if np.ndim(a) else np.broadcast_to(a, len(group_idx))
    funcname = func.__name__ if callable(func) else func
    if funcname == "nancumsum":
        # the nans count as zero in the nan-version, while the plain version
        # propagates them (issues #79 and #91)
        a = np.where(np.isnan(a), 0, a)
        func = "cumsum"
        funcname = "cumsum"
    # the nan-version of a function is the plain one over the values without
    # their nans - the filter of _aggregate_base takes care of that, except
    # for the arg-reductions, which mask the nans instead of dropping them
    is_nan_version = funcname.startswith("nan")
    plain = funcname[3:] if is_nan_version else funcname

    if callable(func):
        # the callable gets the whole group, not its elements: implode() makes
        # a list of them, and the return type has to be told, since polars
        # cannot infer it from the callable
        expression = pl.col("a").implode().map_elements(func, return_dtype=pl.Float64)
    else:
        expression = _expression(plain, ddof)
    # the position of every value in the input, which the arg-reductions name
    frame = pl.DataFrame({"g": group_idx, "a": a, "i": np.arange(len(group_idx))})
    if plain.startswith("cum"):
        # the cumulative functions emit one value per input item, so they are
        # a window over the labels and not an aggregation of them - the window
        # keeps the order of the input
        result = frame.with_columns(expression.over("g").alias("v"))
        return result["v"].to_numpy()
    grouped = frame.group_by("g", maintain_order=True)
    result = grouped.agg(expression.alias("v"))

    # the sum overflow guess needs the maximum number of values that can be
    # summed into a single group, i.e. the input length - not the output size
    dtype = check_dtype(dtype, plain, a, len(group_idx))
    ret = np.full(size, fill_value, dtype=dtype)
    ret[result["g"].to_numpy()] = result["v"].to_numpy()
    if plain in _POISONED_BY_NAN and not is_nan_version and a.dtype.kind in "fc":
        # polars skips the nans in the extremes, the package does not: a group
        # holding one has no extreme at all
        poisoned = np.bincount(group_idx, weights=np.isnan(a), minlength=size) > 0
        if plain in ("argmax", "argmin"):
            # the position of an extreme that does not exist
            ret[poisoned] = fill_value
        else:
            ret[poisoned] = np.nan
    return ret


_supported_funcs = [
    "sum",
    "prod",
    "all",
    "any",
    "min",
    "max",
    "mean",
    "median",
    "var",
    "std",
    "first",
    "last",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "len",
    "anynan",
    "allnan",
    "argmax",
    "argmin",
]
_op_dict: dict[str, Callable[..., Any]] = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_op_dict.update(
    ("nan" + fn, partial(_wrapper, func="nan" + fn)) for fn in _supported_funcs if fn not in funcs_no_separate_nan
)
# the nan-versions of the extremes are the plain ones over the values without
# their nans - the filter of _aggregate_base takes care of that
_op_dict["nancumsum"] = partial(_wrapper, func="nancumsum")


def _custom_operation(func: Any) -> Resolved:
    """The operation of a callable of the caller's own.

    polars' group_by machinery takes any callable, so the wrapper of this
    backend runs it.
    """
    callable_ = custom_callable(func)
    return Resolved(callable_, partial(_wrapper, func=callable_))


_dispatch = Registry(_op_dict, aliasing, _custom_operation)


def aggregate(
    group_idx: npt.NDArray[np.integer],
    a: npt.NDArray[np.number],
    func: str | Callable[..., Any] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: Any = DEFAULT_FILL_VALUE,
    order: RavelOrder = "C",
    dtype: npt.DTypeLike = None,
    axis: int | None = None,
    **kwargs: Any,
) -> Any:
    if iscomplexobj(a):
        # polars' group_by kernels have no signatures for complex dtypes - the
        # numpy implementation handles complex values, scalars included
        return _aggregate_numpy(
            group_idx,
            a,
            func=func,
            size=size,
            fill_value=fill_value,
            order=order,
            dtype=dtype,
            axis=axis,
            **kwargs,
        )
    return _aggregate_base(
        group_idx,
        a,
        size=size,
        fill_value=fill_value,
        order=order,
        dtype=dtype,
        func=func,
        axis=axis,
        _dispatch=_dispatch,
        len_counts_values=True,
        **kwargs,
    )


aggregate.__doc__ = (
    """
    This is the polars implementation of aggregate. It makes use of
    `polars`' group_by machinery and is mainly used for reference
    and benchmarking.
    """
    + aggregate_common_doc
)
