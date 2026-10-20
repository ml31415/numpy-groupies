"""The polars participant of the benchmark: its group_by machinery.

Like the pandas one, this is a reference implementation for the benchmark, not
a backend of the package: it exists to have something to compare the numbers
of the real implementations with.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, cast

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
    AFillValue,
    AResult,
    AScalar,
    RavelOrder,
    aliasing,
    check_dtype,
    iscomplexobj,
)

# the engines of a lazy query, the ones polars knows since 1.29
Engine = Literal["in-memory", "streaming"]

# the expressions of the group_by machinery, per function name
_EXPRESSIONS: dict[str, Callable[[], pl.Expr]] = {
    "sum": lambda: pl.col("a").sum(),
    "prod": lambda: pl.col("a").product(),
    "min": lambda: pl.col("a").min(),
    "max": lambda: pl.col("a").max(),
    "mean": lambda: pl.col("a").mean(),
    "median": lambda: pl.col("a").median(),
    # polars interpolates the quantile to the nearest value by default, while
    # numpy - and the package - interpolate it linearly
    "quantile": lambda: pl.col("a").quantile(0.5, interpolation="linear"),
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
    # is a column of its own and the index of the extreme picks from it (the
    # index is that of the group, so the position is fetched from it)
    "argmax": lambda: pl.col("i").get(pl.col("a").arg_max()),
    "argmin": lambda: pl.col("i").get(pl.col("a").arg_min()),
}

# the functions whose result polars reports per group, and that the group
# holding a nan poisons: polars skips the nans in the extremes and in the
# median, while the package counts a nan as a value that makes the whole group
# answer with the nan
_POISONED_BY_NAN = frozenset(["min", "max", "median", "quantile", "argmax", "argmin"])

# the reductions whose aggregation is faster in the in-memory engine than in
# the streaming one: the product multiplies over the whole group and the
# median sorts it, and the streaming engine keeps neither in one piece
# (measured 6.5ms against 4.8ms for the product, 4.4ms against 4.0ms for the
# median, of 500k values in 1000 groups)
_IN_MEMORY = frozenset(["prod", "median", "quantile"])

# the positions of the values in the input, per length: the arg-reductions
# name them, and building them anew for every call costs 4MB of memory and
# 0.3ms of a cell of 4.6ms
_POSITIONS: dict[int, npt.NDArray[np.intp]] = {}


def _positions(size: int) -> npt.NDArray[np.intp]:
    """The position of every value in the input, cached per length."""
    positions = _POSITIONS.get(size)
    if positions is None:
        if len(_POSITIONS) > 8:
            _POSITIONS.clear()
        positions = np.arange(size)
        _POSITIONS[size] = positions
    return positions


def _knows_engines() -> bool:
    """Whether the polars in use can pick the engine of a query.

    The argument came with polars 1.29; before that the query knew a
    ``streaming`` flag, and before polars 1.0 there was no streaming at all.
    """
    try:
        pl.LazyFrame({"a": [1.0]}).collect(engine="in-memory")
    except TypeError:
        return False
    return True


_HAS_ENGINES = _knows_engines()


def _engine(plain: str) -> Engine:
    """The engine that gathers the aggregation of a function fastest."""
    return "in-memory" if plain in _IN_MEMORY else "streaming"


def _collect(query: pl.LazyFrame, engine: Engine) -> pl.DataFrame:
    """The result of a lazy query, gathered by the engine that suits it.

    The lazy frame is the faster way into the group_by machinery: the eager
    one collects its query with the in-memory engine, which measured 3.7ms for
    a sum where the streaming engine needs 2.3ms.
    """
    if _HAS_ENGINES:
        return query.collect(engine=engine)
    # polars before 1.29 knows no engine, but a streaming flag - and before
    # polars 1.0 there was no streaming at all
    collect: Any = query.collect
    try:
        return cast("pl.DataFrame", collect(streaming=engine == "streaming"))
    except TypeError:
        return cast("pl.DataFrame", collect())


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
    func: str | Callable[..., AScalar] = "sum",
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
    # the columns of the query: the labels and the values, and the position of
    # every value in the input - only the arg-reductions name it, so only they
    # get the column.  A sum does not read positions, and the 4MB of them
    # (measured 0.3ms of a 4.6ms cell, 2.2ms in the first calls of a process,
    # where their pages still have to arrive) are theirs to pay for
    columns: dict[str, pl.Series] = {
        "g": pl.Series("g", group_idx),
        "a": pl.Series("a", a),
    }
    if plain.startswith("arg"):
        columns["i"] = pl.Series("i", _positions(len(group_idx)))
    query = pl.LazyFrame(columns)
    if plain.startswith("cum"):
        # the cumulative functions emit one value per input item, so they are
        # a window over the labels and not an aggregation of them - the window
        # keeps the order of the input
        window = query.with_columns(expression.over("g").alias("v"))
        # the in-memory engine, which keeps the order of the rows
        return _collect(window, "in-memory")["v"].to_numpy()
    # the groups holding a nan have no extreme in the package, where the nan is
    # a value that makes the whole group answer with the nan - polars asks for
    # the flag of the group in the same aggregation, which is cheaper than a
    # pass over the whole input in numpy (0.4ms of a cell of 4.6ms)
    poisoned = plain in _POISONED_BY_NAN and not is_nan_version and a.dtype.kind in "fc"
    # the groups come in no order of their own: the result is scattered into
    # the output by the labels, so the order of the groups does not matter -
    # and not keeping it is faster (2.2ms against 3.1ms for a sum)
    grouped = query.group_by("g", maintain_order=False)
    if poisoned:
        aggregated = grouped.agg(expression.alias("v"), pl.col("a").is_nan().any().alias("n"))
    else:
        aggregated = grouped.agg(expression.alias("v"))
    result = _collect(aggregated, _engine(plain))

    # the dtype of the output: the base has resolved it for the known functions
    # already, only a callable of the caller's own comes without one
    if dtype is None:
        dtype = check_dtype(dtype, plain, a, len(group_idx))
    ret = np.full(size, fill_value, dtype=dtype)
    labels = result["g"].to_numpy()
    ret[labels] = result["v"].to_numpy()
    if poisoned:
        # the groups holding a nan: polars skips the nans in the extremes, the
        # package does not, so such a group has no extreme at all
        bad = labels[result["n"].to_numpy()]
        if plain in ("argmax", "argmin"):
            # the position of an extreme that does not exist
            ret[bad] = fill_value
        else:
            ret[bad] = np.nan
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
    "quantile",
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
_op_dict: dict[str, Callable[..., AResult]] = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_op_dict.update(
    ("nan" + fn, partial(_wrapper, func="nan" + fn)) for fn in _supported_funcs if fn not in funcs_no_separate_nan
)
# the nan-versions of the extremes are the plain ones over the values without
# their nans - the filter of _aggregate_base takes care of that
_op_dict["nancumsum"] = partial(_wrapper, func="nancumsum")


def _custom_operation(func: Callable[..., AScalar]) -> Resolved:
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
    func: str | Callable[..., AScalar] = "sum",
    size: int | Sequence[int] | None = None,
    fill_value: AFillValue = DEFAULT_FILL_VALUE,
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
