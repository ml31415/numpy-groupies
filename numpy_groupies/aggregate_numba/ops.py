"""The python-level wrappers of the numba implementation, and their registry.

``create_aggregate_op`` builds the python-level wrapper of one operation: it
runs the validation and the output-dtype/fill-value resolution, then hands
everything to the compiled kernel driver of the operation (see ``compose``).
There is no per-operation machinery left - plain 1d input runs entirely
inside the driver, everything else (2d/axis, order, out-of-range sizes,
unravelling for the argreductions) is answered on the slow path.  The
registry at the bottom of this module creates a wrapper for every
registered operation, so that ``aggregate()`` resolves with a single lookup.
"""

import functools
import typing

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_common import (
    DEFAULT_FILL_VALUE,
    build_dispatch,
    funcs_no_separate_nan,
    is_scalar_input,
    resolve_fill_value,
)
from numpy_groupies.aggregate_numba import argreductions, averaging, cumulative, reductions, specials
from numpy_groupies.aggregate_numba.compose import KernelHandle
from numpy_groupies.aggregate_numba.steps import step_indices
from numpy_groupies.utils import (
    aliasing,
    check_dtype,
    check_fill_value,
    check_nton_shape,
    input_validation,
    resolve_output_dtype,
    scalar_dtype,
)

# input dtypes for which the 'sum' output dtype depends on the input length
# (the overflow guesses in resolve_output_dtype)
_N_DEPENDENT_ADTYPES = frozenset(
    np.dtype(t) for t in ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "uint64")
)


@functools.lru_cache(maxsize=256)
def _dtype_fill_plan(
    func: str,
    a_dtype: np.dtype,
    dtype: npt.DTypeLike,
    fill_value: typing.Any,
    n: int,
    scalar: bool,
) -> tuple[np.dtype, typing.Any]:
    """Memoized output-dtype and fill-value resolution.

    ``a_dtype`` is the dtype of the aggregated data, ``scalar`` whether that
    data is a single value - the callers resolve both with ``scalar_dtype``.
    ``n`` carries ``len(group_idx)`` only for the length-dependent ``sum``
    dtype guesses and is 0 otherwise.  Unhashable arguments fail the cache
    lookup with a TypeError, which the caller answers with the uncached
    resolution.
    """
    if scalar and func not in ("sum", "prod", "len"):
        raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
    out_dtype = resolve_output_dtype(dtype, func, a_dtype, n)
    if func == "sort":
        # sort writes the input values back unchanged - the numpy twin
        # allocates ``np.empty_like(a)`` and the pure python one rebuilds the
        # list, so both hand out the input dtype whatever was requested, and
        # the compiled rule of the kernel (``rule_same``) says the same
        out_dtype = a_dtype
    fill_value = resolve_fill_value(func, fill_value, out_dtype)
    check_fill_value(fill_value, out_dtype, func=func)
    return out_dtype, fill_value


def _aux_dtype(handle: KernelHandle, a_dtype: np.dtype, out_dtype: np.dtype) -> np.dtype:
    """The auxiliary (widened accumulation) dtype of the dispersion kernels."""
    if handle.aux_widen:
        return np.result_type(a_dtype, np.float64)
    return out_dtype


def build_aggregate_op(func: str, handle: KernelHandle) -> typing.Callable[..., typing.Any]:
    """Create the python-level wrapper around one compiled kernel.

    The wrapper holds no kernel machinery of its own: it resolves dtype and
    fill value (memoized for the plain 1d fast path), materializes scalar
    input and delegates the actual aggregation to the jitted driver.  Input
    dtypes that the compiled driver cannot handle go to the operation's
    python-level ``fallback`` (complex medians to the numpy implementation).
    """

    def aggregate(
        group_idx: npt.ArrayLike,
        a: npt.ArrayLike,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: typing.Any = DEFAULT_FILL_VALUE,
        order: str = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        ddof: float = 0,
    ) -> npt.NDArray[np.number]:
        # dtypes the compiled driver cannot handle, handed to the numpy twin
        if handle.fallback is not None and getattr(getattr(a, "dtype", None), "kind", "") == "c":
            return typing.cast(
                "npt.NDArray[np.number]",
                handle.fallback(group_idx, a, func, size, fill_value, order, dtype, axis),
            )
        # fast path: plain 1d-in/1d-out - the jitted driver does the bounds
        # check, size detection, allocation and the kernel in one compiled
        # call, everything else falls through to the full validation machinery
        if (
            # the cumulatives emit one value per input item and derive the
            # group count from the indices when no size is given, so they
            # take the fast path as well - without it they pay a second
            # max scan plus the validation machinery (~0.09ms per 500k)
            (size is None or (type(size) is int and not handle.nton))
            and axis is None
            and type(group_idx) is np.ndarray
            and group_idx.ndim == 1
            and group_idx.dtype.kind in "iu"
            and group_idx.size > 0
            and type(a) is np.ndarray
            and a.ndim == 1
            and a.shape[0] == group_idx.size
        ):
            a_key = a.dtype
            n_key = len(group_idx) if (func == "sum" and a_key in _N_DEPENDENT_ADTYPES) else 0
            try:
                # a dtype that is no key of the cache raises there, which is caught
                plan_dtype, plan_fill = _dtype_fill_plan(
                    func, a_key, typing.cast("typing.Any", dtype), fill_value, n_key, False
                )
            except TypeError:
                # unhashable dtype/fill_value - resolve on the slow path
                plan_dtype = None
            if plan_dtype is not None:
                return typing.cast(
                    "npt.NDArray[np.number]",
                    handle.driver(
                        group_idx,
                        a,
                        -1 if size is None else size,
                        plan_dtype,
                        _aux_dtype(handle, a_key, plan_dtype),
                        plan_fill,
                        ddof,
                    ),
                )

        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False, func=func)
        group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

        # TODO: The typecheck should be done by the wrapper itself, not by check_dtype
        scalar = is_scalar_input(a)
        # a scalar reaches the compiled driver as a full column of that one
        # value, so its dtype is all that is left to resolve the plan from -
        # for a python number the type stands in, see scalar_dtype
        a_key = scalar_dtype(a) if scalar else typing.cast("npt.NDArray[np.number]", a).dtype
        n_key = len(group_idx) if (func == "sum" and a_key in _N_DEPENDENT_ADTYPES) else 0
        try:
            # a dtype that is no key of the cache raises there, which is caught
            dtype, fill_value = _dtype_fill_plan(
                func, a_key, typing.cast("typing.Any", dtype), fill_value, n_key, scalar
            )
        except TypeError:
            # unhashable dtype/fill_value - resolve without the cache
            dtype = check_dtype(dtype, func, a, len(group_idx))
            fill_value = resolve_fill_value(func, fill_value, dtype)
            check_fill_value(fill_value, dtype, func=func)
        if scalar:
            # scalar input is only supported by sum/prod/len (the plan raises
            # for everything else) - materialize it for the jitted driver
            a = np.full(len(group_idx), a)
        group_idx = np.ascontiguousarray(group_idx)
        a = np.ascontiguousarray(a)

        ret = typing.cast(
            "npt.NDArray[np.number]",
            handle.driver(group_idx, a, flat_size, dtype, _aux_dtype(handle, a_key, dtype), fill_value, ddof),
        )

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            if unravel_shape is not None:
                # argreductions only
                mask = ret == fill_value
                ret[mask] = 0
                ret = typing.cast(
                    "npt.NDArray[np.number]",
                    np.unravel_index(typing.cast("npt.NDArray[np.integer]", ret), unravel_shape)[
                        typing.cast("int", axis)
                    ],
                )
                ret[mask] = fill_value
            check_nton_shape(ret, size, func)
            ret = typing.cast(
                "npt.NDArray[np.number]",
                ret.reshape(typing.cast("typing.Any", size), order=typing.cast("typing.Any", order)),
            )
        return ret

    return aggregate


def _compiled_group_loop(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Compile a user callable and wrap the per-group loop around it."""
    nb_func = nb.njit(func)

    @nb.njit
    def loop(sortidx: typing.Any, group_idx: typing.Any, a: npt.NDArray[np.number], ret: typing.Any) -> None:
        size = len(ret)
        group_idx_srt = group_idx[sortidx]
        a_srt = a[sortidx]

        indices = step_indices(group_idx_srt)
        for i in range(len(indices) - 1):
            start_idx, stop_idx = indices[i], indices[i + 1]
            ri = group_idx_srt[start_idx]
            if ri < 0:
                raise ValueError("negative indices not supported")
            if ri >= size:
                raise ValueError("one or more indices in group_idx are too large")
            ret[ri] = nb_func(a_srt[start_idx:stop_idx])

    return loop


def build_generic_aggregate(func: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Create the aggregation loop for an arbitrary user callable.

    The callable is applied to the values of every group, the data is
    sorted by group index first.  No on-disk caching here: the captured user
    callable has no stable disk-cache key (and potentially an unstable repr),
    so numba's on-disk cache could neither be relied upon to hit, nor be
    trusted against false hits.
    """
    jitfunc = _compiled_group_loop(func)

    def aggregate(
        group_idx: npt.ArrayLike,
        a: npt.ArrayLike,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: typing.Any = DEFAULT_FILL_VALUE,
        order: str = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        ddof: float = 0,
    ) -> npt.NDArray[np.number]:
        # a func reaching this loop is no key of the dispatch table, so it names
        # no function and cannot be an arg reduction or a one-to-one answer
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False, func=None)
        group_idx, a, flat_size, ndim_idx, size, _ = iv

        # TODO: The typecheck should be done by the wrapper itself, not by check_dtype
        dtype = check_dtype(dtype, typing.cast("typing.Any", func), typing.cast("typing.Any", a), len(group_idx))
        fill_value = resolve_fill_value(func, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=typing.cast("typing.Any", func))
        ret = np.full(flat_size, fill_value, dtype=dtype)
        group_idx = np.ascontiguousarray(group_idx)

        sortidx = np.argsort(group_idx, kind="stable")
        jitfunc(sortidx, group_idx, a, ret)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            check_nton_shape(ret, size, func)
            ret = typing.cast(
                "npt.NDArray[np.number]",
                ret.reshape(typing.cast("typing.Any", size), order=typing.cast("typing.Any", order)),
            )
        return ret

    return aggregate


# --------------------------------------------------------------- registry
#
# (name, kernel pair) - the order is the registration order
# the pairs are reached through their modules - binding names like 'len',
# 'sum' or 'min' into this module would shadow the builtins used above
_TABLE = (
    ("sum", reductions.sum),
    ("prod", reductions.prod),
    ("len", reductions.len),
    ("all", reductions.all),
    ("any", reductions.any),
    ("last", reductions.last),
    ("first", reductions.first),
    ("allnan", reductions.allnan),
    ("anynan", reductions.anynan),
    ("min", reductions.min),
    ("max", reductions.max),
    ("argmin", argreductions.argmin),
    ("argmax", argreductions.argmax),
    ("mean", averaging.mean),
    ("median", specials.median),
    ("trapezoid", specials.trapezoid),
    ("std", averaging.std),
    ("var", averaging.var),
    ("sumofsquares", reductions.sumofsquares),
    ("cumsum", cumulative.cumsum),
    ("cumprod", cumulative.cumprod),
    ("cummax", cumulative.cummax),
    ("cummin", cumulative.cummin),
    ("sort", specials.sort),
)


def _dx_argument(op: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Expose the trapezoid's sample spacing under its public name ``dx``.

    The kernels receive it in the position otherwise used for ``ddof``.
    """

    def trapezoid_op(
        group_idx: npt.ArrayLike,
        a: npt.ArrayLike,
        size: int | tuple[int, ...] | None = None,
        fill_value: typing.Any = DEFAULT_FILL_VALUE,
        order: str = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        dx: float = 1.0,
    ) -> npt.NDArray[np.number]:
        return typing.cast("npt.NDArray[np.number]", op(group_idx, a, size, fill_value, order, dtype, axis, dx))

    return trapezoid_op


def _sort_arguments(op: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Expose ``reverse`` and keep the driver arguments jittable.

    The flag reaches the kernel in the position otherwise used for ``ddof``,
    and the empty list the shared fill value resolution hands out for ``sort``
    - it tells the python twins to leave an absent group alone - becomes the
    scalar the compiled driver expects, since sort emits one value per input
    item and never fills a hole.
    """

    def sort_op(
        group_idx: npt.ArrayLike,
        a: npt.ArrayLike,
        size: int | tuple[int, ...] | None = None,
        fill_value: typing.Any = DEFAULT_FILL_VALUE,
        order: str = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        reverse: bool = False,
    ) -> npt.NDArray[np.number]:
        if fill_value is DEFAULT_FILL_VALUE:
            fill_value = 0
        return typing.cast(
            "npt.NDArray[np.number]", op(group_idx, a, size, fill_value, order, dtype, axis, 1.0 if reverse else 0.0)
        )

    return sort_op


# operations whose public keyword argument is not the plain ddof of the kernel
_ARGUMENT_WRAPPERS = {"trapezoid": _dx_argument, "sort": _sort_arguments}


def get_funcs() -> dict[str, typing.Callable[..., typing.Any]]:
    """Create the wrapper function of every registered operation variant."""
    funcs: dict[str, typing.Callable[..., typing.Any]] = {}
    for funcname, pair in _TABLE:
        wrap = _ARGUMENT_WRAPPERS.get(funcname)
        op = build_aggregate_op(funcname, pair.plain)
        funcs[funcname] = op if wrap is None else wrap(op)
        if funcname not in funcs_no_separate_nan:
            op = build_aggregate_op("nan" + funcname, pair.nan)  # type: ignore[arg-type]
            funcs["nan" + funcname] = op if wrap is None else wrap(op)
    return funcs


_impl_dict = get_funcs()

# alias (string or callable) -> (canonical name, wrapper function), every alias
# of every known function and so one dict lookup on the call path - names this
# backend has no wrapper for come along holding None, so the lookup answers that
# question too instead of a second table having to be asked
_dispatch = build_dispatch(_impl_dict, aliasing)
