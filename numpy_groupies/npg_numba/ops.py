"""The python-level wrappers of the numba implementation, and their registry.

``build_aggregate_op`` wraps one operation: it runs the validation and the
output-dtype/fill-value resolution, then hands everything to the compiled
driver.  Plain 1d input runs entirely inside the driver, everything else
(2d/axis, order, out-of-range sizes, unravelling) takes the slow path; the
registry at the bottom builds a wrapper per registered operation, so that
``aggregate()`` resolves with a single lookup.
"""

import functools
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, TypeAlias, cast

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.driver import aliasing
from numpy_groupies.dtypes import (
    check_dtype,
    resolve_output_dtype,
    scalar_dtype,
)
from numpy_groupies.fills import (
    DEFAULT_FILL_VALUE,
    DefaultFillMarker,
    resolve_fill_value,
)
from numpy_groupies.functions import QUANTILE_METHODS, func_extras
from numpy_groupies.inputs import (
    check_nton_shape,
    input_validation,
)
from numpy_groupies.npg_numba import argreductions, averaging, collect, cumulative, reductions, specials
from numpy_groupies.npg_numba.compose import KernelHandle, _round_float16
from numpy_groupies.npg_numba.mt import (
    GROUP_MT_MIN_VALUES,
    group_chunk_count,
    mt_thread_count,
    should_use_group_mt,
)
from numpy_groupies.npg_numba.steps import group_count, group_place, group_ranges, group_starts
from numpy_groupies.registry import (
    Registry,
    Resolved,
    can_name_a_function,
    custom_callable,
    funcs_no_separate_nan,
    function_names,
    resolve_name,
)
from numpy_groupies.temporal import build_temporal_op
from numpy_groupies.typing import AFillValue, AResult, AScalar, AValue, RavelOrder
from numpy_groupies.validation import (
    check_fill_value,
    check_topk_input,
    check_topk_k,
    check_topk_shape,
    is_scalar_input,
)

# input dtypes whose 'sum' output dtype depends on the input length (the
# overflow guesses in resolve_output_dtype)
_N_DEPENDENT_ADTYPES = frozenset(
    np.dtype(t) for t in ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "uint64")
)

# functions whose answer dtype climbs with the number of the values, like
# the sum
_N_DEPENDENT_FUNCS = frozenset(("sum", "nansum", "cumsum", "nancumsum"))

# the dtypes and fill values that can be a key of the dtype plan cache:
# the hashable ones (a list is not, the cache raises a TypeError for it)
_KeyedDType: TypeAlias = str | type | np.dtype | None
_HashableFillValue: TypeAlias = AScalar | None | tuple[AScalar, ...] | DefaultFillMarker


# the cost per value of a callable of the caller's own is unknown - the
# median's number stands in: a cheaper callable is dominated by the counting
# sort anyway, a costlier one only makes the parallel path pay off sooner
_NS_PER_VALUE_CUSTOM = 18.0


@functools.lru_cache(maxsize=256)
def _dtype_fill_plan_cached(
    func: str,
    a_dtype: np.dtype,
    dtype: _KeyedDType,
    fill_value: _HashableFillValue,
    n: int,
    scalar: bool,
) -> tuple[np.dtype, AFillValue]:
    if scalar and func not in ("sum", "prod", "len"):
        raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
    out_dtype = resolve_output_dtype(dtype, func, a_dtype, n)
    if func in ("sort", "topk"):
        # sort and topk hand out the values themselves - the input dtype,
        # whatever was requested (the numpy and the pure python twins do the
        # same, and the kernel's ``rule_same`` says so)
        out_dtype = a_dtype
    resolved = resolve_fill_value(func, fill_value, out_dtype)
    check_fill_value(resolved, out_dtype, func=func)
    return out_dtype, resolved


def _dtype_fill_plan(
    func: str,
    a_dtype: np.dtype,
    dtype: npt.DTypeLike | None,
    fill_value: AFillValue,
    n: int,
    scalar: bool,
) -> tuple[np.dtype, AFillValue]:
    """Memoized output-dtype and fill-value resolution.

    ``n`` carries ``len(group_idx)`` only for the length-dependent ``sum``
    dtype guesses; unhashable arguments fail the cache lookup with a
    TypeError, which the caller answers with the uncached resolution.
    """
    return _dtype_fill_plan_cached(
        func, a_dtype, cast("_KeyedDType", dtype), cast("_HashableFillValue", fill_value), n, scalar
    )


def _aux_dtype(handle: KernelHandle, a_dtype: np.dtype, out_dtype: np.dtype) -> np.dtype:
    """The auxiliary (widened accumulation) dtype of the dispersion kernels."""
    if handle.aux_widen:
        return np.result_type(a_dtype, np.float64)
    return out_dtype


def _needs_fallback(handle: KernelHandle, a: AValue) -> bool:
    """Whether the input is one the compiled driver cannot handle - the
    complex input by default, which the median cannot partition; an operation
    that knows more says so with the ``placeable`` of its handle."""
    if handle.placeable is not None:
        return not handle.placeable(getattr(a, "dtype", None))
    return getattr(getattr(a, "dtype", None), "kind", "") == "c"


def _driver_fill(handle: KernelHandle, fill_value: AFillValue) -> AScalar:
    """The fill value the driver takes - an assembling driver gets a
    placeholder instead, the fill value is the business of its post hook."""
    # only the collecting function fills with a sequence, and it assembles
    return 0.0 if handle.assembles else cast("AScalar", fill_value)


def _driver_call(handle: KernelHandle) -> Callable[..., Any]:
    """Call the compiled driver of one variant with the arguments of a
    python-level call.

    The argument the operation has of its own travels last, in the slot the
    driver is written for - ``ddof`` and ``q`` have slots of their own, the
    ``dx`` and the ``reverse`` arrive through the ``ddof`` slot (see
    ``_ARGUMENT_WRAPPERS``).  A parallel driver takes the thread count as its
    first argument, read of every call: a number captured when the wrapper was
    built would break numba's on-disk cache (a dynamic global) and freeze the
    ``numba.set_num_threads`` knob - and it is read only where the call is big
    enough for the parallel path at all.  The driver itself decides whether
    the parallel path pays off (the cost model of ``npg_numba.mt``).
    """
    driver = handle.driver
    mt_driver = handle.mt_driver
    min_values = handle.mt_min_values

    if handle.variant == "none":

        def call_nothing(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: AScalar,
            ddof: float,
            q: float,
            method: str = "linear",
        ) -> AResult | list[AResult]:
            if mt_driver is not None and group_idx.size >= min_values:
                # the thread count is asked of every call, see _driver_call
                n_threads = mt_thread_count()
                if n_threads > 1:
                    # without a second thread the parallel region is pure
                    # overhead
                    return mt_driver(n_threads, group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value)

        return call_nothing

    if handle.variant == "q_method":
        # the quantile and its method arrive in the slots of their own

        def call_q_method(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: AScalar,
            ddof: float,
            q: float,
            method: str,
        ) -> AResult | list[AResult]:
            if mt_driver is not None and group_idx.size >= min_values:
                n_threads = mt_thread_count()
                if n_threads > 1:
                    return mt_driver(n_threads, group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, (q, method))
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, q, method)

        return call_q_method

    if handle.variant == "topk":
        # k and largest arrive in the slots of the ddof and the q

        def call_topk(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: AScalar,
            ddof: float,
            q: float,
            method: str = "linear",
        ) -> AResult | list[AResult]:
            if mt_driver is not None and group_idx.size >= min_values:
                n_threads = mt_thread_count()
                if n_threads > 1:
                    return mt_driver(n_threads, group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, (ddof, q))
            return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, (ddof, q))

        return call_topk

    # ddof, dx and reverse: the three arrive in the ddof of the call

    def call_ddof(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: AScalar,
        ddof: float,
        q: float,
        method: str = "linear",
    ) -> AResult | list[AResult]:
        if mt_driver is not None and group_idx.size >= min_values:
            n_threads = mt_thread_count()
            if n_threads > 1:
                return mt_driver(n_threads, group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof)
        return driver(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof)

    return call_ddof


def build_aggregate_op(func: str, handle: KernelHandle) -> Callable[..., AResult]:
    """Create the python-level wrapper around one compiled kernel.

    The wrapper holds no kernel machinery of its own: it resolves dtype and
    fill value (memoized on the plain 1d fast path) and delegates to the
    jitted driver; inputs the driver cannot handle go to the operation's
    python-level ``fallback``, or raise its ``refusal``.
    """
    driver_call = _driver_call(handle)
    # the names of the arguments the operation has of its own - only those
    # reach the fallback of the backend, under their own names
    extra_names = {name for name, _annotation, _default in func_extras(func)}

    def aggregate(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        ddof: float = 0,
        q: float = 0.5,
        method: str = "linear",
        **kwargs: Any,
    ) -> AResult:
        # numba has no float16: the values survive the widening cast to
        # float32 exactly, an inexact result is rounded back once, after the
        # kernel - the collecting function hands its values out themselves
        # instead, and a float16 input is always an array (the getattr gate
        # does not narrow the union)
        is16 = not handle.assembles and getattr(a, "dtype", None) == np.float16
        if is16:
            a = cast("npt.NDArray[np.number]", a).astype(np.float32)
        # the dtype the caller asked for - the slow path resolves it in place
        user_dtype = dtype
        # dtypes the compiled driver cannot handle, handed to the numpy twin
        if (handle.fallback is not None or handle.refusal is not None) and _needs_fallback(handle, a):
            if handle.fallback is not None:
                # the fallback (the numpy implementation) takes the arguments
                # of the operation under their own names
                extra = {name: value for name, value in (("ddof", ddof), ("q", q)) if name in extra_names}
                extra.update(kwargs)
                return handle.fallback(group_idx, a, func, size, fill_value, order, dtype, axis, **extra)
            # the refusal is thrown while the call is still being prepared
            raise TypeError(handle.refusal)
        # fast path: plain 1d-in/1d-out - bounds check, size detection,
        # allocation and kernel in one compiled call, everything else falls
        # through to the full validation machinery
        if (
            # the cumulatives derive the group count from the indices
            # themselves, so size=None takes the fast path too - a second max
            # scan plus the validation machinery would cost ~0.09ms per 500k
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
            n_key = len(group_idx) if (func in _N_DEPENDENT_FUNCS and a_key in _N_DEPENDENT_ADTYPES) else 0
            try:
                # a dtype that is no key of the cache raises there, which is caught
                plan_dtype, plan_fill = _dtype_fill_plan(func, a_key, dtype, fill_value, n_key, False)
            except TypeError:
                # unhashable dtype/fill_value - resolve on the slow path
                plan_dtype = None
            if plan_dtype is not None:
                ret = driver_call(
                    group_idx,
                    a,
                    -1 if size is None else size,
                    plan_dtype,
                    _aux_dtype(handle, a_key, plan_dtype),
                    _driver_fill(handle, plan_fill),
                    ddof,
                    q,
                    method,
                )
                if handle.post is not None:
                    ret = handle.post(ret, group_idx, plan_fill)
                else:
                    # only the collecting operation assembles a list
                    ret = cast(AResult, ret)
                if is16 and dtype is None:
                    ret = cast(AResult, _round_float16(ret))
                return ret

        iv = input_validation(
            group_idx,
            a,
            size=size,
            order=order,
            axis=axis,
            check_bounds=False,
            func_name=resolve_name(func, aliasing),
        )
        group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv
        if func == "topk":
            # the topk answers a (size, k) matrix: the axis of a 1d value
            # leaves the answer at its (size, k) and is granted, a group
            # index of more dimensions is refused - the same calls the numpy
            # backend grants and refuses
            check_topk_input(ndim_idx)

        # TODO: The typecheck should be done by the wrapper itself, not by check_dtype
        scalar = is_scalar_input(a)
        # a scalar reaches the compiled driver as a full column of that one
        # value, so its dtype is all that is left to resolve the plan from
        a_key = scalar_dtype(a) if scalar else cast("npt.NDArray[np.generic]", a).dtype
        n_key = len(group_idx) if (func in _N_DEPENDENT_FUNCS and a_key in _N_DEPENDENT_ADTYPES) else 0
        try:
            # a dtype that is no key of the cache raises there, which is caught
            dtype, fill_value = _dtype_fill_plan(func, a_key, dtype, fill_value, n_key, scalar)
        except TypeError:
            # unhashable dtype/fill_value - resolve without the cache
            dtype = check_dtype(dtype, func, a, len(group_idx))
            fill_value = resolve_fill_value(func, fill_value, dtype)
            check_fill_value(fill_value, dtype, func=func)
        if scalar:
            # scalar input is sum/prod/len only (the plan raises otherwise) -
            # materialize it for the jitted driver
            a = np.full(len(group_idx), a)
        group_idx = np.ascontiguousarray(group_idx)
        a = np.ascontiguousarray(a)

        ret = driver_call(
            group_idx,
            a,
            flat_size,
            dtype,
            _aux_dtype(handle, a_key, dtype),
            _driver_fill(handle, fill_value),
            ddof,
            q,
            method,
        )
        if handle.post is not None:
            # the post hook makes what the driver assembled the result of the
            # caller
            ret = handle.post(ret, group_idx, fill_value)
        else:
            # only the collecting operation assembles a list
            ret = cast(AResult, ret)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            if unravel_shape is not None:
                # argreductions only
                mask = ret == fill_value
                ret[mask] = 0
                # the axis is not None whenever the shape to unravel to is known
                assert axis is not None
                ret = cast(
                    "AResult",
                    np.unravel_index(cast("npt.NDArray[np.integer]", ret), unravel_shape)[axis],
                )
                ret[mask] = cast("AScalar", fill_value)
            check_nton_shape(ret, size, func)
            # numpy takes the shape as an int, a sequence of ints or an array
            # of them
            ret = cast("AResult", ret.reshape(size, order=order))
        elif func == "topk":
            # the topk answers (size, k), k travelling in the slot of the ddof
            check_topk_shape(ret, flat_size, int(ddof))
        if is16 and user_dtype is None:
            ret = cast(AResult, _round_float16(ret))
        return ret

    return aggregate


def _compiled_group_loop(func: Callable[..., AScalar]) -> tuple[Callable[..., None], Callable[..., None]]:
    """Compile a user callable and wrap the per-group loop around it: the
    sequential one, and the parallel one that cuts the *groups* into ranges
    the threads work on one after another.  The counting sort stays single
    threaded in both - a third of the work, and the values of a group have to
    be together where the callable is called on them."""
    nb_func = nb.njit(func)
    # the helpers are compiled for this loop alone - a compiled specialization
    # must not reach into the dispatchers of another operation
    count = nb.njit(group_count)
    starts_of = nb.njit(group_starts)
    place = nb.njit(group_place)
    ranges = nb.njit(group_ranges)

    @nb.njit
    def place_values(
        group_idx: npt.NDArray[np.integer], a: npt.NDArray[np.number], size: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.number]]:
        """The counting sort that places the values of every group in a
        contiguous block, in input order (a stable argsort's order at a
        fraction of its cost: 1.3ms against 33ms per 500k values), handing
        back the counts and the block starts besides."""
        counts = count(group_idx, size)
        starts = starts_of(counts)
        a_sorted = np.empty(a.size, a.dtype)
        place(group_idx, a, starts.copy(), a_sorted)
        return counts, starts, a_sorted

    @nb.njit
    def apply_range(
        first: int,
        last: int,
        counts: npt.NDArray[np.int64],
        starts: npt.NDArray[np.int64],
        a_sorted: npt.NDArray[np.number],
        ret: npt.NDArray[np.number],
    ) -> None:
        """The callable of the caller, applied to the values of a range of
        groups - a call per group costs more than a cheap callable (6.0ms
        against 4.3ms per million values)."""
        for g in range(first, last):
            n = counts[g]
            if n > 0:
                ret[g] = nb_func(a_sorted[starts[g] : starts[g] + n])

    @nb.njit
    def loop(group_idx: npt.NDArray[np.integer], a: npt.NDArray[np.number], ret: npt.NDArray[np.number]) -> None:
        size = len(ret)
        counts, starts, a_sorted = place_values(group_idx, a, size)
        apply_range(0, size, counts, starts, a_sorted, ret)

    @nb.njit(parallel=True)
    def mt_loop(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        ret: npt.NDArray[np.number],
        n_threads: int,
    ) -> None:
        size = len(ret)
        counts, starts, a_sorted = place_values(group_idx, a, size)
        if not should_use_group_mt(size, n_threads, group_idx.shape[0], _NS_PER_VALUE_CUSTOM):
            # not worth it, or one thread only: the sequential loop, bit for
            # bit
            apply_range(0, size, counts, starts, a_sorted, ret)
            return
        # the groups are cut into ranges of about equal value count, every
        # group in exactly one - no two threads touch one group, nothing is
        # folded afterwards, the result is the sequential loop's bit for bit
        n_ranges = group_chunk_count(n_threads, size)
        # the largest block of every range is not needed here: the callable is
        # called on the block of the group itself
        edges, _largest = ranges(counts, n_ranges)
        for r in nb.prange(n_ranges):
            apply_range(edges[r], edges[r + 1], counts, starts, a_sorted, ret)

    return loop, mt_loop


def build_generic_aggregate(func: Callable[..., AScalar]) -> Callable[..., AResult]:
    """Create the aggregation loop for an arbitrary user callable, applied to
    the values of every group, the data sorted by group index first.  No
    on-disk caching here: the captured user callable has no stable cache key
    (and potentially an unstable repr), so numba's cache could neither be
    relied on to hit, nor trusted against false hits.
    """
    jitfunc, mt_jitfunc = _compiled_group_loop(func)

    def aggregate(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        ddof: float = 0,
    ) -> AResult:
        # this loop runs only for funcs that are no key of the dispatch table:
        # no arg reduction, no one-to-one answer
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False, func_name=None)
        group_idx, a, flat_size, ndim_idx, size, _ = iv

        # without a dtype of the caller, the answer of a callable of the
        # caller is read as a float - what the numpy based backends answer in
        dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
        fill_value = resolve_fill_value(func, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=func)
        ret = np.full(flat_size, fill_value, dtype=dtype)
        group_idx = np.ascontiguousarray(group_idx)

        # the thread count is asked of every call that is big enough for the
        # parallel path at all - a small call should not pay for the runtime
        # call
        if len(group_idx) >= GROUP_MT_MIN_VALUES:
            n_threads = mt_thread_count()
        else:
            n_threads = 1
        if n_threads > 1:
            mt_jitfunc(group_idx, a, ret, n_threads)
        else:
            jitfunc(group_idx, a, ret)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            check_nton_shape(ret, size, func)
            ret = ret.reshape(size, order=order)
        return ret

    return aggregate


@functools.lru_cache(maxsize=128)
def _get_cached_generic(func: Callable[..., AScalar]) -> Callable[..., AResult]:
    """LRU-cache generic aggregate loops per user callable (bounded)."""
    return build_generic_aggregate(func)


def _custom_operation(func: Callable[..., AScalar]) -> Resolved:
    """The operation of a callable of the caller's own: compiled into a loop
    over the groups, memoized in the bounded LRU of this backend (the
    compilation is the expensive part).
    """
    callable_ = custom_callable(func)
    if not can_name_a_function(callable_):
        raise TypeError(
            "numba can only compile a function as func, a callable object is not one of those (the numpy implementation runs any callable)"
        ) from None

    def op(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        **kwargs: Any,
    ) -> AResult:
        # numba has no float16: the values survive the widening cast to
        # float32 exactly, an inexact result is rounded back to the dtype of
        # the input
        is16 = getattr(a, "dtype", None) == np.float16
        if is16:
            a = cast("npt.NDArray[np.number]", a).astype(np.float32)
        loop = _get_cached_generic(callable_)
        ret = loop(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
        return cast("AResult", _round_float16(ret)) if is16 and dtype is None else ret

    return Resolved(callable_, op)


# --------------------------------------------------------------- registry
#
# (name, kernel pair) - the order is the registration order; the pairs come
# from their modules - binding 'len', 'sum' or 'min' here would shadow the
# builtins used above
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
    ("quantile", specials.quantile),
    ("percentile", specials.percentile),
    ("trapezoid", specials.trapezoid),
    ("std", averaging.std),
    ("var", averaging.var),
    ("skew", averaging.skew),
    ("kurt", averaging.kurt),
    ("sumofsquares", reductions.sumofsquares),
    ("cumsum", cumulative.cumsum),
    ("cumprod", cumulative.cumprod),
    ("cummax", cumulative.cummax),
    ("cummin", cumulative.cummin),
    ("sort", specials.sort),
    ("topk", specials.topk),
    ("nunique", specials.nunique),
    ("mode", specials.mode),
    # the collecting function: the driver assembles a sequence per group into
    # the object array the caller expects
    ("array", collect.array),
)


def _topk_arguments(op: Callable[..., AResult]) -> Callable[..., AResult]:
    """Expose the arguments of the topk under their public names - the number
    of the values and the direction flag reach the driver in the positions
    otherwise used for ``ddof`` and ``q``."""

    def topk_op(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | npt.NDArray[np.integer] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        k: int = 5,
        largest: bool = True,
    ) -> AResult:
        # a fraction or a negative k would fail to type inside the compiled
        # driver, saying nothing a caller can act on
        k = check_topk_k(k)
        # the arguments travel twice: in the slots of the signature, how the
        # compiled driver takes them, and under their own names, how the
        # fallback of the backend takes them
        return op(group_idx, a, size, fill_value, order, dtype, axis, k, largest, k=k, largest=largest)

    return topk_op


def _dx_argument(op: Callable[..., AResult]) -> Callable[..., AResult]:
    """Expose the trapezoid's sample spacing under its public name ``dx``.

    The default is the integer 1 and not the float 1.0: a float in the slot
    compiles the kernel a few percent slower (see the note at the kernel
    builder of the trapezoid).
    """

    def trapezoid_op(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        dx: float = 1,
    ) -> AResult:
        return op(group_idx, a, size, fill_value, order, dtype, axis, dx)

    return trapezoid_op


def _sort_arguments(op: Callable[..., AResult]) -> Callable[..., AResult]:
    """Expose the reverse flag of the sort under its public name.

    The flag reaches the driver in the position otherwise used for ``ddof``;
    the empty list the shared fill value resolution hands out for ``sort``
    becomes the scalar the compiled driver expects - sort emits one value per
    input item and never fills a hole.
    """

    def sort_op(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        reverse: bool = False,
    ) -> AResult:
        if fill_value is DEFAULT_FILL_VALUE:
            fill_value = 0
        return op(group_idx, a, size, fill_value, order, dtype, axis, reverse, reverse=reverse)

    return sort_op


def _quantile_arguments(op: Callable[..., AResult], *, default: float, limit: float = 1.0) -> Callable[..., AResult]:
    """Expose the quantile argument of the quantile and of the percentile.

    The kernels take it in the unit of their own - the quantile a fraction,
    the percentile a percentage, whose kernel scales the argument down
    itself; the range of the quantile and the names of the methods are asked
    here, once per call.
    """

    def quantile_op(
        group_idx: npt.NDArray[np.integer],
        a: AValue,
        size: int | tuple[int, ...] | None = None,
        fill_value: AFillValue = DEFAULT_FILL_VALUE,
        order: RavelOrder = "C",
        dtype: npt.DTypeLike | None = None,
        axis: int | None = None,
        q: float = default,
        method: str = "linear",
        **kwargs: Any,
    ) -> AResult:
        if not 0.0 <= q <= limit:
            raise ValueError(f"Quantiles must be in the range [0, {round(limit)}]")
        if method not in QUANTILE_METHODS:
            raise ValueError(f"unknown interpolation method '{method}' - use one of {sorted(QUANTILE_METHODS)}")
        return op(group_idx, a, size, fill_value, order, dtype, axis, q=q, method=method, **kwargs)

    return quantile_op


# operations whose public keyword argument is not the plain ddof of the kernel
_ARGUMENT_WRAPPERS: dict[str, Callable[[Callable[..., AResult]], Callable[..., AResult]]] = {
    "trapezoid": _dx_argument,
    "sort": _sort_arguments,
    "topk": _topk_arguments,
    "quantile": partial(_quantile_arguments, default=0.5),
    "nanquantile": partial(_quantile_arguments, default=0.5),
    "percentile": partial(_quantile_arguments, default=50.0, limit=100.0),
    "nanpercentile": partial(_quantile_arguments, default=50.0, limit=100.0),
}


def get_ops() -> dict[str, Callable[..., AResult]]:
    """Create the wrapper function of every registered operation variant."""
    funcs: dict[str, Callable[..., AResult]] = {}
    for funcname, pair in _TABLE:
        wrap = _ARGUMENT_WRAPPERS.get(funcname)
        op = build_aggregate_op(funcname, pair.plain)
        funcs[funcname] = op if wrap is None else wrap(op)
        if funcname not in funcs_no_separate_nan:
            op = build_aggregate_op("nan" + funcname, pair.nan)
            funcs["nan" + funcname] = op if wrap is None else wrap(op)
    return funcs


_op_dict = get_ops()

# the operations of the backend on the int64 ticks of temporal values - the
# mean, the median and the quantile, whose numeric kernels are not exact on
# the ticks: the float64 mean rounds, the int64 median promotes to float64
_TICK_TABLE = (
    ("mean", averaging.tick_mean),
    ("median", specials.tick_median),
    ("quantile", specials.tick_quantile),
)


def get_tick_ops() -> dict[str, Callable[..., AResult]]:
    """The operations of the backend on the int64 ticks of temporal values.

    One per function name, built by the same factory as the numeric ones -
    the wrapper of the quantile is the one of the numeric values, so range
    and method names are asked of both alike.
    """
    ops: dict[str, Callable[..., AResult]] = {}
    for funcname, pair in _TICK_TABLE:
        ops[funcname] = build_aggregate_op("tick_" + funcname, pair.plain)
        ops["nan" + funcname] = build_aggregate_op("tick_" + funcname, pair.nan)
        wrap = _ARGUMENT_WRAPPERS.get(funcname)
        if wrap is not None:
            ops[funcname] = wrap(ops[funcname])
            ops["nan" + funcname] = wrap(ops["nan" + funcname])
    # the percentile is the quantile in percent - the kernel scales the
    # argument down itself, so the wrapper is the one of the quantile
    ops["percentile"] = ops["quantile"]
    ops["nanpercentile"] = ops["nanquantile"]
    return ops


_tick_ops = get_tick_ops()


def _call_operation(
    op: Callable[..., AResult],
    group_idx: npt.NDArray[np.integer],
    values: AValue,
    size: int,
    fill_value: AFillValue,
    dtype: npt.DTypeLike | None,
    order: RavelOrder,
    axis: int | None,
    **kwargs: Any,
) -> AResult:
    """The call of an operation of this backend: the flattening order and the
    axis travel between the fill value and the dtype, where the temporal
    policy of the package does not have to know them."""
    return op(group_idx, values, size, fill_value, order, dtype, axis, **kwargs)


def get_temporal_ops(
    op_dict: Mapping[str, Callable[..., AResult]],
    tick_ops: Mapping[str, Callable[..., AResult]],
) -> dict[str, Callable[..., tuple[AResult, AFillValue]]]:
    """The temporal operation of every function of the backend: the policy of
    the function and the operations of the backend, composed into the
    operation that answers the datetime64 and the timedelta64 input.  The
    compiled kernels of this backend take the temporal values as they are, so
    they are the native operation of the functions that have one."""

    def resolve(name: str) -> Callable[..., AResult] | None:
        return op_dict.get(name)

    ops: dict[str, Callable[..., tuple[AResult, AFillValue]]] = {}
    for name in function_names():
        ops[name] = build_temporal_op(
            name,
            op_dict.get(name),
            tick_op=tick_ops.get(name),
            # the compiled kernels take the temporal values as they are
            native=True,
            call=_call_operation,
            resolve=resolve,
            # the tick kernels carry the NaT policy of the function
            # themselves: the plain form poisons the group of a NaT, the nan
            # form leaves it out - the engine hands the rows over as they are
            tick_handles_nat=tick_ops.get(name) is not None,
        )
    return ops


_temporal_ops = get_temporal_ops(_op_dict, _tick_ops)

# alias (string or callable) -> (canonical name, wrapper function), every
# alias of every known function and so one dict lookup on the call path -
# names this backend has no wrapper for come along holding None
_dispatch = Registry(_op_dict, aliasing, _custom_operation, _temporal_ops)
