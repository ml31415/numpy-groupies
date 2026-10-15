"""The python-level wrapper of the numba implementation.

``AggregateOp`` instances sit in the dispatch table of ``aggregate()``: they
run the validation and the output-dtype/fill-value resolution, then hand
everything to the compiled kernel driver of the operation (see ``compose``).
There is no per-class kernel machinery left - plain 1d input runs entirely
inside the driver, everything else (2d/axis, order, out-of-range sizes,
unravelling for the argreductions) is answered on the slow path.
"""

import functools

import numba as nb
import numpy as np

from ..aggregate_common import DEFAULT_FILL_VALUE, resolve_fill_value
from ..utils import (
    check_dtype,
    check_fill_value,
    check_nton_shape,
    input_validation,
    resolve_output_dtype,
)
from .steps import step_indices

# input dtypes for which the 'sum' output dtype depends on the input length
# (the overflow guesses in resolve_output_dtype)
_N_DEPENDENT_ADTYPES = frozenset(
    np.dtype(t) for t in ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "uint64")
)
_SCALAR_BOOL_TYPES = frozenset((bool, np.bool_))


@functools.lru_cache(maxsize=256)
def _dtype_fill_plan(op, a_key, dtype, fill_value, n):
    """Memoized output-dtype and fill-value resolution.

    ``a_key`` is ``type(a)`` for scalar inputs and ``a.dtype`` otherwise.
    ``n`` carries ``len(group_idx)`` only for the length-dependent ``sum``
    dtype guesses and is 0 otherwise.  Unhashable arguments fail the cache
    lookup with a TypeError, which the caller answers with the uncached
    resolution.
    """
    func = op.func
    if isinstance(a_key, type):
        if func not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', 'prod' and 'len'")
        a_dtype = np.dtype(a_key)
    else:
        a_dtype = a_key
    out_dtype = resolve_output_dtype(dtype, func, a_dtype, n)
    fill_value = resolve_fill_value(func, fill_value, out_dtype)
    check_fill_value(fill_value, out_dtype, func=func)
    return out_dtype, fill_value


def _aux_dtype(handle, a_dtype, out_dtype):
    """The auxiliary (widened accumulation) dtype of the dispersion kernels."""
    if handle.aux_widen:
        return np.result_type(a_dtype, np.float64)
    return out_dtype


class AggregateOp:
    """Python-level wrapper around the compiled kernel of one operation.

    The wrapper holds no kernel machinery of its own: ``__call__`` resolves
    dtype and fill value (memoized for the plain 1d fast path), materializes
    scalar input and delegates the actual aggregation to the jitted driver.
    """

    def __init__(self, func, handle, nans=False):
        self.func = func
        self.nans = nans
        self.handle = handle

    def __call__(
        self,
        group_idx,
        a,
        size=None,
        fill_value=DEFAULT_FILL_VALUE,
        order="C",
        dtype=None,
        axis=None,
        ddof=0,
    ):
        handle = self.handle
        # fast path: plain 1d-in/1d-out - the jitted driver does the bounds
        # check, size detection, allocation and the kernel in one compiled
        # call, everything else falls through to the full validation machinery
        if (
            not handle.nton
            and axis is None
            and (size is None or type(size) is int)
            and type(group_idx) is np.ndarray
            and group_idx.ndim == 1
            and group_idx.dtype.kind in "iu"
            and group_idx.size > 0
            and type(a) is np.ndarray
            and a.ndim == 1
            and a.shape[0] == group_idx.size
        ):
            a_key = a.dtype
            n_key = len(group_idx) if (self.func == "sum" and a_key in _N_DEPENDENT_ADTYPES) else 0
            try:
                plan_dtype, plan_fill = _dtype_fill_plan(self, a_key, dtype, fill_value, n_key)
            except TypeError:
                # unhashable dtype/fill_value - resolve on the slow path
                plan_dtype = None
            if plan_dtype is not None:
                return handle.driver(
                    group_idx,
                    a,
                    -1 if size is None else size,
                    plan_dtype,
                    _aux_dtype(handle, a_key, plan_dtype),
                    plan_fill,
                    ddof,
                )

        iv = input_validation(
            group_idx,
            a,
            size=size,
            order=order,
            axis=axis,
            check_bounds=False,
            func=self.func,
        )
        group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

        # TODO: The typecheck should be done by the class itself, not by check_dtype
        a_shape = getattr(a, "shape", None)
        scalar = a_shape is None or not a_shape
        if scalar:
            # scalar input: the type stands in for the dtype, and only the
            # bool 'sum' dtype guess depends on the input length
            a_key = type(a)
            n_key = len(group_idx) if (self.func == "sum" and a_key in _SCALAR_BOOL_TYPES) else 0
        else:
            a_key = a.dtype
            n_key = len(group_idx) if (self.func == "sum" and a_key in _N_DEPENDENT_ADTYPES) else 0
        try:
            dtype, fill_value = _dtype_fill_plan(self, a_key, dtype, fill_value, n_key)
        except TypeError:
            # unhashable dtype/fill_value (or a genuine TypeError from the
            # checks, e.g. 0-d array input) - resolve without the cache
            dtype = check_dtype(dtype, self.func, a, len(group_idx))
            fill_value = resolve_fill_value(self.func, fill_value, dtype)
            check_fill_value(fill_value, dtype, func=self.func)
        if scalar:
            # scalar input is only supported by sum/prod/len (the plan raises
            # for everything else) - materialize it for the jitted driver
            a = np.full(len(group_idx), a)
        group_idx = np.ascontiguousarray(group_idx)
        a = np.ascontiguousarray(a)

        ret = handle.driver(group_idx, a, flat_size, dtype, _aux_dtype(handle, a_key, dtype), fill_value, ddof)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            if unravel_shape is not None:
                # argreductions only
                mask = ret == fill_value
                ret[mask] = 0
                ret = np.unravel_index(ret, unravel_shape)[axis]
                ret[mask] = fill_value
            check_nton_shape(ret, size, self.func)
            ret = ret.reshape(size, order=order)
        return ret


class AggregateGeneric:
    """Wrapper for jitting arbitrary functions.

    The user callable is applied to the values of every group, the data is
    sorted by group index first.  No on-disk caching here: the captured user
    callable has no stable disk-cache key (and potentially an unstable repr),
    so numba's on-disk cache could neither be relied upon to hit, nor be
    trusted against false hits.
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.__dict__.update(kwargs)
        self._jitfunc = self._callable()

    def __call__(
        self,
        group_idx,
        a,
        size=None,
        fill_value=DEFAULT_FILL_VALUE,
        order="C",
        dtype=None,
        axis=None,
        ddof=0,
    ):
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False)
        group_idx, a, flat_size, ndim_idx, size, _ = iv

        # TODO: The typecheck should be done by the class itself, not by check_dtype
        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        fill_value = resolve_fill_value(self.func, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=self.func)
        ret = np.full(flat_size, fill_value, dtype=dtype)
        group_idx = np.ascontiguousarray(group_idx)

        sortidx = np.argsort(group_idx, kind="stable")
        self._jitfunc(sortidx, group_idx, a, ret)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            check_nton_shape(ret, size, self.func)
            ret = ret.reshape(size, order=order)
        return ret

    def _callable(self):
        """Compile a jitted function and loop it over the sorted data."""
        func = nb.njit(self.func)

        @nb.njit
        def loop(sortidx, group_idx, a, ret):
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
                ret[ri] = func(a_srt[start_idx:stop_idx])

        return loop
