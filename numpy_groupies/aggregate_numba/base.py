"""The aggregation machinery: the operation class hierarchy and the call pipeline.

Every aggregation operation is a small class deriving from ``AggregateOp``
or one of its special bases (two-pass averaging, N-to-N cumulatives, and
jitted user callables).  The classes only describe the per-value kernel and
its bookkeeping; ``__call__``/``entry`` handle validation, allocation, the
jitted loops and the compilation caching around them.
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
from .jit import njit_stable
from .steps import step_indices


def complex_less(x, y):
    """The ordering every reduction kernel of this module compares with.

    For real values this is plain ``<``.  numpy orders complex values
    lexicographically - the real part first, the imaginary part only on a tie -
    and compares the parts with ordinary float comparisons rather than with a
    total order, so a nan in either part compares neither smaller nor greater,
    just like a nan of a real dtype.  Note that numpy does not look at the
    imaginary part at all once the real parts differ, but it does require both
    imaginary parts to be comparable for the real part to decide the order -
    which makes ``1+nanj < 2+1j`` False instead of True.
    """
    raise NotImplementedError  # replaced by the overload below


@nb.extending.overload(complex_less)
def _complex_less_overload(x, y):
    if isinstance(x, nb.types.Complex) and isinstance(y, nb.types.Complex):

        def _impl(x, y):
            if x.real < y.real:
                return not (np.isnan(x.imag) or np.isnan(y.imag))
            if x.real == y.real:
                return x.imag < y.imag
            return False

        return _impl

    def _impl(x, y):
        return x < y

    return _impl


class AggregateOp:
    """
    Every subclass of AggregateOp handles a different aggregation operation. There are
    several private class methods that need to be overwritten by the subclasses
    in order to implement different functionality.

    On object instantiation, all necessary static methods are compiled together into
    two jitted callables, one for scalar arguments, and one for arrays. Calling the
    instantiated object picks the right cached callable, does some further preprocessing
    and then executes the actual aggregation operation.

    The compiled kernels are created with numba's on-disk cache enabled, so
    the compilation cost is only paid once per machine.  All dispatchers are
    built through ``njit_stable``, which enables the cache by default and
    pins a content-derived uuid: closures over njit dispatchers would
    otherwise hash to new bytes in every process, breaking the on-disk
    cache and accumulating cache files on disk (see numba #6522).
    """

    disk_cache: bool = True
    forced_fill_value: int | float | None = None
    counter_fill_value: int | None = 1
    counter_dtype: type = bool
    mean_fill_value: float | None = None
    mean_dtype: type | None = np.float64
    outer: bool = False
    # Mean: track the global min/max inside the accumulation pass and check
    # the integer exact-mean limit between the passes. Std/Var: additionally
    # center integer input on the first value of each group (see
    # ``integer_centering``). The ``outer`` slot carries the per-group origins
    # plus two min/max scratch slots at the end.
    integer_gate: bool = False
    # Std/Var only: accumulate exact deviations of integer input instead of
    # raw values
    integer_centering: bool = False
    reverse: bool = False
    nans: bool = False

    def __init__(self, func=None, **kwargs):
        if func is None:
            func = type(self).__name__.lower()
        self.func = func
        self.__dict__.update(kwargs)
        # Cache the compiled functions, so they don't have to be recompiled on every call
        self._jit_scalar = self.callable(self.nans, self.reverse, scalar=True)
        self._jit_non_scalar = self.callable(self.nans, self.reverse, scalar=False)
        self._jit_entry = self.entry()
        # fixed mean dtype as a proper np.dtype instance - the jitted entry
        # must never see a dtype *class* (numba types it as a slow type-ref)
        self._mean_dtype = np.dtype(self.mean_dtype) if self.mean_dtype is not None else None

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
        # fast path: plain 1d-in/1d-out without custom fill handling - the
        # jitted entry does the bounds check, size detection, allocation and
        # the kernel in one compiled call, everything else falls through to
        # the full validation machinery below
        if (
            axis is None
            and not self.outer
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
            if plan_dtype is not None and (
                self.forced_fill_value is None
                or plan_fill == self.forced_fill_value
                or isinstance(self, Aggregate2pass)
            ):
                # 2pass kernels copy the fill_value to empty groups in their
                # second pass, so they never need _finalize
                # a float64 accumulator widened to the complex counterpart
                # keeps the running mean lossless for complex input as well
                mean_dtype = a_key if self._mean_dtype is None else np.result_type(a_key, self._mean_dtype)
                return self._jit_entry(
                    group_idx,
                    a,
                    -1 if size is None else size,
                    plan_dtype,
                    mean_dtype,
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
        if a_shape is None or not a_shape:
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
        input_dtype = a_key
        ret, counter, mean, outer = self._initialize(flat_size, fill_value, dtype, input_dtype, group_idx.size)
        group_idx = np.ascontiguousarray(group_idx)

        if isinstance(a, np.generic) or a_shape is None:
            jitfunc = self._jit_scalar
        else:
            a = np.ascontiguousarray(a)
            jitfunc = self._jit_non_scalar
        jitfunc(group_idx, a, ret, counter, mean, outer, fill_value, ddof)
        self._finalize(ret, counter, fill_value)

        if self.outer:
            ret = outer

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

    @classmethod
    def _initialize(cls, flat_size, fill_value, dtype, input_dtype, input_size):
        if cls.forced_fill_value is None:
            ret = np.full(flat_size, fill_value, dtype=dtype)
        else:
            ret = np.full(flat_size, cls.forced_fill_value, dtype=dtype)

        counter = mean = outer = None
        if cls.counter_fill_value is not None:
            counter = np.full_like(ret, cls.counter_fill_value, dtype=cls.counter_dtype)
        if cls.mean_fill_value is not None:
            # a float64 accumulator widened to the complex counterpart keeps
            # the running mean lossless for complex input as well
            dtype = input_dtype if cls.mean_dtype is None else np.result_type(input_dtype, cls.mean_dtype)
            mean = np.full_like(ret, cls.mean_fill_value, dtype=dtype)
        if cls.outer:
            outer = np.full(input_size, fill_value, dtype=dtype)
        elif cls.integer_gate or cls.integer_centering:
            # per-group centering origins (Std/Var) plus two min/max scratch
            # slots at the end; input_dtype is always a real dtype here since
            # scalar input never reaches the averaging kernels
            outer = np.empty(flat_size + 2, dtype=input_dtype)

        return ret, counter, mean, outer

    @classmethod
    def _finalize(cls, ret, counter, fill_value):
        if cls.forced_fill_value is not None and fill_value != cls.forced_fill_value:
            if cls.counter_dtype == bool:
                ret[counter] = fill_value
            else:
                ret[~counter.astype(bool)] = fill_value

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        """Compile a jitted function doing the hard part of the job"""
        _valgetter = cls._valgetter_scalar if scalar else cls._valgetter
        valgetter = njit_stable(_valgetter)
        outersetter = njit_stable(cls._outersetter)

        if not nans:
            inner = njit_stable(cls._inner)
        else:
            cls_inner = njit_stable(cls._inner)
            cls_nan_check = njit_stable(cls._nan_check)

            @njit_stable
            def inner(ri, val, ret, counter, mean, fill_value):
                if not cls_nan_check(val):
                    cls_inner(ri, val, ret, counter, mean, fill_value)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # ddof needs to be present for being exchangeable with loop_2pass
            size = len(ret)
            rng = range(len(group_idx) - 1, -1, -1) if reverse else range(len(group_idx))
            for i in rng:
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                inner(ri, val, ret, counter, mean, fill_value)
                outersetter(outer, i, ret[ri])

        return loop

    def entry(self):
        """Compile the jitted fast-path entry used by ``__call__``.

        Detects the output size with a bounds check in the same pass,
        allocates the working arrays and runs the compiled kernel loop, all
        inside one call - skipping the python-level validation and
        allocation machinery for the plain 1d case.  Arrays the operation
        does not use get a one-element placeholder, mirroring how
        ``_initialize`` leaves them as None: the kernels only ever touch the
        arrays their operation needs (``mean`` is accessed exactly when
        ``mean_fill_value`` is set, ``counter`` when ``counter_fill_value``
        is set).
        """
        loop = self.callable(self.nans, self.reverse, scalar=False)
        forced = self.forced_fill_value
        counter_fill = self.counter_fill_value
        counter_dtype = np.dtype(self.counter_dtype)
        mean_fill = self.mean_fill_value
        origin_ints = self.integer_gate or self.integer_centering

        @njit_stable
        def fast_entry(group_idx, a, size_arg, ret_dtype, mean_dtype, fill_value, ddof):
            # numba's np.max reduction is several times faster than a scalar
            # loop here; negative indices slip through and are caught by the
            # bounds check inside the kernel loop below
            size = np.max(group_idx) + 1
            if size_arg >= 0:
                # an explicit output size may be larger than the largest
                # index, but never smaller
                if size > size_arg:
                    raise ValueError("one or more indices in group_idx are too large")
                size = size_arg
            if forced is None:
                ret = np.full(size, fill_value, ret_dtype)
            elif forced == 0:
                ret = np.zeros(size, ret_dtype)
            else:
                ret = np.full(size, forced, ret_dtype)
            if counter_fill is None:
                counter = np.zeros(1, counter_dtype)
            elif counter_fill == 0:
                counter = np.zeros(size, counter_dtype)
            else:
                counter = np.ones(size, counter_dtype)
            if mean_fill is None:
                mean = np.zeros(1, mean_dtype)
            elif mean_fill == 0:
                mean = np.zeros(size, mean_dtype)
            else:
                mean = np.full(size, mean_fill, mean_dtype)
            if origin_ints:
                # per-group centering origins plus min/max scratch slots, all
                # seeded inside the kernel (non-integer input never reads the
                # scratch); every other operation keeps the None placeholder
                origin = np.empty(size + 2, a.dtype)
            else:
                origin = None
            loop(group_idx, a, ret, counter, mean, origin, fill_value, ddof)
            return ret

        return fast_entry

    @staticmethod
    def _valgetter(a, i):
        return a[i]

    @staticmethod
    def _valgetter_scalar(a, i):
        return a

    @staticmethod
    def _nan_check(val):
        return val != val

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        raise NotImplementedError("subclasses need to overwrite _inner")

    @staticmethod
    def _outersetter(outer, i, val):
        pass


class Aggregate2pass(AggregateOp):
    """Base class for everything that needs to process the data twice like mean, var and std."""

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # the 1st pass mirrors the generic loop, but hands the ``outer`` slot
        # to the inner callable as well - Std uses it to carry the per-group
        # centering origin of integer input (these ops never use outersetter,
        # their outer flag is always False)
        _valgetter = cls._valgetter_scalar if scalar else cls._valgetter
        valgetter = njit_stable(_valgetter)
        cls_inner = njit_stable(cls._inner)
        cls_nan_check = njit_stable(cls._nan_check)
        gate_check = njit_stable(cls._gate_check)

        if nans:

            @njit_stable
            def inner(ri, val, ret, counter, mean, origin, fill_value):
                if not cls_nan_check(val):
                    cls_inner(ri, val, ret, counter, mean, origin, fill_value)

        else:
            inner = cls_inner

        @njit_stable
        def loop_1st(group_idx, a, ret, counter, mean, origin, fill_value):
            size = len(ret)
            if len(group_idx) > 0:
                # seed the min/max scratch slots with the first value
                origin[size] = a[0]
                origin[size + 1] = a[0]
            rng = range(len(group_idx) - 1, -1, -1) if reverse else range(len(group_idx))
            for i in rng:
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                inner(ri, val, ret, counter, mean, origin, fill_value)

        _2pass_inner = njit_stable(cls._2pass_inner)

        @njit_stable
        def loop_2nd(ret, counter, mean, fill_value, ddof):
            for ri in range(len(ret)):
                if counter[ri] > ddof:
                    ret[ri] = _2pass_inner(ri, ret, counter, mean, ddof)
                else:
                    ret[ri] = fill_value

        @njit_stable
        def loop_2pass(group_idx, a, ret, counter, mean, origin, fill_value, ddof):
            loop_1st(group_idx, a, ret, counter, mean, origin, fill_value)
            size = len(ret)
            # the accumulation pass tracked the global min/max in the scratch
            # slots behind the per-group data - raise before the 2nd pass
            gate_check(origin[size], origin[size + 1], len(group_idx))
            loop_2nd(ret, counter, mean, fill_value, ddof)

        return loop_2pass

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        raise NotImplementedError("subclasses need to overwrite _2pass_inner")

    @classmethod
    def _finalize(cls, ret, counter, fill_value):
        """Copying the fill value is already done in the 2nd pass"""


class AggregateNtoN(AggregateOp):
    """Base class for cumulative functions, where the output size matches the input size."""

    outer = True

    @staticmethod
    def _outersetter(outer, i, val):
        outer[i] = val


class AggregateGeneric(AggregateOp):
    """Base class for jitting arbitrary functions.

    ``disk_cache`` stays off here: the captured user callable has no stable
    disk-cache key (and potentially an unstable repr), so numba's on-disk
    cache could neither be relied upon to hit, nor be trusted against false
    hits.
    """

    counter_fill_value = None
    disk_cache = False

    def __init__(self, func, **kwargs):
        self.func = func
        self.__dict__.update(kwargs)
        self._jitfunc = self.callable(self.nans)

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
        input_dtype = type(a) if np.isscalar(a) else a.dtype
        ret, _, _, _ = self._initialize(flat_size, fill_value, dtype, input_dtype, group_idx.size)
        group_idx = np.ascontiguousarray(group_idx)

        sortidx = np.argsort(group_idx, kind="stable")
        self._jitfunc(sortidx, group_idx, a, ret)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            check_nton_shape(ret, size, self.func)
            ret = ret.reshape(size, order=order)
        return ret

    def callable(self, nans=False):
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


# input dtypes for which the 'sum' output dtype depends on the input length
# (the overflow guesses in resolve_output_dtype)
_N_DEPENDENT_ADTYPES = frozenset(
    np.dtype(t) for t in ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "uint64")
)
_SCALAR_BOOL_TYPES = frozenset((bool, np.bool_))


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
