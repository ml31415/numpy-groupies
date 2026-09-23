import functools
import hashlib
import inspect

import numba as nb
import numpy as np

from .aggregate_numpy import _aggregate_base
from .utils import (
    DEFAULT_FILL_VALUE,
    _no_complex_order,
    aggregate_common_doc,
    aliasing,
    build_dispatch,
    check_dtype,
    check_dtype_support,
    check_fill_value,
    check_nton_shape,
    funcs_no_separate_nan,
    get_func,
    input_validation,
    resolve_fill_value,
    resolve_output_dtype,
)


def njit_stable(py_func=None, cache=True, **options):
    """``nb.njit`` plus a deterministic dispatcher identity.

    Numba draws a dispatcher's uuid lazily from ``uuid.uuid4()`` and embeds
    it in the cloudpickle representation of the dispatcher.  Closures over
    dispatchers therefore hash to new bytes in every interpreter process,
    so numba's on-disk cache key of the enclosing function never hits and
    every run appends a new cache file instead (see numba #6522).  Pinning
    a content-derived uuid keeps the closure hash stable across processes,
    which makes the on-disk cache effective for the factory-nested loops
    of this module.

    The fingerprint covers module, qualname, source, options and the
    closure contents (dispatchers contribute their already pinned uuid),
    so dispatchers compiled from the same factory lines for different
    operations receive distinct identities.  Usable as
    ``njit_stable(func, **options)`` or ``@njit_stable`` /
    ``@njit_stable(**options)``.  Disk caching is enabled by default
    (``cache=True``), which is the main point of this wrapper.
    """

    def wrap(func):
        disp = nb.njit(func, cache=cache, **options)
        try:
            cells = []
            if func.__closure__ is not None:
                for cell in func.__closure__:
                    try:
                        content = cell.cell_contents
                    except ValueError:  # pragma: no cover - empty cell
                        content = "<empty cell>"
                    # dispatchers were wrapped earlier and already carry
                    # their pinned uuid, closures are created in order
                    cells.append(getattr(content, "_uuid", repr(content)))
            fingerprint = "\n".join(
                [
                    func.__module__,
                    func.__qualname__,
                    inspect.getsource(func),
                    repr(sorted(options.items())),
                    repr(cells),
                ]
            )
            disp._set_uuid(hashlib.sha256(fingerprint.encode()).hexdigest()[:16])
        except Exception:  # noqa: BLE001 - any numba internals change must degrade to no caching
            # numba internals changed: without a stable uuid the closure
            # hash differs in every process, the cache could never hit and
            # would grow without bound - disable caching for this
            # dispatcher instead of polluting the disk (numba #6522)
            disp.targetoptions["cache"] = False
        return disp

    if py_func is None:
        return wrap
    return wrap(py_func)


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

    disk_cache = True
    forced_fill_value = None
    counter_fill_value = 1
    counter_dtype = bool
    mean_fill_value = None
    mean_dtype = np.float64
    outer = False
    reverse = False
    nans = False

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
            # order-dependent kernels cannot compile complex values - fall
            # through to the slow path, which rejects them with a clear error
            and not (self.func in _no_complex_order and a.dtype.kind == "c")
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
        # reject order-dependent reductions of complex input before any dtype
        # planning - np.issubdtype inside accepts plain types as well
        check_dtype_support(self.func, a_key, "numba")
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
            loop(group_idx, a, ret, counter, mean, None, fill_value, ddof)
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
        # Careful, cls needs to be passed, so that the overwritten methods remain available in
        # AggregateOp.callable
        loop_1st = super().callable(nans=nans, reverse=reverse, scalar=scalar)

        _2pass_inner = njit_stable(cls._2pass_inner)

        @njit_stable
        def loop_2nd(ret, counter, mean, fill_value, ddof):
            for ri in range(len(ret)):
                if counter[ri] > ddof:
                    ret[ri] = _2pass_inner(ri, ret, counter, mean, ddof)
                else:
                    ret[ri] = fill_value

        @njit_stable
        def loop_2pass(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            loop_1st(group_idx, a, ret, counter, mean, outer, fill_value, ddof)
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
        check_dtype_support(self.func, np.dtype(type(a)) if np.isscalar(a) else a.dtype, "numba")
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


class Sum(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += val


class Prod(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] *= val


class Len(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += 1


class All(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] &= bool(val)


class Any(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] |= bool(val)


class Last(AggregateOp):
    counter_fill_value = None

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        ret[ri] = val


class First(Last):
    reverse = True


class AllNan(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] &= val != val


class AnyNan(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] |= val != val


class Max(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        # select-form on purpose: the branchy equivalent (see below) compiles
        # ~15x slower under numba >= 0.66 when reached as a separate njit call.
        # Note: ret[ri] must be read into a local first - repeating the
        # subscript expression keeps the slow codegen (aliasing).
        # val != val has to be tested explicitly - a nan compares neither
        # smaller nor greater, so `cur < val` alone would leave a group whose
        # nan arrives after its first value at the finite value, unlike np.max
        # (and unlike this module's own arg kernels, which do report such a
        # group as invalid).  The nan variants are unaffected, they filter the
        # nans out before the kernel runs.  The extra comparison is the
        # cheapest way to get there: np.maximum in the non-first branch
        # (which propagates nans on its own) and the branchy form below both
        # measured ~45% slower than this, while for integer dtypes LLVM folds
        # the test away entirely.
        # if counter[ri]:
        #     ret[ri] = val
        #     counter[ri] = 0
        # elif ret[ri] < val:
        #     ret[ri] = val
        first = counter[ri]
        counter[ri] = 0
        cur = ret[ri]
        ret[ri] = val if (first or val != val or cur < val) else cur


class Min(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        # select-form on purpose, see Max._inner (including why val != val is
        # part of the condition and what it costs)
        # if counter[ri]:
        #     ret[ri] = val
        #     counter[ri] = 0
        # elif ret[ri] > val:
        #     ret[ri] = val
        first = counter[ri]
        counter[ri] = 0
        cur = ret[ri]
        ret[ri] = val if (first or val != val or cur > val) else cur


class ArgMax(AggregateOp):
    mean_fill_value = np.nan

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # The loop body is inlined on purpose: with numba >= 0.67 the generic
        # loop plus a separate njit `_inner` call compiles ~20x slower for this
        # reduction. The inlined logic is equivalent to the former _inner:
        #   first value of a group seeds ret/mean, nans reset the group to
        #   fill_value, and for nans=True nans are skipped entirely.
        if scalar or reverse:
            # scalar/reverse argreductions are not meaningfully supported;
            # keep the generic (lazy) path for unchanged behaviour.
            return AggregateOp.callable(nans=nans, reverse=reverse, scalar=scalar)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            size = len(ret)
            for i in range(len(group_idx)):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                cmp_val = a[i]
                if cmp_val != cmp_val:
                    if nans:
                        continue
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = fill_value
                    continue
                if counter[ri]:
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = i
                elif mean[ri] < cmp_val:
                    mean[ri] = cmp_val
                    ret[ri] = i

        return loop


class ArgMin(ArgMax):
    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # inlined on purpose, see ArgMax.callable
        if scalar or reverse:
            return AggregateOp.callable(nans=nans, reverse=reverse, scalar=scalar)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            size = len(ret)
            for i in range(len(group_idx)):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                cmp_val = a[i]
                if cmp_val != cmp_val:
                    if nans:
                        continue
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = fill_value
                    continue
                if counter[ri]:
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = i
                elif mean[ri] > cmp_val:
                    mean[ri] = cmp_val
                    ret[ri] = i

        return loop


class SumOfSquares(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        # the squared magnitude of val - complex values give a real, sqrt-free
        # result like np.var; bool has no .real in numba and keeps the plain
        # val * val product (isinstance resolves at compile time per dtype)
        if isinstance(val, bool):
            ret[ri] += val * val
        else:
            ret[ri] += val.real * val.real + val.imag * val.imag


class Mean(Aggregate2pass):
    forced_fill_value = 0
    counter_fill_value = 0
    counter_dtype = int

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] += 1
        ret[ri] += val

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        return ret[ri] / counter[ri]


class Std(Mean):
    mean_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] += 1
        mean[ri] += val
        # the squared deviation of complex values is its magnitude squared,
        # so ret stays real for complex input - matching np.var; bool has no
        # .real in numba and keeps the plain val * val product (isinstance
        # resolves at compile time per dtype)
        if isinstance(val, bool):
            ret[ri] += val * val
        else:
            ret[ri] += val.real * val.real + val.imag * val.imag

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        m = mean[ri]
        mean2 = m.real * m.real + m.imag * m.imag
        return np.sqrt((ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof))


class Var(Std):
    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        m = mean[ri]
        mean2 = m.real * m.real + m.imag * m.imag
        return (ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof)


class Median(AggregateOp):
    """The median is an order statistic, so unlike the streaming reductions
    it needs group-ordered data: values are placed group by group via
    counting sort (O(n)) and the middle of each group is then *selected*
    via partition (O(n) on average) instead of sorting within the groups."""

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
        if not np.isscalar(a) and np.issubdtype(a.dtype, np.complexfloating):
            # numba cannot partition complex values - the numpy implementation
            # selects the middle of the lexicographically ordered values
            return _aggregate_base(
                group_idx, a, self.func, size=size, fill_value=fill_value, order=order, dtype=dtype, axis=axis
            )
        return super().__call__(group_idx, a, size, fill_value, order, dtype, axis, ddof)

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # median is order-insensitive, so reverse is ignored
        _count = njit_stable(cls._count)
        _starts = njit_stable(cls._starts)
        _place = njit_stable(cls._place)
        _valid = njit_stable(cls._valid)
        _middle = njit_stable(cls._middle)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # signature kept identical to AggregateOp.callable
            size = len(ret)
            counts = _count(group_idx, size)
            starts = _starts(counts)
            # counting-sort placement into contiguous group-blocks (the
            # order within a block is arbitrary, which the median ignores)
            a_sorted = np.empty(group_idx.size, a.dtype)
            _place(group_idx, a, starts.copy(), a_sorted)
            buf = np.empty(group_idx.size, a.dtype)
            for g in range(size):
                if counts[g] == 0:
                    continue  # untouched group, keeps the fill value
                m = _valid(g, a_sorted, counts, starts, buf, nans)
                if m == 0:
                    # all-nan group for nanmedian, or a nan-poisoned group
                    # for plain median (where any nan is contagious)
                    ret[g] = fill_value if nans else np.nan
                else:
                    ret[g] = _middle(buf[:m])
                    counter[g] = True

        return loop

    @staticmethod
    def _count(group_idx, size):
        """bounds-checked per-group element counts"""
        counts = np.zeros(size, np.int64)
        for i in range(len(group_idx)):
            g = group_idx[i]
            if g < 0:
                raise ValueError("negative indices not supported")
            if g >= size:
                raise ValueError("one or more indices in group_idx are too large")
            counts[g] += 1
        return counts

    @staticmethod
    def _starts(counts):
        """first index of each group-block"""
        return np.cumsum(counts) - counts

    @staticmethod
    def _place(group_idx, a, cursor, a_sorted):
        """scatter the values into their group-blocks"""
        for i in range(len(group_idx)):
            g = group_idx[i]
            a_sorted[cursor[g]] = a[i]
            cursor[g] += 1

    @staticmethod
    def _valid(g, a_sorted, counts, starts, buf, nans):
        """copy group g into buf, returning the number of usable values;
        groups with nans yield 0 for plain median (nan is contagious),
        while nanmedian keeps the group with the nans filtered out"""
        begin, size_g = starts[g], counts[g]
        if not nans:
            buf[:size_g] = a_sorted[begin : begin + size_g]
            if np.sum(buf[:size_g] != buf[:size_g]) > 0:
                return 0
            return size_g
        m = 0
        for i in range(size_g):
            v = a_sorted[begin + i]
            if v == v:
                buf[m] = v
                m += 1
        return m

    @staticmethod
    def _middle(values):
        """select the middle value(s) of a group - no sorting needed"""
        m = len(values)
        part = np.partition(values, m // 2)
        if m % 2 == 1:
            return part[m // 2]
        # the lower middle is the maximum of the unsorted left half; the 1.0
        # promotes integral input to float64, just like np.median promotes
        # before averaging - summing the two middles in their own dtype
        # overflows (int64 2**62 + 2**62) and adds bool as a logical or
        return (np.max(part[: m // 2]) * 1.0 + part[m // 2]) / 2


class Trapezoid(AggregateOp):
    """Trapezoidal integration of each group, in the order the samples appear.
    The area accumulates in ret, the previous sample is kept in mean and
    counter doubles as the "group started" flag, so one pass over the data
    is enough."""

    forced_fill_value = 0
    mean_fill_value = 0
    mean_dtype = None

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # the integral follows the order of the input, so reverse is ignored
        valgetter = njit_stable(cls._valgetter_scalar if scalar else cls._valgetter)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # ddof carries the sample spacing dx, see Trapezoid.__call__
            size = len(ret)
            for i in range(len(group_idx)):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                if nans and val != val:
                    continue
                if counter[ri]:
                    counter[ri] = False
                else:
                    ret[ri] += 0.5 * ddof * (mean[ri] + val)
                mean[ri] = val

        return loop

    def __call__(
        self,
        group_idx,
        a,
        size=None,
        fill_value=DEFAULT_FILL_VALUE,
        order="C",
        dtype=None,
        axis=None,
        dx=1.0,
    ):
        # the one float argument of this operation is the sample spacing,
        # which the kernels expect in the position otherwise used for ddof
        return super().__call__(group_idx, a, size, fill_value, order, dtype, axis, dx)


class CumSum(AggregateNtoN, Sum):
    pass


class CumProd(AggregateNtoN, Prod):
    pass


class CumMax(AggregateNtoN, Max):
    pass


class CumMin(AggregateNtoN, Min):
    pass


def get_funcs():
    funcs = {}
    for op in (
        Sum,
        Prod,
        Len,
        All,
        Any,
        Last,
        First,
        AllNan,
        AnyNan,
        Min,
        Max,
        ArgMin,
        ArgMax,
        Mean,
        Median,
        Trapezoid,
        Std,
        Var,
        SumOfSquares,
        CumSum,
        CumProd,
        CumMax,
        CumMin,
    ):
        funcname = op.__name__.lower()
        funcs[funcname] = op(funcname)
        if funcname not in funcs_no_separate_nan:
            funcname = "nan" + funcname
            funcs[funcname] = op(funcname, nans=True)
    return funcs


_impl_dict = get_funcs()

# alias (string or callable) -> compiled op instance, so that the common call
# path resolves with a single dict lookup instead of the two-step aliasing +
# implementation resolution in get_func
_dispatch = build_dispatch(_impl_dict, aliasing)

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


def aggregate(
    group_idx,
    a,
    func="sum",
    size=None,
    fill_value=DEFAULT_FILL_VALUE,
    order="C",
    dtype=None,
    axis=None,
    out=None,
    cache=True,
    **kwargs,
):
    try:
        aggregate_op = _dispatch[func][1]
    except (KeyError, TypeError):
        # a custom callable, or an unknown name that get_func complains about
        if out is not None:
            raise NotImplementedError("out= is only supported for named functions") from None
        return _aggregate_custom(group_idx, a, func, size, fill_value, order, dtype, axis, cache, **kwargs)
    ret = aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    if out is not None:
        if out.shape != ret.shape or out.dtype != ret.dtype:
            raise TypeError(
                f"out must have shape {ret.shape} and dtype {ret.dtype}, got shape {out.shape} and dtype {out.dtype}"
            )
        out[...] = ret
        return out
    return ret


def _aggregate_custom(
    group_idx,
    a,
    func,
    size,
    fill_value,
    order,
    dtype,
    axis,
    cache,
    **kwargs,
):
    """Slow dispatch path for custom callables and unknown function names."""
    func = get_func(func, aliasing, _impl_dict)
    if not isinstance(func, str):
        if cache in (None, False):
            # Keep None and False in order to accept empty dictionaries
            aggregate_op = AggregateGeneric(func)
        elif cache is True:
            aggregate_op = _get_cached_generic(func)
        else:
            aggregate_op = cache.get(func)
            if aggregate_op is None:
                aggregate_op = AggregateGeneric(func)
                cache[func] = aggregate_op
        return aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    else:
        func = _impl_dict[func]
        return func(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)


@functools.lru_cache(maxsize=128)
def _get_cached_generic(func):
    """LRU-cache AggregateGeneric instances per user callable (bounded)."""
    return AggregateGeneric(func)


aggregate.__doc__ = (
    """
    This is the numba implementation of aggregate.

    This implementation accepts two additional keyword arguments:

    out: default=None
        an array the packed result is copied into (matching shape and dtype
        are required).  The array is returned, so calls can be chained.
    cache: default=True
        when aggregating with a custom callable ``func``, the compiled
        implementation is cached so that subsequent calls with the same
        function are fast.  Set to ``False`` to disable caching, or pass
        your own dictionary to control the caching manually.  With caching
        enabled (the default), callables are kept in a bounded LRU cache of
        128 entries.
    """
    + aggregate_common_doc
)


@nb.njit(cache=True)
def step_count(group_idx):
    """Return the amount of index changes within group_idx."""
    cmp_pos = 0
    steps = 1
    if len(group_idx) < 1:
        return 0
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            steps += 1
    return steps


@nb.njit(cache=True)
def step_indices(group_idx):
    """Return the edges of areas within group_idx, which are filled with the same value."""
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, np.int64)
    indices[0] = 0
    indices[-1] = group_idx.size
    cmp_pos = 0
    ri = 1
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            indices[ri] = i
            ri += 1
    return indices


@njit_stable
def _gather_1d(ret, group_idx, out):
    for i in range(group_idx.size):
        out[i] = ret[group_idx[i]]
    return out


def unpack_into(group_idx, ret, out):
    """Like ``unpack``, but gathers into a caller-provided array.

    ``out`` must match the shape of ``ret[group_idx]`` and the dtype of
    ``ret``.  Contiguous 1d inputs are gathered by a jitted loop (noticeably
    faster than fancy indexing at large sizes), anything else falls back to
    fancy indexing assignment.
    """
    group_idx = np.asanyarray(group_idx)
    shape = group_idx.shape + ret.shape[1:]
    if out.shape != shape:
        raise ValueError(f"out with shape {out.shape} does not match the expected shape {shape}")
    if out.dtype != ret.dtype:
        raise TypeError(f"out dtype {out.dtype} does not match the result dtype {ret.dtype}")
    if group_idx.ndim == 1 and ret.ndim == 1 and group_idx.flags.c_contiguous and out.flags.c_contiguous:
        return _gather_1d(ret, group_idx, out)
    out[...] = ret[group_idx]
    return out
