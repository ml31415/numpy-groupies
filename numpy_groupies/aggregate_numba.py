import functools
import hashlib
import inspect

import numba as nb
import numpy as np

from .utils import (
    DEFAULT_FILL_VALUE,
    aggregate_common_doc,
    aliasing,
    check_dtype,
    check_fill_value,
    check_nton_shape,
    funcs_no_separate_nan,
    get_func,
    input_validation,
    resolve_fill_value,
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
        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        fill_value = resolve_fill_value(self.func, fill_value, dtype)
        check_fill_value(fill_value, dtype, func=self.func)
        input_dtype = type(a) if np.isscalar(a) else a.dtype
        ret, counter, mean, outer = self._initialize(flat_size, fill_value, dtype, input_dtype, group_idx.size)
        group_idx = np.ascontiguousarray(group_idx)

        if not np.isscalar(a):
            a = np.ascontiguousarray(a)
            jitfunc = self._jit_non_scalar
        else:
            jitfunc = self._jit_scalar
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
            dtype = cls.mean_dtype if cls.mean_dtype else input_dtype
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
        # if counter[ri]:
        #     ret[ri] = val
        #     counter[ri] = 0
        # elif ret[ri] < val:
        #     ret[ri] = val
        first = counter[ri]
        counter[ri] = 0
        cur = ret[ri]
        ret[ri] = val if (first or cur < val) else cur


class Min(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        # select-form on purpose, see Max._inner
        # if counter[ri]:
        #     ret[ri] = val
        #     counter[ri] = 0
        # elif ret[ri] > val:
        #     ret[ri] = val
        first = counter[ri]
        counter[ri] = 0
        cur = ret[ri]
        ret[ri] = val if (first or cur > val) else cur


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
        ret[ri] += val * val


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
        ret[ri] += val * val

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        mean2 = mean[ri] * mean[ri]
        return np.sqrt((ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof))


class Var(Std):
    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        mean2 = mean[ri] * mean[ri]
        return (ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof)


class Median(AggregateOp):
    """The median is an order statistic, so unlike the streaming reductions
    it needs group-ordered data: values are placed group by group via
    counting sort (O(n)) and the middle of each group is then *selected*
    via partition (O(n) on average) instead of sorting within the groups."""

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
        # the lower middle is the maximum of the unsorted left half
        return (np.max(part[: m // 2]) + part[m // 2]) / 2


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


def aggregate(
    group_idx,
    a,
    func="sum",
    size=None,
    fill_value=DEFAULT_FILL_VALUE,
    order="C",
    dtype=None,
    axis=None,
    cache=True,
    **kwargs,
):
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

    This implementation accepts one additional keyword argument:

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
