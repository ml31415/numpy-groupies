from __future__ import division

import numba as nb
import numpy as np

from .utils import aggregate_common_doc, funcs_no_separate_nan, get_func, isstr
from .utils_numpy import aliasing, check_dtype, check_fill_value, input_validation


class AggregateOp(object):
    """
    Every subclass of AggregateOp handles a different aggregation operation. There are
    several private class methods that need to be overwritten by the subclasses
    in order to implement different functionality.

    On object instantiation, all necessary static methods are compiled together into
    two jitted callables, one for scalar arguments, and one for arrays. Calling the
    instantiated object picks the right cached callable, does some further preprocessing
    and then executes the actual aggregation operation.
    """

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

    def __call__(self, group_idx, a, size=None, fill_value=0, order='C',
                 dtype=None, axis=None, ddof=0):
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False, func=self.func)
        group_idx, a, flat_size, ndim_idx, size, unravel_shape = iv

        # TODO: The typecheck should be done by the class itself, not by check_dtype
        dtype = check_dtype(dtype, self.func, a, len(group_idx))
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
        """ Compile a jitted function doing the hard part of the job """
        _valgetter = cls._valgetter_scalar if scalar else cls._valgetter
        valgetter = nb.njit(_valgetter)
        outersetter = nb.njit(cls._outersetter)

        _cls_inner = nb.njit(cls._inner)
        if nans:
            def _inner(ri, val, ret, counter, mean):
                if not np.isnan(val):
                    _cls_inner(ri, val, ret, counter, mean)
            inner = nb.njit(_inner)
        else:
            inner = _cls_inner

        def _loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # fill_value and ddof need to be present for being exchangeable with loop_2pass
            size = len(ret)
            rng = range(len(group_idx) - 1, -1, -1) if reverse else range(len(group_idx))
            for i in rng:
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                inner(ri, val, ret, counter, mean)
                outersetter(outer, i, ret[ri])
        return nb.njit(_loop, nogil=True)

    @staticmethod
    def _valgetter(a, i):
        return a[i]

    @staticmethod
    def _valgetter_scalar(a, i):
        return a

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
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
        loop = super(Aggregate2pass, cls).callable(nans=nans, reverse=reverse, scalar=scalar)

        _2pass_inner = nb.njit(cls._2pass_inner)

        def _loop2(ret, counter, mean, fill_value, ddof):
            for ri in range(len(ret)):
                if counter[ri] > ddof:
                    ret[ri] = _2pass_inner(ri, ret, counter, mean, ddof)
                else:
                    ret[ri] = fill_value
        loop2 = nb.njit(_loop2)

        def _loop_2pass(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof)
            loop2(ret, counter, mean, fill_value, ddof)
        return nb.njit(_loop_2pass)

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        raise NotImplementedError("subclasses need to overwrite _2pass_inner")

    @classmethod
    def _finalize(cls, ret, counter, fill_value):
        """Copying the fill value is already done in the 2nd pass"""
        pass


class AggregateNtoN(AggregateOp):
    """Base class for cumulative functions, where the output size matches the input size."""
    outer = True

    @staticmethod
    def _outersetter(outer, i, val):
        outer[i] = val


class AggregateGeneric(AggregateOp):
    """Base class for jitting arbitrary functions."""
    counter_fill_value = None

    def __init__(self, func, **kwargs):
        self.func = func
        self.__dict__.update(kwargs)
        self._jitfunc = self.callable(self.nans)

    def __call__(self, group_idx, a, size=None, fill_value=0, order='C',
                 dtype=None, axis=None, ddof=0):
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis, check_bounds=False)
        group_idx, a, flat_size, ndim_idx, size, _ = iv

        # TODO: The typecheck should be done by the class itself, not by check_dtype
        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        check_fill_value(fill_value, dtype, func=self.func)
        input_dtype = type(a) if np.isscalar(a) else a.dtype
        ret, _, _, _ = self._initialize(flat_size, fill_value, dtype, input_dtype, group_idx.size)
        group_idx = np.ascontiguousarray(group_idx)

        sortidx = np.argsort(group_idx, kind='mergesort')
        self._jitfunc(sortidx, group_idx, a, ret)

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            ret = ret.reshape(size, order=order)
        return ret

    def callable(self, nans=False):
        """Compile a jitted function and loop it over the sorted data."""
        jitfunc = nb.njit(self.func, nogil=True)

        def _loop(sortidx, group_idx, a, ret):
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
                ret[ri] = jitfunc(a_srt[start_idx:stop_idx])
        return nb.njit(_loop, nogil=True)


class Sum(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] += val


class Prod(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] *= val


class Len(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] += 1


class All(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] &= bool(val)


class Any(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] |= bool(val)


class Last(AggregateOp):
    counter_fill_value = None

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        ret[ri] = val


class First(Last):
    reverse = True


class AllNan(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] &= val == val


class AnyNan(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] = 0
        ret[ri] |= val != val


class Max(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        if counter[ri]:
            ret[ri] = val
            counter[ri] = 0
        elif ret[ri] < val:
            ret[ri] = val


class Min(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        if counter[ri]:
            ret[ri] = val
            counter[ri] = 0
        elif ret[ri] > val:
            ret[ri] = val


class ArgMax(AggregateOp):
    mean_fill_value = np.nan

    @staticmethod
    def _valgetter(a, i):
        return a[i], i

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        cmp_val, arg = val
        if counter[ri]:
            mean[ri] = cmp_val
            ret[ri] = arg
            counter[ri] = 0
        elif mean[ri] < cmp_val:
            mean[ri] = cmp_val
            ret[ri] = arg


class ArgMin(ArgMax):
    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        cmp_val, arg = val
        if counter[ri]:
            mean[ri] = cmp_val
            ret[ri] = arg
            counter[ri] = 0
        elif mean[ri] > cmp_val:
            mean[ri] = cmp_val
            ret[ri] = arg


class Mean(Aggregate2pass):
    forced_fill_value = 0
    counter_fill_value = 0
    counter_dtype = int

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] += 1
        ret[ri] += val

    @staticmethod
    def _2pass_inner(ri, ret, counter, mean, ddof):
        return ret[ri] / counter[ri]


class Std(Mean):
    mean_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
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


class CumSum(AggregateNtoN, Sum):
    pass


class CumProd(AggregateNtoN, Prod):
    pass


class CumMax(AggregateNtoN, Max):
    pass


class CumMin(AggregateNtoN, Min):
    pass


def get_funcs():
    funcs = dict()
    for op in (Sum, Prod, Len, All, Any, Last, First, AllNan, AnyNan, Min, Max,
               ArgMin, ArgMax, Mean, Std, Var,
               CumSum, CumProd, CumMax, CumMin):
        funcname = op.__name__.lower()
        funcs[funcname] = op(funcname)
        if funcname not in funcs_no_separate_nan:
            funcname = 'nan' + funcname
            funcs[funcname] = op(funcname, nans=True)
    return funcs


_impl_dict = get_funcs()
_default_cache = {}


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, cache=None, **kwargs):
    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        if cache in (None, False):
            aggregate_op = AggregateGeneric(func)
        else:
            if cache is True:
                cache = _default_cache
            aggregate_op = cache.setdefault(func, AggregateGeneric(func))
        return aggregate_op(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)
    else:
        func = _impl_dict[func]
        return func(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)


aggregate.__doc__ = """
    This is the numba implementation of aggregate.
    """ + aggregate_common_doc


@nb.njit(nogil=True, cache=True)
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


@nb.njit(nogil=True, cache=True)
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
