from __future__ import division
import numba as nb
import numpy as np

from .utils import (aliasing, get_func, input_validation, check_dtype,
                    _doc_str, isstr, check_fill_value, _no_separate_nan_version)


class AggregateOp(object):
    forced_fill_value = None
    counter_fill_value = 1
    counter_dtype = bool
    mean_fill_value = None
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
        iv = input_validation(group_idx, a, size=size, order=order, axis=axis)
        group_idx, a, flat_size, ndim_idx, size = iv

        # TODO: The typecheck should be done by the class itself, not by check_dtype
        dtype = check_dtype(dtype, self.func, a, len(group_idx))
        check_fill_value(fill_value, dtype)
        ret, counter, mean = self._initialize(flat_size, fill_value, dtype)
        group_idx = np.ascontiguousarray(group_idx)

        if not np.isscalar(a):
            a = np.ascontiguousarray(a)
            jitfunc = self._jit_non_scalar
        else:
            jitfunc = self._jit_scalar
        jitfunc(group_idx, a, ret, counter, mean, fill_value, ddof)
        self._finalize(ret, counter, mean, fill_value)

        if self.nans:
            ret = ret[1:]

        # Deal with ndimensional indexing
        if ndim_idx > 1:
            ret = ret.reshape(size, order=order)
        return ret

    def _initialize(self, flat_size, fill_value, dtype):
        if self.nans:
            # For avoiding branches
            flat_size += 1
        if self.forced_fill_value is None:
            ret = np.full(flat_size, fill_value, dtype=dtype)
        else:
            ret = np.full(flat_size, self.forced_fill_value, dtype=dtype)
        counter = np.full_like(ret, self.counter_fill_value, dtype=self.counter_dtype)
        if self.mean_fill_value is not None:
            mean = np.full_like(ret, self.mean_fill_value, dtype=ret.dtype)
        else:
            mean = None
        return ret, counter, mean

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        """ Compile a jitted function doing the hard part of the job """
        if scalar:
            def _valgetter(a, i):
                return a
        else:
            def _valgetter(a, i):
                return a[i]
        valgetter = nb.njit(_valgetter)

        if nans:
            def _ri_redir(i, val):
                """ Redirect any write access to the output array to it's 
                    first field, if we encounter a nan value. This first field
                    was reserved in advance for dummy access. Shift the index
                    by 1, if we don't have a nan value.
                """
                return (i + 1) * (val == val)
        else:
            def _ri_redir(i, val):
                return i
        ri_redir = nb.njit(_ri_redir)

        inner = _inner = nb.njit(cls._inner)

        def _loop(group_idx, a, ret, counter, mean, fill_value, ddof):
            rng = range(len(group_idx) - 1, -1 , -1) if reverse else range(len(group_idx))
            for i in rng:
                val = valgetter(a, i)
                ri = ri_redir(group_idx[i], val)
                inner(ri, val, ret, counter, mean)
        loop = nb.njit(_loop, nogil=True, cache=True)

        _2pass = cls._2pass()
        if _2pass is None:
            return loop

        _2pass = nb.njit(_2pass, nogil=True, cache=True)
        def loop_2pass(group_idx, a, ret, counter, mean, fill_value, ddof):
            loop(group_idx, a, ret, counter, mean, fill_value, ddof)
            _2pass(ret, counter, mean, fill_value, ddof)
        return loop_2pass

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        raise NotImplementedError("subclasses need to overwrite _inner")

    @classmethod
    def _2pass(cls):
        return

    @classmethod
    def _finalize(cls, ret, counter, mean, fill_value):
        if cls.forced_fill_value is not None and fill_value != cls.forced_fill_value:
            ret[counter] = fill_value


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
        cp = (ret[ri] < val) | counter[ri]
        ncp = bool(not cp)
        counter[ri] &= ncp
        ret[ri] = cp * val + ncp * ret[ri]


class Min(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        cp = (ret[ri] > val) | counter[ri]
        ncp = bool(not cp)
        counter[ri] &= ncp
        ret[ri] = cp * val + ncp * ret[ri]


class Mean(AggregateOp):
    counter_fill_value = 0
    counter_dtype = int

    @staticmethod
    def _inner(ri, val, ret, counter, mean):
        counter[ri] += 1
        ret[ri] += val

    @classmethod
    def _2pass(cls):
        _2pass_inner = nb.njit(cls._2pass_inner)
        def _2pass_loop(ret, counter, mean, fill_value, ddof):
            for ri in range(len(ret)):
                if counter[ri]:
                    ret[ri] = _2pass_inner(ri, ret, counter, mean, ddof)
                else:
                    ret[ri] = fill_value
        return _2pass_loop

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



def get_funcs():
    funcs = dict()
    for op in (Sum, Prod, Len, All, Any, Last, First, AllNan, AnyNan, Min, Max, Mean, Std, Var):
        funcname = op.__name__.lower()
        funcs[funcname] = op(funcname)
        if funcname not in _no_separate_nan_version:
            funcname = 'nan' + funcname
            funcs[funcname] = op(funcname, nans=True)
    return funcs


_impl_dict = get_funcs()

def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        raise NotImplementedError("generic functions not supported in numba"
                                  " implementation of aggregate.")
    else:
        func = _impl_dict[func]
        return func(group_idx, a, size, fill_value, order, dtype, axis, **kwargs)

aggregate.__doc__ = """
    This is the numba implementation of aggregate.

    """ + _doc_str


@nb.njit(nogil=True, cache=True)
def step_count(group_idx):
    """ Determine the size of the result array
        for contiguous data
    """
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
def _step_indices_loop(group_idx, indices):
    cmp_pos = 0
    ri = 1
    for i in range(1, len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            indices[ri] = i
            ri += 1


def step_indices(group_idx):
    """ Get the edges of areas within group_idx, which are filled 
        with the same value
    """
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, int)
    indices[0] = 0
    indices[-1] = group_idx.size
    _step_indices_loop(group_idx, indices)
    return indices
