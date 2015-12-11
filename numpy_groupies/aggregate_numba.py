import numpy as np

from .utils import (get_func, aliasing, input_validation, check_dtype,
                    _doc_str, isstr)
import numba

"""
================================
Written for numba version 0.22.1
================================

Some notes on numba quirks and performance hacks....

Note that numba currently does not implement bounds checking but it does
implement wraparound indexing...which is a bit strange.
Use unsigned ints to tell the compiler your inds are non-negative. Bounds
checking should be essentially free at run-time due to branch-prediction.

We rely on the fact that True is precisely 1 and False precisely 0. In fact,
not only do we rely on this, but we have to explicitly convert truthy-but-not-1
values to be exactly 1 (see finish_bool_func).
I think this is all safe to do: http://stackoverflow.com/q/5369770/2399799

We rely on the fact that nans don't compare equal (and the above 1/0 thing)
in order to write a manual version of isnan, which somehow makes numba much
happier than using np.isnan:  a_ii != a_ii, and we have to manually
inline it too, which is wierd.
  
Note that numba requires you to jit-decorate any functions you want to call
from inside another jit-decorated function.

For some reason, if you put x.view(some_dtype) inside a loop, numba is unable
to move it outside the loop, which makes the loop super slow.  It's also not 
possible to have an ndarray take one of two  different dtypes dependant on
a compile-time (i.e. closure variable) flag.  Thus we end up wrapping the jit
function in another function which handles the flag outside numba.

Although numba supports zip, it's a lot faster to use range(n) and indexing.

Note that sometimes you have to spend a bit of time looking for the correct
way of doing something, e.g. numba's implementation of python's builtin min
function is about 10x faster than np.minimum or an if statement, or a 
crazy 2-element lookup table + 1-bit-bool-indexing ...but I only disovered
that the builtin after messing around for a while with the other versions.
"""

mode_equals_fill = -100
mode_pos_inf = -101
mode_neg_inf = -102
            
def jitted_loop(initial_value_mode=0, intrusive_used=False, 
                end_func=None, result_view_dtype=None, iter_reversed=False,
                int_version=None):
    """
    initial_value_mode - a num > -10 or one of the enum values (see code).
    
    intrusive_used - if True the iter_func records the is_used/count stuff
        "intrusively" in result.

    result_view_dtype - if not None, then a view of result is created with this
        other type. see note about x.view above.    
        
    end_func - lambda result_view, fill_value, need_to_fill, initial_value: None.
    
    iter_reversed - True reverses iteration order (used by _first)

    int_version - an alternative jitted function to use if input is int type    
    """
    def iter_decorator(iter_func):
        iter_func = numba.jit(iter_func, nopython=True) 
        if end_func:
            end_func_ = numba.jit(end_func, nopython=True)
        else:
            @numba.jit(nopython=True)
            def end_func_(res, fill_value, is_unused, initial_value):
                if initial_value == fill_value:
                    return
                res[is_unused] = fill_value
            
        @numba.jit(nopython=True)
        def jitted_loop(group_idx, a, result, result_view, size, fill_value,
                        initial_value, result_c_order=True, ddof=0):
            """
            result is an empty 1D array of the chosen dtype.
            TODO: ideally, wherever we need counts/is_used flag we would
                  prefer it to be intrusive, unless that screws up alignment.
                  at the end we can collapse the extra-large array down to 
                  it's final size, possibly even doing it in-place.
            """
            result[:] = initial_value
            if not intrusive_used and initial_value != fill_value:
                need_to_fill = np.ones(len(result), dtype=np.bool_)
            else:
                need_to_fill = np.ones(0, dtype=np.bool_) # using None causes problems
                
            # TODO: reshape result according to size and order_out
                
            # Form 1
            assert(group_idx.ndim == 1 and a.ndim == 1)
            
            n = len(group_idx)
            for ii in range(n):
                if iter_reversed:
                    gidx_ii = group_idx[-1-ii]
                    a_ii = a[-1-ii]
                else:
                    gidx_ii = group_idx[ii]
                    a_ii = a[ii]
                assert(0 <= gidx_ii < len(result))                 
                if not intrusive_used and initial_value != fill_value:
                    need_to_fill[gidx_ii] = False
                iter_func(gidx_ii, a_ii, result_view, ii)
        
            # apply fill_value if neccessary and do any final processing
            end_func_(result_view, fill_value, need_to_fill, initial_value) 
                
            return result  
        # end jitted_loop
        
        def jitted_loop_wrapped(group_idx, a, result, size, fill_value,
                                *args, **kwargs):
            if int_version and issubclass(a.dtype.type, np.integer):
                # delegate to alternative jit function
                print "delegating to int version"
                return int_version(group_idx, a, result, size, fill_value,
                                   *args, **kwargs)
                
            # See note at top about x.view
            result_view = result if result_view_dtype is None else \
                            result.view(result_view_dtype)
            # This bit also is difficult/impossible to do inside numba...
            if initial_value_mode > -10:
                initial_value = initial_value_mode
            elif initial_value_mode == mode_equals_fill:
                initial_value = fill_value
            elif initial_value_mode == mode_pos_inf:
                initial_value = np.inf if not issubclass(result.dtype.type, np.integer)\
                                  else np.iinfo(result.dtype).max 
            elif initial_value_mode == mode_neg_inf:
                initial_value = -np.inf if not issubclass(result.dtype.type, np.integer)\
                                  else np.iinfo(result.dtype).min
            else:
                raise TypeError("unkown initial_value_mode")

            return jitted_loop(group_idx, a, result, result_view, size, 
                                   fill_value, initial_value, *args, **kwargs)
        # end jitted_loop_wrapped
                
        return jitted_loop_wrapped
    return iter_decorator
    
"""
Our bool functions store an "intrusive" flag that records
for each group  whether or not it has been used, i.e. if 0
then it needs to be set to fill_value at the end.  They can do
this because their actual data consists of just 1 bit. 
"""
special_shift = 7 # takes group_used to group_true
group_true = 1
group_used = group_true << special_shift
def finish_bool_func(res, fill_value, None_, initial_value):
    """
    numba strugles to remove branches for if statements, so we
    do the hard work manually....
    
    fill_value equals   fill_value is 1      fill_value is 0
    intial_value        intial_value is 0    initial_value is 1
    ----------------    -----------------    -----------------
                        used | x | res       used | x  | res
                        1      1    1        1      1    1
         easy           1      0    0        1      0    0
                        0      1    *        0      1    0
                        0      0    1        0      0    *
    ----------------    -----------------    -----------------
    res: x              res: used==x         res: used & res
    
    * cannot occur due to initial_value and used=0
    """  
    for ii, res_ii in enumerate(res):
        if fill_value == initial_value:
            res[ii] = res_ii & group_true
        elif fill_value:
            res[ii] = (res_ii >> special_shift) == (res_ii & group_true) 
        else:
            res[ii] = (res_ii >> special_shift) & (res_ii & group_true) 
            
            
@jitted_loop(initial_value_mode=0)
def _sum(gidx_ii, a_ii, res, ii):
    res[gidx_ii] += a_ii

@jitted_loop(initial_value_mode=0, int_version=_sum)
def _nansum(gidx_ii, a_ii, res, ii):
    # you can only have nans in floats, so ints can do basic sum

    # hack to avoid branching on nans..store nan values in 0 group
    # and everything else shifted by one position to the right.
    is_nan = a_ii != a_ii
    res[(gidx_ii + 1) * (not is_nan)] += a_ii
    # TODO: need to finish this by shifting values back by one, or
    # changing view of array...also need to actually request the extra
    # 1 value in the length of result.
    
    
@jitted_loop(initial_value_mode=1)
def _prod(gidx_ii, a_ii, res, ii):
    res[gidx_ii] *= a_ii

@jitted_loop(initial_value_mode=mode_pos_inf)
def _min(gidx_ii, a_ii, res, ii):
    res[gidx_ii] = min(a_ii, res[gidx_ii])

@jitted_loop(initial_value_mode=mode_neg_inf) 
def _max(gidx_ii, a_ii, res, ii):
    res[gidx_ii] = max(a_ii, res[gidx_ii])

@jitted_loop(initial_value_mode=mode_equals_fill, iter_reversed=True)
def _first(gidx_ii, a_ii, res, ii):
    res[gidx_ii] = a_ii
    
@jitted_loop(initial_value_mode=mode_equals_fill)
def _last(gidx_ii, a_ii, res, ii):
    res[gidx_ii] = a_ii
    
@jitted_loop(initial_value_mode=group_true, intrusive_used=True, 
             end_func=finish_bool_func, result_view_dtype=np.uint8)
def _all(gidx_ii, a_ii, res, ii):
    res[gidx_ii] = group_used | (res[gidx_ii] & (a_ii != 0))

@jitted_loop(initial_value_mode=0, intrusive_used=True,
             end_func=finish_bool_func, result_view_dtype=np.uint8)
def _any(gidx_ii, a_ii, res, ii):
    res[gidx_ii] |= group_used | (a_ii != 0)
        
@jitted_loop(initial_value_mode=1, intrusive_used=True,
             end_func=finish_bool_func, result_view_dtype=np.uint8)
def _allnan(gidx_ii, a_ii, res, ii):
    res[gidx_ii] = group_used | (res[gidx_ii] & (a_ii != a_ii))
        
@jitted_loop(initial_value_mode=0, intrusive_used=True,
             end_func=finish_bool_func, result_view_dtype=np.uint8)
def _anynan(gidx_ii, a_ii, res, ii):
    res[gidx_ii] |= group_used | (a_ii != a_ii)

    

"""
def _iter_mean(g_idx, a_ii, res, *args):
    counter[gidx_ii] += 1
    res[gidx_ii] += a_ii

def _iter_std(gidx_ii, a_ii, res, *args):
    counter[gidx_ii] += 1
    tmp[gidx_ii] += a_ii
    res[gidx_ii] += a_ii * a_ii

def _finish_mean(res, counter, tmp, fillvalue):
    for i in range(len(res)):
        if counter[i]:
            res[i] /= counter[i]
        else:
            res[i] = fillvalue

def _finish_std(res, counter, tmp, fillvalue):
    for i in range(len(res)):
        if counter[i]:
            mean = tmp[i] / counter[i]
            res[i] = np.sqrt(res[i] / counter[i] - mean * mean)
        else:
            res[i] = fillvalue


"""

_impl_dict = dict(sum=_sum, nansum=_nansum, prod=_prod, min=_min, max=_max,
                  last=_last, first=_first, all=_all, any=_any, allnan=_allnan, 
                  anynan=_anynan)

def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):

    group_idx, a, flat_size, ndim_idx, size = input_validation(group_idx, a,
                size=size, order=order, axis=axis, check_bounds=False,
                ravel_group_idx=False)
    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        # TODO: this isn't hard to fix, but I'll leave it for now
        raise NotImplementedError("generic functions not supported in numba"
                                  " implementation of aggregate.")
    else:
        # final prep and launch the function.    
        dtype = check_dtype(dtype, func, a, flat_size)
        func = _impl_dict[func]
        empty_result = np.empty(flat_size, dtype=dtype)
        ret = func(group_idx, a, empty_result, size, fill_value=fill_value, 
                   result_c_order=order=='C', **kwargs)

    return ret
    
    
aggregate.__doc__ =  """
    This is the numba implementation of aggregate.

    You can use negative indexing in this version.
    
    If you provide a size value we will be able to iterate over ``a`` and 
    ``group_idx`` exactly once.
    
    TODO: make this true even for the case where we broadcast ``group_idx``
    with an ``axis`` argument.
    """ + _doc_str


