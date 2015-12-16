import numpy as np
import numba # version 0.22.1
from .utils import (get_func, aliasing, input_validation, check_dtype,
                    _doc_str, isstr)

    
""" =======================================================================
    The decorator and enums used by the various functions 
    ======================================================================= """

# intial_value_mode enums (can also be a number > -10)
MODE_EQUALS_FILL = -100
MODE_POS_INF = -101
MODE_NEG_INF = -102
            
# mode enum
MODE_SIMPLE = 0
MODE_BOOL = 1
MODE_SPECIAL = 2

def _unbound_to_free(unbound):
    # a hack to allow us to abuse the class definition syntax
    return getattr(unbound, 'im_func', unbound.__func__) # python 2, or 3?
    
def _aggregate_loop(mode=MODE_SIMPLE,
                    initial_value_mode=0,   
                    dtype_func=lambda dt, fill_value: dt,
                    iter_reversed=False,
                    _nan_to_0=False,
                    _int_version=None,
                    ):
    """
    This decorator is only intended for use within this file, but it is a bit
    complicated, so merits some explanation:

    mode: this takes one of three values, and controls how the decorator works:
        
    MODE_SIMPLE - the decorator will record is_used if 
                  initial_value != fill_value, and it will do the filling after 
                  the loop. 
                    
    MODE_BOOL - if initial_value != fill_value, then iter_func needs to set 
                both the group_used bit and the group true bit, provided as an
                argument to iter_func. The decorator will do the post-loop 
                stuff. The work array is of dtype=uint8.
    
    MODE_SPECIAL -  For this case, the wrapped object should be a class, not a
                function, with four methods named: work_dtype_func, start_func,
                iter_func, and end_func. We abuse the python class def syntax,
                using it only to collect together related functions, i.e. we
                do not use any instance binding. iter_func is the same as for 
                the other modes, but see the code or example usage for details
                of the others.  The idea is that in this case the deorator 
                leaves most of the work to the wrapped object.
        
    initial_value_mode - number > -10; or MODE_EQUALS_FILL to match fill_value;
                     or MODE_NEG_INF/MODE_POS_INF for dtype's max/min values.
        
    iter_reversed - True reverses iteration order (used by _first)

    _nan_to_0    -  if _nan_to_0 is True the work array is one element longer 
    _int_version    than the actual result, with all nan data being diverted 
                    into element 0.  Rather than supplying this value directly 
                    in the decorator, use the get_nan_version() method on the 
                    decorated function, which will build a second version of 
                    the function with nan_to_0 set to True.  It will also divert
                    integer input to the non-nan version, because ints can't 
                    be nan.
    
    """
    def iter_decorator(decorated_obj):            
        # private const for MODE_BOOL             
        IS_USED_B = 7 # the index of the bit indicating is_used

        # if decorated object is a class def, then extract the functions
        if mode == MODE_SPECIAL:
            iter_func = _unbound_to_free(decorated_obj.iter_func)
            start_func = _unbound_to_free(decorated_obj.start_func)
            end_func = _unbound_to_free(decorated_obj.end_func)
            work_dtype_func = _unbound_to_free(decorated_obj.work_dtype_func)
            dtype_func_ = _unbound_to_free(decorated_obj.dtype_func) if \
                          hasattr(decorated_obj, 'dtype_func') else dtype_func
        else:
            iter_func = decorated_obj
            dtype_func_ = dtype_func
            
        # jit the main iter function
        iter_func = numba.jit(iter_func, nopython=True)         
    
        # jit the start and end functions, which depends on mode, _nan_to_0 
        # and possibly the end_func arg passed into the decorator
        if mode == MODE_SIMPLE:
            @numba.jit(nopython=True)
            def start_func_(work, initial_value):
                work[:] = initial_value
                
            @numba.jit(nopython=True)
            def end_func_(work, is_used, result, initial_value, fill_value, ddof):
                if initial_value == fill_value:
                    for ii in range(len(result)):
                        if _nan_to_0:
                            result[ii] = work[ii+_nan_to_0]
                        else:
                            result[ii] = (work[ii+_nan_to_0] if is_used[ii+_nan_to_0]
                                          else fill_value)
                            
        elif mode == MODE_BOOL: 
            assert(not _nan_to_0)
            @numba.jit(nopython=True)
            def start_func_(work, initial_value):
                work[:] = initial_value
                
            @numba.jit(nopython=True)
            def end_func_(work, ignore, result, initial_value, fill_value, ddof):
                # see notes at bottom of file
                if initial_value != fill_value:
                    for ii in range(len(result)):
                        if fill_value:
                            work[ii] = ((work[ii] >> IS_USED_B) == (work[ii] & 1))
                        else: 
                            work[ii] = ((work[ii] >> IS_USED_B) & (work[ii] & 1))
                            
        else: # mode == MODE_SPECIAL
            start_func_ = numba.jit(start_func, nopython=True)
                
            end_iter = numba.jit(end_func, nopython=True)
            @numba.jit(nopython=True)
            def end_func_(work, ignore, result, initial_value, fill_value, ddof):            
                for ii in range(len(result)):
                    result[ii] = end_iter(work[ii + _nan_to_0], initial_value, 
                                            fill_value, ddof)
                    
        # this is the main function we generate, though it will get wrapped
        @numba.jit(nopython=True)
        def jitted_loop(group_idx, a, work, is_used, result, 
                        fill_value, initial_value, ddof):

            # prepare the array
            start_func_(work, initial_value)
            
            # TODO: reshape result according to size and order_out
            assert(group_idx.ndim == 1 and a.ndim == 1) # for now we do Form 1
            
            # if we need it, this encodes is_used as a single bit
            is_used_mask = (initial_value != fill_value) << IS_USED_B
            
            # the main loop, this is what we're here for...
            for ii in range(len(group_idx)):
                gidx_ii = group_idx[-1-ii] if iter_reversed else group_idx[ii]
                a_ii = a[-1-ii] if iter_reversed else a[ii]
                assert(0 <= gidx_ii < len(result))
                if _nan_to_0:
                    gidx_ii = (gidx_ii + 1) * (a_ii == a_ii) 
                if len(is_used):
                    is_used[gidx_ii] = True
                iter_func(gidx_ii, a_ii, work, ii, is_used_mask)
        
            # post-loop operations on work and result...
            end_func_(work, is_used, result, initial_value, fill_value, ddof) 
        # end _aggregate_loop -----------------------------------------------------
        
        # The above numba function needs to be wrapped in python prep code
        def jitted_loop_wrapped(group_idx, a, flat_size, res_dtype, size,
                                fill_value, ddof=0):
            if _int_version and issubclass(a.dtype.type, np.integer):
                # delegate to alternative jit function
                return _int_version(group_idx, a, flat_size, res_dtype, size, 
                                   fill_value, ddof)
                
            # get sensible result dtype
            res_dtype = dtype_func_(res_dtype, fill_value)
            
            # select the intial value
            if initial_value_mode > -10:
                initial_value = initial_value_mode
            elif initial_value_mode == MODE_EQUALS_FILL:
                initial_value = fill_value
            elif initial_value_mode == MODE_POS_INF:
                initial_value = np.inf if not issubclass(res_dtype.type, np.integer)\
                                  else np.iinfo(res_dtype).max 
            elif initial_value_mode == MODE_NEG_INF:
                initial_value = -np.inf if not issubclass(res_dtype.type, np.integer)\
                                  else np.iinfo(res_dtype).min
            else:
                raise TypeError("unkown initial_value_mode")
            
            # allocate memory/views for work and result
            # note that result may overlap with work
            work_size = flat_size + _nan_to_0
            is_used = np.empty(0, dtype=bool)
            if mode == MODE_SPECIAL:
                work = np.empty(work_size, dtype=work_dtype_func(res_dtype, a.dtype))
                result = np.empty(flat_size, dtype=res_dtype)
            elif mode == MODE_BOOL:
                work = np.empty(work_size, np.uint8)
                result = work[:flat_size].view(dtype=np.bool_)
            elif mode == MODE_SIMPLE:
                work = np.empty(work_size, dtype=res_dtype)                
                if initial_value != fill_value:
                    is_used = np.zeros(work_size, dtype=bool)
                result = work[:flat_size]
            else:
                raise TypeError("unknown mode")

            jitted_loop(group_idx, a, work, is_used, result, # return value in result
                        fill_value, initial_value, ddof)
            return result
        # end _aggregate_loop_wrapped ---------------------------------------------

        # add a method for building a nan-version (not valid in all cases)            
        jitted_loop_wrapped.get_nan_version = lambda: _aggregate_loop(
            mode=mode, initial_value_mode=initial_value_mode,
            iter_reversed=iter_reversed, _nan_to_0=True,
            _int_version=jitted_loop_wrapped)(decorated_obj)
        
        return jitted_loop_wrapped
    return iter_decorator
           
           
""" =======================================================================
    Functions with mode == MODE_SIMPLE
    ======================================================================= """

@_aggregate_loop(initial_value_mode=0)
def _sum(gidx_ii, a_ii, work, *args):
    work[gidx_ii] += a_ii
    
@_aggregate_loop(initial_value_mode=1)
def _prod(gidx_ii, a_ii, work, *args):
    work[gidx_ii] *= a_ii
    
@_aggregate_loop(initial_value_mode=MODE_POS_INF)
def _min(gidx_ii, a_ii, work, *args):
    work[gidx_ii] = min(a_ii, work[gidx_ii]) # if a is nan, min is work[g_idx]
        
@_aggregate_loop(initial_value_mode=MODE_NEG_INF) 
def _max(gidx_ii, a_ii, work, *args):
    work[gidx_ii] = max(a_ii, work[gidx_ii]) # if a is nan, min is work[g_idx]
    
@_aggregate_loop(initial_value_mode=MODE_EQUALS_FILL, iter_reversed=True)
def _first(gidx_ii, a_ii, work, *args):
    work[gidx_ii] = a_ii
    
@_aggregate_loop(initial_value_mode=MODE_EQUALS_FILL)
def _last(gidx_ii, a_ii, work, *args):
    work[gidx_ii] = a_ii
    
""" =======================================================================
    Functions with mode == MODE_BOOL
    ======================================================================= """

@_aggregate_loop(initial_value_mode=1, mode=MODE_BOOL)
def _all(gidx_ii, a_ii, work, ignore, IS_USED_B):
    work[gidx_ii] = IS_USED_B | (work[gidx_ii] & (a_ii != 0))

@_aggregate_loop(initial_value_mode=1, mode=MODE_BOOL)
def _allnan(gidx_ii, a_ii, work, ignore, IS_USED_B):
    work[gidx_ii] = IS_USED_B | (work[gidx_ii] & (a_ii != a_ii))
        
@_aggregate_loop(initial_value_mode=0, mode=MODE_BOOL)
def _any(gidx_ii, a_ii, work, ignore, IS_USED_B):
    work[gidx_ii] |= IS_USED_B | (a_ii != 0)
        
@_aggregate_loop(initial_value_mode=0, mode=MODE_BOOL)
def _anynan(gidx_ii, a_ii, work, ignore, IS_USED_B):
    work[gidx_ii] |= IS_USED_B | (a_ii != a_ii)

""" =======================================================================
    Functions with mode == MODE_SPECIAL
    ======================================================================= """

@_aggregate_loop(mode=MODE_SPECIAL)
class _mean:
    def work_dtype_func(res, a):
        return np.dtype([('sum', res), ('count', np.uint32)])
        
    def start_func(work, ignore):
        for ii in range(len(work)):
            work[ii].sum = 0
            work[ii].count = 0
            
    def iter_func(gidx_ii, a_ii, work, *args): 
        work_g = work[gidx_ii]
        work_g.sum += a_ii
        work_g.count += 1        
        
    def end_func(work, ignore, fill):
        # TODO: if fill is zero, cant we take advantage of 0/0=0?  
        return work.sum/work.count if work.count > 0 else fill
      

@_aggregate_loop(mode=MODE_SPECIAL)
class _var:
    def work_dtype_func(res, a):
        return np.dtype([('sum', res),('sum_sqrs', res),
                         ('shift', res), ('count', np.uint32)])
        
    def start_func(work, ignore):
        for ii in range(len(work)):
            work[ii].count = 0
            work[ii].sum_sqrs = 0
            work[ii].sum = 0
            work[ii].shift = 0
            
    def iter_func(gidx_ii, a_ii, work, *args): 
        work_g = work[gidx_ii]
        work_g.shift = a_ii * (work_g.count == 0)
        a_ii_shifted = a_ii - work_g.shift # see note at bottom of file..TODO: check this is correct
        work_g.sum += a_ii_shifted
        work_g.sum_sqrs += a_ii_shifted**2
        work_g.count +=1  
        
    def end_func(work, ignore, fill, ddof=0):
        return (work.sum_sqrs - (work.sum**2)/work.count) / (work.count-ddof) \
                if work.count > 0 else fill

# TODO: find a way to avoid copy-paste var to get std (sqrt in end_func is the
# only change made).
@_aggregate_loop(mode=MODE_SPECIAL)
class _std:
    def work_dtype_func(res, a):
        return np.dtype([('sum', res),('sum_sqrs', res),
                         ('shift', res), ('count', np.uint32)])
        
    def start_func(work, ignore):
        for ii in range(len(work)):
            work[ii].count = 0
            work[ii].sum_sqrs = 0
            work[ii].sum = 0
            work[ii].shift = 0
            
    def iter_func(gidx_ii, a_ii, work, *args): 
        work_g = work[gidx_ii]
        work_g.shift = a_ii * (work_g.count == 0)
        a_ii_shifted = a_ii - work_g.shift # see note at bottom of file..TODO: check this is correct
        work_g.sum += a_ii_shifted
        work_g.sum_sqrs += a_ii_shifted**2
        work_g.count +=1  
        
    def end_func(work, ignore, fill, ddof=0):
        return np.sqrt((work.sum_sqrs - (work.sum**2)/work.count) 
                       / (work.count-ddof)) if work.count > 0 else fill
                    
@_aggregate_loop(mode=MODE_SPECIAL, initial_value_mode=MODE_POS_INF)
class _argmin:
    def dtype_func(*args):
        return np.dtype('uint32')
        
    def work_dtype_func(res, a):
        return np.dtype([('best', a), ('idx', np.uint32)])
        
    def start_func(work, initial):
        for ii in range(len(work)):
            work[ii].best = initial # pos_inf
            work[ii].idx = -1 # but as uint
            
    def iter_func(gidx_ii, a_ii, work, ii, *args): 
        work_g = work[gidx_ii]
        new_best = min(a_ii, work_g.best)
        work_g.idx += (ii - work_g.idx) * (new_best == a_ii)
        work_g.best = new_best
        
    def end_func(work, ignore, fill, ddof=0):
        # TODO: if fill is -1, we can avoid branch.
        return work.idx if work.idx != -1 else fill
    
    
@_aggregate_loop(mode=MODE_SPECIAL, initial_value_mode=MODE_NEG_INF)
class _argmax:
    def dtype_func(*args):
        return np.dtype('uint32')
        
    def work_dtype_func(res, a):
        return np.dtype([('best', a), ('idx', np.uint32)])
        
    def start_func(work, initial):
        for ii in range(len(work)):
            work[ii].best = initial # pos_inf
            work[ii].idx = -1 # but as uint
            
    def iter_func(gidx_ii, a_ii, work, ii, *args): 
        work_g = work[gidx_ii]
        new_best = max(a_ii, work_g.best)
        work_g.idx += (ii - work_g.idx) * (new_best == a_ii)
        work_g.best = new_best
        
    def end_func(work, ignore, fill, ddof=0):
        # TODO: if fill is -1, we can avoid branch.
        return work.idx if work.idx != -1 else fill
        


""" =======================================================================
    User-facing aggregate implementation
    ======================================================================= """


_impl_dict = dict(sum=_sum, nansum=_sum.get_nan_version(),
                  prod=_prod, nanprod=_prod.get_nan_version(),
                  min=_min, nanmin=_min, max=_max, nanmax=_max,
                  last=_last, nanlast=_last.get_nan_version(),
                  first=_first, nanfirst=_first.get_nan_version(),
                  all=_all, any=_any, allnan=_allnan, anynan=_anynan,
                  mean=_mean, nanmean=_mean.get_nan_version(),
                  var=_var, nanvar=_var.get_nan_version(),
                  std=_std, nanstd=_std.get_nan_version(),
                  argmin=_argmin, nanargmin=_argmin,
                  argmax=_argmax, nanargmax=_argmax)

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
        ret = func(group_idx, a, flat_size, dtype, size, fill_value=fill_value, 
                    **kwargs)

    return ret
    
    
aggregate.__doc__ =  """
    This is the numba implementation of aggregate.
    
    If you provide a size value we will be able to iterate over ``a`` and 
    ``group_idx`` exactly once.
    
    TODO: make this true even for the case where we broadcast ``group_idx``
    with an ``axis`` argument.
    """ + _doc_str





""" ===================================================================
    ADDITIONAL NOTES
    ===================================================================
    
Written for numba version 0.22.1.

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

We use a hack to avoid branching in nan-prefixed functions:
if a is nan then we assign to group[0] otherwise we assign to the correct
group, or rather we assign to the correct group plus one.  At the end we have
to correct for this.

Because ints cant represent floats, we can get away with using the non-nan
version of functions for ints, i.e. nansum(<ints>) = sum(<ints>).

In python min(99, nan) is 99, but min(nan, 99) is nan. In numba it seems
like this isn't always true.  For our use of min and max, for some reason
we need to do it the other way around.  TODO: investigate further, and
write a proper test for this...the minimal test I tried seemed to agree
with python.  If need be, use the nansum-style hack.
    
Note that sometimes you have to spend a bit of time looking for the correct
way of doing something, e.g. numba's implementation of python's builtin min
function is about 10x faster than np.minimum or an if statement, or a 
crazy 2-element lookup table + 1-bit-bool-indexing ...but I only disovered
that the builtin after messing around for a while with the other versions.

==================

TODO: the end_func stuff should possibly use numpy logical indexing rather
than expect numba to do a good job of jitting branches.

==================

TODO: note that argmin/argmax in numpy give the wrong indices as output: they
provide indices into the squashed array, not the original array.
==================

The post-loop stuff for MODE_BOOL, when fill_value != initial_value, has
to perform the following...(numba strugles to optimize so we do it manually)...
    
    fill_value is 1      fill_value is 0
    intial_value is 0    initial_value is 1
    -----------------    -----------------
    used | x | res       used | x  | res
    1      1    1        1      1    1
    1      0    0        1      0    0
    0      1    *        0      1    0
    0      0    1        0      0    *
    -----------------    -----------------
    res: used==x         res: used & res    
    * cannot occur due to initial_value and used=0
    
    
====================

Computation of Variance
https://en.wikipedia.org/wiki/...
                    Algorithms_for_calculating_variance#Computing_shifted_data
For each group, subtract the first value from all subsequent values before
storing sum and sum of squares, also store count.
..thus 4 items need to be stored.
..we could possibly use the mean of the first two numbers as the shift, because
you only need to store one number still to make that work, but the branching 
requirements could get (more) complex and slow.

Note that numpy's var does two passes, first computing the sum, to get the mean
and then the sum of squared deviations from the mean.  To avoid the second pass
we use the shifted method, which isn't quite as good, but in all real-world
cases will be fine. 

For the mean, numpy just takes the sum and divides by n, so we'll do that too.
Note that this can lead to overflow (which would normally be reasonably 
evident to the end user), but is otherwise ok (I think).

"""


