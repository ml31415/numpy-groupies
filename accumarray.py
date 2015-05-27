'''
Accumulation functions similar to Matlab's `accumarray` function.

Parameters
----------
accmap : ndarray
    This is the "accumulation map". It maps input (i.e. indices into
    `a`) to their destination in the output array.  The dimensions 
    of `accmap` must be the same as `a.shape`.
a : ndarray
    The input data to be accumulated.
func : callable or None
    The accumulation function. The function will be passed a list of 
    values from `a` to be accumulated. If None, np.sum is assumed.
dtype : numpy data type, or None
    The data type of the output array. If None, the data type of
    `a` is used.
mode : 
    incontiguous : Normal operation like used from matlab

    contiguous : Any change of accmap values creates a new output
    field. The size of the output is not defined by the biggest
    value within accmap, but determined by the number of value
    changes within accmap.
    
    `unpack` is provided, to map outputs created that way back
    to their original size.
    
    downscaled : Like running np.unique on accmap, before
    using it. `unpack` provides the same parameter, to
    map the output back to original size. Less performant
    than the other modes.
    

Returns
-------
out : ndarray
    The accumulated results.

Examples
--------
>>> from numpy import array, prod
>>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
>>> a
array([[ 1,  2,  3],
       [ 4, -1,  6],
       [-1,  8,  9]])
>>> # Sum the diagonals.
>>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
>>> s = accum(accmap, a)
array([9, 7, 15])
>>> # Accumulate using a product.
>>> accum(accmap, a, func=prod, dtype=float)
array([[ -8.,  18.],
       [ -8.,   9.]])
'''

from itertools import product
import numpy as np
from scipy.weave import inline

__all__ = ['accum', 'accum_np', 'accum_py', 'unpack', 'step_indices', 'step_count']


optimized_funcs = {'sum', 'min', 'max', 'amin', 'amax', 'mean', 'std', 'prod',
                   'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanstd', 'nanprod',
                   'all', 'any', 'allnan', 'anynan'}

dtype_by_func = {list: 'object',
                 tuple: 'object',
                 sorted: 'object',
                 np.array: 'object',
                 np.sort: 'object',
                 np.mean: 'float',
                 np.std: 'float',
                 np.all: 'bool',
                 np.any: 'bool',
                 all: 'bool',
                 any: 'bool',
                 'mean': 'float',
                 'std': 'float',
                 'nanmean': 'float',
                 'nanstd': 'float',
                 'all': 'bool',
                 'any': 'bool',
                 'allnan': 'bool',
                 'anynan': 'bool',
                 }

# c_funcs will contain all generated c code, so it can be read easily for debugging
c_funcs = dict()
c_iter = dict()
c_finish = dict()

# Set this, to fail deprecated C-API calls
# c_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
c_macros = []

def c_size(varname):
    return r"""
    long L%(varname)s = 1;
    for (int n=0; n<D%(varname)s; n++) L%(varname)s *= N%(varname)s[n];""" % dict(varname=varname)

def c_init(varnames):
    return '    ' + ''.join(c_size(varname) for varname in varnames).lstrip() + """

    long write_idx = 0;
    long cmp_pos = 0;"""

def c_nan_iter(c_iter):
    return r"""
        if (a[i] == a[i]) {%s
        }""" % '\n'.join('    ' + line for line in c_iter.splitlines())


c_minmax = r"""
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )"""

c_base = r"""%(init)s

    for (long i=0; i<Laccmap; i++) {
        write_idx = accmap[i];
        %(iter)s
    }
    %(finish)s
    """

c_base_contiguous = r"""%(init)s

    for (long i=0; i<Laccmap; i++) {
        if (accmap[cmp_pos] != accmap[i]) {
            cmp_pos = i;
            write_idx++;
        }
        %(iter)s
    }
    %(finish)s
    """

c_iter['sum'] = r"""
        if (counter[write_idx] == 0) {
            vals[write_idx] = a[i];
            counter[write_idx] = 1;
        } 
        else vals[write_idx] += a[i];"""

c_iter['prod'] = r"""
        if (counter[write_idx] == 0) {
            vals[write_idx] = a[i];
            counter[write_idx] = 1;
        } 
        else vals[write_idx] *= a[i];"""

c_iter['max'] = r"""
        if (counter[write_idx] == 0) {
            vals[write_idx] = a[i];
            counter[write_idx] = 1;
        } 
        else if (vals[write_idx] < a[i]) vals[write_idx] = a[i];"""

c_iter['min'] = r"""
        if (counter[write_idx] == 0) {
            vals[write_idx] = a[i];
            counter[write_idx] = 1;
        } 
        else if (vals[write_idx] > a[i]) vals[write_idx] = a[i];"""

c_iter['mean'] = r"""
        counter[write_idx]++;
        vals[write_idx] += a[i];"""

c_finish['mean'] = r"""
    for (long i=0; i<Lvals; i++) {
        if (counter[i] != 0) vals[i] = vals[i] / counter[i];
        else vals[i] = fillvalue;
    }"""

c_iter['std'] = r"""
        counter[write_idx]++;
        means[write_idx] += a[i];
        vals[write_idx] += a[i] * a[i];"""

c_finish['std'] = r"""
    double mean = 0;
    for (long i=0; i<Lvals; i++) {
        if (counter[i] != 0) {
            mean = means[i] / counter[i];
            vals[i] = sqrt(vals[i] / counter[i] - mean * mean);
        }
        else vals[i] = fillvalue;
    }"""

c_iter['all'] = r"""
        if (counter[write_idx] == 0) vals[write_idx] = 1;
        counter[write_idx] = 1;
        if (a[i] == 0) vals[write_idx] = 0;"""

c_iter['any'] = r"""
        if (counter[write_idx] == 0) vals[write_idx] = 0;
        counter[write_idx] = 1;
        if (a[i] != 0) vals[write_idx] = 1;"""

c_iter['allnan'] = r"""
        if (counter[write_idx] == 0) vals[write_idx] = 1;
        counter[write_idx] = 1;
        if (a[i] == a[i]) vals[write_idx] = 0;"""

c_iter['anynan'] = r"""
        if (counter[write_idx] == 0) vals[write_idx] = 0;
        counter[write_idx] = 1;
        if (a[i] != a[i]) vals[write_idx] = 1;"""


# Fill c_funcs with constructed code from the templates
for mode in ('contiguous', ''):
    codebase = c_base_contiguous if mode == 'contiguous' else c_base
    mode_postfix = '_' + mode if mode else ''
    varnames = ['accmap', 'a', 'vals', 'counter']
    for funcname in c_iter:
        code = codebase % dict(init=c_init(varnames), iter=c_iter[funcname],
                               finish=c_finish.get(funcname, ''))
        c_funcs[funcname + mode_postfix] = code
        if not 'nan' in funcname:
            code = codebase % dict(init=c_init(varnames), iter=c_nan_iter(c_iter[funcname]),
                                   finish=c_finish.get(funcname, ''))
            c_funcs['nan' + funcname + mode_postfix] = code



c_funcs['step_count'] = c_size('accmap') + r"""
    long cmp_pos = 0;
    long steps = 1;
    if (Laccmap < 1) return_val = 0;
    else {
        for (long i=0; i<Laccmap; i++) {
            if (accmap[cmp_pos] != accmap[i]) {
                cmp_pos = i;
                steps++;
            }
        }
        return_val = steps;
    }"""

def step_count(accmap):
    """ Determine the size of the result array
        for contiguous data
    """
    return inline(c_funcs['step_count'], ['accmap'], define_macros=c_macros)


c_funcs['step_indices'] = c_size('accmap') + r"""
    long cmp_pos = 0;
    long write_idx = 1; 
    for (long i=1; i<Laccmap; i++) {
        if (accmap[cmp_pos] != accmap[i]) {
            cmp_pos = i;
            indices[write_idx++] = i;
        }
    }"""

def step_indices(accmap):
    """ Get the edges of areas within accmap, which are filled 
        with the same value
    """
    ilen = step_count(accmap) + 1
    indices = np.empty(ilen, int)
    indices[0] = 0
    indices[-1] = accmap.size
    inline(c_funcs['step_indices'], ['accmap', 'indices'], define_macros=c_macros)
    return indices


def _check_accmap(accmap, a=None, check_min=True):
    if a is not None and accmap.shape != a.shape:
        raise ValueError("The dimensions of accmap must be the same as a.shape")
    if not issubclass(accmap.dtype.type, np.integer):
        raise TypeError("Accmap must be of integer type")
    if check_min and np.min(accmap) < 0:
        raise ValueError("Accmap contains negative indices")


def _check_mode(mode):
    if mode not in {'contiguous', 'incontiguous', 'downscaled'}:
        raise ValueError("Unknown accumulation mode: %s" % mode)


def accum(accmap, a, func='sum', dtype=None, fillvalue=0, mode='incontiguous'):
    """ For most common cases, operates like usual matlab accumarray
        http://www.mathworks.com/help/matlab/ref/accumarray.html
    
        accmap and a are generally treated as flattened arrays.
        
        Contiguous:
        Same values within accmap can be expected to be grouped
        or be treated as new values starting a new group, in 
        case they should appear another time
        E.g. accmap = [1 1 2 2 2 1 1 3 3] with contiguous set will 
        be treated the same way as [0 0 1 1 1 2 2 3 3]
        That way, feeding data through np.unique, maintaining order
        etc. can be omitted. It also gives a nice speed boost, as
        np.argsort of accmap can also be omitted.
    """
    if not isinstance(func, basestring):
        if getattr(func, '__name__', None) in optimized_funcs:
            func = func.__name__
            func = dict(amin='min', amax='max').get(func, func)
        else:
            # Fall back to acuum_np if no optimized C version available
            return accum_np(accmap, a, func=func, dtype=dtype,
                            fillvalue=fillvalue, mode=mode)
    elif func not in optimized_funcs:
        raise ValueError("No optimized function %s available" % func)

    if mode == 'downscaled':
        accmap = np.unique(accmap, return_inverse=True)[1]
    _check_accmap(accmap, a)
    _check_mode(mode)

    dtype = dtype or dtype_by_func.get(func, a.dtype)
    if mode == 'contiguous':
        vals_len = step_count(accmap)
    else:
        vals_len = np.max(accmap) + 1
    vals = np.zeros(vals_len, dtype=dtype)
    # Fill if required and function does no second path
    if fillvalue != 0 and func not in {'mean', 'std', 'nanmean', 'nanstd'}:
        vals.fill(fillvalue)

    # In case we should get some ugly fortran arrays, convert them
    vals_dict = dict(accmap=np.ascontiguousarray(accmap), a=np.ascontiguousarray(a),
                     vals=vals, fillvalue=fillvalue)
    if func in ('std', 'nanstd'):
        vals_dict['means'] = np.zeros_like(vals)
        vals_dict['counter'] = np.zeros_like(vals, dtype=int)
    elif func in ('mean', 'nanmean'):
        vals_dict['counter'] = np.zeros_like(vals, dtype=int)
    else:
        vals_dict['counter'] = np.zeros_like(vals, dtype=bool)

    if mode == 'contiguous':
        func += '_' + mode
    inline(c_funcs[func], vals_dict.keys(), local_dict=vals_dict, define_macros=c_macros)
    return vals


def accum_np(accmap, a, func=np.sum, dtype=None, fillvalue=0, mode='incontiguous'):
    """ Pure numpy solution without the need for a compiler.
        This implementation is used, if no optimized 
        grouping function is found.
    """
    if mode == 'downscaled':
        accmap = np.unique(accmap, return_inverse=True)[1]
    _check_accmap(accmap, a, check_min=False)
    _check_mode(mode)

    dtype = dtype or dtype_by_func.get(func, a.dtype)
    if mode == 'contiguous':
        indices = np.where(np.ediff1d(accmap, to_begin=[1], to_end=[1]))[0]
        vals_len = len(indices) - 1
        vals = np.zeros(vals_len, dtype=dtype)
        a_f = a.flat
        for i in xrange(vals_len):
            vals[i] = func(a_f[indices[i]:indices[i + 1]])
    else:
        # Mergesort does a stable search, so grouping
        # functions can rely on the sort order
        rev = np.argsort(accmap.flat, kind='mergesort')
        accmap_rev = accmap.flat[rev]
        if accmap_rev[0] < 0:
            raise ValueError("Accmap contains negative indices")
        indices = np.where(np.ediff1d(accmap_rev, to_begin=[1], to_end=[1]))[0]

        vals_len = accmap_rev[-1] + 1
        vals = np.zeros(vals_len, dtype=dtype)
        if fillvalue is not 0:
            vals.fill(fillvalue)

        a_rev = a.flat[rev]
        for i in xrange(len(indices) - 1):
            indices_i = indices[i]
            vals[accmap_rev[indices_i]] = func(a_rev[indices_i:indices[i + 1]])
    return vals


def accum_py(accmap, a, func=np.sum, size=None, fillvalue=0, dtype=None, mode='incontiguous'):
    """ Slow python solution from http://www.scipy.org/Cookbook/AccumarrayLike
    """
    if mode == 'downscaled':
        _, accmap = np.unique(accmap, return_inverse=True)
    _check_accmap(accmap, a)
    _check_mode(mode)

    dtype = dtype or dtype_by_func.get(func, a.dtype)
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)

    adims = tuple(xrange(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[xrange(k) for k in size]):
        # All fields in vals
        vals[s] = []

    for s in product(*[xrange(k) for k in a.shape]):
        # All fields in a
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.zeros(size, dtype=dtype)
    for s in product(*[xrange(k) for k in size]):
        # All fields in vals
        if vals[s] == []:
            out[s] = fillvalue
        else:
            out[s] = func(vals[s])

    return out


############ ufuncs implementation #############

def _fill_untouched(accmap, vals, fillvalue):
    # Probably there is something faster ...
    erase = np.ones_like(vals, dtype=bool)
    erase[accmap] = 0
    vals[erase] = fillvalue
    return vals


def _nanmask(arr, replacement, nans):
    if nans:
        return np.where(np.isnan(arr), replacement, arr)
    else:
        return arr


def _sum(accmap, a, vals, fillvalue, dtype=None, nans=False):
    a = _nanmask(a, 0, nans)
    vals = np.bincount(accmap, weights=a, minlength=len(vals)).astype(dtype)
    if fillvalue != 0:
        vals = _fill_untouched(accmap, vals, fillvalue)
    return vals


def _last(accmap, a, vals, fillvalue, dtype=None, nans=False):
    if fillvalue != 0:
        vals.fill(fillvalue)
    vals[accmap] = a
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    return vals


def _first(accmap, a, vals, fillvalue, dtype=None, nans=False):
    if fillvalue != 0:
        vals.fill(fillvalue)
    vals[accmap[::-1]] = a[::-1]  # same trick as above, but in reverse
    return vals


def _prod(accmap, a, vals, fillvalue, dtype=None, nans=False):
    a = _nanmask(a, 1, nans)
    vals.fill(1)
    np.multiply.at(vals, accmap, a)
    if fillvalue != 1:
        _fill_untouched(accmap, vals, fillvalue)
    return vals


def _all(accmap, a, vals, fillvalue, dtype=None, nans=False):
    a = _nanmask(a, 1, nans)
    vals.fill(1)
    np.logical_and.at(vals, accmap, a)
    if fillvalue != 1:
        _fill_untouched(accmap, vals, fillvalue)
    return vals


def _any(accmap, a, vals, fillvalue, dtype=None, nans=False):
    a = _nanmask(a, 0, nans)
    vals.fill(0)
    np.logical_or.at(vals, accmap, a)
    if fillvalue != 0:
        _fill_untouched(accmap, vals, fillvalue)
    return vals


def _min(accmap, a, vals, fillvalue, dtype=None, nans=False):
    minfill = np.iinfo(vals.dtype).max if isinstance(vals.dtype, np.integer) else np.finfo(vals.dtype).max
    a = _nanmask(a, minfill, nans)
    vals.fill(minfill)
    np.minimum.at(vals, accmap, a)
    _fill_untouched(accmap, vals, fillvalue)
    return vals


def _max(accmap, a, vals, fillvalue, dtype=None, nans=False):
    maxfill = np.iinfo(vals.dtype).min if isinstance(vals.dtype, np.integer) else np.finfo(vals.dtype).min
    a = _nanmask(a, maxfill, nans)
    vals.fill(maxfill)
    np.maximum.at(vals, accmap, a)
    _fill_untouched(accmap, vals, fillvalue)
    return vals


def _mean(accmap, a, vals, fillvalue, dtype=None, nans=False):
    counts = np.bincount(accmap)
    if nans:
        nancount = np.bincount(accmap, weights=np.isnan(a))
        counts -= nancount

    a = _nanmask(a, 0, nans)
    sums = np.bincount(accmap, weights=a)
    with np.errstate(divide='ignore'):
        vals[:len(sums)] = sums / counts
    vals[counts == 0] = fillvalue
    return vals


def _std(accmap, a, vals, fillvalue, dtype=None, nans=False):
    counts = np.bincount(accmap)
    if nans:
        nancount = np.bincount(accmap, weights=np.isnan(a))
        counts -= nancount

    a = _nanmask(a, 0, nans)
    sums = np.bincount(accmap, weights=a)
    sq_sums = np.bincount(accmap, weights=a * a)
    with np.errstate(divide='ignore'):
        E_x = sums / counts
        E_x2 = E_x * E_x
        vals[:len(sums)] = np.sqrt(sq_sums / counts - E_x2)
    vals[counts == 0] = fillvalue
    return vals


def _allnan(accmap, a, vals, fillvalue, dtype=None, nans=False):
    vals.fill(1)
    np.logical_and.at(vals, accmap, np.isnan(a))
    if fillvalue != 1:
        _fill_untouched(accmap, vals, fillvalue)
    return vals


def _anynan(accmap, a, vals, fillvalue, dtype=None, nans=False):
    np.logical_or.at(vals, accmap, np.isnan(a))
    if fillvalue != 0:
        _fill_untouched(accmap, vals, fillvalue)
    return vals


_accum_funcs = dict(min=_min, amin=_min, max=_max, amax=_max, sum=_sum, add=_sum, prod=_prod, multiply=_prod,
                    last=_last, first=_first, all=_all, any=_any, mean=_mean, std=_std,
                    anynan=_anynan, allnan=_allnan)

_accum_mask_vals = dict(prod=1, all=1)

def accum_ufunc(accmap, a, func=np.sum, dtype=None, fillvalue=0, mode='incontiguous'):
    _check_accmap(accmap, a, check_min=False)
    a = np.asanyarray(a)

    func_str = func.lower() if isinstance(func, basestring) else func.__name__
    if func_str in {'list', 'array', '<lambda>', 'sort'}:
        # Fallback solution for backwards compatibility
        return accum_np(accmap, a, func=func, fillvalue=fillvalue)
    else:
        dtype = dtype or dtype_by_func.get(func, a.dtype)
        vals_len = np.max(accmap) + 1
        vals = np.zeros(vals_len, dtype=dtype)

        if not func_str.startswith('nan'):
            nans = False
            func = _accum_funcs.get(func_str)
        else:
            nans = True
            func = _accum_funcs.get(func_str[3:])

        if func:
            return func(accmap, a, vals, fillvalue, dtype=dtype, nans=nans)
        else:
            # The general case for arbitrary ufuncs
            if isinstance(func, basestring):
                raise NotImplementedError("Function string '%s' not recognised" % func_str)
            try:
                func = getattr(func, 'at')
            except AttributeError:
                # No such ufunc available
                raise NotImplementedError("ufunc %s does not provide broadcasting" % func)
            else:
                func(vals, accmap, a)
            if fillvalue != 0:
                vals.fill(fillvalue)
        return vals


c_funcs['unpack_contiguous'] = c_minmax + c_size('accmap') + c_size('vals') + r"""
    long cmp_pos = 0;
    long val_cnt = 0;
    unpacked[0] = vals[0];
    for (long i=1; i<Laccmap; i++) {
        if (accmap[cmp_pos] != accmap[i]) {
            cmp_pos = i;
            val_cnt = min((val_cnt + 1), (Lvals - 1));
        }
        unpacked[i] = vals[val_cnt];
    }"""


def unpack(accmap, vals, mode='incontiguous'):
    """ Take an accum packed array and uncompress it to the size of accmap. 
        This is equivalent to vals[accmap], but gives a more than 
        3-fold speedup.
    """
    if mode == 'downscaled':
        accmap = np.unique(accmap, return_inverse=True)[1]
    if mode != 'contiguous':
        # Numpy internal version got faster recently, so let's just use this if possible
        return vals[accmap]
    _check_accmap(accmap)
    _check_mode(mode)
    unpacked = np.zeros_like(accmap, dtype=vals.dtype)
    inline(c_funcs['unpack_contiguous'], ['accmap', 'vals', 'unpacked'], define_macros=c_macros)
    return unpacked

if __name__ == '__main__':
    accmap = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
    a = np.arange(accmap.size, dtype=float)
    mode = 'contiguous'
    for fn in (np.mean, np.std, 'allnan', 'anynan'):
        vals = accum(accmap, a, mode=mode, func=fn)
        print vals
