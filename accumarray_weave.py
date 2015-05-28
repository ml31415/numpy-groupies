from itertools import product
import numpy as np
from scipy.weave import inline

__all__ = ['accum', 'accum_np', 'accum_ufunc', 'accum_py', 'unpack', 'step_indices', 'step_count']


optimized_funcs = {'sum', 'min', 'max', 'amin', 'amax', 'mean', 'std', 'prod',
                   'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanstd', 'nanprod',
                   'all', 'any', 'allnan', 'anynan'}

_dtypes_by_func_base = {list: 'object', tuple: 'object', sorted: 'object',
                        np.array: 'object',
                        np.mean: 'float', np.std: 'float',
                        np.all: 'bool', np.any: 'bool',
                        all: 'bool', any: 'bool',
                        np.nanmean: 'float', np.nanstd: 'float'}
_dtypes_by_func = dict((k.__name__, v) for k, v in _dtypes_by_func_base.iteritems())
_dtypes_by_func.update(_dtypes_by_func_base)
_dtypes_by_func.update(allnan='bool', anynan='bool')

def _dtype_by_func(func, a):
    try:
        return _dtypes_by_func[func]
    except KeyError:
        func_str = func.lower() if isinstance(func, basestring) else func.__name__
        return _dtypes_by_func.get(func_str, a.dtype)


# c_funcs will contain all generated c code, so it can be read easily for debugging
c_funcs = dict()
c_iter = dict()
c_finish = dict()

# Set this for testing, to fail deprecated C-API calls
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


def accumarray(accmap, a, func='sum', dtype=None, fillvalue=0, mode='incontiguous'):
    '''
    Accumulation function similar to Matlab's `accumarray` function.
    
    See readme file at https://github.com/ml31415/accumarray for 
    full description.

    This implementation is by @ml31415.

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
        
    Additional Notes
    --------
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

    dtype = dtype or _dtype_by_func(func, a)
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
    print "TODO: make proper use of testing and benchmarking scripts"
    accmap = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
    a = np.arange(accmap.size, dtype=float)
    mode = 'contiguous'
    for fn in (np.mean, np.std, 'allnan', 'anynan'):
        vals = accum(accmap, a, mode=mode, func=fn)
        print vals
