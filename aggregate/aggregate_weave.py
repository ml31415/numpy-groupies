import numpy as np
from scipy.weave import inline

from .utils import check_group_idx, check_dtype, aliasing_numpy as aliasing, check_fill_value
from .aggregate_numpy import aggregate as aggregate_np


optimized_funcs = {'sum', 'min', 'max', 'amin', 'amax', 'mean', 'std', 'prod',
                   'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanstd', 'nanprod',
                   'all', 'any', 'allnan', 'anynan'}


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

    for (long i=0; i<Lgroup_idx; i++) {
        write_idx = group_idx[i];
        %(iter)s
    }
    %(finish)s
    """

c_base_contiguous = r"""%(init)s

    for (long i=0; i<Lgroup_idx; i++) {
        if (group_idx[cmp_pos] != group_idx[i]) {
            cmp_pos = i;
            write_idx++;
        }
        %(iter)s
    }
    %(finish)s
    """

c_iter['sum'] = r"""
        counter[write_idx] = 0;
        vals[write_idx] += a[i];"""

c_iter['prod'] = r"""
        counter[write_idx] = 0;
        vals[write_idx] *= a[i];"""

c_iter['all'] = r"""
        counter[write_idx] = 0;
        if (a[i] == 0) vals[write_idx] = 0;"""

c_iter['any'] = r"""
        counter[write_idx] = 0;
        if (a[i] != 0) vals[write_idx] = 1;"""

c_iter['allnan'] = r"""
        counter[write_idx] = 0;
        if (a[i] == a[i]) vals[write_idx] = 0;"""

c_iter['anynan'] = r"""
        counter[write_idx] = 0;
        if (a[i] != a[i]) vals[write_idx] = 1;"""

c_iter['max'] = r"""
        if (counter[write_idx] == 1) {
            vals[write_idx] = a[i];
            counter[write_idx] = 0;
        } 
        else if (vals[write_idx] < a[i]) vals[write_idx] = a[i];"""

c_iter['min'] = r"""
        if (counter[write_idx] == 1) {
            vals[write_idx] = a[i];
            counter[write_idx] = 0;
        } 
        else if (vals[write_idx] > a[i]) vals[write_idx] = a[i];"""

c_iter['mean'] = r"""
        counter[write_idx]++;
        vals[write_idx] += a[i];"""

c_finish['mean'] = r"""
    for (long i=0; i<Lvals; i++) {
        if (counter[i] != 0) vals[i] = vals[i] / counter[i];
        else vals[i] = fill_value;
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
        else vals[i] = fill_value;
    }"""


# Fill c_funcs with constructed code from the templates
for mode in ('contiguous', ''):
    codebase = c_base_contiguous if mode == 'contiguous' else c_base
    mode_postfix = '_' + mode if mode else ''
    varnames = ['group_idx', 'a', 'vals', 'counter']
    for funcname in c_iter:
        code = codebase % dict(init=c_init(varnames), iter=c_iter[funcname],
                               finish=c_finish.get(funcname, ''))
        c_funcs[funcname + mode_postfix] = code
        if not 'nan' in funcname:
            code = codebase % dict(init=c_init(varnames), iter=c_nan_iter(c_iter[funcname]),
                                   finish=c_finish.get(funcname, ''))
            c_funcs['nan' + funcname + mode_postfix] = code



c_funcs['step_count'] = c_size('group_idx') + r"""
    long cmp_pos = 0;
    long steps = 1;
    if (Lgroup_idx < 1) return_val = 0;
    else {
        for (long i=0; i<Lgroup_idx; i++) {
            if (group_idx[cmp_pos] != group_idx[i]) {
                cmp_pos = i;
                steps++;
            }
        }
        return_val = steps;
    }"""

def step_count(group_idx):
    """ Determine the size of the result array
        for contiguous data
    """
    return inline(c_funcs['step_count'], ['group_idx'], define_macros=c_macros)


c_funcs['step_indices'] = c_size('group_idx') + r"""
    long cmp_pos = 0;
    long write_idx = 1; 
    for (long i=1; i<Lgroup_idx; i++) {
        if (group_idx[cmp_pos] != group_idx[i]) {
            cmp_pos = i;
            indices[write_idx++] = i;
        }
    }"""

def step_indices(group_idx):
    """ Get the edges of areas within group_idx, which are filled 
        with the same value
    """
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, int)
    indices[0] = 0
    indices[-1] = group_idx.size
    inline(c_funcs['step_indices'], ['group_idx', 'indices'], define_macros=c_macros)
    return indices


def aggregate(group_idx, a, func='sum', size=None, dtype=None, fill_value=0):
    '''
    Aggregation function similar to Matlab's `accumarray`.
    
    See readme file at https://github.com/ml31415/accumarray for 
    full description.

    Parameters
    ----------
    group_idx : ndarray
        This is the "aggregation map". It maps input (i.e. indices into
        `a`) to their destination in the output array.  The dimensions 
        of `group_idx` must be the same as `a.shape`.
    a : ndarray
        The input data to be aggregated.
    func : callable or None
        The aggregation function. The function will be passed a list of 
        values from `a` to be aggregated. If None, np.sum is assumed.
    dtype : numpy data type, or None
        The data type of the output array. If None, the data type of
        `a` is used.
        
    Additional Notes
    --------
    group_idx and a are generally treated as flattened arrays.
    
    Contiguous:
    Same values within group_idx can be expected to be grouped
    or be treated as new values starting a new group, in 
    case they should appear another time
    E.g. group_idx = [1 1 2 2 2 1 1 3 3] with contiguous set will 
    be treated the same way as [0 0 1 1 1 2 2 3 3]
    That way, feeding data through np.unique, maintaining order
    etc. can be omitted. It also gives a nice speed boost, as
    np.argsort of group_idx can also be omitted.

    Returns
    -------
    out : ndarray
        The aggregated results.

    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> group_idx = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = aggregate(group_idx, a)
    array([9, 7, 15])
    >>> # Aggregate using a product.
    >>> aggregate(group_idx, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    '''

    func = aliasing.get(func, func)
    if not isinstance(func, basestring):
        # Fall back to acuum_np if no optimized C version available
        return aggregate_np(group_idx, a, func=func, dtype=dtype,
                            fill_value=fill_value)
    elif func not in optimized_funcs:
        raise NotImplementedError("No optimized function for %s available" % func)

    check_group_idx(group_idx, a)
    dtype = check_dtype(dtype, func, a)
    check_fill_value(fill_value, dtype)
    size = size or np.max(group_idx) + 1
    if func in ('sum', 'any', 'anynan', 'nansum'):
        vals = np.zeros(size, dtype=dtype)
    elif func in ('prod', 'all', 'allnan', 'nanprod'):
        vals = np.ones(size, dtype=dtype)
    else:
        vals = np.full(size, fill_value, dtype=dtype)

    # In case we should get some ugly fortran arrays, convert them
    vals_dict = dict(group_idx=np.ascontiguousarray(group_idx), a=np.ascontiguousarray(a),
                     vals=vals, fill_value=fill_value)
    if func in ('std', 'nanstd'):
        counter = np.zeros_like(vals, dtype=int)
        vals_dict['means'] = np.zeros_like(vals)
    elif func in ('mean', 'nanmean'):
        counter = np.zeros_like(vals, dtype=int)
    else:
        # Using inverse logic, marking anyting touched with zero for later removal
        counter = np.ones_like(vals, dtype=bool)
    vals_dict['counter'] = counter

    inline(c_funcs[func], vals_dict.keys(), local_dict=vals_dict, define_macros=c_macros)

    # Postprocessing
    if func in ('sum', 'any', 'anynan', 'nansum') and fill_value != 0:
        vals[counter] = fill_value
    elif func in ('prod', 'all', 'allnan', 'nanprod') and fill_value != 1:
        vals[counter] = fill_value
    return vals



c_funcs['unpack_contiguous'] = c_minmax + c_size('group_idx') + c_size('vals') + r"""
    long cmp_pos = 0;
    long val_cnt = 0;
    unpacked[0] = vals[0];
    for (long i=1; i<Lgroup_idx; i++) {
        if (group_idx[cmp_pos] != group_idx[i]) {
            cmp_pos = i;
            val_cnt = min((val_cnt + 1), (Lvals - 1));
        }
        unpacked[i] = vals[val_cnt];
    }"""


def unpack(group_idx, vals, mode='incontiguous'):
    """ Take an aggregate packed array and uncompress it to the size of group_idx. 
        This is equivalent to vals[group_idx], but gives a more than 
        3-fold speedup.
    """
    if mode == 'downscaled':
        group_idx = np.unique(group_idx, return_inverse=True)[1]
    if mode != 'contiguous':
        # Numpy internal version got faster recently, so let's just use this if possible
        return vals[group_idx]
    check_group_idx(group_idx)
    unpacked = np.zeros_like(group_idx, dtype=vals.dtype)
    inline(c_funcs['unpack_contiguous'], ['group_idx', 'vals', 'unpacked'], define_macros=c_macros)
    return unpacked

if __name__ == '__main__':
    print "TODO: make proper use of testing and benchmarking scripts"
    group_idx = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
    a = np.arange(group_idx.size, dtype=float)
    mode = 'contiguous'
    for fn in (np.mean, np.std, 'allnan', 'anynan'):
        vals = aggregate(group_idx, a, mode=mode, func=fn)
        print vals


# For a future downscale wrapper:
#        group_idx = np.unique(group_idx, return_inverse=True)[1]
