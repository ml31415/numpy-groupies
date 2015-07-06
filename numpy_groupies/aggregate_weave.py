import numpy as np
from scipy.weave import inline

from .utils import (_no_separate_nan_version, get_func, check_dtype,
                    aliasing, check_fill_value, input_validation, _doc_str, isstr)
from .aggregate_numpy import aggregate as aggregate_np


optimized_funcs = {'sum', 'min', 'max', 'amin', 'amax', 'mean', 'var', 'std', 'prod',
                   'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanvar', 'nanstd', 'nanprod',
                   'all', 'any', 'allnan', 'anynan',
                   'first', 'last', 'nanfirst', 'nanlast'}

# c_funcs will contain all generated c code, so it can be read easily for debugging
c_funcs = dict()
c_iter = dict()
c_iter_scalar = dict()
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


c_base = r"""%(init)s

    for (long i=0; i<Lgroup_idx; i++) {
        write_idx = group_idx[i];
        %(iter)s
    }
    %(finish)s
    """

c_base_reverse = r"""%(init)s

    for (long i=Lgroup_idx-1; i>=0; i--) {
        write_idx = group_idx[i];
        %(iter)s
    }
    %(finish)s
    """

c_iter['sum'] = r"""
        counter[write_idx] = 0;
        ret[write_idx] += a[i];"""

c_iter_scalar['sum'] = r"""
        counter[write_idx] = 0;
        ret[write_idx] += a;"""

c_iter['prod'] = r"""
        counter[write_idx] = 0;
        ret[write_idx] *= a[i];"""

c_iter_scalar['prod'] = r"""
        counter[write_idx] = 0;
        ret[write_idx] *= a;"""

c_iter['all'] = r"""
        counter[write_idx] = 0;
        if (a[i] == 0) ret[write_idx] = 0;"""

c_iter['any'] = r"""
        counter[write_idx] = 0;
        if (a[i] != 0) ret[write_idx] = 1;"""

c_iter['last'] = r"""
        ret[write_idx] = a[i];"""

c_iter['allnan'] = r"""
        counter[write_idx] = 0;
        if (a[i] == a[i]) ret[write_idx] = 0;"""

c_iter['anynan'] = r"""
        counter[write_idx] = 0;
        if (a[i] != a[i]) ret[write_idx] = 1;"""

c_iter['max'] = r"""
        if (counter[write_idx] == 1) {
            ret[write_idx] = a[i];
            counter[write_idx] = 0;
        } 
        else if (ret[write_idx] < a[i]) ret[write_idx] = a[i];"""

c_iter['min'] = r"""
        if (counter[write_idx] == 1) {
            ret[write_idx] = a[i];
            counter[write_idx] = 0;
        } 
        else if (ret[write_idx] > a[i]) ret[write_idx] = a[i];"""

c_iter['mean'] = r"""
        counter[write_idx]++;
        ret[write_idx] += a[i];"""

c_finish['mean'] = r"""
    for (long i=0; i<Lret; i++) {
        if (counter[i] != 0) ret[i] = ret[i] / counter[i];
        else ret[i] = fill_value;
    }"""

c_iter['std'] = r"""
        counter[write_idx]++;
        means[write_idx] += a[i];
        ret[write_idx] += a[i] * a[i];"""

c_finish['std'] = r"""
    double mean2 = 0;
    for (long i=0; i<Lret; i++) {
        if (counter[i] != 0) {
            mean2 = means[i] * means[i];
            ret[i] = sqrt((ret[i] - mean2 / counter[i]) / (counter[i] - ddof));
        }
        else ret[i] = fill_value;
    }"""

c_iter['var'] = c_iter['std']

c_finish['var'] = r"""
    double mean2 = 0;
    for (long i=0; i<Lret; i++) {
        if (counter[i] != 0) {
            mean2 = means[i] * means[i];
            ret[i] = (ret[i] - mean2 / counter[i]) / (counter[i] - ddof);
        }
        else ret[i] = fill_value;
    }"""



def c_func(funcname, reverse=False, nans=False, scalar=False):
    """ Fill c_funcs with constructed code from the templates """
    varnames = ['group_idx', 'a', 'ret', 'counter']
    codebase = c_base_reverse if reverse else c_base
    iterbase = c_iter_scalar[funcname] if scalar else c_iter[funcname]
    iteration = c_nan_iter(iterbase) if nans else iterbase
    if scalar:
        varnames.remove('a')
    return codebase % dict(init=c_init(varnames), iter=iteration,
                           finish=c_finish.get(funcname, ''))

def get_cfuncs():
    c_funcs = dict()
    for funcname in c_iter:
        c_funcs[funcname] = c_func(funcname)
        if funcname not in _no_separate_nan_version:
            c_funcs['nan' + funcname] = c_func(funcname, nans=True)
    for funcname in c_iter_scalar:
        c_funcs[funcname + 'scalar'] = c_func(funcname, scalar=True)
    c_funcs['first'] = c_func('last', reverse=True)
    c_funcs['nanfirst'] = c_func('last', reverse=True, nans=True)
    return c_funcs


c_funcs = get_cfuncs()


c_step_count = c_size('group_idx') + r"""
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
    return inline(c_step_count, ['group_idx'], define_macros=c_macros)


c_step_indices = c_size('group_idx') + r"""
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
    inline(c_step_indices, ['group_idx', 'indices'], define_macros=c_macros)
    return indices

        
def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C', dtype=None, **kwargs):
    func = get_func(func, aliasing, optimized_funcs)
    if not isstr(func):
        # Fall back to acuum_np if no optimized C version is available
        return aggregate_np(group_idx, a, func=func, dtype=dtype,
                            fill_value=fill_value)

    # Preparations for optimized processing
    group_idx, a, flat_size, ndim_idx, size = input_validation(group_idx, a,
                                                               size=size,
                                                               order=order)
    dtype = check_dtype(dtype, func, a, len(group_idx))
    check_fill_value(fill_value, dtype)

    if func in ('sum', 'any', 'anynan', 'nansum'):
        ret = np.zeros(flat_size, dtype=dtype)
    elif func in ('prod', 'all', 'allnan', 'nanprod'):
        ret = np.ones(flat_size, dtype=dtype)
    else:
        ret = np.full(flat_size, fill_value, dtype=dtype)

    # In case we should get some ugly fortran arrays, convert them
    inline_vars = dict(group_idx=np.ascontiguousarray(group_idx), a=np.ascontiguousarray(a),
                      ret=ret, fill_value=fill_value)
    # TODO: Have this fixed by proper raveling
    # inline_vars = dict(group_idx=group_idx, a=a, ret=ret, fill_value=fill_value)
    if func in ('std', 'var', 'nanstd', 'nanvar'):
        counter = np.zeros_like(ret, dtype=int)
        inline_vars['means'] = np.zeros_like(ret)
        inline_vars['ddof'] = kwargs.pop('ddof', 0)
    elif func in ('mean', 'nanmean'):
        counter = np.zeros_like(ret, dtype=int)
    else:
        # Using inverse logic, marking anyting touched with zero for later removal
        counter = np.ones_like(ret, dtype=bool)
    inline_vars['counter'] = counter

    if np.isscalar(a):
        func += 'scalar'
        inline_vars['a'] = a
    inline(c_funcs[func], inline_vars.keys(), local_dict=inline_vars, define_macros=c_macros, extra_compile_args=['-O3'])

    # Postprocessing
    if func in ('sum', 'any', 'anynan', 'nansum') and fill_value != 0:
        ret[counter] = fill_value
    elif func in ('prod', 'all', 'allnan', 'nanprod') and fill_value != 1:
        ret[counter] = fill_value

    # deal with ndimensional indexing
    if ndim_idx > 1:
        ret = ret.reshape(size, order=order)
    return ret

aggregate.__doc__ = """
    This is the scipy.weave implementation of aggregate.

    **NOTE:** If weave is installed but fails to run (probably because you
    have not setup a suitable compiler) then you can manually select the numpy
    implementation by using::


        import numpy_groupies as npg
        # NOT THIS: npg.aggregate(...)
        npg.aggregate_np(...)


    """ + _doc_str
