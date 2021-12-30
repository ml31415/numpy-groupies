import numpy as np

try:
    from weave import inline
except ImportError:
    from scipy.weave import inline

from .utils import aggregate_common_doc, check_boolean, funcs_no_separate_nan, get_func, isstr
from .utils_numpy import aliasing, check_dtype, check_fill_value, input_validation

optimized_funcs = {'sum', 'min', 'max', 'amin', 'amax', 'mean', 'var', 'std', 'prod', 'len',
                   'nansum', 'nanmin', 'nanmax', 'nanmean', 'nanvar', 'nanstd', 'nanprod', 'nanlen',
                   'all', 'any', 'nanall', 'nanany', 'allnan', 'anynan',
                   'first', 'last', 'nanfirst', 'nanlast'}

# c_funcs will contain all generated c code, so it can be read easily for debugging
c_funcs = dict()
c_iter = dict()
c_iter_scalar = dict()
c_finish = dict()

# Set this for testing, to fail deprecated C-API calls
#c_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
c_macros = []
c_args = ['-Wno-cpp']  # Suppress the deprecation warnings created by weave


def c_size(varname):
    return r"""
    long L%(varname)s = 1;
    for (int n=0; n<D%(varname)s; n++) L%(varname)s *= N%(varname)s[n];""" % dict(varname=varname)


def c_init(varnames):
    return '    ' + ''.join(c_size(varname) for varname in varnames).lstrip() + """

    long ri = 0;
    long cmp_pos = 0;"""


c_ri = """ri = group_idx[i];"""
c_ri_redir = """ri = (group_idx[i] + 1) * (a[i] == a[i]);"""

c_base = r"""%(init)s

    for (long i=0; i<Lgroup_idx; i++) {
        %(ri_redir)s
        %(iter)s
    }
    %(finish)s
    """

c_base_reverse = r"""%(init)s

    for (long i=Lgroup_idx-1; i>=0; i--) {
        %(ri_redir)s
        %(iter)s
    }
    %(finish)s
    """

c_iter['sum'] = r"""
        counter[ri] = 0;
        ret[ri] += a[i];"""

c_iter_scalar['sum'] = r"""
        counter[ri] = 0;
        ret[ri] += a;"""

c_iter['prod'] = r"""
        counter[ri] = 0;
        ret[ri] *= a[i];"""

c_iter_scalar['prod'] = r"""
        counter[ri] = 0;
        ret[ri] *= a;"""

c_iter['len'] = r"""
        counter[ri] = 0;
        ret[ri] += 1;"""

c_iter_scalar['len'] = r"""
        counter[ri] = 0;
        ret[ri] += 1;"""

c_iter['all'] = r"""
        counter[ri] = 0;
        ret[ri] &= (a[i] != 0);"""

c_iter['any'] = r"""
        counter[ri] = 0;
        ret[ri] |= (a[i] != 0);"""

c_iter['last'] = r"""
        ret[ri] = a[i];"""

c_iter['allnan'] = r"""
        counter[ri] = 0;
        ret[ri] &= (a[i] == a[i]);"""

c_iter['anynan'] = r"""
        counter[ri] = 0;
        ret[ri] |= (a[i] == a[i]);"""

c_iter['max'] = r"""
        if (counter[ri]) {
            ret[ri] = a[i];
            counter[ri] = 0;
        }
        else if (ret[ri] < a[i]) ret[ri] = a[i];"""

c_iter['min'] = r"""
        if (counter[ri]) {
            ret[ri] = a[i];
            counter[ri] = 0;
        }
        else if (ret[ri] > a[i]) ret[ri] = a[i];"""

c_iter['mean'] = r"""
        counter[ri]++;
        ret[ri] += a[i];"""

c_finish['mean'] = r"""
    for (long ri=0; ri<Lret; ri++) {
        if (counter[ri]) ret[ri] = ret[ri] / counter[ri];
        else ret[ri] = fill_value;
    }"""

c_iter['std'] = r"""
        counter[ri]++;
        means[ri] += a[i];
        ret[ri] += a[i] * a[i];"""

c_finish['std'] = r"""
    double mean2 = 0;
    for (long ri=0; ri<Lret; ri++) {
        if (counter[ri]) {
            mean2 = means[ri] * means[ri];
            ret[ri] = sqrt((ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof));
        }
        else ret[ri] = fill_value;
    }"""

c_iter['var'] = c_iter['std']

c_finish['var'] = r"""
    double mean2 = 0;
    for (long ri=0; ri<Lret; ri++) {
        if (counter[ri]) {
            mean2 = means[ri] * means[ri];
            ret[ri] = (ret[ri] - mean2 / counter[ri]) / (counter[ri] - ddof);
        }
        else ret[ri] = fill_value;
    }"""


def c_func(funcname, reverse=False, nans=False, scalar=False):
    """ Fill c_funcs with constructed code from the templates """
    varnames = ['group_idx', 'a', 'ret', 'counter']
    codebase = c_base_reverse if reverse else c_base
    iteration = c_iter_scalar[funcname] if scalar else c_iter[funcname]
    if scalar:
        varnames.remove('a')
    return codebase % dict(init=c_init(varnames), iter=iteration,
                           finish=c_finish.get(funcname, ''),
                           ri_redir=(c_ri_redir if nans else c_ri))


def get_cfuncs():
    c_funcs = dict()
    for funcname in c_iter:
        c_funcs[funcname] = c_func(funcname)
        if funcname not in funcs_no_separate_nan:
            c_funcs['nan' + funcname] = c_func(funcname, nans=True)
    for funcname in c_iter_scalar:
        c_funcs[funcname + 'scalar'] = c_func(funcname, scalar=True)
    c_funcs['first'] = c_func('last', reverse=True)
    c_funcs['nanfirst'] = c_func('last', reverse=True, nans=True)
    return c_funcs


c_funcs.update(get_cfuncs())


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
    return inline(c_step_count, ['group_idx'], define_macros=c_macros, extra_compile_args=c_args)


c_step_indices = c_size('group_idx') + r"""
    long cmp_pos = 0;
    long ri = 1;
    for (long i=1; i<Lgroup_idx; i++) {
        if (group_idx[cmp_pos] != group_idx[i]) {
            cmp_pos = i;
            indices[ri++] = i;
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
    inline(c_step_indices, ['group_idx', 'indices'], define_macros=c_macros, extra_compile_args=c_args)
    return indices


_force_fill_0 = frozenset({'sum', 'any', 'len', 'anynan', 'mean', 'std', 'var',
                          'nansum', 'nanlen', 'nanmean', 'nanstd', 'nanvar'})
_force_fill_1 = frozenset({'prod', 'all', 'allnan', 'nanprod'})


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    func = get_func(func, aliasing, optimized_funcs)
    if not isstr(func):
        raise NotImplementedError("generic functions not supported, in the weave implementation of aggregate")

    # Preparations for optimized processing
    group_idx, a, flat_size, ndim_idx, size = input_validation(group_idx, a,
                                                               size=size,
                                                               order=order,
                                                               axis=axis)
    dtype = check_dtype(dtype, func, a, len(group_idx))
    check_fill_value(fill_value, dtype, func=func)
    nans = func.startswith('nan')

    if nans:
        flat_size += 1

    if func in _force_fill_0:
        ret = np.zeros(flat_size, dtype=dtype)
    elif func in _force_fill_1:
        ret = np.ones(flat_size, dtype=dtype)
    else:
        ret = np.full(flat_size, fill_value, dtype=dtype)

    # In case we should get some ugly fortran arrays, convert them
    inline_vars = dict(group_idx=np.ascontiguousarray(group_idx), a=np.ascontiguousarray(a),
                       ret=ret, fill_value=fill_value)
    # TODO: Have this fixed by proper raveling
    if func in {'std', 'var', 'nanstd', 'nanvar'}:
        counter = np.zeros_like(ret, dtype=int)
        inline_vars['means'] = np.zeros_like(ret)
        inline_vars['ddof'] = kwargs.pop('ddof', 0)
    elif func in {'mean', 'nanmean'}:
        counter = np.zeros_like(ret, dtype=int)
    else:
        # Using inverse logic, marking anything touched with zero for later removal
        counter = np.ones_like(ret, dtype=bool)
    inline_vars['counter'] = counter

    if np.isscalar(a):
        func += 'scalar'
        inline_vars['a'] = a
    inline(c_funcs[func], inline_vars.keys(), local_dict=inline_vars, define_macros=c_macros, extra_compile_args=c_args)

    # Postprocessing
    if func in _force_fill_0 and fill_value != 0 or func in _force_fill_1 and fill_value != 1:
        if counter.dtype == np.bool_:
            ret[counter] = fill_value
        else:
            ret[~counter.astype(bool)] = fill_value

    if nans:
        # Restore the shifted return array
        ret = ret[1:]

    # Deal with ndimensional indexing
    if ndim_idx > 1:
        ret = ret.reshape(size, order=order)
    return ret


aggregate.__doc__ = """
    This is the weave based implementation of aggregate.

    **NOTE:** If weave is installed but fails to run (probably because you
    have not setup a suitable compiler) then you can manually select the numpy
    implementation by using::


        import numpy_groupies as npg
        # NOT THIS: npg.aggregate(...)
        npg.aggregate_np(...)


    """ + aggregate_common_doc
