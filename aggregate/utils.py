try:
    import numpy as np
except ImportError:
    np = None


def check_group_idx(group_idx, a=None, check_min=True):
    if a is not None and group_idx.size != a.size:
        raise ValueError("The size of group_idx must be the same as a.size")
    if not issubclass(group_idx.dtype.type, np.integer):
        raise TypeError("group_idx must be of integer type")
    if check_min and np.min(group_idx) < 0:
        raise ValueError("group_idx contains negative indices")


def check_boolean(x):
    if x not in (0, 1):
        raise ValueError("Value not boolean")


def fill_untouched(idx, ret, fill_value):
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value

_funcs_common = 'first last mean var std allnan anynan'.split()

_alias_str = {
    'or': 'any',
    'and': 'all',
    'add': 'sum',
    'plus': 'sum',
    'multiply': 'prod',
    'product': 'prod',
    'times': 'prod',
    'amax': 'max',
    'maximum': 'max',
    'amin': 'min',
    'minimum': 'min',
    'split': 'array',
    'splice': 'array',
    'sorted': 'sort',
    'asort': 'sort',
    'asorted': 'sort',
    'rsorted': 'rsort',
    'dsort': 'rsort',
    'dsorted': 'rsort',
}

_alias_builtin = {
    all: 'all',
    any: 'any',
    max: 'max',
    min: 'min',
    sum: 'sum',
    sorted: 'sort',
    slice: 'array',
    list: 'array',
}

_no_separate_nan_version = {'sort', 'rsort', 'array', 'allnan', 'anynan'}


def get_aliasing(*extra):
    """This should be called only once by an aggregate_implementation.py file,
        i.e. it should be called at the point when the given implementation is imported.

        It returns two things. The first is a dict mapping strings and functions
        to the list of supported funciton names:     
            e.g. alias['add'] = 'sum'  and alias[sorted] = 'sort'   
        The second output is a list of functions names which should not support
        nan- prefixing.
    """
    alias = dict((k, k) for k in _funcs_common)
    alias.update(_alias_str)
    alias.update((fn, fn) for fn in _alias_builtin.values())
    alias.update(_alias_builtin)
    for d in extra:
        alias.update(d)
    alias.update((k, k) for k in set(alias.values()))
    # Treat nan-functions as firstclass member and add them directly
    for key in set(alias.values()):
        if key not in _no_separate_nan_version:
            key = 'nan' + key
            alias[key] = key


    return alias

aliasing = get_aliasing()


def get_func(func, aliasing, implementations):
    """ Return the key of a found implementation or the func itself """
    try:
        func_str = aliasing[func]
    except KeyError:
        if callable(func):
            return func
    else:
        if func_str in implementations:
            return func_str
        if func_str.startswith('nan') and func_str[3:] in _no_separate_nan_version:
            raise ValueError("%s does not have a nan-version" % func_str[3:])
        else:
            raise NotImplementedError("No such function available")
    raise ValueError("func %s is neither a valid function string nor a callable object" % func)
