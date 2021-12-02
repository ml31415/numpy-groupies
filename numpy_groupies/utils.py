"""Common helpers without certain dependencies."""

aggregate_common_doc = """
    See readme file at https://github.com/ml31415/numpy-groupies for a full
    description.  Below we reproduce the "Full description of inputs"
    section from that readme, note that the text below makes references to
    other portions of the readme that are not shown here.

    group_idx:
        this is an array of non-negative integers, to be used as the "labels"
        with which to group the values in ``a``. Although we have so far
        assumed that ``group_idx`` is one-dimesnaional, and the same length as
        ``a``, it can in fact be two-dimensional (or some form of nested
        sequences that can be converted to 2D).  When ``group_idx`` is 2D, the
        size of the 0th dimension corresponds to the number of dimesnions in
        the output, i.e. ``group_idx[i,j]`` gives the index into the ith
        dimension in the output
        for ``a[j]``.  Note that ``a`` should still be 1D (or scalar), with
        length matching ``group_idx.shape[1]``.
    a:
        this is the array of values to be aggregated.  See above for a
        simple demonstration of what this means.  ``a`` will normally be a
        one-dimensional array, however it can also be a scalar in some cases.
    func: default='sum'
        the function to use for aggregation.  See the section above for
        details. Note that the simplest way to specify the function is using a
        string (e.g. ``func='max'``) however a number of aliases are also
        defined (e.g. you can use the ``func=np.max``, or even ``func=max``,
        where ``max`` is the
        builtin function).  To check the available aliases see ``utils.py``.
    size: default=None
        the shape of the output array. If ``None``, the maximum value in
        ``group_idx`` will set the size of the output.  Note that for
        multidimensional output you need to list the size of each dimension
        here, or give ``None``.
    fill_value: default=0
        in the example above, group 2 does not have any data, so requires some
        kind of filling value - in this case the default of ``0`` is used.  If
        you had set ``fill_value=nan`` or something else, that value would
        appear instead of ``0`` for the 2 element in the output.  Note that
        there are some subtle interactions between what is permitted for
        ``fill_value`` and the input/output ``dtype`` - exceptions should be
        raised in most cases to alert the programmer if issue arrise.
    order: default='C'
        this is relevant only for multimensional output.  It controls the
        layout of the output array in memory, can be ``'F'`` for fortran-style.
    dtype: default=None
        the ``dtype`` of the output.  By default something sensible is chosen
        based on the input, aggregation function, and ``fill_value``.
    ddof: default=0
        passed through into calculations of variance and standard deviation
        (see above).
"""

funcs_common = 'first last len mean var std allnan anynan max min argmax argmin cumsum cumprod cummax cummin'.split()
funcs_no_separate_nan = frozenset(['sort', 'rsort', 'array', 'allnan', 'anynan'])


_alias_str = {
    'or': 'any',
    'and': 'all',
    'add': 'sum',
    'count': 'len',
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
    'rsorted': 'sort',
    'dsort': 'sort',
    'dsorted': 'rsort',
}

_alias_builtin = {
    all: 'all',
    any: 'any',
    len: 'len',
    max: 'max',
    min: 'min',
    sum: 'sum',
    sorted: 'sort',
    slice: 'array',
    list: 'array',
}


def get_aliasing(*extra):
    """The assembles the dict mapping strings and functions to the list of
    supported function names:
            e.g. alias['add'] = 'sum'  and alias[sorted] = 'sort'
    This funciton should only be called during import.
    """
    alias = dict((k, k) for k in funcs_common)
    alias.update(_alias_str)
    alias.update((fn, fn) for fn in _alias_builtin.values())
    alias.update(_alias_builtin)
    for d in extra:
        alias.update(d)
    alias.update((k, k) for k in set(alias.values()))
    # Treat nan-functions as firstclass member and add them directly
    for key in set(alias.values()):
        if key not in funcs_no_separate_nan:
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
        if func_str.startswith('nan') and \
                func_str[3:] in funcs_no_separate_nan:
            raise ValueError("%s does not have a nan-version".format(func_str[3:]))
        else:
            raise NotImplementedError("No such function available")
    raise ValueError("func %s is neither a valid function string nor a "
                     "callable object".format(func))


def check_boolean(x):
    if x not in (0, 1):
        raise ValueError("Value not boolean")


try:
    basestring  # Attempt to evaluate basestring

    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    # Probably Python 3.x
    def isstr(s):
        return isinstance(s, str)
