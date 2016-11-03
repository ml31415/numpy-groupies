import math

_doc_str = """
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

_funcs_common = 'first last len mean var std allnan anynan max min argmax argmin'.split()
_no_separate_nan_version = {'sort', 'rsort', 'array', 'allnan', 'anynan'}


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
    'rsorted': 'rsort',
    'dsort': 'rsort',
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

aliasing_purepy = get_aliasing()


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
                func_str[3:] in _no_separate_nan_version:
            raise ValueError("%s does not have a nan-version" % func_str[3:])
        else:
            raise NotImplementedError("No such function available")
    raise ValueError("func %s is neither a valid function string nor a "
                     "callable object" % func)


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


try:
    import numpy as np
except ImportError:
    pass
else:
    _alias_numpy = {
        np.add: 'sum',
        np.sum: 'sum',
        np.any: 'any',
        np.all: 'all',
        np.multiply: 'prod',
        np.prod: 'prod',
        np.amin: 'min',
        np.min: 'min',
        np.minimum: 'min',
        np.amax: 'max',
        np.max: 'max',
        np.maximum: 'max',
        np.argmax: 'argmax',
        np.argmin: 'argmin',
        np.mean: 'mean',
        np.std: 'std',
        np.var: 'var',
        np.array: 'array',
        np.asarray: 'array',
        np.sort: 'sort',
        np.nansum: 'nansum',
        np.nanprod: 'nanprod',
        np.nanmean: 'nanmean',
        np.nanvar: 'nanvar',
        np.nanmax: 'nanmax',
        np.nanmin: 'nanmin',
        np.nanstd: 'nanstd',
        np.nanargmax: 'nanargmax',
        np.nanargmin: 'nanargmin'
    }

    try:
        import bottleneck as bn
    except ImportError:
        _alias_bottleneck = {}
    else:
        _bn_funcs = 'allnan anynan nansum nanmin nanmax nanmean nanvar nanstd'
        _alias_bottleneck = dict((getattr(bn, fn), fn)
                                 for fn in _bn_funcs.split())

    aliasing = get_aliasing(_alias_numpy, _alias_bottleneck)

    def fill_untouched(idx, ret, fill_value):
        """any elements of ret not indexed by idx are set to fill_value."""
        untouched = np.ones_like(ret, dtype=bool)
        untouched[idx] = False
        ret[untouched] = fill_value

    _next_int_dtype = dict(
        bool=np.int8,
        uint8=np.int16,
        int8=np.int16,
        uint16=np.int32,
        int16=np.int32,
        uint32=np.int64,
        int32=np.int64
    )

    _next_float_dtype = dict(
        float16=np.float32,
        float32=np.float64,
        float64=np.complex64,
        complex64=np.complex128
    )

    def minimum_dtype(x, dtype=np.bool_):
        """returns the "most basic" dtype which represents `x` properly, which
        provides at least the same value range as the specified dtype."""

        def check_type(x, dtype):
            try:
                converted = dtype.type(x)
            except (ValueError, OverflowError):
                return False
            # False if some overflow has happened
            return converted == x or math.isnan(x)

        def type_loop(x, dtype, dtype_dict, default=None):
            while True:
                try:
                    dtype = np.dtype(dtype_dict[dtype.name])
                    if check_type(x, dtype):
                        return np.dtype(dtype)
                except KeyError:
                    if default is not None:
                        return np.dtype(default)
                    raise ValueError("Can not determine dtype of %r" % x)

        dtype = np.dtype(dtype)
        if check_type(x, dtype):
            return dtype

        if np.issubdtype(dtype, np.inexact):
            return type_loop(x, dtype, _next_float_dtype)
        else:
            return type_loop(x, dtype, _next_int_dtype, default=np.float32)


    def minimum_dtype_scalar(x, dtype, a):
        if dtype is None:
            dtype = np.dtype(type(a)) if isinstance(a, (int, float))\
                                      else a.dtype
        return minimum_dtype(x, dtype)

    _forced_types = {
        'array': np.object,
        'all': np.bool_,
        'any': np.bool_,
        'nanall': np.bool_,
        'nanany': np.bool_,
        'len': np.int64,
        'nanlen': np.int64,
        'allnan': np.bool_,
        'anynan': np.bool_,
    }
    _forced_float_types = {'mean', 'var', 'std', 'nanmean', 'nanvar', 'nanstd'}
    _forced_same_type = {'min', 'max', 'first', 'last', 'nanmin', 'nanmax',
                         'nanfirst', 'nanlast'}


    def check_dtype(dtype, func_str, a, n):
        if np.isscalar(a) or not a.shape:
            if func_str not in ("sum", "prod", "len"):
                raise ValueError("scalar inputs are supported only for 'sum', "
                                 "'prod' and 'len'")
            a_dtype = np.dtype(type(a))
        else:
            a_dtype = a.dtype

        if dtype is not None:
            # dtype set by the user
            # Careful here: np.bool != np.bool_ !
            if np.issubdtype(dtype, np.bool_) and \
                    not('all' in func_str or 'any' in func_str):
                raise TypeError("function %s requires a more complex datatype "
                                "than bool" % func_str)
            if not np.issubdtype(dtype, np.integer) and func_str in ('len', 'nanlen'):
                raise TypeError("function %s requires an integer datatype" % func_str)
            # TODO: Maybe have some more checks here
            return np.dtype(dtype)
        else:
            try:
                return np.dtype(_forced_types[func_str])
            except KeyError:
                if func_str in _forced_float_types:
                    if np.issubdtype(a_dtype, np.floating):
                        return a_dtype
                    else:
                        return np.dtype(np.float64)
                else:
                    if func_str == 'sum':
                        # Try to guess the minimally required int size
                        if np.issubdtype(a_dtype, np.int64):
                            # It's not getting bigger anymore
                            # TODO: strictly speaking it might need float
                            return np.dtype(np.int64)
                        elif np.issubdtype(a_dtype, np.integer):
                            maxval = np.iinfo(a_dtype).max * n
                            return minimum_dtype(maxval, a_dtype)
                        elif np.issubdtype(a_dtype, np.bool_):
                            return minimum_dtype(n, a_dtype)
                        else:
                            # floating, inexact, whatever
                            return a_dtype
                    elif func_str in _forced_same_type:
                        return a_dtype
                    else:
                        if isinstance(a_dtype, np.integer):
                            return np.dtype(np.int64)
                        else:
                            return a_dtype


    def check_fill_value(fill_value, dtype):
        try:
            return dtype.type(fill_value)
        except ValueError:
            raise ValueError("fill_value must be convertible into %s"
                             % dtype.type.__name__)


    def check_group_idx(group_idx, a=None, check_min=True):
        if a is not None and group_idx.size != a.size:
            raise ValueError("The size of group_idx must be the same as "
                             "a.size")
        if not issubclass(group_idx.dtype.type, np.integer):
            raise TypeError("group_idx must be of integer type")
        if check_min and np.min(group_idx) < 0:
            raise ValueError("group_idx contains negative indices")


    def input_validation(group_idx, a, size=None, order='C', axis=None,
                         ravel_group_idx=True, check_bounds=True):
        """ Do some fairly extensive checking of group_idx and a, trying to
        give the user as much help as possible with what is wrong. Also,
        convert ndim-indexing to 1d indexing.
        """
        if not isinstance(a, (int, float, complex)):
            a = np.asanyarray(a)
        group_idx = np.asanyarray(group_idx)

        if not np.issubdtype(group_idx.dtype, np.integer):
            raise TypeError("group_idx must be of integer type")

        # This check works for multidimensional indexing as well
        if check_bounds and np.any(group_idx < 0):
            raise ValueError("negative indices not supported")

        ndim_idx = np.ndim(group_idx)
        ndim_a = np.ndim(a)

        # Deal with the axis arg: if present, then turn 1d indexing into
        # multi-dimensional indexing along the specified axis.
        if axis is None:
            if ndim_a > 1:
                raise ValueError("a must be scalar or 1 dimensional, use .ravel to"
                                 " flatten. Alternatively specify axis.")
        elif axis >= ndim_a or axis < -ndim_a:
            raise ValueError("axis arg too large for np.ndim(a)")
        else:
            axis = axis if axis >= 0 else ndim_a + axis  # negative indexing
            if ndim_idx > 1:
                # TODO: we could support a sequence of axis values for multiple
                # dimensions of group_idx.
                raise NotImplementedError("only 1d indexing currently"
                                          "supported with axis arg.")
            elif a.shape[axis] != len(group_idx):
                raise ValueError("a.shape[axis] doesn't match length of group_idx.")
            elif size is not None and not np.isscalar(size):
                raise NotImplementedError("when using axis arg, size must be"
                                          "None or scalar.")
            else:
                # Create the broadcast-ready multidimensional indexing.
                # Note the user could do this themselves, so this is
                # very much just a convenience.
                size_in = np.max(group_idx) + 1 if size is None else size
                group_idx_in = group_idx
                group_idx = []
                size = []
                for ii, s in enumerate(a.shape):
                    ii_idx = group_idx_in if ii == axis else np.arange(s)
                    ii_shape = [1] * ndim_a
                    ii_shape[ii] = s
                    group_idx.append(ii_idx.reshape(ii_shape))
                    size.append(size_in if ii == axis else s)
                # Use the indexing, and return. It's a bit simpler than
                # using trying to keep all the logic below happy
                group_idx = np.ravel_multi_index(group_idx, size, order=order,
                                                 mode='raise')
                flat_size = np.prod(size)
                ndim_idx = ndim_a
                return group_idx.ravel(), a.ravel(), flat_size, ndim_idx, size

        if ndim_idx == 1:
            if size is None:
                size = np.max(group_idx) + 1
            else:
                if not np.isscalar(size):
                    raise ValueError("output size must be scalar or None")
                if check_bounds and np.any(group_idx > size - 1):
                    raise ValueError("one or more indices are too large for "
                                     "size %d" % size)
            flat_size = size
        else:
            if size is None:
                size = np.max(group_idx, axis=1) + 1
            elif np.isscalar(size):
                raise ValueError("output size must be of length %d"
                                 % len(group_idx))
            elif len(size) != len(group_idx):
                raise ValueError("%d sizes given, but %d output dimensions "
                                 "specified in index" % (len(size),
                                                         len(group_idx)))
            if ravel_group_idx:
                group_idx = np.ravel_multi_index(group_idx, size, order=order,
                                                 mode='raise')
            flat_size = np.prod(size)

        if not (np.ndim(a) == 0 or len(a) == group_idx.size):
            raise ValueError("group_idx and a must be of the same length, or a"
                             " can be scalar")

        return group_idx, a, flat_size, ndim_idx, size
