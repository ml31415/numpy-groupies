import math
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

if np is None:
    _alias_numpy = {}
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
        np.mean: 'mean',
        np.std: 'std',
        np.var: 'var',
        np.array: 'array',
        np.asarray: 'array',
        np.sort: 'sort',
        np.nansum: 'nansum',
        np.nanmean: 'nanmean',
        np.nanvar: 'nanvar',
        np.nanmax: 'nanmax',
        np.nanmin: 'nanmin',
        np.nanstd: 'nanstd',
    }

_no_separate_nan_version = {'sort', 'rsort', 'array', 'allnan', 'anynan', 'list'}


def get_aliasing(with_numpy=False):
    """This should be called only once by an aggregate_implementation.py file,
        i.e. it should be called at the point when the given implementation is imported.

        It returns two things. The first is a dict mapping strings and functions
        to the list of supported funciton names:     
            e.g. alias['add'] = 'sum'  and alias[sorted] = 'sort'   
        The second output is a list of functions names which should not support
        nan- prefixing.
    """
    alias = dict((k, k) for k in ['first', 'last'])
    alias.update(_alias_str)
    alias.update(_alias_builtin)
    if with_numpy:
        alias.update(_alias_numpy)
    alias.update((k, k) for k in set(alias.values()))
    # Treat nan-functions as firstclass member and add them directly
    for key in set(alias.values()):
        if key not in _no_separate_nan_version:
            key = 'nan' + key
            alias[key] = key


    return alias

aliasing = get_aliasing(with_numpy=False)
aliasing_numpy = get_aliasing(with_numpy=True)


def get_func_str(aliasing, func):
    try:
        return aliasing[func]
    except KeyError:
        if isinstance(func, basestring):
            func_str = func
        elif callable(func):
            func_str = func.__name__
        else:
            raise ValueError("func is neither a string nor a callable object")
        # Tried everything to resolve to string, if still not found,
        # no optimized version is available and simple loop is required
        # KeyError is thrown in that case
        if func_str.startswith('nan') and func_str[3:] in _no_separate_nan_version:
            raise ValueError("%s does not have a nan-version" % func_str[3:])
        return aliasing.get(func_str, func_str)


if np is not None:
    def allnan(x):
        return np.all(np.isnan(x))


    def anynan(x):
        return np.any(np.isnan(x))


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
        """returns the "most basic" dtype which represents `x` properly, which is
        at least as "complicated" as the specified dtype."""

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
            return type_loop(x, dtype, _next_int_dtype, default=np.int64)

    _forced_types = {
        'array': np.object,
        'all': np.bool_,
        'any': np.bool_,
        'allnan': np.bool_,
        'anynan': np.bool_,
    }
    _forced_float_types = {'mean', 'var', 'std', 'nanmean', 'nanvar', 'nanstd'}
    _forced_same_type = {'min', 'max', 'first', 'last', 'nanmin', 'nanmax', 'nanfirst', 'nanlast'}

    def check_dtype(dtype, func_str, a):
        if dtype is not None:
            # dtype set by the user
            # Careful here: np.bool != np.bool_ !
            if np.issubdtype(dtype, np.bool_) and not ('all' in func_str or 'any' in func_str):
                raise TypeError("function %s requires a more complex datatype than bool" % func_str)
            # TODO: Maybe have some more checks here, if the user is doing some sane thing
            return np.dtype(dtype)
        else:
            try:
                return np.dtype(_forced_types[func_str])
            except KeyError:
                if func_str in _forced_float_types:
                    if not np.issubdtype(dtype, np.floating):
                        return np.dtype(np.float64)
                    else:
                        return a.dtype
                else:
                    if func_str == 'sum':
                        # Try to guess the minimally required int size
                        if np.issubdtype(a.dtype, np.int64):
                            # It's not getting bigger anymore, so let's shortcut this
                            return np.dtype(np.int64)
                        elif np.issubdtype(a.dtype, np.integer):
                            maxval = np.iinfo(a.dtype).max * len(a)
                            return minimum_dtype(maxval, a.dtype)
                        elif np.issubdtype(dtype, np.bool_):
                            return minimum_dtype(len(a), a.dtype)
                        else:
                            # floating, inexact, whatever
                            return a.dtype
                    elif func_str in _forced_same_type:
                        return a.dtype
                    else:
                        if isinstance(a.dtype, np.integer):
                            return np.dtype(np.int64)
                        else:
                            return a.dtype

    def check_fill_value(fill_value, dtype):
        try:
            return dtype.type(fill_value)
        except ValueError:
            raise ValueError("fill_value must be convertible into %s" % dtype.type.__name__)
