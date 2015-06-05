import math
import numpy as np

from .utils import get_aliasing, check_boolean, get_func


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


try:
    import bottleneck as bn
except ImportError:
    _alias_bottleneck = {}
else:
    _bn_funcs = 'allnan anynan nansum nanmin nanmax nanmean nanvar nanstd'.split()
    _alias_bottleneck = dict((getattr(bn, fn), fn) for fn in _bn_funcs)

aliasing = get_aliasing(_alias_numpy, _alias_bottleneck)


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


def input_validation(group_idx, a, size=None, order='C'):
    """ Do some fairly extensive checking of group_idx and a, trying to give the user
        as much help as possible with what is wrong. Also, convert ndim-indexing to 1d indexing.
    """
    a = np.asanyarray(a)
    group_idx = np.asanyarray(group_idx)

    if not issubclass(group_idx.dtype.type, np.integer):
        raise TypeError("group_idx must be of integer type")
    if np.ndim(a) > 1:
        raise ValueError("a must be scalar or 1 dimensional, use .ravel to flatten.")

    ndim_idx = np.ndim(group_idx)
    if ndim_idx == 1:
        if np.any(group_idx < 0):
            raise ValueError("Negative indices not supported.")
        if size is None:
            size = np.max(group_idx) + 1
        else:
            if not np.isscalar(size):
                raise ValueError("Output size must be scalar or None")
            if np.any(group_idx > size - 1):
                raise ValueError("One or more indices are too large for size %d." % size)
        flat_size = size
    else:
        if size is None:
            size = np.max(group_idx, axis=1) + 1
        else:
            if np.isscalar(size):
                raise ValueError("Output size must be None or 1d sequence of length %d" % group_idx.shape[0])
            if len(size) != group_idx.shape[0]:
                raise ValueError("%d sizes given, but %d output dimensions specified in index" % (len(size), group_idx.shape[0]))

        group_idx = np.ravel_multi_index(tuple(group_idx), size, order=order, mode='raise')
        flat_size = np.prod(size)

    if not (np.ndim(a) == 0 or len(a) == group_idx.size):
        raise ValueError("group_idx and a must be of the same length, or a can be scalar")

    return group_idx, a, flat_size, ndim_idx