from functools import partial
import numpy as np
import pandas as pd

from .utils_numpy import check_dtype, allnan, anynan, _no_separate_nan_version
from .aggregate_numpy import aggregate as aggregate_np


def _wrapper(group_idx, a, size, fill_value, func='sum', dtype=None, ddof=0):
    kwargs = dict()
    if func in ('var', 'std'):
        kwargs['ddof'] = ddof
    if isinstance(func, basestring):
        grouped = getattr(pd.DataFrame({'group_idx': group_idx, 'a': a}).groupby('group_idx'), func)(**kwargs)
    else:
        grouped = pd.DataFrame({'group_idx': group_idx, 'a': a}).groupby('group_idx').aggregate(func, **kwargs)

    dtype = check_dtype(dtype, func, a)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[grouped.index] = grouped
    return ret

_supported_funcs = 'min max sum prod mean var std first last all any'.split()
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_impl_dict.update(('nan' + fn, partial(_wrapper, func=fn)) for fn in _supported_funcs if fn not in _no_separate_nan_version)
_impl_dict.update(allnan=partial(_wrapper, func=allnan), anynan=partial(_wrapper, func=anynan))


def aggregate(*args, **kwargs):
    """
    Aggregation similar to Matlab's `accumarray` function.
    
    See readme file at https://github.com/ml31415/accumarray for 
    full description.  Or see ``accumarray`` in ``accumarray_numpy.py``.

    This function makes use of `pandas`'s groupby machienery, it is slightly
    faster than the numpy implementation for `max`, `min`, and `prod`, but slower
    for other functions.
   
    Note that this implementation piggybacks on the main error checking and
    argument parsing etc. in ``accumarray_numpy.py``.
    """
    return aggregate_np(*args, _impl_dict=_impl_dict, **kwargs)


