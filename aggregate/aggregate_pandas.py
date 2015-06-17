from functools import partial
import numpy as np
import pandas as pd

from .utils import (check_dtype, allnan, anynan, _no_separate_nan_version,
                    _doc_str)
from .aggregate_numpy import _aggregate_base


def _wrapper(group_idx, a, size, fill_value, func='sum', dtype=None, ddof=0):
    kwargs = dict()
    if func in ('var', 'std'):
        kwargs['ddof'] = ddof
    if isinstance(func, basestring):
        grouped = getattr(pd.DataFrame({'group_idx': group_idx, 'a': a})
                          .groupby('group_idx'), func)(**kwargs)
    else:
        grouped = pd.DataFrame({'group_idx': group_idx, 'a': a})\
                    .groupby('group_idx').aggregate(func, **kwargs)

    dtype = check_dtype(dtype, getattr(func, '__name__', func), a, size)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[grouped.index] = grouped
    return ret

_supported_funcs = 'min max sum prod mean var std first last all any'.split()
_impl_dict = dict(**{fn: partial(_wrapper, func=fn)
                     for fn in _supported_funcs})
_impl_dict.update(('nan' + fn, partial(_wrapper, func=fn))
                  for fn in _supported_funcs
                  if fn not in _no_separate_nan_version)
_impl_dict.update(allnan=partial(_wrapper, func=allnan),
                  anynan=partial(_wrapper, func=anynan))


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, **kwargs):
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func,
                           _impl_dict=_impl_dict, _nansqueeze=False, **kwargs)


aggregate.__doc__ = """
    This function makes use of `pandas`'s groupby machienery, it is slightly
    faster than the numpy implementation for `max`, `min`, and `prod`, but
    slower for other functions.
    """ + _doc_str
