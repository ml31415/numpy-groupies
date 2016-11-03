from functools import partial
import numpy as np
import pandas as pd

from .utils import (check_dtype, _no_separate_nan_version,
                    _doc_str, isstr)
from .misc_tools_numpy import allnan, anynan
from .aggregate_numpy import _aggregate_base


def _wrapper(group_idx, a, size, fill_value, func='sum', dtype=None, ddof=0):
    kwargs = dict()
    if func in ('var', 'std'):
        kwargs['ddof'] = ddof
    if isstr(func):
        grouped = getattr(pd.DataFrame({'group_idx': group_idx, 'a': a})
                          .groupby('group_idx'), func)(**kwargs)
    else:
        grouped = pd.DataFrame({'group_idx': group_idx, 'a': a})\
                    .groupby('group_idx').aggregate(func, **kwargs)

    dtype = check_dtype(dtype, getattr(func, '__name__', func), a, size)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[grouped.index] = grouped
    return ret

_supported_funcs = 'sum prod all any min max mean var std first last'.split()
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_impl_dict.update(('nan' + fn, partial(_wrapper, func=fn))
                  for fn in _supported_funcs
                  if fn not in _no_separate_nan_version)
_impl_dict.update(allnan=partial(_wrapper, func=allnan),
                  anynan=partial(_wrapper, func=anynan),
                  len=partial(_wrapper, func='count'),
                  nanlen=partial(_wrapper, func='count'))


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    nansqueeze = isstr(func) and func.startswith('nan')
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func, axis=axis,
                           _impl_dict=_impl_dict, _nansqueeze=nansqueeze, **kwargs)


aggregate.__doc__ = """
    This function makes use of `pandas`'s groupby machienery, it is slightly
    faster than the numpy implementation for `max`, `min`, and `prod`, but
    slower for other functions.
    """ + _doc_str
