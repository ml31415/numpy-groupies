from functools import partial

import numpy as np
import pandas as pd

from .aggregate_numpy import _aggregate_base
from .utils import aggregate_common_doc, funcs_no_separate_nan, isstr
from .utils_numpy import allnan, anynan, check_dtype


def _wrapper(group_idx, a, size, fill_value, func='sum', dtype=None, ddof=0, **kwargs):
    funcname = func.__name__ if callable(func) else func
    kwargs = dict()
    if funcname in ('var', 'std'):
        kwargs['ddof'] = ddof
    df = pd.DataFrame({'group_idx': group_idx, 'a': a})
    if func == "sort":
        grouped = df.groupby('group_idx', sort=True)
    else:
        grouped = df.groupby('group_idx', sort=False).aggregate(func, **kwargs)

    dtype = check_dtype(dtype, getattr(func, '__name__', funcname), a, size)
    if funcname.startswith('cum'):
        ret = grouped.values[:, 0]
    else:
        ret = np.full(size, fill_value, dtype=dtype)
        ret[grouped.index] = grouped.values[:, 0]
    return ret


_supported_funcs = 'sum prod all any min max mean var std first last cumsum cumprod cummax cummin'.split()
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_impl_dict.update(('nan' + fn, partial(_wrapper, func=fn))
                  for fn in _supported_funcs
                  if fn not in funcs_no_separate_nan)
_impl_dict.update(allnan=partial(_wrapper, func=allnan),
                  anynan=partial(_wrapper, func=anynan),
                  len=partial(_wrapper, func='count'),
                  nanlen=partial(_wrapper, func='count'),
                  argmax=partial(_wrapper, func='idxmax'),
                  argmin=partial(_wrapper, func='idxmin'),
                  generic=_wrapper)


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    nansqueeze = isstr(func) and func.startswith('nan')
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func, axis=axis,
                           _impl_dict=_impl_dict, _nansqueeze=nansqueeze, **kwargs)


aggregate.__doc__ = """
    This is the pandas implementation of aggregate. It makes use of
    `pandas`'s groupby machienery and is mainly used for reference
    and benchmarking.
    """ + aggregate_common_doc
