from functools import partial

import numpy as np
import polars as pl

from .aggregate_numpy import _aggregate_base
from .utils import aggregate_common_doc, funcs_no_separate_nan, isstr
from .utils_numpy import check_dtype


def _wrapper(group_idx, a, size, fill_value, func='sum', dtype=None, ddof=0, **kwargs):
    if isinstance(a, (int, float)):
        a = np.full_like(group_idx, fill_value=a)
    df = pl.DataFrame({'group_idx': group_idx, 'a': a})
    funcname = func.__name__ if callable(func) else func
    if funcname in ("array", "list"):
        grouped = df.groupby('group_idx', maintain_order=True).agg_list()
    elif func == funcname:
        grouped = getattr(df.groupby('group_idx'), funcname)()
    else:
        grouped = df.groupby('group_idx').apply(func)
    dtype = check_dtype(dtype, getattr(func, '__name__', funcname), a, size)
    ret = np.full(size, fill_value, dtype=dtype)
    res_field = 'count' if funcname == 'count' else 'a'
    ret[grouped["group_idx"]] = grouped[res_field]
    return ret


_supported_funcs = 'sum min max mean first last'.split()
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}
_impl_dict.update(('nan' + fn, partial(_wrapper, func=fn))
                  for fn in _supported_funcs
                  if fn not in funcs_no_separate_nan)
_impl_dict.update(len=partial(_wrapper, func='count'),)


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func, axis=axis,
                           _impl_dict=_impl_dict, **kwargs)


aggregate.__doc__ = """
    This is the polars implementation of aggregate. It makes use of
    `polars`'s groupby machienery and is mainly used for reference
    and benchmarking.
    """ + aggregate_common_doc
