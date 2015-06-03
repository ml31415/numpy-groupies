from functools import partial
import numpy as np
import pandas as pd

from .aggregate_numpy import aggregate as aggregate_np


def _wrapper(group_idx, a, n, fill_value, func='sum', dtype=None):
    ret = np.full(n, fill_value)
    grouped = getattr(pd.DataFrame({'group_idx': group_idx, 'a': a}).groupby('group_idx'), func)()
    ret[grouped.index] = grouped
    return ret


_supported_funcs = 'min max sum prod mean first last all any'.split()
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}


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


