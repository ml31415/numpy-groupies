# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from functools import partial
from accumarray_numpy import (accumarray as accumarray_np)


def _wrapper(idx, vals, n, fillvalue, dtype=None, func='sum'):
    ret = np.full(n, fillvalue)    
    grouped = getattr(pd.DataFrame({'idx': idx, 'vals': vals}).groupby('idx'),func)()
    ret[grouped.index] = grouped
    return ret


_supported_funcs = 'min max sum prod mean first last all any'.split(' ')
_impl_dict = {fn: partial(_wrapper, func=fn) for fn in _supported_funcs}


def accumarray(*args, **kwargs):
    """
    Accumulation function similar to Matlab's `accumarray` function.
    
    See readme file at https://github.com/ml31415/accumarray for 
    full description.  Or see ``accumarray`` in ``accumarray_numpy.py``.

    This implementation is by DM, May 2015.

    This function makes use of `pandas`'s groupby machienery, it is slightly
    faster than the numpy implementation for `max`, `min`, and `prod`, but slower
    for other functions.
   
    Note that this implementation piggybacks on the main error checking and
    argument parsing etc. in ``accumarray_numpy.py``.
    """
    return accumarray_np(*args, impl_dict=_impl_dict, **kwargs)


