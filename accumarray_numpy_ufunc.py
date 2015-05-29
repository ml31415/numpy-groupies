# -*- coding: utf-8 -*-

import numpy as np

from accumarray_numpy import (accumarray as accumarray_np, 
                              _get_minimum_dtype, _fill_untouched, _check_boolean)

def _dummy(idx, vals, n, fillvalue, dtype=None):
    raise NotImplementedError("No ufunc for this, is there?")

def _anynan(idx, vals, n, fillvalue, dtype=None):
    return _any(idx, np.isnan(vals), n, fillvalue=fillvalue, dtype=dtype)
    
def _allnan(idx, vals, n, fillvalue, dtype=None):
    return _all(idx, np.isnan(vals), n, fillvalue=fillvalue, dtype=dtype)

def _any(idx, vals, n, fillvalue, dtype=None):
    _check_boolean(fillvalue, name="fillvalue")
    ret = np.full(n, fillvalue, dtype=bool)
    if fillvalue:
        ret[idx] = False # any-test should start from False
    np.logical_or.at(ret, idx, vals)
    return ret

def _all(idx, vals, n, fillvalue, dtype=None):
    _check_boolean(fillvalue, name="fillvalue")
    ret = np.full(n, fillvalue, dtype=bool)
    if not fillvalue:
        ret[idx] = True # all-test should start from True
    np.logical_and.at(ret, idx, vals)
    return ret

def _sum(idx, vals, n, fillvalue, dtype=None):
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    ret = np.full(n, fillvalue, dtype=dtype)
    if fillvalue != 0:
        ret[idx] = 0 # sums should start at 0
    np.add.at(ret, idx, vals)
    return ret


def _prod(idx, vals, n, fillvalue, dtype=None):
    """Same as accumarray_numpy.py"""
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    ret = np.full(n, fillvalue, dtype=dtype)
    if fillvalue != 1:
        ret[idx] = 1 # product should start from 1
    np.multiply.at(ret, idx, vals)
    return ret

def _min(idx, vals, n, fillvalue, dtype=None):
    """Same as accumarray_numpy.py"""
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    dmax = np.iinfo(vals.dtype).max if issubclass(vals.dtype.type, np.integer) else np.finfo(vals.dtype).max
    ret = np.full(n, fillvalue, dtype=dtype)
    if fillvalue != dmax:
        ret[idx] = dmax # min starts from maximum 
    np.minimum.at(ret, idx, vals)
    return ret

def _max(idx, vals, n, fillvalue, dtype=None):
    """Same as accumarray_numpy.py"""
    dtype = _get_minimum_dtype(fillvalue, dtype or vals.dtype)
    dmin = np.iinfo(vals.dtype).min if issubclass(vals.dtype.type, np.integer) else np.finfo(vals.dtype).min
    ret = np.full(n, fillvalue, dtype=dtype)
    if fillvalue != dmin:
        ret[idx] = dmin # max starts from minimum
    np.maximum.at(ret, idx, vals)
    return ret
    


_impl_dict = dict(min=_min, max=_max, sum=_sum, prod=_prod, last=_dummy, first=_dummy,
                all=_all, any=_any, mean=_dummy, std=_dummy, var=_dummy,
                anynan=_dummy, allnan=_dummy, sort=_dummy, rsort=_dummy, 
                array=_dummy)


def accumarray(*args, **kwargs):
    """
    Accumulation function similar to Matlab's `accumarray` function.
    
    See readme file at https://github.com/ml31415/accumarray for 
    full description.  Or see ``accumarray`` in ``accumarray_numpy.py``.

    This implementation is by DM, May 2015.

    Unlike the ``accumarray_numpy.py``, which in most cases does some custom 
    oprtimisations, this version simply uses ``numpy``'s ``ufunc.at``. 
    
    As of version 1.9 this gives fairly poor performance.

    Note that this implementation piggybacks on the main error checking and
    argument parsing etc. in ``accumarray_numpy.py``.
    """
    return accumarray_np(*args, impl_dict=_impl_dict, **kwargs)

