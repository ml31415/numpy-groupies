"""
In this test, aggregate_numpy is taken as a reference implementation and this
results are compared against the results of the other implementations. Implementations
may throw NotImplementedError in order to show missing functionality without throwing
test errors. 
"""
import itertools
import numpy as np
import pytest

from .. import (aggregate_py, aggregate_ufunc, aggregate_np as aggregate_numpy,
                aggregate_weave, aggregate_pd as aggregate_pandas)
from .test_generic import wrap_aggregate_xfail

class AttrDict(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture(params=['np/py', 'c/np', 'ufunc/np', 'pandas/np'], scope='module')
def aggregate_cmp(request):
    if request.param == 'np/py':
        func = aggregate_numpy
        func_ref = wrap_aggregate_xfail(aggregate_py)
        group_cnt = 100
    else:
        group_cnt = 3000
        func_ref = aggregate_numpy
        if 'ufunc' in request.param:
            func = aggregate_ufunc
        elif 'pandas' in request.param:
            func = aggregate_pandas
        else:
            func = aggregate_weave

    if not func:
        pytest.xfail("Implementation not available")
    func = wrap_aggregate_xfail(func)

    # Gives 100000 duplicates of size 10 each
    group_idx = np.repeat(np.arange(group_cnt), 2)
    np.random.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 10)

    a = np.random.randn(group_idx.size)
    nana = a.copy()
    nana[::3] = np.nan
    somea = a.copy()
    somea[somea < 0.3] = 0
    somea[::31] = np.nan
    return AttrDict(locals())


def func_arbitrary(iterator):
    tmp = 0
    for x in iterator:
        tmp += x * x
    return tmp

def func_preserve_order(iterator):
    tmp = 0
    for i, x in enumerate(iterator, 1):
        tmp += x ** i
    return tmp

func_list = (np.sum, np.min, np.max, np.prod, np.all, np.any, np.mean, np.std,
             np.nansum, np.nanmin, np.nanmax, np.nanmean, np.nanstd,
             'anynan', 'allnan', func_arbitrary, func_preserve_order)

@pytest.mark.parametrize("func", func_list, ids=lambda x: getattr(x, '__name__', x))
def test_cmp(aggregate_cmp, func, decimal=14):
    a = aggregate_cmp.nana if 'nan' in getattr(func, '__name__', func) else aggregate_cmp.a
    res = aggregate_cmp.func(aggregate_cmp.group_idx, a, func=func)
    ref = aggregate_cmp.func_ref(aggregate_cmp.group_idx, a, func=func)
    np.testing.assert_array_almost_equal(res, ref, decimal=decimal)


@pytest.mark.parametrize(["ndim", "order"], itertools.product([2, 3], ["C", "F"]))
def test_cmp_ndim(aggregate_cmp, ndim, order, outsize=100, decimal=14):
    nindices = int(outsize ** ndim)
    outshape = tuple([outsize] * ndim)
    group_idx = np.random.randint(0, outsize, size=(ndim, nindices))
    a = np.random.random(group_idx.shape[1])

    res = aggregate_cmp.func(group_idx, a, size=outshape, order=order)
    ref = aggregate_cmp.func_ref(group_idx, a, size=outshape, order=order)
    if ndim > 1 and order == 'F':
        # 1d arrays always return False here
        assert np.isfortran(res)
    else:
        assert not np.isfortran(res)
    assert res.shape == outshape
    np.testing.assert_array_almost_equal(res, ref, decimal=decimal)
