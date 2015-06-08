"""
Implementation of tests, that are run against all implemented versions of aggregate.
"""

import itertools
import numpy as np
import pytest

from .. import (aggregate_py, aggregate_ufunc, aggregate_np as aggregate_numpy,
                aggregate_weave, aggregate_pd as aggregate_pandas)


_implementations = ['aggregate_' + impl for impl in "py ufunc numpy weave pandas".split()]
aggregate_implementations = dict((impl, globals()[impl]) for impl in _implementations)


@pytest.fixture(params=_implementations, ids=lambda x: x.split('_')[1])
def aggregate_all(request):
    impl = aggregate_implementations[request.param]
    if impl is None:
        pytest.xfail("Implementation not available")
    return impl


def test_preserve_missing(aggregate_all):
    res = aggregate_all(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int))
    np.testing.assert_array_equal(res, np.array([101, 206, 0, 208]))
    if aggregate_all != aggregate_py:
        assert 'int' in res.dtype.name


def test_start_with_offset(aggregate_all):
    group_idx = np.array([1, 1, 2, 2, 2, 2, 4, 4])
    res = aggregate_all(group_idx, np.ones(group_idx.size), dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 2, 4, 0, 2]))
    if aggregate_all != aggregate_py:
        assert 'int' in res.dtype.name


@pytest.mark.parametrize("floatfunc", [np.std, np.var, np.mean], ids=lambda x: x.__name__)
def test_float_enforcement(aggregate_all, floatfunc):
    group_idx = np.arange(10).repeat(3)
    a = np.arange(group_idx.size)
    try:
        res = aggregate_all(group_idx, a, floatfunc)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    if aggregate_all != aggregate_py:
        assert 'float' in res.dtype.name
    assert np.all(res > 0)


def test_start_with_offset_prod(aggregate_all):
    group_idx = np.array([2, 2, 4, 4, 4, 7, 7, 7])
    res = aggregate_all(group_idx, group_idx, func=np.prod, dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 0, 4, 0, 64, 0, 0, 343]))


def test_no_negative_indices(aggregate_all):
    pytest.raises(ValueError, aggregate_all, np.arange(-10, 10), np.arange(20))


def test_parameter_missing(aggregate_all):
    pytest.raises(TypeError, aggregate_all, np.arange(5))


def test_shape_mismatch(aggregate_all):
    pytest.raises(ValueError, aggregate_all, np.array((1, 2, 3)), np.array((1, 2)))


def test_create_lists(aggregate_all):
    try:
        res = aggregate_all(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int), func=list)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        np.testing.assert_array_equal(np.array(res[0]), np.array([101]))
        assert res[2] == 0
        np.testing.assert_array_equal(np.array(res[3]), np.array([103, 105]))


@pytest.mark.parametrize("sort_order", ["normal", "reverse"])
def test_stable_sort(aggregate_all, sort_order):
    group_idx = np.repeat(np.arange(5), 4)
    a = np.arange(group_idx.size)
    if sort_order == "reverse":
        a = a[::-1]
    ref = a[:4]

    try:
        res = aggregate_all(group_idx, a, func=list)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        np.testing.assert_array_equal(np.array(res[0]), ref)


def test_item_counting(aggregate_all):
    group_idx = np.array([0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 5, 4, 3, 8, 8])
    a = np.arange(group_idx.size)
    try:
        res = aggregate_all(group_idx, a, func=lambda x: len(x) > 1)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        np.testing.assert_array_equal(res, np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))


@pytest.mark.parametrize(["func", "fill_value"], [(np.array, None), (np.sum, -1)], ids=["array", "sum"])
def test_fill_value(aggregate_all, func, fill_value):
    group_idx = np.array([0, 2, 2], dtype=int)
    try:
        res = aggregate_all(group_idx, np.arange(len(group_idx), dtype=int), func=func, fill_value=fill_value)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        assert res[1] == fill_value


@pytest.mark.parametrize("order", ["C", "F"])
def test_array_ordering(aggregate_all, order, size=10):
    mat = np.zeros((size, size), order=order, dtype=float)
    mat.flat[:] = np.arange(size * size)
    assert aggregate_all(np.zeros(size, dtype=int), mat[0, :])[0] == sum(range(size))


@pytest.mark.parametrize("first_last", ["first", "last"])
def test_first_last(aggregate_all, first_last):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(5)
    a = np.arange(group_idx.size)
    try:
        res = aggregate_all(group_idx, a, func=first_last, fill_value=-1)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    ref = np.zeros(np.max(group_idx) + 1)
    ref.fill(-1)
    ref[::2] = np.arange(0 if first_last == 'first' else 4, group_idx.size, 5, dtype=int)
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize(["first_last", "nanoffset"], itertools.product(["nanfirst", "nanlast"], [2, 0, 4]))
def test_nan_first_last(aggregate_all, first_last, nanoffset):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(5)
    a = np.arange(group_idx.size, dtype=float)

    a[nanoffset::5] = np.nan
    try:
        res = aggregate_all(group_idx, a, func=first_last, fill_value=-1)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        ref = np.zeros(np.max(group_idx) + 1)
        ref.fill(-1)

        if first_last == "nanfirst":
            ref_offset = 1 if nanoffset == 0 else 0
        else:
            ref_offset = 3 if nanoffset == 4 else 4
        ref[::2] = np.arange(ref_offset, group_idx.size, 5, dtype=int)
        np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize(["func", "ddof"], itertools.product(["var", "std"], [0, 1, 2]))
def test_ddof(aggregate_all, func, ddof, size=20):
    func = {'std': np.std, 'var': np.var}.get(func)
    group_idx = np.zeros(20, dtype=int)
    a = np.random.random(group_idx.size)
    try:
        res = aggregate_all(group_idx, a, func, ddof=ddof)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        assert abs(res[0] - func(a, ddof=ddof)) < 1e-10
