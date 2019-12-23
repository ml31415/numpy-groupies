""" Tests, that are run against all implemented versions of aggregate. """

import itertools
import numpy as np
import pytest

from . import _implementations, _impl_name, _wrap_notimplemented_xfail


@pytest.fixture(params=_implementations, ids=_impl_name)
def aggregate_all(request):
    impl = request.param
    if impl is None:
        pytest.xfail("Implementation not available")
    name = _impl_name(impl)
    return _wrap_notimplemented_xfail(impl.aggregate, 'aggregate_' + name)


def test_preserve_missing(aggregate_all):
    res = aggregate_all(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int))
    np.testing.assert_array_equal(res, np.array([101, 206, 0, 208]))
    if not isinstance(res, list):
        assert 'int' in res.dtype.name


def test_start_with_offset(aggregate_all):
    group_idx = np.array([1, 1, 2, 2, 2, 2, 4, 4])
    res = aggregate_all(group_idx, np.ones(group_idx.size), dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 2, 4, 0, 2]))
    if not isinstance(res, list):
        assert 'int' in res.dtype.name


@pytest.mark.parametrize("floatfunc", [np.std, np.var, np.mean], ids=lambda x: x.__name__)
def test_float_enforcement(aggregate_all, floatfunc):
    group_idx = np.arange(10).repeat(3)
    a = np.arange(group_idx.size)
    res = aggregate_all(group_idx, a, floatfunc)
    if not isinstance(res, list):
        assert 'float' in res.dtype.name
    assert np.all(np.array(res) > 0)


def test_start_with_offset_prod(aggregate_all):
    group_idx = np.array([2, 2, 4, 4, 4, 7, 7, 7])
    res = aggregate_all(group_idx, group_idx, func=np.prod, dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 0, 4, 0, 64, 0, 0, 343]))


def test_no_negative_indices(aggregate_all):
    for pos in (0, 10, -1):
        group_idx = np.arange(5).repeat(5)
        group_idx[pos] = -1
        pytest.raises(ValueError, aggregate_all, group_idx, np.arange(len(group_idx)))


def test_parameter_missing(aggregate_all):
    pytest.raises(TypeError, aggregate_all, np.arange(5))


def test_shape_mismatch(aggregate_all):
    pytest.raises(ValueError, aggregate_all, np.array((1, 2, 3)), np.array((1, 2)))


def test_create_lists(aggregate_all):
    res = aggregate_all(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int), func=list)
    np.testing.assert_array_equal(np.array(res[0]), np.array([101]))
    assert res[2] == 0
    np.testing.assert_array_equal(np.array(res[3]), np.array([103, 105]))


def test_item_counting(aggregate_all):
    group_idx = np.array([0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 5, 4, 3, 8, 8])
    a = np.arange(group_idx.size)
    res = aggregate_all(group_idx, a, func=lambda x: len(x) > 1)
    np.testing.assert_array_equal(res, np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))


@pytest.mark.parametrize(["func", "fill_value"], [(np.array, None), (np.sum, -1)], ids=["array", "sum"])
def test_fill_value(aggregate_all, func, fill_value):
    group_idx = np.array([0, 2, 2], dtype=int)
    res = aggregate_all(group_idx, np.arange(len(group_idx), dtype=int), func=func, fill_value=fill_value)
    assert res[1] == fill_value


@pytest.mark.parametrize("order", ["C", "F"])
def test_array_ordering(aggregate_all, order, size=10):
    mat = np.zeros((size, size), order=order, dtype=float)
    mat.flat[:] = np.arange(size * size)
    assert aggregate_all(np.zeros(size, dtype=int), mat[0, :], order=order)[0] == sum(range(size))


@pytest.mark.parametrize(["ndim", "order"], itertools.product([1, 2, 3], ["C", "F"]))
def test_ndim_indexing(aggregate_all, ndim, order, outsize=10):
    nindices = int(outsize ** ndim)
    outshape = tuple([outsize] * ndim)
    group_idx = np.random.randint(0, outsize, size=(ndim, nindices))
    a = np.random.random(group_idx.shape[1])
    res = aggregate_all(group_idx, a, size=outshape, order=order)
    if ndim > 1 and order == 'F':
        # 1d arrays always return False here
        assert np.isfortran(res)
    else:
        assert not np.isfortran(res)
    assert res.shape == outshape


def test_len(aggregate_all, group_size=5):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(group_size)
    a = np.arange(group_idx.size)
    res = aggregate_all(group_idx, a, func='len')
    ref = aggregate_all(group_idx, 1, func='sum')
    if isinstance(res, np.ndarray):
        assert issubclass(res.dtype.type, np.integer)
    else:
        assert isinstance(res[0], int)
    np.testing.assert_array_equal(res, ref)
    group_idx = np.arange(0, 100, dtype=int).repeat(group_size)
    a = np.arange(group_idx.size)
    res = aggregate_all(group_idx, a, func=len)
    if isinstance(res, np.ndarray):
        assert np.all(res == group_size)
    else:
        assert all(x == group_size for x in res)


def test_nan_len(aggregate_all):
    group_idx = np.arange(0, 20, 2, dtype=int).repeat(5)
    a = np.random.random(group_idx.size)
    a[::4] = np.nan
    a[::5] = np.nan
    res = aggregate_all(group_idx, a, func='nanlen')
    ref = aggregate_all(group_idx[~np.isnan(a)], 1, func='sum')
    if isinstance(res, np.ndarray):
        assert issubclass(res.dtype.type, np.integer)
    else:
        assert isinstance(res[0], int)
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize("first_last", ["first", "last"])
def test_first_last(aggregate_all, first_last):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(5)
    a = np.arange(group_idx.size)
    res = aggregate_all(group_idx, a, func=first_last, fill_value=-1)
    ref = np.zeros(np.max(group_idx) + 1)
    ref.fill(-1)
    ref[::2] = np.arange(0 if first_last == 'first' else 4, group_idx.size, 5, dtype=int)
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize(["first_last", "nanoffset"], itertools.product(["nanfirst", "nanlast"], [0, 2, 4]))
def test_nan_first_last(aggregate_all, first_last, nanoffset):
    group_idx = np.arange(0, 100, 2, dtype=int).repeat(5)
    a = np.arange(group_idx.size, dtype=float)

    a[nanoffset::5] = np.nan
    res = aggregate_all(group_idx, a, func=first_last, fill_value=-1)
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
    group_idx = np.zeros(20, dtype=int)
    a = np.random.random(group_idx.size)
    res = aggregate_all(group_idx, a, func, ddof=ddof)
    ref_func = {'std': np.std, 'var': np.var}.get(func)
    ref = ref_func(a, ddof=ddof)
    assert abs(res[0] - ref) < 1e-10


@pytest.mark.parametrize("func", ["sum", "prod", "mean", "var", "std"])
def test_scalar_input(aggregate_all, func):
    group_idx = np.arange(0, 100, dtype=int).repeat(5)
    if func not in ("sum", "prod"):
        pytest.raises((ValueError, NotImplementedError), aggregate_all, group_idx, 1, func=func)
    else:
        res = aggregate_all(group_idx, 1, func=func)
        ref = aggregate_all(group_idx, np.ones_like(group_idx, dtype=int), func=func)
        np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize("func", ["sum", "prod", "mean", "var", "std" , "all", "any"])
def test_nan_input(aggregate_all, func, groups=100):
    if aggregate_all.__name__.endswith('pandas'):
        pytest.skip("pandas automatically skip nan values")
    group_idx = np.arange(0, groups, dtype=int).repeat(5)
    a = np.random.random(group_idx.size)
    a[::2] = np.nan

    if func in ('all', 'any'):
        ref = np.ones(groups, dtype=bool)
    else:
        ref = np.full(groups, np.nan, dtype=float)
    res = aggregate_all(group_idx, a, func=func)
    np.testing.assert_array_equal(res, ref)


def test_nan_input_len(aggregate_all, groups=100, group_size=5):
    if aggregate_all.__name__.endswith('pandas'):
        pytest.skip("pandas always skips nan values")
    group_idx = np.arange(0, groups, dtype=int).repeat(group_size)
    a = np.random.random(len(group_idx))
    a[::2] = np.nan
    ref = np.full(groups, group_size, dtype=int)
    res = aggregate_all(group_idx, a, func=len)
    np.testing.assert_array_equal(res, ref)


def test_argmin_argmax(aggregate_all):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.array([4, 4, 3, 1, 10, 9, 9, 11])

    res = aggregate_all(group_idx, a, func="argmax", fill_value=-1)
    np.testing.assert_array_equal(res, [0, -1, -1, 7])

    res = aggregate_all(group_idx, a, func="argmin", fill_value=-1)
    np.testing.assert_array_equal(res, [3, -1, -1, 5])


def test_mean(aggregate_all):
    group_idx = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    a = np.arange(len(group_idx))

    res = aggregate_all(group_idx, a, func="mean")
    np.testing.assert_array_equal(res, [1.5, 0, 0, 5.5])


def test_cumsum(aggregate_all):
    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a =         np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    ref =       np.array([3, 4, 5, 6,15, 9,15,22, 7, 0,15,17, 6,14,31,39])

    res = aggregate_all(group_idx, a, func="cumsum")
    np.testing.assert_array_equal(res, ref)


def test_cummax(aggregate_all):
    group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    a =         np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    ref =       np.array([3, 4, 4, 3, 9, 9, 9, 9, 7, 0, 8, 9, 4, 8, 9, 9])

    res = aggregate_all(group_idx, a, func="cummax")
    np.testing.assert_array_equal(res, ref)


@pytest.mark.parametrize("order", ["normal", "reverse"])
def test_list_ordering(aggregate_all, order):
    group_idx = np.repeat(np.arange(5), 4)
    a = np.arange(group_idx.size)
    if order == "reverse":
        a = a[::-1]
    ref = a[:4]

    try:
        res = aggregate_all(group_idx, a, func=list)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        np.testing.assert_array_equal(np.array(res[0]), ref)


@pytest.mark.parametrize("order", ["normal", "reverse"])
def test_sort(aggregate_all, order):
    group_idx =   np.array([3, 3, 3, 2, 2, 2, 1, 1, 1])
    a =           np.array([3, 2, 1, 3, 4, 5, 5,10, 1])
    ref_normal =  np.array([1, 2, 3, 3, 4, 5, 1, 5,10])
    ref_reverse = np.array([3, 2, 1, 5, 4, 3,10, 5, 1])
    reverse = order == "reverse"
    ref = ref_reverse if reverse else ref_normal

    res = aggregate_all(group_idx, a, func="sort", reverse=reverse)
    np.testing.assert_array_equal(res, ref)
