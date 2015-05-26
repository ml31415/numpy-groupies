import timeit
import numpy as np
import pytest

from accumarray import accum_py, accum_np, accum, accum_ufunc, unpack, step_indices, step_count

class AttrDict(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture(params=[accum_py, accum_np, accum, accum_ufunc], ids=lambda x: x.func_name)
def accmap_all(request):
    return request.param


def test_preserve_missing(accmap_all):
    res = accmap_all(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int))
    np.testing.assert_array_equal(res, np.array([101, 206, 0, 208]))
    assert 'int' in res.dtype.name


def test_start_with_offset(accmap_all):
    accmap = np.array([1, 1, 2, 2, 2, 2, 4, 4])
    res = accmap_all(accmap, np.ones(accmap.size), dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 2, 4, 0, 2]))
    assert 'int' in res.dtype.name


def test_start_with_offset_prod(accmap_all):
    accmap = np.array([2, 2, 4, 4, 4, 7, 7, 7])
    res = accmap_all(accmap, accmap, func=np.prod, dtype=int)
    np.testing.assert_array_equal(res, np.array([0, 0, 4, 0, 64, 0, 0, 343]))


def test_no_negative_indices(accmap_all):
    pytest.raises(ValueError, accmap_all, np.arange(-10, 10), np.arange(20))


def test_parameter_missing(accmap_all):
    pytest.raises(TypeError, accmap_all, np.arange(5))


def test_shape_mismatch(accmap_all):
    pytest.raises(ValueError, accmap_all, np.array((1, 2, 3)), np.array((1, 2)))


def test_create_lists(accmap_all):
    res = accmap_all(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int), func=list)
    np.testing.assert_array_equal(np.array(res[0]), np.array([101]))
    assert res[2] == 0
    np.testing.assert_array_equal(np.array(res[3]), np.array([103, 105]))


def test_stable_sort(accmap_all):
    accmap = np.repeat(np.arange(5), 4)
    a = np.arange(accmap.size)
    res = accmap_all(accmap, a, func=list)
    np.testing.assert_array_equal(np.array(res[0]), np.array([0, 1, 2, 3]))
    a = np.arange(accmap.size)[::-1]
    res = accmap_all(accmap, a, func=list)
    np.testing.assert_array_equal(np.array(res[0]), np.array([19, 18, 17, 16]))


def test_item_counting(accmap_all):
    accmap = np.array([0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 5, 4, 3, 8, 8])
    a = np.arange(accmap.size)
    res = accmap_all(accmap, a, func=lambda x: len(x) > 1)
    np.testing.assert_array_equal(res, np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))


def test_fillvalue(accmap_all):
    accmap = np.array([0, 2, 2], dtype=int)
    for aggfunc, fillval in [(np.array, None), (np.sum, -1)]:
        res = accmap_all(accmap, np.arange(len(accmap), dtype=int), func=aggfunc, fillvalue=fillval)
        assert res[1] == fillval


def test_contiguous_equality(accmap_all):
    """ In case, accmap contains all numbers in range
        0 < n < max(accmap), and the values are sorted,
        the result of contiguous and incontiguous have
        to be equal.
    """
    accmap = np.repeat(np.arange(10), 3)
    a = np.random.randn(accmap.size)
    res_cont = accmap_all(accmap, a, mode='contiguous')
    res_incont = accmap_all(accmap, a)
    np.testing.assert_array_almost_equal(res_cont, res_incont, decimal=11)


def test_fortran_arrays(accmap_all):
    """ Numpy handles C and Fortran style indices. Optimized accum has to
        convert the Fortran matrices to C style, before doing it's job.
    """
    t = 10
    for order_style in ('C', 'F'):
        mat = np.zeros((t, t), order=order_style, dtype=float)
        mat.flat[:] = np.arange(t * t)
        assert accmap_all(np.zeros(t, dtype=int), mat[0, :])[0] == sum(range(t))


@pytest.fixture(params=['np/py', 'c/np', 'c/np_contiguous', 'ufunc/np'], scope='module')
def accmap_compare(request):
    if request.param == 'np/py':
        func = accum_np
        func_ref = accum_py
        group_cnt = 100
    else:
        group_cnt = 3000
        if 'ufunc' in request.param:
            func = accum_ufunc
            func_ref = accum_np
        else:
            func = accum
            func_ref = accum_np

    if request.param.endswith('contiguous'):
        accmap = np.repeat(np.arange(group_cnt), 20)
        mode = 'contiguous'
    else:
        # Gives 100000 duplicates of size 10 each
        accmap = np.repeat(np.arange(group_cnt), 2)
        np.random.shuffle(accmap)
        accmap = np.repeat(accmap, 10)
        mode = 'incontiguous'

    a = np.random.randn(accmap.size)
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


def allnan(x):
    return np.all(np.isnan(x))

def anynan(x):
    return np.any(np.isnan(x))

func_list = (np.sum, np.min, np.max, np.prod, np.all, np.any, np.mean, np.std,
             np.nansum, np.nanmin, np.nanmax, np.nanmean, np.nanstd,
             anynan, allnan, func_arbitrary, func_preserve_order)


@pytest.mark.parametrize("func", func_list, ids=lambda x: getattr(x, '__name__', x))
def test_compare(accmap_compare, func, decimal=14):
    mode = accmap_compare.mode
    a = accmap_compare.nana if 'nan' in getattr(func, '__name__', func) else accmap_compare.a
    ref = accmap_compare.func_ref(accmap_compare.accmap, a, func=func, mode=mode)
    try:
        res = accmap_compare.func(accmap_compare.accmap, a, func=func, mode=mode)
    except NotImplementedError:
        pytest.xfail("Function not yet implemented")
    else:
        np.testing.assert_array_almost_equal(res, ref, decimal=decimal)


def test_timing_sum(accmap_compare):
    t0 = timeit.Timer(lambda: accmap_compare.func_ref(accmap_compare.accmap, accmap_compare.a, mode=accmap_compare.mode)).timeit(number=3)
    t1 = timeit.Timer(lambda: accmap_compare.func(accmap_compare.accmap, accmap_compare.a, mode=accmap_compare.mode)).timeit(number=3)
    assert t0 > t1
    print "%s/%s speedup: %.3f" % (accmap_compare.func.func_name, accmap_compare.func_ref.func_name, t0 / t1)


def test_timing_std(accmap_compare):
    t0 = timeit.Timer(lambda: accmap_compare.func_ref(accmap_compare.accmap, accmap_compare.a, func=np.std, mode=accmap_compare.mode)).timeit(number=3)
    t1 = timeit.Timer(lambda: accmap_compare.func(accmap_compare.accmap, accmap_compare.a, func=np.std, mode=accmap_compare.mode)).timeit(number=3)
    assert t0 > t1
    print "%s/%s speedup: %.3f" % (accmap_compare.func.func_name, accmap_compare.func_ref.func_name, t0 / t1)


def test_step_indices_length():
    accmap = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
    for _ in xrange(20):
        np.random.shuffle(accmap)
        step_cnt_ref = np.count_nonzero(np.diff(accmap))
        assert step_count(accmap) == step_cnt_ref + 1
        assert len(step_indices(accmap)) == step_cnt_ref + 2


def test_step_indices_fields():
    accmap = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
    steps = step_indices(accmap)
    np.testing.assert_array_equal(steps, np.array([ 0, 3, 5, 7, 8, 9, 11]))


def test_unpack_compare():
    accmap = np.arange(10)
    np.random.shuffle(accmap)
    accmap = np.repeat(accmap, 3)
    a = np.random.randn(accmap.size)
    np.testing.assert_array_equal(unpack(accmap, a), a[accmap])


def test_unpack_contiguous():
    accmap = np.arange(10)
    np.random.shuffle(accmap)
    accmap = np.repeat(accmap, 3)
    a = np.random.randn(accmap.size)

    vals = unpack(accmap, accum(accmap, a))
    vals_cont = unpack(accmap, accum(accmap, a, mode='contiguous'), mode='contiguous')
    np.testing.assert_array_almost_equal(vals, vals_cont, decimal=10)


def test_unpack_simple():
    accmap = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    vals = accum(accmap, np.arange(accmap.size))
    unpacked = unpack(accmap, vals)
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21, 30, 30, 30]))


def test_unpack_incontiguous_a():
    accmap = np.array([5, 5, 3, 3, 1, 1, 4, 4])
    vals = accum(accmap, np.arange(accmap.size))
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])


def test_unpack_incontiguous_b():
    accmap = np.array([5, 5, 12, 5, 9, 12, 9])
    x = np.array([1, 2, 3, 24, 15, 6, 17])
    vals = accum(accmap, x)
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])


def test_unpack_long():
    accmap = np.repeat(np.arange(10000), 20)
    a = np.arange(accmap.size, dtype=int)
    vals = accum(accmap, a)
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])


def test_unpack_timing():
    # Unpacking should not be considerably slower than usual indexing
    accmap = np.repeat(np.arange(10000), 20)
    a = np.arange(accmap.size, dtype=int)
    vals = accum(accmap, a)

    t0 = timeit.Timer(lambda: vals[accmap]).timeit(number=100)
    t1 = timeit.Timer(lambda: unpack(accmap, vals)).timeit(number=100)
    np.testing.assert_array_equal(unpack(accmap, vals), vals[accmap])
    # This was a speedup once, but using openblas speeds up numpy greatly
    # So let's just make sure it's not a too big drawback
    assert t0 / t1 > 0.5


def test_unpack_downscaled():
    accmap = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
    vals = accum(accmap, np.arange(accmap.size), mode='downscaled')
    unpacked = unpack(accmap, vals, mode='downscaled')
    np.testing.assert_array_equal(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21]))


def benchmark(group_cnt=10000):
    accmap = np.repeat(np.arange(group_cnt), 2)
    np.random.shuffle(accmap)
    accmap = np.repeat(accmap, 10)
    a = np.random.randn(accmap.size)

    for func in func_list:
        print func.__name__ + ' ' + '-' * 50
        for accumfunc in (accum_np, accum_ufunc, accum):
            try:
                res = accumfunc(accmap, a, func=func)
            except NotImplementedError:
                continue
            t0 = timeit.Timer(lambda: accumfunc(accmap, a, func=func)).timeit(number=10)
            print "%s %s ... %s" % (accumfunc.__name__.ljust(13), ("%.3f" % (t0 * 1000)).rjust(8), res[:4])


if __name__ == '__main__':
    benchmark()
