'''
Created on 20.02.2013

TestCases taken from octave

@author: nimrod
'''
import logging
import unittest
import timeit
import numpy as np
from accumarray import accum_py, accum_np, accum, unpack, step_indices, _count_steps


class TestSequence(unittest.TestCase):
    def assertArrayEqual(self, x, y):
        assert np.all((x == y) | ((x != x) & (y != y)))

    def assertArrayAlmostEqual(self, x, y, delta=0.000000001):
        assert np.all((np.abs(x - y) < delta) | ((x != x) & (y != y)))


class TestAccumPy(TestSequence):
    delta = 0.000000000001

    @property
    def func(self):
        return accum_py

    def test_preserve_missing(self):
        res = self.func(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int))
        self.assertArrayEqual(res, np.array([101, 206, 0, 208]))
        self.assertIn('int', res.dtype.name)

    def test_start_with_offset(self):
        accmap = np.array([1, 1, 2, 2, 2, 2, 4, 4])
        res = self.func(accmap, np.ones(accmap.size), dtype=int)
        self.assertArrayEqual(res, np.array([0, 2, 4, 0, 2]))
        self.assertIn('int', res.dtype.name)

    def test_start_with_offset_prod(self):
        accmap = np.array([2, 2, 4, 4, 4, 7, 7, 7])
        res = self.func(accmap, accmap, func=np.prod, dtype=int)
        self.assertArrayEqual(res, np.array([0, 0, 4, 0, 64, 0, 0, 343]))

    def test_no_negative_indices(self):
        self.assertRaises(ValueError, self.func, np.arange(-10, 10), np.arange(20))

    def test_parameter_missing(self):
        self.assertRaises(TypeError, self.func, np.arange(5))

    def test_shape_mismatch(self):
        self.assertRaises(ValueError, self.func, np.array((1, 2, 3)), np.array((1, 2)))

    def test_create_lists(self):
        res = self.func(np.array([0, 1, 3, 1, 3]), np.arange(101, 106, dtype=int), func=list)
        self.assertArrayEqual(np.array(res[0]), np.array([101]))
        self.assertEqual(res[2], 0)
        self.assertArrayEqual(np.array(res[3]), np.array([103, 105]))

    def test_stable_sort(self):
        accmap = np.repeat(np.arange(5), 4)
        a = np.arange(accmap.size)
        res = self.func(accmap, a, func=list)
        self.assertArrayEqual(np.array(res[0]), np.array([0, 1, 2, 3]))
        a = np.arange(accmap.size)[::-1]
        res = self.func(accmap, a, func=list)
        self.assertArrayEqual(np.array(res[0]), np.array([19, 18, 17, 16]))

    def test_item_counting(self):
        accmap = np.array([0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 5, 4, 3, 8, 8])
        a = np.arange(accmap.size)
        res = self.func(accmap, a, func=lambda x: len(x) > 1)
        self.assertArrayEqual(res, np.array([0, 0, 0, 1, 1, 1, 0, 0, 1]))

    def test_fillvalue(self):
        accmap = np.array([0, 2, 2], dtype=int)
        for aggfunc, fillval in [(np.array, None), (np.sum, -1)]:
            res = self.func(accmap, np.arange(len(accmap), dtype=int), func=aggfunc, fillvalue=fillval)
            self.assertEqual(res[1], fillval)

    def test_contiguous_equality(self):
        """ In case, accmap contains all numbers in range
            0 < n < max(accmap), and the values are sorted,
            the result of contiguous and incontiguous have
            to be equal.
        """
        accmap = np.repeat(np.arange(10), 3)
        a = np.random.randn(accmap.size)
        res_cont = self.func(accmap, a, mode='contiguous')
        res_incont = self.func(accmap, a)
        self.assertArrayAlmostEqual(res_cont, res_incont, delta=self.delta)

    def test_fortran_arrays(self):
        """ Numpy handles C and Fortran style indices. Optimized accum has to
            convert the Fortran matrices to C style, before doing it's job.
        """
        t = 10
        for order_style in ('C', 'F'):
            mat = np.zeros((t, t), order=order_style, dtype=float)
            mat.flat[:] = np.arange(t * t)
            self.assertEqual(self.func(np.zeros(t, dtype=int), mat[0, :])[0], sum(range(t)))


class TestAccumNumpy(TestAccumPy):
    @property
    def func(self):
        return accum_np

    def test_unpack(self):
        accmap = np.arange(10)
        np.random.shuffle(accmap)
        accmap = np.repeat(accmap, 3)
        a = np.random.randn(accmap.size)
        self.assertArrayEqual(unpack(accmap, a), a[accmap])


class TestAccumOptimized(TestAccumNumpy):
    @property
    def func(self):
        return accum

    def test_unpack_contiguous(self):
        accmap = np.arange(10)
        np.random.shuffle(accmap)
        accmap = np.repeat(accmap, 3)
        a = np.random.randn(accmap.size)

        vals = unpack(accmap, self.func(accmap, a))
        vals_cont = unpack(accmap, self.func(accmap, a, mode='contiguous'), mode='contiguous')
        self.assertArrayAlmostEqual(vals, vals_cont, delta=self.delta)


class TestComparing(TestSequence):
    """ Verify the results of one implementation of accum with another one """

    group_cnt = 1000
    delta = 0.00000000000001
    mode = 'incontiguous'

    @property
    def func(self):
        return accum_np

    @property
    def func_ref(self):
        return accum_py

    @classmethod
    def setUpClass(cls):
        if cls.mode == 'contiguous':
            cls.accmap = np.repeat(np.arange(cls.group_cnt), 20)
        else:
            # Gives 100000 duplicates of size 10 each
            cls.accmap = np.repeat(np.arange(cls.group_cnt), 2)
            np.random.shuffle(cls.accmap)
            cls.accmap = np.repeat(cls.accmap, 10)
        cls.a = np.random.randn(cls.accmap.size)
        cls.nana = cls.a.copy()
        cls.nana[::3] = np.nan
        cls.somea = cls.a.copy()
        cls.somea[cls.somea < 0.3] = 0
        cls.somea[::31] = np.nan

    def test_sum(self):
        self.assertArrayAlmostEqual(self.func(self.accmap, self.a, np.sum, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.sum, mode=self.mode),
                              delta=self.delta)

    def test_min(self):
        self.assertArrayEqual(self.func(self.accmap, self.a, np.min, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.min, mode=self.mode))

    def test_max(self):
        self.assertArrayEqual(self.func(self.accmap, self.a, np.max, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.max, mode=self.mode))

    def test_prod(self):
        # Multiplications drastically increases arithmetic errors, so we
        # need to be more generous here
        self.assertArrayAlmostEqual(self.func(self.accmap, self.a, np.prod, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.prod, mode=self.mode),
                              delta=self.delta * 100)

    def test_all(self):
        self.assertArrayEqual(self.func(self.accmap, self.somea, np.all, mode=self.mode),
                              self.func_ref(self.accmap, self.somea, np.all, mode=self.mode))

    def test_any(self):
        self.assertArrayEqual(self.func(self.accmap, self.somea, np.any, mode=self.mode),
                              self.func_ref(self.accmap, self.somea, np.any, mode=self.mode))

    def test_mean(self):
        self.assertArrayAlmostEqual(self.func(self.accmap, self.a, np.mean, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.mean, mode=self.mode),
                              delta=self.delta)

    def test_std(self):
        self.assertArrayAlmostEqual(self.func(self.accmap, self.a, np.std, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.std, mode=self.mode),
                              delta=self.delta)

    def test_nansum(self):
        self.assertArrayAlmostEqual(self.func(self.accmap, self.a, np.sum, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.sum, mode=self.mode),
                              delta=self.delta)

    def test_nanmin(self):
        self.assertArrayEqual(self.func(self.accmap, self.a, np.nanmin, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.nanmin, mode=self.mode))

    def test_nanmax(self):
        self.assertArrayEqual(self.func(self.accmap, self.a, np.nanmax, mode=self.mode),
                              self.func_ref(self.accmap, self.a, np.nanmax, mode=self.mode))

    def test_arbitrary_func(self):
        def testfunc(iterator):
            tmp = 0
            for x in iterator:
                tmp += x * x
            return tmp

        res = self.func(self.accmap, self.a, testfunc, mode=self.mode)
        ref = self.func_ref(self.accmap, self.a, testfunc, mode=self.mode)
        self.assertArrayAlmostEqual(res, ref, delta=self.delta * 10)

    def test_preserve_order(self):
        def testfunc(iterator):
            tmp = 0
            for i, x in enumerate(iterator, 1):
                tmp += x ** i
            return tmp

        res = self.func(self.accmap, self.a, testfunc, mode=self.mode)
        ref = self.func_ref(self.accmap, self.a, testfunc, mode=self.mode)
        self.assertArrayAlmostEqual(res, ref, delta=self.delta * 10)

    def test_timing_sum(self):
        t0 = timeit.Timer(lambda: self.func_ref(self.accmap, self.a, mode=self.mode)).timeit(number=5)
        t1 = timeit.Timer(lambda: self.func(self.accmap, self.a, mode=self.mode)).timeit(number=5)
        logging.info("%s/%s speedup: %.3f", self.func_ref.func_name, self.func.func_name, t0 / t1)

    def test_timing_std(self):
        t0 = timeit.Timer(lambda: self.func_ref(self.accmap, self.a, func=np.std, mode=self.mode)).timeit(number=5)
        t1 = timeit.Timer(lambda: self.func(self.accmap, self.a, func=np.std, mode=self.mode)).timeit(number=5)
        logging.info("%s/%s speedup: %.3f", self.func_ref.func_name, self.func.func_name, t0 / t1)


class TestOptimization(TestComparing):
    group_cnt = 1000
    delta = 0.00000000000001
    mode = 'incontiguous'

    @property
    def func(self):
        return accum

    @property
    def func_ref(self):
        return accum_np

    def test_nanmean(self):
        self.assertArrayAlmostEqual(self.func(self.accmap, self.nana, 'nanmean', mode=self.mode),
                              self.func_ref(self.accmap, self.nana, lambda x: np.mean(x[~np.isnan(x)]), mode=self.mode),
                              delta=self.delta)

    def test_nanstd(self):
        self.assertArrayAlmostEqual(self.func(self.accmap, self.nana, 'nanstd', mode=self.mode),
                              self.func_ref(self.accmap, self.nana, lambda x: np.std(x[~np.isnan(x)]), mode=self.mode),
                              delta=self.delta)

    def test_nanprod(self):
        # Multiplications drastically increases arithmetic errors, so we
        # need to be more generous here
        self.assertArrayAlmostEqual(self.func(self.accmap, self.nana, 'nanprod', mode=self.mode),
                              self.func_ref(self.accmap, self.nana, lambda x: np.prod(x[~np.isnan(x)]), mode=self.mode),
                              delta=self.delta * 100)

    def test_allnan(self):
        self.assertArrayEqual(self.func(self.accmap, self.nana, 'allnan', mode=self.mode),
                              self.func_ref(self.accmap, self.nana, lambda x: np.all(np.isnan(x)),
                                            mode=self.mode, dtype=bool))

    def test_anynan(self):
        self.assertArrayEqual(self.func(self.accmap, self.nana, 'anynan', mode=self.mode),
                              self.func_ref(self.accmap, self.nana, lambda x: np.any(np.isnan(x)),
                                            mode=self.mode, dtype=bool))

class TestOptimizationContiguous(TestOptimization):
    mode = 'contiguous'


class TestUnpack(TestSequence):
    def test_simple(self):
        accmap = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        vals = accum(accmap, np.arange(accmap.size))
        unpacked = unpack(accmap, vals)
        self.assertArrayEqual(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21, 30, 30, 30]))

    def test_incontiguous_a(self):
        accmap = np.array([5, 5, 3, 3, 1, 1, 4, 4])
        vals = accum(accmap, np.arange(accmap.size))
        self.assertArrayEqual(unpack(accmap, vals), vals[accmap])

    def test_incontiguous_b(self):
        accmap = np.array([5, 5, 12, 5, 9, 12, 9])
        x = np.array([1, 2, 3, 24, 15, 6, 17])
        vals = accum(accmap, x)
        self.assertArrayEqual(unpack(accmap, vals), vals[accmap])

    def test_long(self):
        accmap = np.repeat(np.arange(10000), 20)
        a = np.arange(accmap.size, dtype=int)
        vals = accum(accmap, a)
        self.assertArrayEqual(unpack(accmap, vals), vals[accmap])

    def test_timing(self):
        accmap = np.repeat(np.arange(10000), 20)
        a = np.arange(accmap.size, dtype=int)
        vals = accum(accmap, a)

        t0 = timeit.Timer(lambda: vals[accmap]).timeit(number=100)
        t1 = timeit.Timer(lambda: unpack(accmap, vals)).timeit(number=100)
        self.assertArrayEqual(unpack(accmap, vals), vals[accmap])
        # Require 2-fold speedup, it's actually 2.5 on my machine
        self.assertGreater(t0 / t1, 2)

    def test_downscaled(self):
        accmap = np.array([4, 4, 4, 1, 1, 1, 2, 2, 2])
        vals = accum(accmap, np.arange(accmap.size), mode='downscaled')
        unpacked = unpack(accmap, vals, mode='downscaled')
        self.assertArrayEqual(unpacked, np.array([3, 3, 3, 12, 12, 12, 21, 21, 21]))


class TestStepIndices(TestSequence):
    def test_length(self):
        accmap = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2], dtype=int)
        for _ in xrange(20):
            np.random.shuffle(accmap)
            step_cnt_ref = np.count_nonzero(np.diff(accmap))
            self.assertEqual(_count_steps(accmap), step_cnt_ref + 1)
            self.assertEqual(len(step_indices(accmap)), step_cnt_ref + 2)

    def test_fields(self):
        accmap = np.array([1, 1, 1, 2, 2, 3, 3, 4, 5, 2, 2], dtype=int)
        steps = step_indices(accmap)
        self.assertArrayEqual(steps, np.array([ 0, 3, 5, 7, 8, 9, 11]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2, failfast=False, exit=False)
