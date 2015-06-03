#!/usr/bin/python -B

import timeit
import numpy as np

from aggregate import (aggregate_py, aggregate_ufunc, aggregate_np as aggregate_numpy,
                aggregate_weave, aggregate_pd as aggregate_pandas)

_implementations = ['aggregate_' + impl for impl in "py ufunc numpy weave pandas".split()]
aggregate_implementations = dict((impl, globals()[impl]) for impl in _implementations)


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


def benchmark(group_cnt=2000):
    group_idx = np.repeat(np.arange(group_cnt), 2)
    np.random.shuffle(group_idx)
    group_idx = np.repeat(group_idx, 10)
    a = np.random.randn(group_idx.size)

    print "function" + ''.join(impl.split('_')[1].rjust(15) for impl in _implementations)
    print "-" * (8 + 15 * len(_implementations))
    for func in func_list[:-2]:
        print func.__name__.ljust(8),
        results = []
        for impl in _implementations:
            aggregatefunc = aggregate_implementations[impl]
            if aggregatefunc is None:
                print '----'.rjust(14),
                continue
            try:
                res = aggregatefunc(group_idx, a, func=func)
            except NotImplementedError:
                print '----'.rjust(14),
                continue
            else:
                results.append(res)
            t0 = timeit.Timer(lambda: aggregatefunc(group_idx, a, func=func)).timeit(number=10)
            print ("%.3f" % (t0 * 1000)).rjust(14),
        print
        for res in results[1:]:
            np.testing.assert_array_almost_equal(res, results[0])

if __name__ == '__main__':
    benchmark()
