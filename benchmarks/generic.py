#!/usr/bin/python -B

import timeit
import numpy as np

from aggregate import (aggregate_py, aggregate_ufunc, aggregate_np as aggregate_numpy,
                       aggregate_weave, aggregate_pd as aggregate_pandas)

_implementations = ['aggregate_' + impl for impl in "py ufunc numpy weave pandas".split()]
aggregate_implementations = dict((impl, globals()[impl]) for impl in _implementations)


func_list = (np.sum, np.min, np.max, np.prod, np.all, np.any, np.mean, np.std,
             np.nansum, np.nanmin, np.nanmax, np.nanmean, np.nanstd,
             'anynan', 'allnan')


def benchmark(group_cnt=6000, repeat=3):
    group_idx = np.arange(group_cnt).repeat(4)
    np.random.shuffle(group_idx)
    group_idx = group_idx.repeat(15)
    a = np.random.random(group_idx.size)
    a[a > 0.8] = 0
    nana = a.copy()
    nana[(nana < 0.2) & (nana != 0)] = np.nan
    nan_share = np.mean(np.isnan(nana))
    assert 0.15 < nan_share < 0.25, "%3f%% nans" % (nan_share * 100)

    print "function" + ''.join(impl.split('_')[1].rjust(15) for impl in _implementations)
    print "-" * (8 + 15 * len(_implementations))
    for func in func_list:
        func_name = getattr(func, '__name__', func)
        print func_name.ljust(8),
        results = []
        used_a = nana if 'nan' in func_name else a

        for impl in _implementations:
            aggregatefunc = aggregate_implementations[impl]
            if aggregatefunc is None:
                print '----'.rjust(14),
                continue

            try:
                res = aggregatefunc(group_idx, used_a, func=func)
            except NotImplementedError:
                print '----'.rjust(14),
                continue
            else:
                results.append(res)
            t0 = min(timeit.Timer(lambda: aggregatefunc(group_idx, used_a, func=func)).repeat(repeat=repeat, number=1))
            print ("%.3f" % (t0 * 1000)).rjust(14),
        print
        for res in results[1:]:
            np.testing.assert_array_almost_equal(res, results[0])

if __name__ == '__main__':
    benchmark()
