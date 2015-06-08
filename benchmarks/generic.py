#!/usr/bin/python -B
import sys
import platform
import timeit
from operator import itemgetter
import numpy as np

from aggregate import (aggregate_py, aggregate_ufunc, aggregate_np as aggregate_numpy,
                       aggregate_weave, aggregate_pd as aggregate_pandas)

from aggregate.utils_numpy import allnan, anynan, nanfirst, nanlast


def aggregate_grouploop(*args, **kwargs):
    """wraps func in lambda which prevents aggregate_numpy from
    recognising and optimising it. Instead it groups and loops."""
    extrafuncs = {'allnan': allnan, 'anynan': anynan,
                  'first': itemgetter(0), 'last': itemgetter(-1),
                  'nanfirst': nanfirst, 'nanlast': nanlast}
    func = kwargs.pop('func')
    func = extrafuncs.get(func, func)
    if isinstance(func, basestring):
        raise NotImplementedError("Grouploop needs to be called with a function")
    return aggregate_numpy(*args, func=lambda x: func(x), **kwargs)


_implementations = ['aggregate_' + impl for impl in "py grouploop numpy weave ufunc pandas".split()]
aggregate_implementations = dict((impl, globals()[impl]) for impl in _implementations)


func_list = (np.sum, np.min, np.max, np.prod, np.all, np.any, np.mean, np.var, np.std, 'first', 'last',
             np.nansum, np.nanmin, np.nanmax, np.nanmean, np.nanvar, np.nanstd, 'nanfirst', 'nanlast',
             'anynan', 'allnan')



def benchmark(implementations, size=5e5, repeat=3):
    group_idx = np.random.randint(0, 1e3, size)
    a = np.random.random(group_idx.size)
    a[a > 0.8] = 0
    nana = a.copy()
    nana[(nana < 0.2) & (nana != 0)] = np.nan
    nan_share = np.mean(np.isnan(nana))
    assert 0.15 < nan_share < 0.25, "%3f%% nans" % (nan_share * 100)

    print "function" + ''.join(impl.split('_')[1].rjust(15) for impl in implementations)
    print "-" * (8 + 15 * len(implementations))
    for func in func_list:
        func_name = getattr(func, '__name__', func)
        print func_name.ljust(8),
        results = []
        used_a = nana if 'nan' in func_name else a

        for impl in implementations:
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
            sys.stdout.flush()
        print
        for res in results[1:]:
            np.testing.assert_array_almost_equal(res, results[0])

    print "%s(%s), Python %s, Numpy %s" % (platform.system(), platform.machine(), sys.version.split()[0], np.version.version)

if __name__ == '__main__':
    implementations = _implementations if '--purepy' in sys.argv else _implementations[1:]
    benchmark(implementations)