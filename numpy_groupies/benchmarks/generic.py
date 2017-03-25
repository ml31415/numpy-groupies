#!/usr/bin/python -B
from __future__ import print_function
import sys
import platform
import timeit
from operator import itemgetter
import numpy as np

from numpy_groupies.tests import _implementations, aggregate_numpy
from numpy_groupies.misc_tools_numpy import allnan, anynan, nanfirst, nanlast


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
    return aggregate_numpy.aggregate(*args, func=lambda x: func(x), **kwargs)


func_list = (np.sum, np.prod, np.min, np.max, len, np.all, np.any, 'anynan', 'allnan',
             np.mean, np.std, np.var, 'first', 'last',
             np.nansum, np.nanprod, np.nanmin, np.nanmax, 'nanlen', 'nanall', 'nanany',
             np.nanmean, np.nanvar, np.nanstd, 'nanfirst', 'nanlast',)



def benchmark(implementations, size=5e5, repeat=3):
    group_idx = np.random.randint(0, int(1e3), int(size))
    a = np.random.random(group_idx.size)
    a[a > 0.8] = 0
    nana = a.copy()
    nana[(nana < 0.2) & (nana != 0)] = np.nan
    nan_share = np.mean(np.isnan(nana))
    assert 0.15 < nan_share < 0.25, "%3f%% nans" % (nan_share * 100)

    print("function" + ''.join(impl.__name__.rsplit('_', 1)[1].rjust(14) for impl in implementations))
    print("-" * (8 + 14 * len(implementations)))
    for func in func_list:
        func_name = getattr(func, '__name__', func)
        print(func_name.ljust(8), end='')
        results = []
        used_a = nana if 'nan' in func_name else a

        for impl in implementations:
            if impl is None:
                print('----'.rjust(14), end='')
                continue
            aggregatefunc = impl.aggregate

            try:
                res = aggregatefunc(group_idx, used_a, func=func)
            except NotImplementedError:
                print('----'.rjust(14), end='')
                continue
            except Exception:
                print('ERROR'.rjust(14), end='')
            else:
                results.append(res)
                try:
                    np.testing.assert_array_almost_equal(res, results[0])
                except AssertionError:
                    print('FAIL'.rjust(14), end='')
                else:
                    t0 = min(timeit.Timer(lambda: aggregatefunc(group_idx, used_a, func=func)).repeat(repeat=repeat, number=1))
                    print(("%.3f" % (t0 * 1000)).rjust(14), end='')
            sys.stdout.flush()
        print()

    print("%s(%s), Python %s, Numpy %s" % (platform.system(), platform.machine(), sys.version.split()[0], np.version.version))

if __name__ == '__main__':
    implementations = _implementations if '--purepy' in sys.argv else _implementations[1:]
    benchmark(implementations)
