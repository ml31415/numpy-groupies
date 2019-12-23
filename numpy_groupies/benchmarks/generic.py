#!/usr/bin/python -B
from __future__ import print_function
import sys
import platform
import timeit
from operator import itemgetter
import numpy as np

from numpy_groupies.tests import _implementations, aggregate_numpy
from numpy_groupies.utils_numpy import allnan, anynan, nanfirst, nanlast


def aggregate_grouploop(*args, **kwargs):
    """wraps func in lambda which prevents aggregate_numpy from
    recognising and optimising it. Instead it groups and loops."""
    extrafuncs = {'allnan': allnan, 'anynan': anynan,
                  'first': itemgetter(0), 'last': itemgetter(-1),
                  'nanfirst': nanfirst, 'nanlast': nanlast}
    func = kwargs.pop('func')
    func = extrafuncs.get(func, func)
    if isinstance(func, str):
        raise NotImplementedError("Grouploop needs to be called with a function")
    return aggregate_numpy.aggregate(*args, func=lambda x: func(x), **kwargs)


def arbitrary(iterator):
    tmp = 0
    for i, x in enumerate(iterator, 1):
        tmp += x ** i
    return tmp


func_list = (np.sum, np.prod, np.min, np.max, len, np.all, np.any, 'anynan', 'allnan',
             np.mean, np.std, np.var, 'first', 'last', 'argmax', 'argmin',
             np.nansum, np.nanprod, np.nanmin, np.nanmax, 'nanlen', 'nanall', 'nanany',
             np.nanmean, np.nanvar, np.nanstd, 'nanfirst', 'nanlast',
             'cumsum', 'cumprod', 'cummax', 'cummin', arbitrary, 'sort')

def benchmark_data(size=5e5, seed=100):
    rnd = np.random.RandomState(seed=seed)
    group_idx = rnd.randint(0, int(1e3), int(size))
    a = rnd.random_sample(group_idx.size)
    a[a > 0.8] = 0
    nana = a.copy()
    nana[(nana < 0.2) & (nana != 0)] = np.nan
    nan_share = np.mean(np.isnan(nana))
    assert 0.15 < nan_share < 0.25, "%3f%% nans" % (nan_share * 100)
    return a, nana, group_idx


def benchmark(implementations, repeat=5, size=5e5, seed=100):
    a, nana, group_idx = benchmark_data(size=size, seed=seed)

    print("function" + ''.join(impl.__name__.rsplit('_', 1)[1].rjust(14) for impl in implementations))
    print("-" * (9 + 14 * len(implementations)))
    for func in func_list:
        func_name = getattr(func, '__name__', func)
        print(func_name.ljust(9), end='')
        results = []
        used_a = nana if 'nan' in func_name else a

        for impl in implementations:
            if impl is None:
                print('----'.rjust(14), end='')
                continue
            aggregatefunc = impl.aggregate

            try:
                res = aggregatefunc(group_idx, used_a, func=func, cache=True)
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
                    t0 = min(timeit.Timer(lambda: aggregatefunc(group_idx, used_a, func=func, cache=True)).repeat(repeat=repeat, number=1))
                    print(("%.3f" % (t0 * 1000)).rjust(14), end='')
            sys.stdout.flush()
        print()

    implementation_names = [impl.__name__.rsplit('_', 1)[1] for impl in implementations]
    postfix = ''
    if 'numba' in implementation_names:
        import numba
        postfix += ', Numba %s' % numba.__version__
    if 'weave' in implementation_names:
        import weave
        postfix += ', Weave %s' % weave.__version__
    if 'pandas' in implementation_names:
        import pandas
        postfix += ', Pandas %s' % pandas.__version__
    print("%s(%s), Python %s, Numpy %s%s" % (platform.system(), platform.machine(), sys.version.split()[0], np.version.version, postfix))

if __name__ == '__main__':
    implementations = _implementations if '--purepy' in sys.argv else _implementations[1:]
    implementations = implementations if '--pandas' in sys.argv else implementations[:-1]
    benchmark(implementations)
