#!/usr/bin/python -B
# -*- coding: utf-8 -*-

from __future__ import print_function
import timeit
import numpy as np


from numpy_groupies.utils import aliasing
from numpy_groupies import aggregate_py, aggregate_np, aggregate_ufunc
from numpy_groupies.aggregate_pandas import aggregate as aggregate_pd


def aggregate_group_loop(*args, **kwargs):
    """wraps func in lambda which prevents aggregate_numpy from
    recognising and optimising it. Instead it groups and loops."""
    func = kwargs['func']
    del kwargs['func']
    return aggregate_np(*args, func=lambda x: func(x), **kwargs)


print("TODO: use more extensive tests")
print("")
print("-----simple examples----------")
test_a = np.array([12.0, 3.2, -15, 88, 12.9])
test_group_idx = np.array([1, 0, 1, 4, 1  ])
print("test_a: ", test_a)
print("test_group_idx: ", test_group_idx)
print("aggregate(test_group_idx, test_a):")
print(aggregate_np(test_group_idx, test_a))  # group vals by idx and sum
# array([3.2, 9.9, 0., 0., 88.])
print("aggregate(test_group_idx, test_a, sz=8, func='min', fill_value=np.nan):")
print(aggregate_np(test_group_idx, test_a, size=8, func='min', fill_value=np.nan))
# array([3.2, -15., nan, 88., nan, nan, nan, nan])
print("aggregate(test_group_idx, test_a, sz=5, func=lambda x: ' + '.join(str(xx) for xx in x),fill_value='')")
print(aggregate_np(test_group_idx, test_a, size=5, func=lambda x: ' + '.join(str(xx) for xx in x), fill_value=''))


print("")
print("---------testing--------------")
print("compare against group-and-loop with numpy")
testable_funcs = {aliasing[f]: f for f in (np.sum, np.prod, np.any, np.all, np.min, np.max, np.std, np.var, np.mean)}
test_group_idx = np.random.randint(0, int(1e3), int(1e5))
test_a = np.random.rand(int(1e5)) * 100 - 50
test_a[test_a > 25] = 0  # for use with bool functions
for name, f in testable_funcs.items():
    numpy_loop_group = aggregate_group_loop(test_group_idx, test_a, func=f)

    for acc_func, acc_name in [(aggregate_np, 'np-optimised'),
                               (aggregate_ufunc, 'np-ufunc-at'),
                               (aggregate_py, 'purepy'),
                               (aggregate_pd, 'pandas')]:
        try:
            test_out = acc_func(test_group_idx, test_a, func=name)
            test_out = np.asarray(test_out)
            if not np.allclose(test_out, numpy_loop_group.astype(test_out.dtype)):
                print(name, acc_name, "FAILED test, output: [" + acc_name + "; correct]...")
                print(np.vstack((test_out, numpy_loop_group)))
            else:
                print(name, acc_name, "PASSED test")
        except NotImplementedError:
            print(name, acc_name, "NOT IMPLEMENTED")

print("")
print("----------benchmarking-------------")
print("Note that the actual observed speedup depends on a variety of properties of the input.")
print("Here we are using 100,000 indices uniformly picked from [0, 1000).")
print("Specifically, about 25% of the values are 0 (for use with bool operations),")
print("the remainder are uniformly distribuited on [-50,25).")
print("Times are scaled to 10 repetitions (actual number of reps used may not be 10).")

print(''.join(['function'.rjust(8), 'pure-py'.rjust(14), 'np-grouploop'.rjust(14),
              'np-ufuncat'.rjust(14), 'np-optimised'.rjust(14), 'pandas'.rjust(14),
               'ratio'.rjust(15)]))

for name, f in testable_funcs.items():
    print(name.rjust(8), end='')
    times = [None] * 5
    for ii, acc_func in enumerate([aggregate_py, aggregate_group_loop,
                                   aggregate_ufunc, aggregate_np,
                                   aggregate_pd]):
        try:
            func = f if acc_func is aggregate_group_loop else name
            reps = 3 if acc_func is aggregate_py else 20
            times[ii] = timeit.Timer(lambda: acc_func(test_group_idx, test_a, func=func)).timeit(number=reps) / reps * 10
            print(("%.1fms" % ((times[ii] * 1000))).rjust(13), end='')
        except NotImplementedError:
            print("no-impl".rjust(13), end='')

    denom = min(t for t in times if t is not None)
    ratios = [("-".center(4) if t is None else str(round(t / denom, 1))).center(5) for t in times]
    print("   ", (":".join(ratios)))
