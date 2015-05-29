# -*- coding: utf-8 -*-

import timeit
import numpy as np


from accumarray_purepy import accumarray as accumarray_pure
from accumarray_numpy import accumarray as accumarray_np
from accumarray_numpy_ufunc import accumarray as accumarray_npu
from accumarray_pandas import accumarray as accumarray_pd

def accumarray_group_loop(*args, **kwargs):
    """wraps func in lambda which prevents accumarray_numpy from
    recognising and optimising it. Instead it groups and loops."""
    func = kwargs['func']
    del kwargs['func']
    return accumarray_np(*args, func=lambda x: func(x), **kwargs)
    
import accumarray_utils as utils

_func_alias, _ = utils.get_alias_info(with_numpy=True)

print "TODO: use more extensive tests in test_accumarray.py"
print ""
print "-----simple examples----------"
test_vals = np.array([12.0, 3.2, -15, 88, 12.9])
test_idx = np.array([1,    0,    1,  4,   1  ])
print "test_vals: ", test_vals
print "test_idx: ", test_idx
print "accumarray(test_idx, test_vals):"
print accumarray_np(test_idx, test_vals) # group vals by idx and sum
# array([3.2, 9.9, 0., 0., 88.])
print "accumarray(test_idx, test_vals, sz=8, func='min', fillvalue=np.nan):"
print accumarray_np(test_idx, test_vals, sz=8, func='min', fillvalue=np.nan)
# array([3.2, -15., nan, 88., nan, nan, nan, nan])
print "accumarray(test_idx, test_vals, sz=5, func=lambda x: ' + '.join(str(xx) for xx in x),fillvalue='')"
print accumarray_np(test_idx, test_vals, sz=5, func=lambda x: ' + '.join(str(xx) for xx in x),fillvalue='')


print ""
print "---------testing--------------"
print "compare against group-and-loop with numpy"
testable_funcs = {_func_alias[f]: f for f in (np.sum, np.prod, np.any, np.all, np.min, np.max, np.std, np.var, np.mean)}
test_idx = np.random.randint(0, 1e3, 1e5)
test_vals = np.random.rand(1e5)*100-50
test_vals[test_vals>25] = 0 # for use with bool functions
for name, f in testable_funcs.items():    
    numpy_loop_group = accumarray_group_loop(test_idx, test_vals, func=f) 
    
    for acc_func, acc_name in [(accumarray_np, 'np-optimised'),
                               (accumarray_npu, 'np-ufunc-at'),
                               (accumarray_pure, 'purepy'),
                               (accumarray_pd, 'pandas')]:
        try:
            test_out = acc_func(test_idx, test_vals, func=name)
            test_out = np.asarray(test_out)
            if not np.allclose(test_out, numpy_loop_group.astype(test_out.dtype)):
                print name, acc_name, "FAILED test, output: [" + acc_name + "; correct]..."
                print np.vstack((test_out, numpy_loop_group))
            else:
                print name, acc_name, "PASSED test"            
        except NotImplementedError:
            print name, acc_name, "NOT IMPLEMENTED"
        
print ""
print "----------benchmarking-------------"
print "Note that the actual observed speedup depends on a variety of properties of the input."
print "Here we are using 100,000 indices uniformly picked from [0, 1000)."
print "Specifically, about 25% of the values are 0 (for use with bool operations),"
print "the remainder are uniformly distribuited on [-50,25)."
        
print ''.join(['function'.rjust(8), 'pure-py'.rjust(14), 'np-grouploop'.rjust(14),
              'np-ufuncat'.rjust(14),'np-optimised'.rjust(14), 'pandas'.rjust(14),
               'ratio'.rjust(15)]) 

for name, f in testable_funcs.items():
    t_np_simple = timeit.Timer(lambda: accumarray_np(test_idx, test_vals, func=lambda x: f(x))).timeit(number=5)

    print name.rjust(8),     
    times = [None]*5
    for ii, acc_func in enumerate([accumarray_pure, accumarray_group_loop,
                                   accumarray_npu, accumarray_np, 
                                   accumarray_pd]):
        try:
            func = f if acc_func is accumarray_group_loop else name
            times[ii] = timeit.Timer(lambda: acc_func(test_idx, test_vals, func=func)).timeit(number=4)            
            print ("%.1fms" %( (times[ii]*1000))).rjust(13),
        except NotImplementedError:
            print "no-impl".rjust(13),
            
    denom = min(t for t in times if t is not None)
    ratios = [("-".center(4) if t is None else str(round(t/denom,1))).center(5) for t in times]
    print "   ", (":".join(ratios))