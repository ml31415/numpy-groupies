# -*- coding: utf-8 -*-

import timeit
import numpy as np

from accumarray_purepy import accumarray as accumarray_pure
from accumarray_numpy import accumarray as accumarray_np
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
    numpy_optimized = accumarray_np(test_idx, test_vals, func=f)
    numpy_simple = accumarray_np(test_idx, test_vals, func=lambda x: f(x)) # wrapping f in lambda forces group-and-loop
    purepy = accumarray_pure(test_idx, test_vals, func=name)
    if not np.allclose(numpy_optimized, numpy_simple.astype(numpy_optimized.dtype)):
        print name, "numpy FAILED test, output: [optimised; correct]..."
        print np.vstack((numpy_optimized, numpy_simple))
    else:
        print name, "numpy PASSED test"
    if not np.allclose(purepy, numpy_simple.astype(numpy_optimized.dtype)):
        print name, "purepy FAILED test, output: [purepy; correct]..."
        print np.vstack((purepy, numpy_simple))
    else:
        print name, "purepy PASSED test"

print ""
print "----------benchmarking-------------"
print "Note that the actual observed speedup depends on a variety of properties of the input."
print "Here we are using 100,000 indices uniformly picked from [0, 1000)."
print "Specifically, about 25% of the values are 0 (for use with bool operations),"
print "the remainder are uniformly distribuited on [-50,25)."
        
print ''.join(f.rjust(15) for f in ['pure-py','np-simple','np-optimized']) + "(np-simple/np-optimized)".rjust(25)
for name, f in testable_funcs.items():
    t_np_optimized = timeit.Timer(lambda: accumarray_np(test_idx, test_vals, func=f)).timeit(number=5)
    t_np_simple = timeit.Timer(lambda: accumarray_np(test_idx, test_vals, func=lambda x: f(x))).timeit(number=5)
    t_purepy = timeit.Timer(lambda: accumarray_pure(test_idx, test_vals, func=name)).timeit(number=2)
    print ("%.1fms" % (t_purepy * 1000)).rjust(14),
    print ("%.1fms" % (t_np_simple * 1000)).rjust(14),
    print ("%.1fms" % (t_np_optimized * 1000)).rjust(14),
    print ("%.1fx" % (t_np_simple/t_np_optimized)).rjust(25)