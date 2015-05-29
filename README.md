#accumarray
*Accumulation function for python. It is named after, and very similar to, Matlab's `accumarray` function - [see Mathworks docs here](http://uk.mathworks.com/help/matlab/ref/accumarray.html?refresh=true). If you are familiar with `pandas`, you could consider `accumarray` to be a light-weight version of the [`groupby` concept](http://pandas.pydata.org/pandas-docs/dev/groupby.html).*

```python
from accumarray_numpy import accumarray 
import numpy as np
idx = np.array([3,0,0,1,0,3,5,5,0,4])
vals = np.array([13.2,3.5,3.5,-8.2,3.0,13.4,99.2,-7.1,0.0,53.7])
result = accumarray(idx, vals, func='sum', fillvalue=np.nan)
# result:  array([10.0, -8.2, 0.0, 26.6, 53.7, 92.1])
```
*`accumarray` can be run with zero dependecies, i.e. using pure python, but a fast `numpy` implementation is also  available. If that's not enough, you can use the super-fast `scipy.weave` version.*

###The main idea of accumarray
Suppose that that you have a list of values, `vals`, and some labels for each of the values, `idx`. The purpose of the `accumarray` function is to aggregate over all the values with the same label, for example taking the `sum` or `mean` (using whatever aggregation function the user requests).  

Here is a simple example with ten `vals` and their paired `idx` (this is the same as the code example above):

![accumarray_diagram](/accumarray_diagram.png)
    
The output is an array, with the ith element giving the `sum`  (or `mean` or whatever) of the relevant items from within `vals`.  By default, any gaps are filled with zero: in the above example, the label `2` does not appear in the `idx` list, so in the output, the element `2` is `0.0`.  If you would prefer to fill the gaps with `nan` (or some other value, e.g. `-1`) you can do this using `filvalue=nan`.

###Multiple implementations of accumarray
This repositorary contains several independant implementations of the same function.
Some of the implementations may throw `NotImplemented` exceptions for certain inputs, 
but whenever a result is produced it should be the same across all implementations
(to within some small floating-point error).  
The simplest implementation, provided in the file **`accumarray_purepy.py`** uses pure python, but is much slower
 than the other implementations.  **`accumarray_numpy.py`** makes use of a variety of `numpy` tricks to try and get as close to
the hardware's optimal performance as possible, however if you really want the absolute best performance possible you will need
 the **`accumarray_weave.py`** version - see the notes on `scipy.weave` below *TODO: this isn't refactored properly yet*.

Note that if you know which implementation you want you only need that one file, plus the `accumarray_utils.py` file.
See below for benchmarking stats.

**Other implementations** The **`accumarray_numpy_ufunc.py`** version is only for testing and benchmarking, though hopefully in future, if numpy
improves, this version will become more relevant.  The **`accumarray_pandas.py`"" version is faster than the numpy version for `prod`, `min`, and `max`,
though only slightly.  Note that not much work has gone into trying to squeeze the most out of pandas, so it may be possible to do better still, especially
for `all` and `any` which are oddly slow.  `std` and `var` are currently not implemented as `pandas` seems to be doing something slightly different.

*TODO: create a meta-implementation which dynamically picks from available implementations based on which is available.*

###Available aggregation functions
Below is a list of the main functions.  Note that you can also provide your own custom function, but obviously it wont run as fast as most of these optimised ones. As shown below, most functions have a "nan- version", for example there is a `"nansum"` function as well as a `"sum"` function. The nan- verions simply drop all the nans before doing the aggregation. This means that any groups consisting only of `nan`s will be given `fillvalue` in the output (rather than `nan`). If you would like to set all-nan groups to have `nan` in the output, do the following:

```python
a = accumarray(idx, vals, func='nanvar') # e.g. get variance ignoring nans
a[accumarray(idx, vals, func='allnan')] = nan # here you set the all-nan groups to be nan
```  

The prefered way of specifying a function is using a string, e.g. `func="sum"`, however in many cases actual function objects will be recognised too:

name     | aliases       | nan-?  |  performance| notes
:-------- |:-------------| --------------  | ----------------------------| --------
`"sum"`   | `"plus"`, `"add"`, `np.sum`, `np.add`, `sum` (inbuilt python) | yes | `numpy`: 5/5, `weave`: 5/5 | `numpy` uses `bincount`
`"mean"` | `np.mean` | yes | `numpy`: 5/5, `weave`: 5/5| `numpy` uses `bincount`
`"var"` | `np.var` | yes | `numpy`: 5/5, `weave`: - | `numpy` uses `bincount`, computed as `sum((vals-means)**2)`. 
`"std"` | `np.std` | yes | `numpy`: 5/5, `weave`: 5/5 | see `"var"`.
`"all"` | `"and"`, `np.all`, `all` (inbuilt python) | yes | `numpy`: 4/5, `weave`: 5/5 | `numpy` uses simple indexing operations
`"any"` | `"or"`, `np.any`, `any` (inbuilt python) | yes | `numpy`: 4/5, `weave`: 5/5 | `numpy` uses simple indexing operations
`"first"` | | yes |  `numpy`: 5/5, `weave`: - | `numpy` uses simple indexing
`"last"` | | yes |  `numpy`: 5/5, `weave`: -  | `numpy` uses simple indexing
`"min"` | `"amin"`, `"minimum"`, `np.min`, `np.amin`, `np.minimum`, `min` (inbuilt python) | yes |  `numpy`: 2/5, `weave`: 5/5  | `numpy` uses `minimum.at` which is slow (as of `v1.9`)
`"max"` | `"amax"`, `"maximum"`, `np.max`, `np.amax`, `np.maxmum`, `max` (inbuilt python) | yes | `numpy`: 2/5, `weave`: 5/5 | `numpy` uses `maximum.at` which is slow (as of `v1.9`)
`"prod"` | `"product"`, `"times"`, `"multiply"`, `np.prod`, `np.multiply` | yes | `numpy`: 2/5, `weave`: 5/5| `numpy` uses `prod.at` which is slow (as of `v1.9`)
`"allnan"` | | no | `numpy`: 4/5, `weave`: 5/5 | `numpy` uses `np.isnan` and then `accumarray`'s `"all"`.
`"anynan"` | | no | `numpy`: 4/5, `weave`: 5/5 | `numpy` uses `np.isnan` and then `accumarray`'s `"any"`.
`"array"` |`"split"`, `"splice"`, `np.array`, `np.asarray` | no | `numpy`: 4/5, `weave`: ?? | output is a `numpy` array with `dtype=object`, each element of which is a `numpy` array (or `fillvalue`). The order of values within each group matches the original order in the full `vals` array.
`"sort"` | `"sorted"`, `"asort"`, `"fsort"`, `np.sort`, `sorted` (inbuilt python) | no |  `numpy`: 4/5, `weave`: ??  | similar to `"array"`, except here the values in each output array are sorted in ascending order.
`"rsort"` | `"rsorted"`, `"dsort"` | no |  `numpy`: 4/5, `weave`: ??  | similar to `"sort"`, except in descending order.
`<custom function>` | | |  `numpy`: 4/5, `weave`: ?? | similar to `"array"`, except the `<custom function>` is evaulated on each group and the return value is placed in the final output array.

Note that the last few functions listed above do not return an array of scalars but an array with `dtype=object`.  
Also, note that as of `numpy v1.9`, the `<custom function>` implementation is only slightly slower than the `ufunc.at` method, so if you want to use a `ufunc` not in the above list, it wont run that much slower when simple supplied as a `<custom function>`, e.g. `func=np.logaddexp`.  There is a [numpy issue](https://github.com/numpy/numpy/issues/5922) trying to get this performance bug fixed - please show interest there if you want to encourage the `numpy` devs to work on that! If, however, for a specific `ufunc`, you know of a fast algorithm which does signficantly better than `ufunc.at` please get in touch and we can incorporate it here.

### Scalar `vals`
Although we have so far assumed that `vals` is a 1d array, it can in fact be a scalar. The most common example of this is using `accumarray` to simply count the number of occurances of each value in `idx`.

```python
accumarray(idx, 1, func='sum') # equivalent to np.bincount(idx)
```

Most other functions do accept a scalar, but the output may be rather meaningless in many cases (e.g. `max`  just returns an array repeating the given scalar and/or `fillvalue`.) .  Scalars are not accepted for "nan- versions" of the functions because either the single scalar value is `nan` or it's not!

### 2D `idx` for multidimensional output
Although we have so far assumed that `idx` is 1D, and the same length as `vals`, it can in fact be 2D (or some form of nested sequences that can be converted to 2D).  When `idx` is 2D, the size of the 0th dimension corresponds to the number of dimesnions in the output, i.e. `idx[i,j]` gives the index into the ith dimension in the output for `val[j]`.  Note that `vals` should still be 1D (or scalar), with length matching `idx.shape[1]`.  When producing multidimensional output you can specify `C` or `Fortran` memory layout using `order='C'` or `order='F'` repsectively.

*TODO: show example*

### Specifying the size of the output array
Sometimes you may want to force the output of `accumarray` to be of a particular length/shape.  You can use the `sz` keyword argument for this. The length of `sz` should match the number of dimesnions in the output. If left as `None`the maximumum values in `idx` will define the size of the output array.


### Some examples

*TODO: show a variety of things, ideally explaining them with some real-world motivation.*

### Benchmarking and testing
Benchmarking and testing scripts are included here.  Here are some benchmarking results:

*TODO: use a range of inputs shapes/types etc. and give more details hardware/software info*

```text
function   pure-py  np-grouploop**  np-ufuncat* np-optimised     pandas          ratio
     std   1737.8ms       171.8ms       no-impl         7.0ms    no-impl     247.1: 24.4:  -  : 1.0 :  -  
     all   1280.8ms        62.2ms        41.8ms         6.6ms    550.7ms     193.5: 9.4 : 6.3 : 1.0 : 83.2
     min   1358.7ms        59.6ms        42.6ms        42.7ms     24.5ms      55.4: 2.4 : 1.7 : 1.7 : 1.0 
     max   1538.3ms        55.9ms        38.8ms        37.5ms     18.8ms      81.9: 3.0 : 2.1 : 2.0 : 1.0 
     sum   1532.8ms        62.6ms        40.6ms         1.9ms     20.4ms     808.5: 33.0: 21.4: 1.0 : 10.7
     var   1756.8ms       146.2ms       no-impl         6.3ms    no-impl     279.1: 23.2:  -  : 1.0 :  -  
    prod   1448.8ms        55.2ms        39.9ms        38.7ms     20.2ms      71.7: 2.7 : 2.0 : 1.9 : 1.0 
     any   1399.5ms        69.1ms        41.1ms         5.7ms    558.8ms     246.2: 12.2: 7.2 : 1.0 : 98.3
    mean   1321.3ms        88.3ms       no-impl         4.0ms     20.9ms     327.6: 21.9:  -  : 1.0 : 5.2 
Python 2.7.9, Numpy 1.9.2, Win7 Core i7.
```

Note that the actual observed speedup depends on a variety of properties of the input.
Here we are using `100,000` indices uniformly picked from `[0, 1000)`.
Specifically, about 25% of the values are `0` (for use with bool operations),
the remainder are uniformly distribuited on `[-50,25)`. 

**The `np-grouploop` implementation shown here uses `accumarray_numpy.py`'s
 generic function menchanism, which groups `vals` by `idx`, and then
loops over each group, applying the specified function (in this case it is a numpy function 
such as `np.add`).  It is only included here for reference, note that the output form
this funciton is considered to be the "correct" answer when used in testing.

*The `np-ufuncat` uses the `accumarray_numpy_ufunc.py` implementation. That implementation
is not intended for mainstream usage, it is only include in the hope that numpy's `ufunc.at`
performance will eventually improve.


### Development
The authors hope that `numpy`'s `ufunc.at` methods will eventually be fast enough that hand-optimisation of individual functions will become unnecccessary.  However even if that does happen, there will still probably be a role for this `accumarray` function as a light-weight wrapper around those functions, and it may well be that `C` code will always be significantly faster than whatever `numpy` can offer.

Maybe at some point a version of `accumarray` will make its way into `numpy` itself (or at least `scipy`).

This project was started by @ml31415 and the `scipy.weave` implementation is by him.  The pure python and `numpy` implementations were written by @d1manson. 

 
