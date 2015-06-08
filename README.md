#aggregate
*Aggregation function for python. It is named after, and very similar to, Matlab's `accumarray` function - [see Mathworks docs here](http://uk.mathworks.com/help/matlab/ref/accumarray.html?refresh=true). If you are familiar with `pandas`, you could consider `aggregate` to be a light-weight version of the [`groupby` concept](http://pandas.pydata.org/pandas-docs/dev/groupby.html).*

```python
from aggregate_numpy import aggregate 
import numpy as np
group_idx = np.array([3,0,0,1,0,3,5,5,0,4])
a = np.array([13.2,3.5,3.5,-8.2,3.0,13.4,99.2,-7.1,0.0,53.7])
result = aggregate(group_idx, a, func='sum', fill_value=np.nan)
# result:  array([10.0, -8.2, 0.0, 26.6, 53.7, 92.1])
```
*`aggregate` can be run with zero dependecies, i.e. using pure python, but a fast `numpy` implementation is also available. If that's not enough, you can use the super-fast `scipy.weave` version.*

###The main idea of aggregate
Suppose that that you have a list of values, `a`, and some labels for each of the values, `group_idx`. The purpose of the `aggregate` function is to aggregate over all the values with the same label, for example taking the `sum` or `mean` (using whatever aggregation function the user requests).  

Here is a simple example with ten values in `a` and their paired `group_idx` (this is the same as the code example above):

![aggregate_diagram](/diagram.png)
    
The output is an array, with the ith element giving the `sum`  (or `mean` or whatever) of the relevant items from within `a`. By default, any gaps are filled with zero: in the above example, the label `2` does not appear in the `group_idx` list, so in the output, the element `2` is `0.0`.  If you would prefer to fill the gaps with `nan` (or some other value, e.g. `-1`) you can do this using `fill_value=nan`.

###Multiple implementations of aggregate
This repository contains several independent implementations of the same function.
Some of the implementations may throw `NotImplementedError` for certain inputs, 
but whenever a result is produced it should be the same across all implementations
(to within some small floating-point error).  
The simplest implementation, provided in the file **`aggregate_purepy.py`** uses pure python, but is much slower
 than the other implementations.  **`aggregate_numpy.py`** makes use of a variety of `numpy` tricks to try and get as close to
the hardware's optimal performance as possible, however if you really want the absolute best performance possible you will need
 the **`aggregate_weave.py`** version - see the notes on `scipy.weave` below *TODO: this isn't refactored properly yet*.

Note that if you know which implementation you want you only need that one file, plus the `utils.py` file.
See below for benchmarking stats.

**Other implementations** The **`aggregate_numpy_ufunc.py`** version is only for testing and benchmarking, though hopefully in future, if numpy
improves, this version will become more relevant.  The **`aggregate_pandas.py`** version is faster than the numpy version for `prod`, `min`, and `max`,
though only slightly.  Note that not much work has gone into trying to squeeze the most out of pandas, so it may be possible to do better still, especially
for `all` and `any` which are oddly slow.  `std` and `var` are currently not implemented as `pandas` seems to be doing something slightly different.

*TODO: create a meta-implementation which dynamically picks from available implementations based on which is available.*

###Available aggregation functions
Below is a list of the main functions.  Note that you can also provide your own custom function, but obviously it wont run as fast as most of these optimised ones. As shown below, most functions have a "nan- version", for example there is a `"nansum"` function as well as a `"sum"` function. The nan- verions simply drop all the nans before doing the aggregation. This means that any groups consisting only of `nan`s will be given `fill_value` in the output (rather than `nan`). If you would like to set all-nan groups to have `nan` in the output, do the following:

```python
# get variance ignoring nans
a = aggregate(group_idx, a, func='nanvar') 
# set the all-nan groups to be nan
a[aggregate(group_idx, a, func='allnan')] = nan
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
`"allnan"` | | no | `numpy`: 4/5, `weave`: 5/5 | `numpy` uses `np.isnan` and then `aggregate`'s `"all"`.
`"anynan"` | | no | `numpy`: 4/5, `weave`: 5/5 | `numpy` uses `np.isnan` and then `aggregate`'s `"any"`.
`"array"` |`"split"`, `"splice"`, `np.array`, `np.asarray` | no | `numpy`: 4/5, `weave`: ?? | output is a `numpy` array with `dtype=object`, each element of which is a `numpy` array (or `fill_value`). The order of values within each group matches the original order in the full `a` array.
`"sort"` | `"sorted"`, `"asort"`, `"fsort"`, `np.sort`, `sorted` (inbuilt python) | no |  `numpy`: 4/5, `weave`: ??  | similar to `"array"`, except here the values in each output array are sorted in ascending order.
`"rsort"` | `"rsorted"`, `"dsort"` | no |  `numpy`: 4/5, `weave`: ??  | similar to `"sort"`, except in descending order.
`<custom function>` | | |  `numpy`: 4/5, `weave`: ?? | similar to `"array"`, except the `<custom function>` is evaulated on each group and the return value is placed in the final output array.

Note that the last few functions listed above do not return an array of scalars but an array with `dtype=object`.  
Also, note that as of `numpy v1.9`, the `<custom function>` implementation is only slightly slower than the `ufunc.at` method, so if you want to use a `ufunc` not in the above list, it wont run that much slower when simple supplied as a `<custom function>`, e.g. `func=np.logaddexp`.  There is a [numpy issue](https://github.com/numpy/numpy/issues/5922) trying to get this performance bug fixed - please show interest there if you want to encourage the `numpy` devs to work on that! If, however, for a specific `ufunc`, you know of a fast algorithm which does signficantly better than `ufunc.at` please get in touch and we can incorporate it here.

### Scalar `a`
Although we have so far assumed that `a` is a 1d array, it can in fact be a scalar. The most common example of this is using `aggregate` to simply count the number of occurances of each value in `group_idx`.

```python
aggregate(group_idx, 1, func='sum') # equivalent to np.bincount(idx)
```

Most other functions do accept a scalar, but the output may be rather meaningless in many cases (e.g. `max`  just returns an array repeating the given scalar and/or `fill_value`). Scalars are not accepted for "nan- versions" of the functions because either the single scalar value is `nan` or it's not!

### 2D `group_idx` for multidimensional output
Although we have so far assumed that `group_idx` is 1D, and the same length as `a`, it can in fact be 2D (or some form of nested sequences that can be converted to 2D).  When `group_idx` is 2D, the size of the 0th dimension corresponds to the number of dimesnions in the output, i.e. `group_idx[i,j]` gives the index into the ith dimension in the output for `a[j]`.  Note that `a` should still be 1D (or scalar), with length matching `group_idx.shape[1]`.  When producing multidimensional output you can specify `C` or `Fortran` memory layout using `order='C'` or `order='F'` respectively.

*TODO: show example*

### Specifying the size of the output array
Sometimes you may want to force the output of `aggregate` to be of a particular length/shape.  You can use the `size` keyword argument for this. The length of `size` should match the number of dimesnions in the output. If left as `None`the maximumum values in `group_idx` will define the size of the output array.


### Some examples

```python
import python as np
from aggregate import aggregate

group_idx = np.arange(5).repeat(3)
a = np.arange(group_idx.size)
aggregate(group_idx, a)
>>> array([ 3, 12, 21, 30, 39])

aggregate(group_idx, a, np.prod)
>>> array([   0,   60,  336,  990, 2184])
```

### Benchmarking and testing
Benchmarking and testing scripts are included. 

Note that the actual observed results depend on a variety of properties of the input.
Here we are using `500,000` indices uniformly picked from `[0, 1000)`.
Specifically, about 20% of the values are set to `0` for use with bool operations.
Nan operations get another 20% of the values set to nan. So the remainder is uniformly 
distribuited on `[0.2,1)` or on `[0.2,0.8)` for nan operations. 

The benchmarking results are given in ms for an i7-5500U@2.40GHz:

```text
function      grouploop          numpy          weave          ufunc         pandas
-----------------------------------------------------------------------------------
sum              53.087          1.846          1.615         36.961         17.032
amin             51.126         37.652          1.609         37.539         16.976
amax             51.221         38.330          1.639         38.447         17.086
prod             51.779         37.284          1.610         37.733         17.091
all              52.555          4.288          2.498         44.059        105.400
any              52.545          6.972          2.493         42.346        102.686
mean             54.946          2.674          1.655           ----         14.309
var              65.339          6.176          1.978           ----         53.387
std              67.685          6.345          1.966           ----         50.349
first            49.388          2.708          1.481           ----         11.920
last             50.277          1.898          1.489           ----         13.914
nansum           58.896          6.427          2.462           ----         14.713
nanmin           56.265         34.821          2.578           ----         14.778
nanmax           56.173         35.606          2.570           ----         14.870
nanmean          78.230          7.004          2.549           ----         14.736
nanvar          103.345          9.977          2.819           ----         47.639
nanstd          105.108         10.081          2.812           ----         47.951
nanfirst         54.182          7.066          2.346           ----         15.226
nanlast          54.418          6.456          2.357           ----         16.277
anynan           53.635          3.178          2.423         36.306         73.342
allnan           53.010          5.300          2.415         37.693         73.106
Linux(x86_64), Python 2.7.6, Numpy 1.9.2
```


The `grouploop` implementation shown here uses `aggregate_numpy.py`'s
generic function menchanism, which groups `a` by `group_idx`, and then
loops over each group, applying the specified function (in this case it is a numpy function 
such as `np.add`). `grouploop` is only included for reference, note that the output from
this function is considered to be the "correct" answer when used in testing.

`ufunc` uses the `aggregate_numpy_ufunc.py` implementation. That implementation is not 
intended for mainstream usage, it is only included in the hope that numpy's `ufunc.at`
performance will eventually improve.

`pandas` does some preprocessing and caching, probably reuses sorting when grouping. The
times given here represent the full time from constructing the `DataFrame`, grouping it and finally
doing the acutal aggregation. Skipping or reusing these steps speeds `pandas` up notably, but for a
fair competition the grouping is done for each separate `aggregate` call.


### Development
The authors hope that `numpy`'s `ufunc.at` methods will eventually be fast enough that hand-optimisation 
of individual functions will become unneccessary. However even if that does happen, there will still 
probably be a role for this `aggregate` function as a light-weight wrapper around those functions, 
and it may well be that `C` code will always be significantly faster than whatever `numpy` can offer.

Maybe at some point a version of `aggregate` will make its way into `numpy` itself (or at least `scipy`).

This project was started by @ml31415 and the `scipy.weave` implementation is by him. The pure python and `numpy` implementations were written by @d1manson. 
