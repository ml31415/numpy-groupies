# aggregate
Aggregation function for python. It is named after, and very similar to, Matlab's `accumarray` function - [see Mathworks docs here](http://uk.mathworks.com/help/matlab/ref/accumarray.html?refresh=true). If you are familiar with `pandas`, you could consider `aggregate` to be a light-weight version of the [`groupby` concept](http://pandas.pydata.org/pandas-docs/dev/groupby.html).

```python
import numpy as np
from aggregate import aggregate 
group_idx = np.array([3,0,0,1,0,3,5,5,0,4])
a = np.array([13.2,3.5,3.5,-8.2,3.0,13.4,99.2,-7.1,0.0,53.7])
aggregate(group_idx, a, func='sum', fill_value=np.nan)
# >>> array([10.0, -8.2, 0.0, 26.6, 53.7, 92.1])
```
`aggregate` can be run with zero dependecies, i.e. using pure python, but a fast `numpy` implementation is also available. If that's not enough, you can use the super-fast `scipy.weave` version.

### The main idea of aggregate
Suppose that that you have a list of values, `a`, and some labels for each of the values, `group_idx`. The purpose of the `aggregate` function is to aggregate over all the values with the same label, for example taking the `sum` or `mean` (using whatever aggregation function the user requests).  

Here is a simple example with ten values in `a` and their paired `group_idx` (this is the same as the code example above):

![aggregate_diagram](/diagram.png)
    
The output is an array, with the ith element giving the `sum`  (or `mean` or whatever) of the relevant items from within `a`. By default, any gaps are filled with zero: in the above example, the label `2` does not appear in the `group_idx` list, so in the output, the element `2` is `0.0`.  If you would prefer to fill the gaps with `nan` (or some other value, e.g. `-1`) you can do this using `fill_value=nan`.

### Multiple implementations of aggregate
This repository contains several independent implementations of the same function.
Some of the implementations may throw `NotImplementedError` for certain inputs, 
but whenever a result is produced it should be the same across all implementations
(to within some small floating-point error).  
The simplest implementation, provided in the file **`aggregate_purepy.py`** uses pure python, but is magnitudes slower
than the other implementations. **`aggregate_numpy.py`** makes use of a variety of `numpy` tricks to try and get as close to
the hardwares optimal performance as possible, however if you really want the absolute best performance possible you will need
the **`aggregate_weave.py`** version, see benchmarking below.

Note that if you know which implementation you want you only need that one file, plus the `utils.py` file.

**Other implementations** The **`aggregate_numpy_ufunc.py`** version is only for testing and benchmarking, though hopefully in future, if numpy
improves, this version will become more relevant.  The **`aggregate_pandas.py`** version is faster than the numpy version for `prod`, `min`, and `max`,
though only slightly.  Note that not much work has gone into trying to squeeze the most out of pandas, so it may be possible to do better still, especially
for `all` and `any` which are oddly slow.

### Available aggregation functions
Below is a list of the main functions. Note that you can also provide your own custom function, but obviously it wont run as fast as most of these optimised ones. As shown below, most functions have a "nan- version", for example there is a `"nansum"` function as well as a `"sum"` function. The nan- verions simply drop all the nans before doing the aggregation. This means that any groups consisting only of `nan`s will be given `fill_value` in the output (rather than `nan`). If you would like to set all-nan groups to have `nan` in the output, do the following:

```python
# get variance ignoring nans
a = aggregate(group_idx, a, func='nanvar') 
# set the all-nan groups to be nan
a[aggregate(group_idx, a, func='allnan')] = nan
```  

The prefered way of specifying a function is using a string, e.g. `func="sum"`, however in many cases actual function objects will be recognised too. For an overview of the performance of the functions, see the benchmarking below.

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

```python
nindices = 100
outsize = (10, 10)
group_idx = np.random.randint(0, 10, size=(len(outsize), nindices))
a = np.random.random(group_idx.shape[1])
res = aggregate(group_idx, a, func="sum", size=outsize, order="F")
res.shape
# >>> (10, 10)
np.isfortran(res)
# >>> True
```

### Specifying the size of the output array
Sometimes you may want to force the output of `aggregate` to be of a particular length/shape.  You can use the `size` keyword argument for this. The length of `size` should match the number of dimesnions in the output. If left as `None`the maximumum values in `group_idx` will define the size of the output array.


### Some examples

```python
group_idx = np.arange(5).repeat(3)
a = np.arange(group_idx.size)
aggregate(group_idx, a)
# >>> array([ 3, 12, 21, 30, 39])

aggregate(group_idx, a, np.prod)
# >>> array([   0,   60,  336,  990, 2184])
```

### Benchmarking and testing
Benchmarking and testing scripts are included. 

Note that the actual observed results depend on a variety of properties of the input.
Here we are using `500,000` indices uniformly picked from `[0, 1000)`.
Specifically, about 20% of the values are set to `0` for use with bool operations.
Nan operations get another 20% of the values set to nan. So the remainder is uniformly 
distribuited on `[0.2,1)` or on `[0.2,0.8)` for nan operations. 

The benchmarking results are given in ms for an i7-5500U running at 2.40GHz:

```text
function      grouploop          numpy          weave          ufunc         pandas
-----------------------------------------------------------------------------------
sum              54.090          1.922          1.616         36.598         17.789
amin             51.318         37.344          1.643         37.354         17.197
amax             51.623         38.418          1.686         38.598         17.238
prod             51.676         37.296          1.675         37.656         17.225
all              52.587          4.343          2.508         44.119        104.996
any              52.284          7.038          2.460         42.797        103.572
mean             55.270          2.720          1.692           ----         14.104
var              66.521          6.468          1.933           ----         53.966
std              67.732          6.269          1.920           ----         50.633
first            49.457          2.768          1.511           ----         11.761
last             49.573          1.966          1.526           ----         13.403
nansum           58.828          6.470          2.443           ----         14.711
nanmin           56.135         35.125          2.581           ----         14.845
nanmax           56.190         35.639          2.636           ----         14.579
nanmean          78.909          7.096          2.592           ----         14.594
nanvar          104.019          9.916          2.845           ----         47.680
nanstd          106.703         10.135          2.872           ----         48.563
nanfirst         53.752          7.213          2.393           ----         14.936
nanlast          54.073          6.529          2.381           ----         16.094
anynan           52.983          3.249          2.441         36.368         73.483
allnan           52.968          5.374          2.539         38.186         73.396
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
