# numpy-groupies

This package consists of a small library of optimised tools for doing things that can roughly 
be considered "group-indexing operations". The most prominent tool is `aggregate`, which is 
descibed in detail further down the page.


## Installation
If you have `pip`, then simply:
```
pip install numpy_groupies
```
Note that `numpy_groupies` doesn't have any compulsory dependencies (even `numpy` is optional) 
so you should be able to install it fairly easily even without a package manager.  If you just 
want one particular implementation of `aggregate` (e.g. `aggregate_numpy.py`), you can download 
that one file, and copy-paste the contents of `utils.py` into the top of that file (replacing 
the `from .utils import (...)` line).


## aggregate

![aggregate_diagram](/diagrams/aggregate.png)
```python
import numpy as np
import numpy_groupies as npg
group_idx = np.array([   3,   0,   0,   1,   0,   3,   5,   5,   0,   4])
a =         np.array([13.2, 3.5, 3.5,-8.2, 3.0,13.4,99.2,-7.1, 0.0,53.7])
npg.aggregate(group_idx, a, func='sum', fill_value=0)
# >>>          array([10.0, -8.2, 0.0, 26.6, 53.7, 92.1])
```
`aggregate` takes an array of values, and an array giving the group number for each of those values. 
It then returns the sum (or mean, or std, or any, ...etc.) of the values in each group. You have 
probably come across this idea before - see [Matlab's `accumarray` function](http://uk.mathworks.com/help/matlab/ref/accumarray.html?refresh=true), or
 [`pandas` groupby concept](http://pandas.pydata.org/pandas-docs/dev/groupby.html), or
 [MapReduce paradigm](http://en.wikipedia.org/wiki/MapReduce), or simply the [basic histogram](https://en.wikipedia.org/wiki/Histogram).

A couple of implemented functions do not reduce the data, instead it calculates values cumulatively
while iterating over the data or permutates them. The output size matches the input size.

```python
group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
a =         np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
npg.aggregate(group_idx, a, func='cumsum')
# >>>          array([3, 4, 5, 6,15, 9,15,22, 7, 0,15,17, 6,14,31,39])
```


### Inputs
The function accepts various different combinations of inputs, producing various different shapes of output. 
We give a brief description of the general meaning of the inputs and then go over the different combinations 
in more detail:

* `group_idx` - array of non-negative integers to be used as the "labels" with which to group the values in `a`.
* `a` - array of values to be aggregated.
* `func='sum'` - the function to use for aggregation.  See the section below for nore details.
* `size=None` - the shape of the output array. If `None`, the maximum value in `group_idx` will set the size of the output.
* `fill_value=0` - value to use for output groups that do not appear anywhere in the `group_idx` input array.
* `order='C'` - for multidimensional output, this controls the layout in memory, can be `'F'` for fortran-style.
* `dtype=None` - the`dtype` of the output. `None` means choose a sensible type for the given `a`, `func`, and `fill_value`.
* `axis=None` - explained below.
* `ddof=0` - passed through into calculations of variance and standard deviation (see section on functions).

![aggregate_dims_diagram](/diagrams/aggregate_dims.png)

* Form 1 is the simplest, taking `group_idx` and `a` of matching 1D lengths, and producing a 1D output.
* Form 2 is similar to Form 1, but takes a scalar `a`, which is broadcast out to the length of `group_idx`. Note that this is generally not that useful.
* Form 3 is more complicated. `group_idx` is the same length as the `a.shape[axis]`. The groups are broadcast out along the other axis/axes of `a`, thus the output is of shape `n_groups x a.shape[0] x ... x a.shape[axis-1] x a.shape[axis+1] x ... a.shape[-1]`, i.e. the output has two or more dimensions.
* Form 4 also produces output with two or more dimensions, but for very different reasons to Form 3.  Here `a` is 1D and `group_idx` is exactly `2D`, whereas in Form 3 `a` is `ND`, `group_idx` is `1D`, and we provide a value for `axis`.  The length of `a` must match `group_idx.shape[1]`, the value of `group_idx.shape[0]` determines the number of dimensions in the ouput, i.e. `group_idx[:,99]` gives the `(x,y,z)` group indices for the `a[99]`.
* Form 5 is the same as Form 4 but with scalar `a`. As with Form 2, this is rarely that helpful.

**Note on performance.** The `order` of the output is unlikely to affect performance of `aggregate` (although it may affect your downstream usage of that output), however the order of multidimensional `a` or `group_idx` can affect performance:  in Form 4 it is best if columns are contiguous in memory within `group_idx`, i.e. `group_idx[:, 99]` corresponds to a contiguous chunk of memory; in Form 3 it's best if all the data in `a` for `group_idx[i]` is contiguous, e.g. if `axis=1` then we want `a[:, 55]` to be contiguous.


### Available functions
By default, `aggregate` assumes you want to sum the values within each group, however you can specify another 
function using the `func` kwarg.  This `func` can be any custom callable, however you will likely want one of
the following optimized functions. Note that not all functions might be provided by all implementations.

* `'sum'` - sum of items within each group (see example above).
* `'prod'` - product of items within each group
* `'mean'` - mean of items within each group
* `'var'`- variance of items within each group. Use `ddof` kwarg for degrees of freedom. The divisor used in calculations is `N - ddof`, where `N` represents the number of elements. By default `ddof` is zero.
* `'std'` - standard deviation of items within each group. Use `ddof` kwarg for degrees of freedom (see `var` above).
* `'min'` - minimum value of items within each group.
* `'max'` - maximum value of items within each group.
* `'first'` - first item in `a` from each group.
* `'last'` - last item in `a` from each group.
* `'argmax'` - the index in `a` of the maximum value in each group.
* `'argmin'` - the index in `a` of the minimum value in each group.

The above functions also have a `nan`-form, which skip the `nan` values instead of propagating them to the result of the calculation:
* `'nansum'`, `'nanprod'`, `'nanmean'`, `'nanvar'`, `'nanstd'`, `'nanmin'`, `'nanmax'`, `'nanfirst'`, `'nanlast'`, ``nanargmax``, ``nanargmin``

The following functions are slightly different in that they always return boolean values. Their treatment of nans is also different from above:
* `'all'` - `True` if all items within a group are truethy. Note that `np.all(nan)` is `True`, i.e. `nan` is actually truethy.
* `'any'` - `True` if any items within a group are truethy.
* `'allnan'` - `True` if all items within a group are `nan`.
* `'anynan'` - `True` if any items within a gorup are `nan`.

The following functions don't reduce the data, but instead produce an output matching the size of the input:
* `cumsum` - cumulative sum of items within each group.
* `cumprod` - cumulative product of items within each group. (numba only)
* `cummin` - cumulative minimum of items within each group. (numba only)
* `cummax` - cumulative maximum of items within each group. (numba only)
* `'sort'` - sort the items within each group in ascending order, use reverse=True to invert the order.

Finally, there are three functions which don't reduce each group to a single value, instead they return the full set of items within the group:
* `'array'` - simply returns the grouped items, using the same order as appeared in `a`. (numpy only)


### Examples
Compute sums of consecutive integers, and then compute products of those consecutive integers.
```python
group_idx = np.arange(5).repeat(3)
# group_idx: array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
a = np.arange(group_idx.size)
# a: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
x = aggregate(group_idx, a) # sum is default
# x: array([ 3, 12, 21, 30, 39])
x = aggregate(group_idx, a, 'prod')
# x: array([ 0, 60, 336, 990, 2184])
```

Get variance ignoring nans, setting all-nan groups to `nan`.
```python
x = aggregate(group_idx, a, func='nanvar', fill_value=nan)
```

Count the number of elements in each group. Note that this is equivalent to doing `np.bincount(group_idx)`, indeed that is how the numpy implementation does it.
```python
x = aggregate(group_idx, 1)
```

Sum 1000 values into a three-dimensional cube of size 15x15x15. Note that in this example all three dimensions have the same size, but that doesn't have to be the case.
```python
group_idx = np.random.randint(0, 15, size=(3, 1000))
a = np.random.random(group_idx.shape[1])
x = aggregate(group_idx, a, func="sum", size=(15,15,15), order="F")
# x.shape: (15, 15, 15)
# np.isfortran(x): True
```

Use a custom function to generate some strings.
```python
group_idx = array([1, 0,  1,  4,  1])
a = array([12.0, 3.2, -15, 88, 12.9])
x = aggregate(group_idx, a,
              func=lambda g: ' or maybe '.join(str(gg) for gg in g), fill_value='')
# x: ['3.2', '12.0 or maybe -15.0 or maybe 12.9', '', '', '88.0']
```

Use the `axis` arg in order to do a sum-aggregation on three rows simultaneously.
```python
a = array([[99, 2,  11, 14,  20],
	   	   [33, 76, 12, 100, 71],
		   [67, 10, -8, 1,   9]])
group_idx = array([[3, 3, 7, 0, 0]])
x = aggregate(group_idx, a, axis=1)
# x : [[ 34, 0, 0, 101, 0, 0, 0, 11],
#      [171, 0, 0, 109, 0, 0, 0, 12],
#      [ 10, 0, 0,  77, 0, 0, 0, -8]]
```


### Multiple implementations
There are multiple implementations of `aggregate` provided. If you use `from numpy_groupies import aggregate`, 
the best available implementation will automatically be selected. Otherwise you can pick a specific version directly 
like `from numpy_groupies import aggregate_nb as aggregate` or by importing aggregate from the implementing module 
`from numpy_groupies.aggregate_weave import aggregate`.

Currently the following implementations exist:
* **numpy** - This is the default implementation. It uses plain `numpy`, mainly relying on `np.bincount` and basic indexing magic. It comes without other dependencies except `numpy` and shows reasonable performance for the occasional usage.
* **numba** - This is the most performant implementation in average, based on jit compilation provided by numba and LLVM.
* **weave** - `weave` compiles C-code on demand at runtime, producing binaries that get executed from within python. The performance of this implementation is comparable to the numba implementation.
* **pure python** - This implementation has no dependencies and uses only the standard library. It's horribly slow and should only be used, if there is no numpy available.
* **numpy ufunc** - *Only for benchmarking.*  This impelmentation uses the `.at` method of numpy's `ufunc`s (e.g. `add.at`), which would appear to be designed for perfoming excactly the same calculation that `aggregate` executes, however the numpy implementation is rather incomplete and slow (as of `v1.14.0`). A [numpy issue](https://github.com/numpy/numpy/issues/5922) has been created to address this issue.
* **pandas** - *Only for reference.*  The pandas' `groupby` concept is the same as the task performed by `aggregate`. However, `pandas` is not actually faster than the default `numpy` implementation. Also, note that there may be room for improvement in the way that `pandas` is utilized here. Most notably, when computing multiple aggregations of the same data (e.g. `'min'` and `'max'`) pandas could potentially be used more efficiently.

All implementations have the same calling syntax and produce the same outputs, to within some floating-point error. 
However some implementations only support a subset of the valid inputs and will sometimes throw `NotImplementedError`.


### Benchmarks
Scripts for testing and benchmarking are included in this repository. For benchmarking, run `python -m numpy_groupies.benchmarks.generic` 
from the root of this repository.

Below we are using `500,000` indices uniformly picked from `[0, 1000)`. The values of `a` are uniformly picked from 
the interval `[0,1)`, with anything less than `0.2` then set to 0 (in order to serve as falsy values in boolean operations). 
For `nan-` operations another 20% of the values are set to nan, leaving the remainder on the interval `[0.2,0.8)`.

The benchmarking results are given in ms for an i7-7560U running at 2.40GHz:
```text
function         ufunc         numpy         numba         weave
-----------------------------------------------------------------
sum              28.763         1.477         0.917         1.167
prod             29.165        29.162         0.919         1.170
amin             33.020        33.134         0.979         1.181
amax             33.150        33.156         1.049         1.216
len              28.594         1.260         0.755         1.023
all              33.493         3.883         0.995         1.214
any              33.308         6.776         1.003         1.216
anynan           28.938         2.472         0.930         1.182
allnan           29.391         5.929         0.931         1.201
mean               ----         2.100         0.972         1.216
std                ----         6.600         1.127         1.370
var                ----         6.684         1.109         1.388
first              ----         2.140         1.067         1.188
last               ----         1.545         0.818         1.086
argmax             ----        33.860         1.016          ----
argmin             ----        36.690         0.981          ----
nansum             ----         4.944         1.722         1.342
nanprod            ----        27.286         1.726         1.369
nanmin             ----        30.238         1.895         1.359
nanmax             ----        30.337         1.939         1.446
nanlen             ----         4.820         1.707         1.312
nanall             ----         9.148         1.786         1.380
nanany             ----        10.157         1.830         1.392
nanmean            ----         5.775         1.758         1.440
nanvar             ----        10.090         1.922         1.651
nanstd             ----        10.308         1.884         1.664
nanfirst           ----         5.481         1.945         1.295
nanlast            ----         4.992         1.735         1.199
cumsum             ----       144.807         1.455          ----
cumprod            ----          ----         1.371          ----
cummax             ----          ----         1.441          ----
cummin             ----          ----         1.340          ----
arbitrary          ----       237.252        79.717          ----
sort               ----       261.951          ----          ----
Linux(x86_64), Python 2.7.12, Numpy 1.16.0, Numba 0.42.0, Weave 0.17.0

```


## Development
This project was started by @ml31415 and the `numba` and `weave` implementations are by him. The pure 
python and `numpy` implementations were written by @d1manson.

The authors hope that `numpy`'s `ufunc.at` methods will eventually be fast enough that hand-optimisation
of individual functions will become unneccessary. However even if that does happen, there may still be a 
role for the `aggregate` function as a light-weight wrapper around those functions.

Maybe at some point a version of `aggregate` will make its way into `numpy` itself (or at least `scipy`).
