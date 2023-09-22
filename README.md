[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/ml31415/numpy-groupies/ci.yaml?branch=master&logo=github&style=flat)](https://github.com/ml31415/numpy-groupies/actions)
[![PyPI](https://img.shields.io/pypi/v/numpy-groupies.svg?style=flat)](https://pypi.org/project/numpy-groupies/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/numpy_groupies.svg?style=flat)](https://anaconda.org/conda-forge/numpy_groupies)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# numpy-groupies

This package consists of a small library of optimised tools for doing things that can roughly 
be considered "group-indexing operations". The most prominent tool is `aggregate`, which is 
described in detail further down the page.


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
* `func='sum'` - the function to use for aggregation. See the section below for more details.
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
* Form 4 also produces output with two or more dimensions, but for very different reasons to Form 3.  Here `a` is 1D and `group_idx` is exactly `2D`, whereas in Form 3 `a` is `ND`, `group_idx` is `1D`, and we provide a value for `axis`.  The length of `a` must match `group_idx.shape[1]`, the value of `group_idx.shape[0]` determines the number of dimensions in the output, i.e. `group_idx[:,99]` gives the `(x,y,z)` group indices for the `a[99]`.
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
* `'nansum'`, `'nanprod'`, `'nanmean'`, `'nanvar'`, `'nanstd'`, `'nanmin'`, `'nanmax'`, `'nanfirst'`, `'nanlast'`, `'nanargmax'`, `'nanargmin'`

The following functions are slightly different in that they always return boolean values. Their treatment of nans is also different from above:
* `'all'` - `True` if all items within a group are truethy. Note that `np.all(nan)` is `True`, i.e. `nan` is actually truethy.
* `'any'` - `True` if any items within a group are truethy.
* `'allnan'` - `True` if all items within a group are `nan`.
* `'anynan'` - `True` if any items within a group are `nan`.

The following functions don't reduce the data, but instead produce an output matching the size of the input:
* `'cumsum'` - cumulative sum of items within each group.
* `'cumprod'` - cumulative product of items within each group. (numba only)
* `'cummin'` - cumulative minimum of items within each group. (numba only)
* `'cummax'` - cumulative maximum of items within each group. (numba only)
* `'sort'` - sort the items within each group in ascending order, use reverse=True to invert the order.

Finally, there are three functions which don't reduce each group to a single value, instead they return the full 
set of items within the group:
* `'array'` - simply returns the grouped items, using the same order as appeared in `a`. (numpy only)


### Examples
Compute sums of consecutive integers, and then compute products of those consecutive integers.
```python
group_idx = np.arange(5).repeat(3)
# group_idx: array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
a = np.arange(group_idx.size)
# a: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
x = npg.aggregate(group_idx, a) # sum is default
# x: array([ 3, 12, 21, 30, 39])
x = npg.aggregate(group_idx, a, 'prod')
# x: array([ 0, 60, 336, 990, 2184])
```

Get variance ignoring nans, setting all-nan groups to `nan`.
```python
x = npg.aggregate(group_idx, a, func='nanvar', fill_value=nan)
```

Count the number of elements in each group. Note that this is equivalent to doing `np.bincount(group_idx)`, 
indeed that is how the numpy implementation does it.
```python
x = npg.aggregate(group_idx, 1)
```

Sum 1000 values into a three-dimensional cube of size 15x15x15. Note that in this example all three dimensions 
have the same size, but that doesn't have to be the case.
```python
group_idx = np.random.randint(0, 15, size=(3, 1000))
a = np.random.random(group_idx.shape[1])
x = npg.aggregate(group_idx, a, func="sum", size=(15,15,15), order="F")
# x.shape: (15, 15, 15)
# np.isfortran(x): True
```

Use a custom function to generate some strings.
```python
group_idx = np.array([1, 0,  1,  4,  1])
a = np.array([12.0, 3.2, -15, 88, 12.9])
x = npg.aggregate(group_idx, a,
              func=lambda g: ' or maybe '.join(str(gg) for gg in g), fill_value='')
# x: ['3.2', '12.0 or maybe -15.0 or maybe 12.9', '', '', '88.0']
```

Use the `axis` arg in order to do a sum-aggregation on three rows simultaneously.
```python
a = np.array([[99, 2,  11, 14,  20],
	   	   [33, 76, 12, 100, 71],
		   [67, 10, -8, 1,   9]])
group_idx = np.array([[3, 3, 7, 0, 0]])
x = npg.aggregate(group_idx, a, axis=1)
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
* **numba** - This is the most performant implementation, based on jit compilation provided by numba and LLVM.
* **pure python** - This implementation has no dependencies and uses only the standard library. It's horribly slow and should only be used, if there is no numpy available.
* **numpy ufunc** - *Only for benchmarking.*  This implementation uses the `.at` method of numpy's `ufunc`s (e.g. `add.at`), which would appear to be designed for performing exactly the same calculation that `aggregate` executes, however the numpy implementation is rather incomplete.
* **pandas** - *Only for reference.*  The pandas' `groupby` concept is the same as the task performed by `aggregate`. However, `pandas` is not actually faster than the default `numpy` implementation. Also, note that there may be room for improvement in the way that `pandas` is utilized here. Most notably, when computing multiple aggregations of the same data (e.g. `'min'` and `'max'`) pandas could potentially be used more efficiently.

All implementations have the same calling syntax and produce the same outputs, to within some floating-point error. 
However some implementations only support a subset of the valid inputs and will sometimes throw `NotImplementedError`.


### Benchmarks
Scripts for testing and benchmarking are included in this repository. For benchmarking, run 
`python -m numpy_groupies.benchmarks.generic` from the root of this repository.

Below we are using `500,000` indices uniformly picked from `[0, 1000)`. The values of `a` are uniformly picked from 
the interval `[0,1)`, with anything less than `0.2` then set to 0 (in order to serve as falsy values in boolean operations). 
For `nan-` operations another 20% of the values are set to nan, leaving the remainder on the interval `[0.2,0.8)`.

The benchmarking results are given in ms for an i7-7560U running at 2.40GHz:

| function  | ufunc   | numpy   | numba   | pandas  |
|-----------|---------|---------|---------|---------|
| sum       | 1.950   | 1.728   | 0.708   | 11.832  |
| prod      | 2.279   | 2.349   | 0.709   | 11.649  |
| min       | 2.472   | 2.489   | 0.716   | 11.686  |
| max       | 2.457   | 2.480   | 0.745   | 11.598  |
| len       | 1.481   | 1.270   | 0.635   | 10.932  |
| all       | 37.186  | 3.054   | 0.892   | 12.587  |
| any       | 35.278  | 5.157   | 0.890   | 12.845  |
| anynan    | 5.783   | 2.126   | 0.762   | 144.740 |
| allnan    | 7.971   | 4.367   | 0.774   | 144.507 |
| mean      | ----    | 2.500   | 0.825   | 13.284  |
| std       | ----    | 4.528   | 0.965   | 12.193  |
| var       | ----    | 4.269   | 0.969   | 12.657  |
| first     | ----    | 1.847   | 0.811   | 11.584  |
| last      | ----    | 1.309   | 0.581   | 11.842  |
| argmax    | ----    | 3.504   | 1.411   | 293.640 |
| argmin    | ----    | 6.996   | 1.347   | 290.977 |
| nansum    | ----    | 5.388   | 1.569   | 15.239  |
| nanprod   | ----    | 5.707   | 1.546   | 15.004  |
| nanmin    | ----    | 5.831   | 1.700   | 14.292  |
| nanmax    | ----    | 5.847   | 1.731   | 14.927  |
| nanlen    | ----    | 3.170   | 1.529   | 14.529  |
| nanall    | ----    | 6.499   | 1.640   | 15.931  |
| nanany    | ----    | 8.041   | 1.656   | 15.839  |
| nanmean   | ----    | 5.636   | 1.583   | 15.185  |
| nanvar    | ----    | 7.514   | 1.682   | 15.643  |
| nanstd    | ----    | 7.292   | 1.666   | 15.104  |
| nanfirst  | ----    | 5.318   | 2.096   | 14.432  |
| nanlast   | ----    | 4.943   | 1.473   | 14.637  |
| nanargmin | ----    | 7.977   | 1.779   | 298.911 |
| nanargmax | ----    | 5.869   | 1.802   | 301.022 |
| cumsum    | ----    | 71.713  | 1.119   | 8.864   |
| cumprod   | ----    | ----    | 1.123   | 12.100  |
| cummax    | ----    | ----    | 1.062   | 12.133  |
| cummin    | ----    | ----    | 0.973   | 11.908  |
| arbitrary | ----    | 147.853 | 46.690  | 129.779 |
| sort      | ----    | 167.699 | ----    | ----    |

_Linux(x86_64), Python 3.10.12, Numpy 1.25.2, Numba 0.58.0, Pandas 2.0.2_

## Development
This project was started by @ml31415 and the `numba` and `weave` implementations are by him. The pure 
python and `numpy` implementations were written by @d1manson.

The authors hope that `numpy`'s `ufunc.at` methods or some other implementation of `aggregate` within
`numpy` or `scipy` will eventually be fast enough, to make this package redundant. Numpy 1.25 actually
contained major [improvements on ufunc speed](https://numpy.org/doc/stable/release/1.25.0-notes.html), 
which reduced the speed gap between numpy and the numba implementation a lot.
