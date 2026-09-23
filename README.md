[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/ml31415/numpy-groupies/ci.yaml?branch=master&logo=github&style=flat)](https://github.com/ml31415/numpy-groupies/actions)
[![PyPI](https://img.shields.io/pypi/v/numpy-groupies.svg?style=flat)](https://pypi.org/project/numpy-groupies/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/numpy_groupies.svg?style=flat)](https://anaconda.org/conda-forge/numpy_groupies)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fml31415%2Fnumpy-groupies%2Fmaster%2Fpyproject.toml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/numpy-groupies)

# numpy-groupies

This package consists of a small library of optimised tools for doing things that can roughly 
be considered "group-indexing operations". The most prominent tool is `aggregate`, which is 
described in detail further down the page.


## Installation
If you have `pip`, then simply:
```
pip install numpy_groupies
```
Note that the package only declares `numpy` as a dependency; the `pure python` implementation of 
`aggregate` additionally works without it. If you just want one particular implementation of 
`aggregate` (e.g. `aggregate_numpy.py`), you can download that one file, and copy-paste the contents 
of `utils.py` into the top of that file (replacing the `from .utils import (...)` line).


## aggregate

![aggregate_diagram](/diagrams/aggregate.png)
```python
import numpy as np
import numpy_groupies as npg

group_idx = np.array([3, 0, 0, 1, 0, 3, 5, 5, 0, 4])
a = np.array([13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7])
npg.aggregate(group_idx, a, func="sum", fill_value=0)
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
a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
npg.aggregate(group_idx, a, func="cumsum")
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
* `fill_value=DEFAULT_FILL_VALUE` - value to use for output groups that do not appear anywhere in the `group_idx` input array.  By default it is chosen per function, see the section below.
* `order='C'` - for multidimensional output, this controls the layout in memory, can be `'F'` for fortran-style.
* `dtype=None` - the`dtype` of the output. `None` means choose a sensible type for the given `a`, `func`, and `fill_value`.
* `axis=None` - explained below.
* `ddof=0` - passed through into calculations of variance and standard deviation (see section on functions).
* `dx=1.0` - passed through into the calculation of the trapezoidal integral (see section on functions), where it is the sample spacing.

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
* `'median'` - median of items within each group
* `'var'`- variance of items within each group. Use `ddof` kwarg for degrees of freedom. The divisor used in calculations is `N - ddof`, where `N` represents the number of elements. By default `ddof` is zero.
* `'std'` - standard deviation of items within each group. Use `ddof` kwarg for degrees of freedom (see `var` above).
* `'min'` - minimum value of items within each group.
* `'max'` - maximum value of items within each group.
* `'first'` - first item in `a` from each group.
* `'last'` - last item in `a` from each group.
* `'argmax'` - the index in `a` of the maximum value in each group.
* `'argmin'` - the index in `a` of the minimum value in each group.
* `'trapezoid'` - trapezoidal integral of the items within each group, taken in the order they appear in `a` (numpy, numba and pure python). Use `dx` kwarg for the sample spacing, which is 1 by default. A group of fewer than two items integrates to zero.

The above functions also have a `nan`-form, which skip the `nan` values instead of propagating them to the result of the calculation (for `nantrapezoid` this means integrating over the items which are left, bridging over the gap the `nan` leaves):
* `'nansum'`, `'nanprod'`, `'nanmean'`, `'nanmedian'`, `'nantrapezoid'`, `'nanvar'`, `'nanstd'`, `'nanmin'`, `'nanmax'`, `'nanfirst'`, `'nanlast'`, `'nanargmax'`, `'nanargmin'`

The following functions are slightly different in that they always return boolean values. Their treatment of nans is also different from above:
* `'all'` - `True` if all items within a group are truthy. Note that `np.all(nan)` is `True`, i.e. `nan` is actually truthy.
* `'any'` - `True` if any items within a group are truthy.
* `'allnan'` - `True` if all items within a group are `nan`.
* `'anynan'` - `True` if any items within a group are `nan`.

The following functions don't reduce the data, but instead produce an output matching the size of the input:
* `'cumsum'` - cumulative sum of items within each group.
* `'cumprod'` - cumulative product of items within each group. (numba and pandas)
* `'cummin'` - cumulative minimum of items within each group. (numba and pandas)
* `'cummax'` - cumulative maximum of items within each group. (numba and pandas)
* `'sort'` - sort the items within each group in ascending order, use reverse=True to invert the order.


There is one function which doesn't reduce each group to a single value, instead it returns the full 
set of items within the group:
* `'array'` - simply returns the grouped items, using the same order as appeared in `a`. (numpy and pure python)


### Fill values

Groups which have no items at all are filled with `fill_value`, which by default is chosen to match
what the corresponding numpy function gives for an empty input:

| functions | default fill value |
|-----------|--------------------|
| `sum`, `len`, `sumofsquares` (and their `nan`-forms) | `0` |
| `prod` (and `nanprod`) | `1` |
| `all`, `any`, `allnan`, `anynan` | `False` |
| `mean`, `median`, `var`, `std` (and their `nan`-forms) | `nan` |
| `min`, `max`, `first`, `last` (and their `nan`-forms) | `nan` for floating input, `0` for integer output, which cannot hold `nan` |
| `argmax`, `argmin` (and their `nan`-forms) | `-1` |
| `trapezoid`, `nantrapezoid` | `0`, as it is for a single sample |
| `array`, `sort` | an empty sequence |
| a custom `func` | `nan` for floating input, `0` otherwise |

The `cum`-functions and `sort` return one value per input item, so they have nothing to fill an
absent group with - asking them to do so raises a `ValueError`.  The default of a particular
function can be queried with `npg.default_fill_value(func, dtype=None)`, which is handy if downstream
code needs to know it without repeating the table.

### Examples
Compute sums of consecutive integers, and then compute products of those consecutive integers.
```python
group_idx = np.arange(5).repeat(3)
# group_idx: array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
a = np.arange(group_idx.size)
# a: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
x = npg.aggregate(group_idx, a)  # sum is default
# x: array([ 3, 12, 21, 30, 39])
x = npg.aggregate(group_idx, a, "prod")
# x: array([ 0, 60, 336, 990, 2184])
```

Get variance ignoring nans.  Groups which are entirely nan end up as `nan`, which is the default
`fill_value` of that function.
```python
x = npg.aggregate(group_idx, a, func="nanvar")
```

Integrate the items of each group with the trapezoidal rule, in the order they appear in `a`. `dx` is the
sample spacing, and `np.trapezoid` may be used in place of the function name.
```python
x = npg.aggregate(group_idx, a, func="trapezoid")
# x: array([ 2.,  8., 14., 20., 26.])
x = npg.aggregate(group_idx, a, func=np.trapezoid, dx=0.5)
# x: array([ 1.,  4.,  7., 10., 13.])
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
x = npg.aggregate(group_idx, a, func="sum", size=(15, 15, 15), order="F")
# x.shape: (15, 15, 15)
# np.isfortran(x): True
```

Use a custom function to generate some strings.  Non-numeric output needs `dtype=object`, and the
`fill_value` of a custom function has to be given explicitly, since it cannot be guessed.
```python
group_idx = np.array([1, 0, 1, 4, 1])
a = np.array([12.0, 3.2, -15, 88, 12.9])
x = npg.aggregate(group_idx, a, func=lambda g: " or maybe ".join(str(gg) for gg in g), fill_value="", dtype=object)
# x: ['3.2', '12.0 or maybe -15.0 or maybe 12.9', '', '', '88.0']
```

Use the `axis` arg in order to do a sum-aggregation on three rows simultaneously.
```python
a = np.array([[99, 2, 11, 14, 20], [33, 76, 12, 100, 71], [67, 10, -8, 1, 9]])
group_idx = np.array([[3, 3, 7, 0, 0]])
x = npg.aggregate(group_idx, a, axis=1)
# x : [[ 34, 0, 0, 101, 0, 0, 0, 11],
#      [171, 0, 0, 109, 0, 0, 0, 12],
#      [ 10, 0, 0,  77, 0, 0, 0, -8]]
```


### Multiple implementations
There are multiple implementations of `aggregate` provided. If you use `from numpy_groupies import aggregate`, 
the best available implementation will automatically be selected (numba if installed, otherwise numpy).
Otherwise you can pick a specific version directly 
like `from numpy_groupies import aggregate_nb as aggregate` or by importing aggregate from the implementing module 
`from numpy_groupies.aggregate_numpy import aggregate`.

Currently the following implementations exist:
* **numpy** - It uses plain `numpy`, mainly relying on `np.bincount` and basic indexing magic. It comes without other dependencies except `numpy` and shows reasonable performance for the occasional usage. This is the default implementation used when numba is not installed.
* **numba** - This is the most performant implementation, based on jit compilation provided by numba and LLVM.
* **pure python** - This implementation has no dependencies and uses only the standard library. It's horribly slow and should only be used, if there is no numpy available.
* **numpy ufunc** - *Only for benchmarking.*  This implementation uses the `.at` method of numpy's `ufunc`s (e.g. `add.at`), which would appear to be designed for performing exactly the same calculation that `aggregate` executes, however this implementation is rather incomplete.
* **pandas** - *Only for reference.*  The pandas' `groupby` concept is the same as the task performed by `aggregate`. However, `pandas` is not actually faster than the default `numpy` implementation. Also, note that there may be room for improvement in the way that `pandas` is utilized here. Most notably, when computing multiple aggregations of the same data (e.g. `'min'` and `'max'`) pandas could potentially be used more efficiently.

All implementations have the same calling syntax and produce the same outputs, to within some floating-point error. 
However some implementations only support a subset of the valid inputs and will sometimes throw `NotImplementedError`.


### Benchmarks
Scripts for testing and benchmarking are included in this repository. For benchmarking, run 
`python -m numpy_groupies.benchmarks.generic` from the root of this repository.

Below we are using `500,000` indices uniformly picked from `[0, 1000)`. The values of `a` are uniformly picked from 
the interval `[0,1)`, with anything less than `0.2` then set to 0 (in order to serve as falsy values in boolean operations). 
For `nan-` operations another 20% of the values are set to nan, leaving the remainder on the interval `[0.2,0.8)`.

The benchmarking results are given in ms for an i7-7560U running at 2.40GHz, taking the minimum over 3 runs:

| function | ufunc  | numpy   | numba  | pandas  |
|-----------|--------|---------|--------|---------|
| sum       |   1.386 |   1.158 |   0.673 |  14.498 |
| prod      |   2.390 |   2.570 |   0.740 |  13.569 |
| min       |   2.593 |   2.625 |   0.788 |  13.560 |
| max       |   2.608 |   2.572 |   0.736 |  13.745 |
| len       |   1.386 |   1.029 |   0.528 |  12.699 |
| all       |  47.042 |   2.494 |   0.859 |  14.314 |
| any       |  43.811 |   3.167 |   0.928 |  14.633 |
| anynan    |   6.835 |   1.399 |   0.809 |  13.659 |
| allnan    |  10.014 |   3.592 |   0.800 |  13.799 |
| mean      |    ---- |   1.836 |   0.770 |  14.771 |
| median    |    ---- |  60.150 |  12.379 |  19.578 |
| trapezoid |    ---- |   4.626 |   1.000 |    ---- |
| std       |    ---- |   4.481 |   0.968 |  15.491 |
| var       |    ---- |   4.327 |   0.981 |  15.588 |
| first     |    ---- |   1.733 |   0.637 |  13.619 |
| last      |    ---- |   1.542 |   0.607 |  14.027 |
| argmax    |    ---- |   3.937 |   0.976 |  13.299 |
| argmin    |    ---- |   6.528 |   0.963 |  13.058 |
| nansum    |    ---- |   5.138 |   1.868 |  21.898 |
| nanprod   |    ---- |   6.571 |   1.854 |  20.524 |
| nanmin    |    ---- |   5.979 |   1.783 |  18.319 |
| nanmax    |    ---- |   6.079 |   1.759 |  18.218 |
| nanlen    |    ---- |   3.041 |   1.626 |  18.105 |
| nanall    |    ---- |   6.001 |   1.709 |  19.557 |
| nanany    |    ---- |   6.694 |   1.716 |  19.712 |
| nanmean   |    ---- |   5.405 |   1.998 |  19.443 |
| nanmedian |    ---- |  53.924 |   9.602 |  24.646 |
| nantrapezoid|    ---- |   8.541 |   2.164 |    ---- |
| nanvar    |    ---- |   7.273 |   2.050 |  20.352 |
| nanstd    |    ---- |   7.589 |   2.100 |  22.989 |
| nanfirst  |    ---- |   5.442 |   1.474 |  19.188 |
| nanlast   |    ---- |   5.370 |   1.484 |  18.711 |
| nanargmin |    ---- |   8.112 |   2.082 |  14.236 |
| nanargmax |    ---- |   5.727 |   2.221 |  14.160 |
| cumsum    |    ---- |  51.117 |   1.155 |   8.894 |
| cumprod   |    ---- |    ---- |   1.125 |  14.006 |
| cummax    |    ---- |    ---- |   1.174 |  13.838 |
| cummin    |    ---- |    ---- |   1.192 |  14.774 |
| arbitrary |    ---- | 145.723 |  49.292 | 140.942 |
| sort      |    ---- | 141.303 |    ---- |    ---- |

_Linux(x86_64), Python 3.14.2, Numpy 2.5.3, Numba 0.67.0, Pandas 3.0.6_

## Development
This project was started by @ml31415 and the `numba` and `weave` implementations are by him. The pure 
python and `numpy` implementations were written by @d1manson.

The authors hope that `numpy`'s `ufunc.at` methods or some other implementation of `aggregate` within
`numpy` or `scipy` will eventually be fast enough, to make this package redundant. Numpy 1.25 actually
contained major [improvements on ufunc speed](https://numpy.org/doc/stable/release/1.25.0-notes.html), 
which reduced the speed gap between numpy and the numba implementation a lot.
