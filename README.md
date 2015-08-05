# numpy-groupies

This package consists of a small library of optimised tools for doing things 
that can roughly be considered "group-indexing operations".  The most prominent
tool is `aggregate`, which is descibed in detail further down the page.

#### Installation 
If you have `pip`, then simply:
```
pip install numpy_groupies
```
Note that `numpy_groupies` doesn't have any compulsorary dependencies (even `numpy`
is optional) so you should be able to install it fairly easily even without a package 
manager.  If you just want one particular implementation of `aggregate` (e.g. `aggregate_numpy.py`), you can
download that one file, and copy-paste the contents of `utils.py` into the top
of that file (replacing the `from .utils import (...)` line).


# Overview of tools
#### aggregate

![aggregate_diagram](/diagrams/aggregate.png)
```python
import numpy as np
import numpy_groupies as npg
group_idx = np.array([3,0,0,1,0,3,5,5,0,4])
a = np.array([13.2,3.5,3.5,-8.2,3.0,13.4,99.2,-7.1,0.0,53.7])
npg.aggregate(group_idx, a, func='sum', fill_value=0) # see below for further examples
# >>> array([10.0, -8.2, 0.0, 26.6, 53.7, 92.1])
```
If you have used [Matlab's `accumarray` function](http://uk.mathworks.com/help/matlab/ref/accumarray.html?refresh=true),
this `aggregate` function should appear fairly familiar to you because the inputs 
and output are almost identical to `accumarray`.  Alternatively, if you are familiar 
with the [`pandas` groupby concept](http://pandas.pydata.org/pandas-docs/dev/groupby.html), 
the purpose, if not the exact syntax, of `aggregate` will be familiar to you.  Failing that, 
if you've come across the [MapReduce paradigm](http://en.wikipedia.org/wiki/MapReduce) you 
should at least appriciate the need for an `aggreagate` function such as this.  Anyone 
still confused should think back to the last time they created a histogram, and then 
look carefully at the above diagram..although if you are reading this the chances 
are you already basically understand what's going on.

#### multi_cumsum [alpha]
![multicumsum_diagram](/diagrams/multi_cumsum.png)
**Warning:** the API for this function has not be stabilized yet and is liable to change.
```python
#TODO: give code example as with aggregate
```

#### multi_arange [alpha]
![multicumsum_diagram](/diagrams/multi_arange.png)
**Warning:** the API for this function has not be stabilized yet and is liable to change.
```python
#TODO: give code example as with aggregate
```

#### label_contiguous_1d [alpha]
![label_contiguous_1d](/diagrams/label_contiguous_1d.png)
**Warning:** the API for this function has not be stabilized yet and is liable to change.
```python
#TODO: give code example as with aggregate
```



# aggregate - full documentation

### Available functions   
By default, `aggregate` assumes you want to sum the values within each group, however you can specify another function using the `func` kwarg.  This `func` can be any custom callable, however in normal use you will probably want one of the following optimized functions:

* `'sum'` - sum of items within each group (see example above).
* `'prod'` - product of items within each group
* `'mean'` - mean of items within each group
* `'var'`- variance of items within each group. Use `ddof` kwarg for degrees of freedom. The divisor used in calculations is `N - ddof`, where `N` represents the number of elements. By default `ddof` is zero.
* `'std'` - standard deviation of items within each group. Use `ddof` kwarg for degrees of freedom (see `var` above).
* `'min'` - minimum value of items within each group.
* `'max'` - maximum value of items within each group.
* `'first'` - first item in `a` from each group.
* `'last'` - last item in `a` from each group.
* ``argmax`` - the index in `a` of the maximum value in each group.
* ``argmin`` - the index in `a` of the minimum value in each group.

The above functions also have a `nan-` form, for which `nan` values are dropped before the aggregation calculation is computed:
* `'nansum'`, `'nanprod'`, `'nanmean'`, `'nanvar'`, `'nanstd'`, `'nanmin'`, `'nanmax'`, `'nanfirst'`, `'nanlast'`, ``nanargmax``, ``nanargmin``   

The following functions are slightly different in that they always return boolean values. Their treatment of nans is also different from above:
* `'all'` - `True` if all items within a group are truethy. Note that `np.all(nan)` is `True`, i.e. `nan` is actually truethy.
* `'any'` - `True` if any items within a group are truethy.
* `allnan` - `True` if all items within a group are `nan`.
* `anynan` - `True` if any items within a gorup are `nan`.

Finally, there are three functions which don't reduce each group to a single value, instead they return the full set of items within the group:  
* `'array'` - simply returns the grouped items, using the same order as appeared in `a`.
* `'sort'` - like `'array'`, above, but the items within each group are now sorted in ascending order.
* `'rsort'` - same as `'sort'`, but in reverse, i.e. descending order.

### Full description of inputs
The first three inputs have already been explained/demonstrated above, but we list them here again for completeness.  
* `group_idx` - this is an array of non-negative integers, to be used as the "labels" with which to group the values in `a`. Although we have so far assumed that `group_idx` is one-dimesnaional, and the same length as `a`, it can in fact be two-dimensional (or some form of nested sequences that can be converted to 2D).  When `group_idx` is 2D, the size of the 0th dimension corresponds to the number of dimesnions in the output, i.e. `group_idx[i,j]` gives the index into the ith dimension in the output for `a[j]`.  Note that `a` should still be 1D (or scalar), with length matching `group_idx.shape[1]`.  One final option for `group_idx` involves the `axis` argument - see description below.
* `a` - this is the array of values to be aggregated.  See the above for a simple demonstration of what this means.  `a` will normally be a one-dimensional array, however it can also be a scalar in some cases.
* `func='sum'` - the function to use for aggregation.  See the section above for details.  Note that the simplest way to specify the function is using a string (e.g. `func='mac'`) however a number of aliases are also defined (e.g. you can use the `func=np.max`, or even `func=max`, where `max` is the builtin function).  To check the available aliases see `utils.py`.
* `size=None` - the shape of the output array. If `None`, the maximum value in `group_idx` will set the size of the output.  Note that for multidimensional output you need to list the size of each dimension here, or give `None`.
* `fill_value=0` - in the example above, group 2 does not have any data, so requires some kind of filling value - in this case the default of `0` is used.  If you had set `fill_value=nan` or something else, that value would appear instead of `0` for the 2 element in the output.  Note that there are some subtle interactions between what is permitted for `fill_value` and the input/output `dtype` - exceptions should be raised in most cases to alert the programmer if issue arrise.
* `order='C'` - this is relevant only for multimensional output.  It controls the layout of the output array in memory, can be `'F'` for fortran-style.
* `dtype=None` - the `dtype` of the output.  By default something sensible is chosen based on the input, aggregation function, and `fill_value`.
* `axis=None` - in addition to the basic 1D form for `a`, you can provide multimensional `a`, with `group_idx` being 1D and corresponding to only one axis of `a`.  Use the `axis` argument to specify which axis `group_idx` correspodns to. See the example below for a demonstration of this.
* `ddof=0` - passed through into calculations of variance and standard deviation (see above).



### Multiple implementations - explanation and benchmark results
There are multiple implementations of `aggregate` provided.  They range from the 
pure-python implementation which has no dependencies at all but is not very fast,
 to the `scipy.weave` implementation which runs fast but requires a working install 
of `scipy.weave`.  For most users, the `aggregate_numpy.py` implementation is probably 
the easiest to install and offers fairly reasonable speed for the majority of aggregation 
functions.  If you download the whole repository and use `from aggregate import aggregate`, 
the best available implementation will automatically be selected.  

**Note:** if you have `weave` installed but no working compiler registered then you 
may run in to problems with the default implementation of `aggregate`.  The workaround
is to explicitly use `npg.aggregate_np` rather than `npg.aggregate`.

Currently the following implementations exist:  
* **pure python**. *Use only if you don't have numpy installed*. This has no dependencies, instead making use of the grouping and sorting functionality provided by the python language itself plus the standard library.
* **numpy ufunc**. *Only for use with testing/benchmarking, i.e. normally do NOT use this.*  This impelmentation uses the `.at` method of numpy's `ufunc`s (e.g. `add.at`), which would appear to be designed for perfoming excactly the same calculation that `aggregate` executes, however the numpy implementation is very slow (as of `v1.9.2`).  A [numpy issue](https://github.com/numpy/numpy/issues/5922) has be created to try and address this performance bug.  Also, note that some of the desired functions do not have suitable `ufunc.at` analogues (e.g. `mean`, `var`).
* **numpy**.  *RECOMMENDED, unless you have weave installed and working.* This also uses `numpy`, but most of the aggregation functions have been hand-optimised using something other than `ufunc.at`.  You can see the tricks employed by reading the code: it mostly relies on `np.bincount` and some basic indexing magic.  For most users this will be the most sensible choice of implementation, at least initially.
* **pandas**. *You probably don't want this - check the benchmarks.*  As mentioned at the top of this page, pandas' `groupby` concept is the same as the task performed by `aggregate`.  Thus, it makes sense to try and piggyback off pandas if it is available. Note however, that `pandas` is not actually any faster than the recommended `numpy` implementation (except in a few cases).  Also, note that there may be room for improvement in the way that `pandas` is utilized here.  Most notably, when computing multiple aggregations of the same data (e.g. `'min'` and `'max'`) pandas could potentially be used much more efficiently - although other implementations could also deal with this case better,[as discussed in this issue](https://github.com/ml31415/accumarray/issues/3).
* **weave** - *If you need absolute speed and have/can get weave working then use this.* Weave does just-in-time compilation of handwritten C-code, producing binaries that can be easily run from python.  In all cases this allows this implementation to be faster than the recommended numpy implementation, especially for `'min'`, `'max'`, and `'prod'`.

All implementations have the same calling syntax and produce the same outputs, to within some floating-point error. However some implementations only support a subset of the valid inputs and will sometimes throw `NotImplementedError`.

Scripts for testing and benchmarking are included in this repository, which you can run yourself if need be.  Note that relative speeds will vary depending on the nature of the inputs.

Below we are using `500,000` indices uniformly picked from `[0, 1000)`.  The values of `a` are uniformly picked from the interval `[0,1)`, with anything less than `0.2` then set to 0 (in order to serve as falsy values in boolean operations). For `nan-` operations another 20% of the values set to nan, leaving the remainder on the interval `[0.2,0.8)`.

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

The `grouploop` implementation shown here uses `aggregate_numpy.py`'s generic function menchanism, which groups `a` by `group_idx`, and then loops over each group, applying the specified function (in this case it is a numpy function such as `np.add`). `grouploop` is only included for reference, note that the output from this function is considered to be the "correct" answer when used in testing.


### Examples
See the example at the top of the page for a super-simple introduction.


* Compute sums of consecutive integers, and then compute products of those consecutive integers.
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

* Get variance ignoring nans, but set all-nan groups to `nan` rather than `fill_value`.
```python
x = aggregate(group_idx, a, func='nanvar', fill_value=0) 
x[aggregate(group_idx, a, func='allnan')] = nan
```  

* Count the number of elements in each group. Note that this is equivalent to doing `np.bincount(group_idx)`, indeed that is how the numpy implementation does it.
```python
x = aggregate(group_idx, 1)
```

* Sum 1000 values into a three-dimensional cube of size 15x15x15. Note that in this example all three dimensions have the same size, but that doesn't have to be the case.
```python
group_idx = np.random.randint(0, 15, size=(3, 1000))
a = np.random.random(group_idx.shape[1])
x = aggregate(group_idx, a, func="sum", size=(15,15,15), order="F")
# x.shape: (15, 15, 15)
# np.isfortran(x): True
```

* Use a custom function to generate some strings.
```python
group_idx = array([1, 0,  1,  4,  1])
a = array([12.0, 3.2, -15, 88, 12.9])
x = aggregate(group_idx, a, 
              func=lambda g: ' or maybe '.join(str(gg) for gg in g), fill_value='')
# x: ['3.2', '12.0 or maybe -15.0 or maybe 12.9', '', '', '88.0']
```

* Use the `axis` arg in order to do a sum-aggregation on three rows simultaneously.
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

### Development
The authors hope that `numpy`'s `ufunc.at` methods will eventually be fast enough that hand-optimisation 
of individual functions will become unneccessary. However even if that does happen, there will still 
probably be a role for this `aggregate` function as a light-weight wrapper around those functions, 
and it may well be that `C` code will always be significantly faster than whatever `numpy` can offer.

Maybe at some point a version of `aggregate` will make its way into `numpy` itself (or at least `scipy`).

This project was started by @ml31415 and the `scipy.weave` implementation is by him. The pure python and `numpy` implementations were written by @d1manson. 
