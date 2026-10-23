# numpy-groupies

[![CI](https://img.shields.io/github/actions/workflow/status/ml31415/numpy-groupies/ci.yaml?branch=master&logo=github&style=flat)](https://github.com/ml31415/numpy-groupies/actions)
[![PyPI](https://img.shields.io/pypi/v/numpy-groupies.svg?style=flat)](https://pypi.org/project/numpy-groupies/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/numpy_groupies.svg?style=flat)](https://anaconda.org/conda-forge/numpy_groupies)
[![Python versions](https://img.shields.io/pypi/pyversions/numpy-groupies.svg?style=flat)](https://pypi.org/project/numpy-groupies/)
[![Downloads](https://img.shields.io/pypi/dm/numpy-groupies.svg?style=flat)](https://pypi.org/project/numpy-groupies/)

**Fast group-by aggregation on plain NumPy arrays.**
Give it values and a group label for each value, get back one result per group — sum, mean, std, median, min/max, argmax, cumulative sums, custom functions and more. No DataFrame required, with an optional [Numba](https://numba.pydata.org/) backend for speed.

```python
import numpy as np
import numpy_groupies as npg

group_idx = np.array([3, 0, 0, 1, 0, 3, 5, 5, 0, 4])
a = np.array([13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7])

npg.aggregate(group_idx, a, func="sum", fill_value=0)
# array([10. , -8.2,  0. , 26.6, 53.7, 92.1])
#  group:  0     1    2     3     4     5
```

Group 2 never occurs in `group_idx`, so its slot is filled with `fill_value`.

![Diagram of aggregate: values are collected by group label and reduced](https://github.com/ml31415/numpy-groupies/raw/master/docs/diagrams/aggregate.png)

**Contents:**
[Why numpy-groupies?](#why-numpy-groupies) ·
[Installation](#installation) ·
[Quickstart](#quickstart) ·
[Input forms](#input-forms) ·
[Functions](#functions) ·
[Helper tools](#helper-tools) ·
[Implementations](#implementations) ·
[Performance](#performance) ·
[Development](#development)

## Why numpy-groupies?

- **Fast.** With Numba installed, `aggregate` is the fastest group-by aggregation for plain NumPy arrays. See [Performance](#performance).
- **One call, many functions.** The usual reductions plus `median`, `quantile`/`percentile`, `skew`/`kurt`, `sort` (ascending or descending), `trapezoid`, the cumulative `cumsum`/`cumprod`/`cummin`/`cummax`, `argmin`/`argmax`, `array` (collect the members of each group) and any custom callable. Every reduction has a `nan…` variant.
- **Array in, array out.** Plain NumPy arrays, no DataFrame and no index. The result is an array indexed by group label, with empty groups filled with the value you choose. Multi-dimensional output (N-D binning) and aggregation along an `axis` are built in.
- **Usable inside your own jitted code.** Every Numba kernel is `@njit`-callable, so group statistics can sit inside your own compiled loops.
- **Numerically careful.** Integer sums, means and variances accumulate in a dtype wide enough not to wrap around, NaN handling follows NumPy, and complex and `float16` input are supported.
- **Light on dependencies.** NumPy is the only requirement; Numba is optional and only makes things faster. A pure-Python implementation is included as a fallback.

## Installation

```sh
pip install numpy_groupies            # NumPy implementation
pip install "numpy_groupies[fast]"    # + Numba for the fastest implementation
conda install -c conda-forge numpy_groupies
```

NumPy is the only declared dependency. Numba is optional and only the fastest implementation needs it; the pure-Python implementation runs without Numba but still imports NumPy (see [Implementations](#implementations)).

## Quickstart

**Counting, and other reductions.** The default function is `sum`; pass a scalar to count items per group (this is what `np.bincount` does under the hood in the NumPy implementation). Labels must be non-negative integers. For strings or other objects, map them first with `np.unique(labels, return_inverse=True)` and pass the inverse as `group_idx`. `a` are the values that you want to run your reductions on.

```python
npg.aggregate(group_idx, 1)  # items per group
npg.aggregate(group_idx, a, func="mean")  # nan for the empty group 2
npg.aggregate(group_idx, a, func="mean", fill_value=-1)
```

**Ignoring NaNs.** Every common reduction has a `nan…` variant, as in NumPy.

```python
group_idx = np.array([0, 0, 0, 1, 1, 2])
a = np.array([1.0, 2.0, np.nan, 4.0, np.nan, np.nan])

npg.aggregate(group_idx, a, func="nanmean")  # array([1.5, 4. , nan])
npg.aggregate(group_idx, a, func="nanvar")  # array([0.25, 0.  , nan])
npg.aggregate(group_idx, a, func="var")  # array([nan, nan, nan])  (plain var propagates NaN)
```

**Cumulative functions** don't reduce; the output has the same length as the input.

```python
npg.aggregate(np.array([0, 1, 0, 1, 0]), np.array([1, 2, 3, 4, 5]), func="cumsum")
# array([1, 2, 4, 6, 9])
```

**Custom functions.** Any callable works. Non-numeric output needs `dtype=object`, and since the default fill value of a custom function cannot be guessed, give one explicitly. Use the NumPy implementation for this: Numba can only compile numeric functions, so with Numba installed the plain `aggregate` fails on the string example below — and it compiles *functions*, so a callable object (a `functools.partial`, a dataclass with `__call__`) is for the NumPy implementation as well.

```python
import numpy_groupies.aggregate_numpy as npg

group_idx = np.array([1, 0, 1, 4, 1])
a = np.array([12.0, 3.2, -15, 88, 12.9])
npg.aggregate(group_idx, a, func=lambda x: " or ".join(map(str, x)), fill_value="", dtype=object)
# array(['3.2', '12.0 or -15.0 or 12.9', '', '', '88.0'], dtype=object)
```

**Integration into own jitted functions.** Every kernel is `@njit`-callable, so group statistics can sit inside a compiled loop of your own. The Numba module carries the kernels as its attributes — they are the one thing jitted code can reach, since Numba cannot reach the attributes of a Python object:

```python
import numba as nb
import numpy as np
import numpy_groupies.aggregate_numba as npg

@nb.njit
def zscore(group_idx, a):
    mean = npg.mean(group_idx, a)
    std = npg.std(group_idx, a)
    scale = np.where(std > 0, std, 1.0)
    return (a - mean[group_idx]) / scale[group_idx]
```

The same call from Python reaches the very same compiled driver — the same arguments, the same result, only no jitted caller around it:

```python
npg.mean(group_idx, a)              # the mean of every group
npg.std(group_idx, a, ddof=1)       # the argument the function has of its own, by its own name
```

A kernel is a lean thing: it takes flat 1-D input with integer indices and runs none of the validation and dtype resolution of `aggregate()` — the narrow integer sum always widens to `int64`, and `dtype`/`order`/`axis`/`out` are not its business. `aggregate()` — `npg.aggregate` of the Numba module, or `npg.aggregate` of the package once Numba is installed — is the one that knows all of that.

**Multi-dimensional output.** Give `group_idx` one row per output dimension — here, 1000 values binned into a 15×15×15 cube:

```python
group_idx = np.random.randint(0, 15, size=(3, 1000))
a = np.random.random(1000)
cube = npg.aggregate(group_idx, a, func="sum", size=(15, 15, 15), order="F")
cube.shape  # (15, 15, 15)
np.isfortran(cube)  # True
```

**Aggregating along an axis** of an N-D array: the groups run along `axis`, everything else is carried along.

```python
a = np.array([[99, 2, 11, 14, 20], [33, 76, 12, 100, 71], [67, 10, -8, 1, 9]])
group_idx = np.array([3, 3, 7, 0, 0])  # one label per column

npg.aggregate(group_idx, a, axis=1)
# array([[ 34,   0,   0, 101,   0,   0,   0,  11],
#        [171,   0,   0, 109,   0,   0,   0,  12],
#        [ 10,   0,   0,  77,   0,   0,   0,  -8]])
```

**Coming from pandas?** These are equivalent:

```python
import pandas as pd

group_idx = np.array([3, 0, 0, 1, 0, 3, 5, 5, 0, 4])
a = np.array([13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7])

npg.aggregate(group_idx, a, func="sum", fill_value=0)
# array([ 10. ,  -8.2,   0. ,  26.6,  53.7,  92.1])
pd.Series(a).groupby(group_idx).sum().reindex(range(group_idx.max() + 1), fill_value=0).to_numpy()
# array([ 10. ,  -8.2,   0. ,  26.6,  53.7,  92.1])
```

## Input forms

`aggregate(group_idx, a, func="sum", size=None, fill_value=DEFAULT, order="C", dtype=None, axis=None, ddof=0, ...)` accepts five combinations of input shapes:

| Form | `group_idx`                          | `a`        | `axis`  | Output                                          |
| :--: | ------------------------------------ | ---------- | ------- | ----------------------------------------------- |
|  1   | 1-D, length *n*                      | 1-D, len *n* | —     | 1-D, one value per group                        |
|  2   | 1-D                                  | scalar     | —       | like form 1, scalar broadcast (e.g. counting)   |
|  3   | 1-D, length `a.shape[axis]` (or broadcastable to `a.shape`) | N-D | required | `a`'s shape with `axis` replaced by the groups |
|  4   | 2-D, shape *(d, n)*                  | 1-D, len *n* | —     | *d*-dimensional; `group_idx[:, i]` is the position of `a[i]` |
|  5   | 2-D                                  | scalar     | —       | like form 4, scalar broadcast                   |

![Diagram of the five input forms](https://github.com/ml31415/numpy-groupies/raw/master/docs/diagrams/aggregate_dims.png)

Output size defaults to `max(group_idx) + 1` per dimension; pass `size=` to fix it. Full parameter descriptions, plus notes on memory layout and performance, are in the [reference](https://github.com/ml31415/numpy-groupies/blob/master/docs/functions.md).

## Functions

`func` can be a name, a NumPy function or builtin (`np.max`, `max`, `len`, …), or any callable. Optimised built-ins:

| Category                        | Functions |
| ------------------------------- | --------- |
| Reductions                      | `sum` `prod` `mean` `median` `quantile` `percentile` `var` `std` `skew` `kurt` `min` `max` `first` `last` `len` `nunique` `sumofsquares` `trapezoid` |
| Index of extremes               | `argmin` `argmax` |
| Boolean                         | `all` `any` `allnan` `anynan` |
| NaN-skipping variants           | `nansum` `nanmean` `nanvar` `nanmax` … — a `nan` prefix on every reduction above except `allnan`/`anynan` |
| Cumulative (size of input)      | `cumsum` `cumprod` `cummin` `cummax` |
| Sorting (size of input)         | `sort` (`reverse=True` for descending) |
| Collect group members           | `array` |

Not every implementation supports every function — the [function reference](https://github.com/ml31415/numpy-groupies/blob/master/docs/functions.md) has the exact semantics, default fill values, complex-number behaviour and a function × implementation matrix.

Every function is also a name of its own, with the function in the name instead of in the `func` argument. The name takes the arguments of its function — the `ddof` of the variance, the `q` of the quantile, the `dx` of the trapezoid, the `reverse` of the sort — and the rest like `aggregate` itself:

```python
npg.mean(group_idx, a)              # the best implementation available
npg.quantile(group_idx, a, q=0.9)   # the same as func="quantile", q=0.9
```

The same names are the attributes of the module of every implementation, so an implementation can be chosen and used the same way:

```python
import numpy_groupies.aggregate_numpy as npg
npg.mean(group_idx, a)              # the numpy implementation
```

The Numba module carries the kernels there, and a kernel is a leaner thing than the aggregate of the module (see [above](#quickstart)).

## Helper tools

`uaggregate(group_idx, a, ...)` takes the same arguments as `aggregate` but broadcasts the result back to the size of `a`, so every element carries the result of its own group — `aggregate(...)[group_idx]` in one call:

```python
group_idx = np.array([3, 0, 0, 1, 0, 3, 5, 5, 0, 4])
a = np.array([13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7])

npg.uaggregate(group_idx, a, func="mean")
# array([13.3 ,  2.5 ,  2.5 , -8.2 ,  2.5 , 13.3 , 46.05, 46.05,  2.5 , 53.7 ])

a - npg.uaggregate(group_idx, a, func="mean")  # demean within each group
```

The rest of the exports are small and situational: `unpack` (broadcast a per-group result back) and `unpack_into` (the same, into a preallocated array — Numba only), `step_count` and `step_indices` (runs of equal labels), `multi_arange`, `label_contiguous_1d`, `relabel_groups_unique` and `relabel_groups_masked` (close gaps in sparse labels), and `default_fill_value`. All of them are described in the [reference](https://github.com/ml31415/numpy-groupies/blob/master/docs/functions.md#helper-functions).

## Implementations

`from numpy_groupies import aggregate` picks the best available implementation: **numba** if installed, otherwise **numpy**. Every implementation is a module of its own, and it is called through its `aggregate` — its functions are its attributes as well:

```python
import numpy_groupies.aggregate_numba as npg   # numba
import numpy_groupies.aggregate_numpy as npg   # numpy
import numpy_groupies.aggregate_purepy as npg  # pure python

npg.aggregate(group_idx, a, func="mean")  # the aggregate of that implementation
npg.mean(group_idx, a)                    # one of its functions
```

In the Numba module, the functions are the kernels — the compiled drivers, which are leaner than the aggregate of the module and are the ones jitted code can reach.

The names the package has always carried name the same implementations directly: `aggregate_np` is the aggregate of the numpy implementation, `aggregate_nb` the one of the numba implementation (`None` where numba is not installed) and `aggregate_py` the one of the pure python implementation.

```python
from numpy_groupies import aggregate_nb as aggregate  # numba
from numpy_groupies import aggregate_np as aggregate  # numpy
from numpy_groupies import aggregate_py as aggregate  # pure python
```

All implementations share the calling syntax and produce the same results up to floating-point error, but some support only a subset of functions and raise `NotImplementedError` otherwise.

| Implementation | Needs        | Notes |
| -------------- | ------------ | ----- |
| **numba**      | numpy, numba | Fastest. Default if numba is installed. |
| **numpy**      | numpy        | Based on `np.bincount` and indexing tricks. Default without numba. |
| **pure python** | numpy        | Plain Python loops, no Numba. Very slow; useful as a fallback or for porting. |
| numpy ufunc    | numpy        | For benchmarking only: built on `ufunc.at` (`np.add.at`, …), in `numpy_groupies.benchmarks`. Incomplete. |
| pandas         | numpy, pandas | For benchmarking only: wraps `groupby` (in `numpy_groupies.benchmarks`). Skips NaN even in the plain functions (except `median`, `quantile`, `cumsum`). |
| polars         | numpy, polars | For benchmarking only: wraps `group_by` (in `numpy_groupies.benchmarks`). |

> **Gotchas:** with Numba installed, custom callables must be Numba-compilable (numeric code). For the cases that are not, call the NumPy implementation: `import numpy_groupies.aggregate_numpy as npg`, then `npg.aggregate(group_idx, a, func=my_callable)`.

## Performance

Best of three passes of five timed blocks of five runs each, in milliseconds, for 500,000 values in 1,000 groups (lower is better):

| function        | numpy  | numba  | numbagg | polars | pandas |
| --------------- | -----: | -----: | ------: | -----: | -----: |
| `sum`            |  0.917 |  0.454 |  0.463 |  2.421 |  6.960 |
| `mean`           |  1.512 |  0.491 |  0.531 |  2.665 |  7.430 |
| `max`            |  1.681 |  0.608 |  0.858 |  3.337 |  6.718 |
| `std`            |  3.060 |  0.653 |  0.680 |  2.481 |  7.676 |
| `nansum`         |  4.099 |  0.429 |  1.754 |  5.915 |  8.288 |
| `median`         | 11.730 |  9.345 |   ---- |  3.675 | 16.229 |
| `skew`            |  5.578 |  2.577 |   ---- |  3.349 |  9.642 |
| `kurt`            |  5.591 |  2.434 |   ---- |  4.125 | 10.717 |
| `quantile`       | 14.050 |  9.469 |   ---- |  4.180 | 22.721 |
| `nunique`        | 14.921 |  8.296 |   ---- | 11.896 | 58.062 |
| `sort`           | 70.184 | 24.658 |   ---- |   ---- |   ---- |
| `cumsum`         | 14.042 |  0.551 |   ---- |  8.483 |  7.283 |
| `array`          |  7.358 |  1.958 |   ---- |   ---- |   ---- |
| custom callable  | 111.000 |  5.961 |   ---- | 79.490 | 95.918 |

The `polars` column uses all the cores of the machine. The `numba` column uses them too, for the operations that can be reduced in chunks - the sums, the products, the counts, the extremes and the mean: their data is split into chunks that are reduced separately and folded afterwards. The operations that need the values of a group together (the order statistics) and the ones that give an output per value (the cumulatives) stay single-threaded. That matters most for the order statistics: with a single thread polars needs 9.7ms for the quantile, against 4.0ms with six, which puts it level with the single-threaded Numba kernel (9.3ms).

Absolute numbers depend on your machine, so compare the columns against each other rather than reading a single one as the price of the operation.

The complete table and the benchmark setup are in [docs/benchmarks.md](https://github.com/ml31415/numpy-groupies/blob/master/docs/benchmarks.md). To run it yourself, from the repository root — `--help` explains which columns to ask for:

```sh
python -m numpy_groupies.benchmarks --numbagg --pandas --polars
```

The `numpy` and `numba` columns above are not the whole story: NumPy 1.25 brought major [speed improvements to ufuncs](https://numpy.org/doc/stable/release/1.25.0-notes.html), which narrowed the gap between the NumPy and Numba implementations considerably. The authors hope that `ufunc.at` or an equivalent in NumPy or SciPy will eventually become fast enough to make this package redundant.

## Development

```sh
git clone https://github.com/ml31415/numpy-groupies
cd numpy-groupies
pip install -e ".[dev]"     # pytest, numba, pandas
pytest
```

The repository uses [pre-commit](https://pre-commit.com/) (`pre-commit install`). Release notes are on the [releases page](https://github.com/ml31415/numpy-groupies/releases).

## License

BSD 2-Clause — see [LICENSE.txt](LICENSE.txt).
