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
a         = np.array([13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7])

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

The idea behind `aggregate` is an old one: Matlab's [`accumarray`](https://www.mathworks.com/help/matlab/ref/accumarray.html), the pandas [`groupby`](https://pandas.pydata.org/docs/user_guide/groupby.html), the [MapReduce](https://en.wikipedia.org/wiki/MapReduce) paradigm, or simply a [histogram](https://en.wikipedia.org/wiki/Histogram). What this package adds is a single, consistent array-in/array-out function that covers many reductions.

|                                    | `np.bincount` | `pandas.groupby` | `numpy_groupies.aggregate` |
| ---------------------------------- | :-----------: | :--------------: | :------------------------: |
| Works on bare NumPy arrays         |      yes      |  via a Series    |            yes             |
| Reductions beyond sum / count      |      no       |       yes        |            yes             |
| Multi-dimensional output (N-D bins)|      no       |   via MultiIndex |            yes             |
| Arbitrary labels (strings, ...)    |      no       |       yes        |   no — non-negative ints   |
| Speed on plain arrays              |     fast      |      slower      |       fast (see below)     |

Use it when your data already lives in NumPy arrays, your groups are (or can cheaply be turned into) integer labels, and you want group statistics without the overhead of building a DataFrame. If your labels are strings or other objects, map them to integers first:

```python
labels = np.array(["b", "a", "b", "c", "a"])
values = np.arange(5.0)
uniques, inverse = np.unique(labels, return_inverse=True)
npg.aggregate(inverse, values)       # array([5., 2., 3.])  ->  groups "a", "b", "c"
```

## Installation

```sh
pip install numpy_groupies            # NumPy implementation
pip install "numpy_groupies[fast]"    # + Numba for the fastest implementation
conda install -c conda-forge numpy_groupies
```

NumPy is the only declared dependency. Numba is optional and only the fastest implementation needs it; the pure-Python implementation runs without Numba but still imports NumPy (see [Implementations](#implementations)).

If you only want one implementation, you can copy a single file (e.g. `aggregate_numpy.py`) into your project: paste the contents of `utils.py` at its top, replacing the `from .utils import (...)` line.

## Quickstart

**Counting, and other reductions.** The default function is `sum`; pass a scalar to count items per group (this is what `np.bincount` does under the hood in the NumPy implementation).

```python
npg.aggregate(group_idx, 1)                     # items per group
npg.aggregate(group_idx, a, func="mean")        # nan for the empty group 2
npg.aggregate(group_idx, a, func="mean", fill_value=-1)
```

**Ignoring NaNs.** Every common reduction has a `nan…` variant, as in NumPy.

```python
g = np.array([0, 0, 0, 1, 1, 2])
v = np.array([1.0, 2.0, np.nan, 4.0, np.nan, np.nan])

npg.aggregate(g, v, func="nanmean")             # array([1.5, 4. , nan])
npg.aggregate(g, v, func="nanvar")              # array([0.25, 0.  , nan])
npg.aggregate(g, v, func="var")                 # array([nan, nan, nan])  (plain var propagates NaN)
```

**Cumulative functions** don't reduce; the output has the same length as the input.

```python
npg.aggregate(np.array([0, 1, 0, 1, 0]), np.array([1, 2, 3, 4, 5]), func="cumsum")
# array([1, 2, 4, 6, 9])
```

**Custom functions.** Any callable works. Non-numeric output needs `dtype=object`, and since the default fill value of a custom function cannot be guessed, give one explicitly. Use the NumPy implementation for this: Numba can only compile numeric functions, so with Numba installed the plain `aggregate` fails on the string example below.

```python
g = np.array([1, 0, 1, 4, 1])
v = np.array([12.0, 3.2, -15, 88, 12.9])
npg.aggregate_np(g, v, func=lambda x: " or ".join(map(str, x)), fill_value="", dtype=object)
# array(['3.2', '12.0 or -15.0 or 12.9', '', '', '88.0'], dtype=object)
```

**Multi-dimensional output.** Give `group_idx` one row per output dimension — here, 1000 values binned into a 15×15×15 cube:

```python
group_idx = np.random.randint(0, 15, size=(3, 1000))
a = np.random.random(1000)
cube = npg.aggregate(group_idx, a, func="sum", size=(15, 15, 15), order="F")
cube.shape            # (15, 15, 15)
np.isfortran(cube)    # True
```

**Aggregating along an axis** of an N-D array: the groups run along `axis`, everything else is carried along.

```python
a = np.array([[99,  2, 11,  14, 20],
              [33, 76, 12, 100, 71],
              [67, 10, -8,   1,  9]])
group_idx = np.array([3, 3, 7, 0, 0])             # one label per column

npg.aggregate(group_idx, a, axis=1)
# array([[ 34,   0,   0, 101,   0,   0,   0,  11],
#        [171,   0,   0, 109,   0,   0,   0,  12],
#        [ 10,   0,   0,  77,   0,   0,   0,  -8]])
```

**Coming from pandas?** These are equivalent:

```python
npg.aggregate(group_idx, a, func="sum", fill_value=0)
pd.Series(a).groupby(group_idx).sum().reindex(range(group_idx.max() + 1), fill_value=0).to_numpy()
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
| Reductions                      | `sum` `prod` `mean` `median` `var` `std` `min` `max` `first` `last` `len` `sumofsquares` `trapezoid` |
| Index of extremes               | `argmin` `argmax` |
| Boolean                         | `all` `any` `allnan` `anynan` |
| NaN-skipping variants           | `nansum` `nanmean` `nanvar` `nanmax` … — a `nan` prefix on every reduction above except `allnan`/`anynan` |
| Cumulative (size of input)      | `cumsum` `cumprod` `cummin` `cummax` |
| Sorting (size of input)         | `sort` (`reverse=True` for descending) |
| Collect group members           | `array` |

Not every implementation supports every function — for example `sort` and `array` are **not** available in the Numba implementation. The [function reference](https://github.com/ml31415/numpy-groupies/blob/master/docs/functions.md) has the exact semantics, default fill values, complex-number behaviour and a function × implementation matrix.

## Helper tools

Besides `aggregate`, the package exports a few small tools that tend to be useful around group operations.

### `uaggregate` — aggregate and broadcast back

Like `aggregate`, but the result is "unpacked" back to the length of the input, so each element receives the result of its group. This is `aggregate(...)[group_idx]` in one call — handy for normalising within groups.

```python
group_idx = np.array([3, 0, 0, 1, 0, 3, 5, 5, 0, 4])
a = np.array([13.2, 3.5, 3.5, -8.2, 3.0, 13.4, 99.2, -7.1, 0.0, 53.7])

npg.uaggregate(group_idx, a, func="mean")
# array([13.3 ,  2.5 ,  2.5 , -8.2 ,  2.5 , 13.3 , 46.05, 46.05,  2.5 , 53.7 ])

a - npg.uaggregate(group_idx, a, func="mean")     # demean within each group
```

With the Numba implementation you can pass `out=` to gather into a preallocated array (shape and dtype must match; other implementations raise `NotImplementedError`):

```python
out = np.empty_like(a)
npg.uaggregate(group_idx, a, func="max", out=out)
```

### `unpack` / `unpack_into`

`unpack(group_idx, ret)` expands a per-group result back to input length — it is simply `ret[group_idx]`. `unpack_into(group_idx, ret, out)` does the same into an existing array, using a jitted loop that is faster than fancy indexing for large 1-D inputs (Numba only).

### `step_count` and `step_indices` — find runs of equal labels *(Numba only)*

For a `group_idx` whose equal values are stored contiguously (e.g. data sorted by group), these find the run boundaries without sorting or hashing:

```python
group_idx = np.array([0, 0, 0, 2, 2, 5, 5, 5, 5, 1])

npg.step_count(group_idx)       # 4   -> number of runs
npg.step_indices(group_idx)     # array([ 0,  3,  5,  9, 10])   -> run edges, incl. start and end

edges = npg.step_indices(group_idx)
[group_idx[i:j] for i, j in zip(edges[:-1], edges[1:])]    # the four runs
```

Note that runs are counted as they appear: a label that shows up in two separate places counts as two runs.

### `multi_arange`

Concatenates `arange(n_i)` for every entry of `n` — a vectorised `np.hstack([np.arange(k) for k in n])`.

```python
npg.multi_arange(np.array([0, 0, 3, 0, 0, 2, 0, 2, 1]))
# array([0, 1, 2, 0, 1, 0, 1, 0])
```

Combined with `np.bincount` it gives the rank of each item *within* its group, when the data is sorted by group:

```python
npg.multi_arange(np.bincount(np.array([0, 0, 0, 1, 1, 2])))
# array([0, 1, 2, 0, 1, 0])
```

### `label_contiguous_1d`

Labels consecutive blocks with 1, 2, 3, … and leaves zeros/`False` as 0. For boolean input each block of `True` gets a label; for other dtypes each block of identical non-zero values does.

```python
npg.label_contiguous_1d(np.array([False, True, True, False, False, True]))
# array([0, 1, 1, 0, 0, 2])
npg.label_contiguous_1d(np.array([0, 3, 3, 0, 0, 5, 5, 5, 1, 1, 0, 2]))
# array([0, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 4])
```

The output is a ready-made `group_idx` for "aggregate over each run of …" questions.

### `relabel_groups_unique` / `relabel_groups_masked`

Output size is `max(group_idx) + 1`, so sparse labels waste memory. These functions close the gaps while preserving order:

```python
g = np.array([0, 3, 3, 3, 0, 2, 5, 2, 0, 1, 1, 0, 3, 5, 5])

npg.relabel_groups_unique(g)
# array([0, 3, 3, 3, 0, 2, 4, 2, 0, 1, 1, 0, 3, 4, 4])     label 4 was unused, so 5 -> 4

keep = np.array([0, 1, 0, 1, 1, 1])                         # drop group 2
npg.relabel_groups_masked(g, keep)
# array([0, 2, 2, 2, 0, 0, 4, 0, 0, 1, 1, 0, 2, 4, 4])     removed items become group 0
```

Group 0 plays a special role here: `keep[0]` is ignored, and removed groups are merged into 0.

### `default_fill_value`

`npg.default_fill_value(func, dtype=None)` returns the fill value `aggregate` would use for empty groups — useful when downstream code needs it without duplicating the [table](https://github.com/ml31415/numpy-groupies/blob/master/docs/functions.md#fill-values).

```python
npg.default_fill_value("max")            # nan
npg.default_fill_value("max", int)       # 0   (integers cannot hold nan)
npg.default_fill_value("argmax")         # -1
```

## Implementations

`from numpy_groupies import aggregate` picks the best available implementation: **numba** if installed, otherwise **numpy**. To choose explicitly:

```python
from numpy_groupies import aggregate_nb as aggregate      # numba
from numpy_groupies import aggregate_np as aggregate      # numpy
from numpy_groupies import aggregate_py as aggregate      # pure python
from numpy_groupies.aggregate_numpy import aggregate      # same, via the module
```

All implementations share the calling syntax and produce the same results up to floating-point error, but some support only a subset of functions and raise `NotImplementedError` otherwise.

| Implementation | Needs        | Notes |
| -------------- | ------------ | ----- |
| **numba**      | numpy, numba | Fastest. Default if numba is installed. Lacks `sort` and `array`. |
| **numpy**      | numpy        | Based on `np.bincount` and indexing tricks. Default without numba. Most complete. |
| **pure python** | numpy        | Plain Python loops, no Numba. Very slow; useful as a fallback or for porting. |
| numpy ufunc    | numpy        | For benchmarking only: built on `ufunc.at` (`np.add.at`, …). Incomplete. |
| pandas         | numpy, pandas | For reference only: wraps `groupby`. Skips NaN even in the plain functions (except `median`, `cumsum`). |

> **Gotchas:** with Numba installed, `aggregate(..., func="sort")` and `func="array"` raise `NotImplementedError`, and custom callables must be Numba-compilable (numeric code). For those cases call the NumPy implementation: `npg.aggregate_np(group_idx, a, func="sort")`.

## Performance

Best of five timed runs, in milliseconds, for 500,000 values in 1,000 groups (lower is better), taken from the maintainers' benchmark on an Intel i7-7560U laptop CPU (Linux, Python 3.14, NumPy 2.5, Numba 0.67, pandas 3.0):

| function | numpy  | numba  | pandas |
| -------- | -----: | -----: | -----: |
| `sum`    |   1.24 |   0.72 |  13.76 |
| `mean`   |   1.82 |   0.99 |  13.91 |
| `max`    |   2.76 |   0.88 |  13.11 |
| `std`    |   4.18 |   1.14 |  14.98 |
| `nansum` |   5.13 |   1.69 |  18.96 |
| `median` |  53.34 |  11.71 |  24.28 |
| `cumsum` |  53.28 |   1.15 |  13.43 |
| custom callable | 161.54 | 50.94 | 131.86 |

In short: the NumPy implementation is about 4–12× faster than pandas on common reductions, and Numba adds another 1.5–5× on top — and about 46× for `cumsum`. `median` and custom callables are the expensive cases everywhere. Absolute numbers depend on your machine, so treat the ratios as the takeaway.

The complete table and the benchmark setup are in [docs/benchmarks.md](https://github.com/ml31415/numpy-groupies/blob/master/docs/benchmarks.md). To run it yourself, from the repository root (add `--pandas` for the pandas column, `--purepy` for the pure-Python one):

```sh
python -m numpy_groupies.benchmarks.generic
```

The `numpy` and `numba` columns above are not the whole story: NumPy 1.25 brought major [speed improvements to ufuncs](https://numpy.org/doc/stable/release/1.25.0-notes.html), which narrowed the gap between the NumPy and Numba implementations considerably. The authors hope that `ufunc.at` or an equivalent in NumPy or SciPy will eventually become fast enough to make this package redundant.

## Development

```sh
git clone https://github.com/ml31415/numpy-groupies
cd numpy-groupies
pip install -e ".[dev]"     # pytest, numba, pandas
pytest
```

The repository uses [pre-commit](https://pre-commit.com/) (`pre-commit install`). Bug reports and pull requests are welcome in the [issue tracker](https://github.com/ml31415/numpy-groupies/issues). Release notes are on the [releases page](https://github.com/ml31415/numpy-groupies/releases).

**Credits.** Started by [@ml31415](https://github.com/ml31415), who wrote the Numba implementation; the pure-Python and NumPy implementations were written by [@d1manson](https://github.com/d1manson). Currently maintained by Deepak Cherian ([@dcherian](https://github.com/dcherian)).

## License

BSD 2-Clause — see [LICENSE.txt](LICENSE.txt).
