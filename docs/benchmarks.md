# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better. The first call of a participant in a process starts the threads of its pool, so it is made once per participant, before the measurement, and not counted.
- The `polars` column uses all the cores of the machine (6 threads here), the `numpy` and `numba` columns are single-threaded - the Numba kernels of this package do not use `prange`. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 9.7ms for the quantile, against 3.8ms and 4.1ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.3ms).
- The `nunique` is the one function where the Numba kernel is slower than the NumPy one: counting the distinct values needs the values of a group in order, and numba's sort is about four times slower than numpy's, which vectorizes it - 23.7ms of the 25.9ms of the Numba kernel are the sort.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --polars --runs=3
```

The participants get the data as it is - they must not write into it, since that would change the results of the runs that follow. The benchmark checks the data after every function, undoes a change if there was one and reports the row as not measured.

`--funcs min,max` measures the named functions only, instead of all of them. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`. The `numbagg`, `pandas` and `polars` columns are the group-by machinery of those libraries, the `ufunc` column is numpy's own `ufunc.at` - the last three are reference implementations for the comparison, not backends of this package.

## Results

| function     |        numpy |        numba |      numbagg |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.923 |        0.458 |        0.455 |        2.740 |        7.278 |        0.902 |
| prod         |        0.932 |        0.455 |        0.443 |        5.722 |        6.548 |        0.876 |
| min          |        1.697 |        0.610 |        0.848 |        3.420 |        6.833 |        1.697 |
| max          |        1.706 |        0.619 |        0.881 |        2.959 |        6.689 |        1.710 |
| len          |        0.791 |        0.381 |        0.476 |        2.616 |        5.663 |        0.906 |
| nunique      |       14.464 |       25.852 |         ---- |       12.384 |       59.935 |         ---- |
| all          |        1.749 |        0.814 |        1.604 |        2.661 |        7.051 |       32.825 |
| any          |        2.450 |        0.691 |        1.577 |        2.788 |        7.206 |       31.431 |
| anynan       |        0.852 |        0.636 |        2.663 |        2.857 |        6.751 |        5.287 |
| allnan       |        2.069 |        0.831 |        3.285 |        2.988 |        7.197 |        7.563 |
| mean         |        1.483 |        0.499 |        0.522 |        2.773 |        7.886 |         ---- |
| median       |       13.141 |        9.467 |         ---- |        4.009 |       15.834 |         ---- |
| quantile     |       14.200 |        9.428 |         ---- |        3.724 |       23.724 |         ---- |
| trapezoid    |        3.809 |        0.654 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.086 |        0.652 |        0.673 |        2.267 |        7.838 |         ---- |
| var          |        3.095 |        0.652 |        0.665 |        2.293 |        7.920 |         ---- |
| first        |        1.780 |        0.303 |        0.305 |        2.471 |        6.471 |         ---- |
| last         |        0.879 |        0.369 |        0.426 |        2.591 |        6.800 |         ---- |
| argmax       |        2.572 |        0.606 |        0.874 |        7.096 |        7.781 |         ---- |
| argmin       |        4.904 |        0.575 |        0.846 |        7.265 |        7.482 |         ---- |
| nansum       |        4.171 |        0.439 |        1.755 |        6.258 |        8.688 |         ---- |
| nanprod      |        4.152 |        0.448 |        1.765 |        8.157 |        7.756 |         ---- |
| nanmin       |        5.051 |        0.955 |        2.017 |        6.315 |        8.597 |         ---- |
| nanmax       |        4.812 |        1.033 |        2.108 |        6.055 |        8.235 |         ---- |
| nanlen       |        2.426 |        0.444 |        1.687 |        5.893 |        7.467 |         ---- |
| nannunique   |       16.037 |       21.088 |         ---- |       13.611 |       30.605 |         ---- |
| nanall       |        4.986 |        1.046 |        3.286 |        6.393 |        8.687 |         ---- |
| nanany       |        5.761 |        0.628 |        2.676 |        6.313 |        8.824 |         ---- |
| nanmean      |        4.677 |        1.615 |        1.764 |        6.284 |        9.102 |         ---- |
| nanmedian    |       14.990 |        9.634 |         ---- |        7.307 |       15.272 |         ---- |
| nanquantile  |       17.191 |        9.552 |         ---- |        8.062 |       20.973 |         ---- |
| nantrapezoid |        6.591 |        1.757 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.175 |        1.848 |        1.923 |        6.714 |        9.439 |         ---- |
| nanstd       |        6.361 |        1.898 |        1.929 |        6.574 |        9.342 |         ---- |
| nanfirst     |        4.825 |        0.358 |        0.316 |        6.627 |        8.042 |         ---- |
| nanlast      |        4.253 |        0.445 |        1.594 |        6.166 |        8.469 |         ---- |
| nanargmin    |        6.193 |        0.763 |        1.988 |        6.386 |        8.634 |         ---- |
| nanargmax    |        3.885 |        0.799 |        2.017 |        6.364 |        8.537 |         ---- |
| cumsum       |       14.943 |        0.566 |         ---- |        8.337 |        7.666 |         ---- |
| cumprod      |         ---- |        0.568 |         ---- |        8.630 |        7.583 |         ---- |
| cummax       |         ---- |        0.782 |         ---- |        8.137 |        6.898 |         ---- |
| cummin       |         ---- |        0.741 |         ---- |        8.597 |        6.788 |         ---- |
| arbitrary    |      111.968 |        6.516 |         ---- |       80.095 |       95.051 |         ---- |
| sort         |       70.255 |       25.889 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.549 |        1.900 |         ---- |         ---- |         ---- |         ---- |

The `array` row is the collecting function, which hands out the values of every group instead of reducing them to one number. The two numpy-based backends differ by a factor of four here: the NumPy implementation sorts the values by their group (a stable argsort of 500,000 labels), the Numba one counts the groups and places the values in their blocks - a counting sort, which is linear and needs no comparison at all. The other implementations have no such function.

The `quantile` row is the order statistic next to the median, and it costs about as much as the median does: the Numba implementation groups the values with a counting sort and then selects the wanted quantile out of the block of every group, the NumPy one sorts the labels and partitions the block. polars sorts the group as well and is the fastest of them; pandas' groupby takes the longest, and the quantile is slower there than the median is.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks --sort --runs=3
```

| groups | values per group | numpy  | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 66.975 | 33.921 |
|   1000 |              500 | 70.681 | 26.099 |
|  10000 |               50 | 71.956 | 21.147 |
| 100000 |                5 | 162.923 | 12.007 |

NumPy slows down as the groups multiply, from 67 to 161 ms, and the last step costs the most: every group is ordered by a call of its own, and 100,000 groups mean 100,000 calls. The Numba column falls the whole way down, to 12 ms: with few large groups the comparison sort of a whole block is what costs, while many tiny groups are ordered by insertion sort in the flat array, without the per-group view that a library sort has to build - which is almost three times the speed of the five-thousand-value groups. The other implementations have no column here because none of them provides `sort`.
