# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better. The first call of a participant in a process starts the threads of its pool, so it is made once per participant, before the measurement, and not counted.
- The `polars` column uses all the cores of the machine (6 threads here), the `numpy` and `numba` columns are single-threaded - the Numba kernels of this package do not use `prange`. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 9.7ms for the quantile, against 3.8ms and 4.1ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.3ms).
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number - 8.3ms for the Numba kernel against 14.5ms for the NumPy one, which has to sort the values of every group (numba's sort is about four times slower than numpy's, which vectorizes it).
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
| sum          |        0.943 |        0.473 |        0.466 |        2.658 |        6.969 |        0.908 |
| prod         |        0.873 |        0.457 |        0.440 |        5.769 |        6.069 |        0.865 |
| min          |        1.672 |        0.607 |        0.827 |        3.076 |        6.148 |        1.662 |
| max          |        1.671 |        0.628 |        0.850 |        2.705 |        6.416 |        1.659 |
| len          |        0.771 |        0.350 |        0.441 |        2.219 |        5.045 |        0.958 |
| nunique      |       14.456 |        8.324 |         ---- |       10.596 |       58.380 |         ---- |
| all          |        1.751 |        0.819 |        1.566 |        2.941 |        7.389 |       32.513 |
| any          |        2.416 |        0.654 |        1.595 |        2.831 |        6.995 |       30.793 |
| anynan       |        0.890 |        0.608 |        2.672 |        2.707 |        6.622 |        5.237 |
| allnan       |        2.029 |        0.833 |        3.258 |        2.774 |        6.749 |        7.585 |
| mean         |        1.488 |        0.491 |        0.538 |        2.575 |        7.105 |         ---- |
| median       |       12.785 |        9.325 |         ---- |        4.037 |       15.632 |         ---- |
| quantile     |       13.997 |        9.274 |         ---- |        4.066 |       22.808 |         ---- |
| trapezoid    |        3.698 |        0.633 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.065 |        0.660 |        0.674 |        2.766 |        7.560 |         ---- |
| var          |        3.049 |        0.650 |        0.667 |        2.620 |        7.263 |         ---- |
| first        |        1.738 |        0.288 |        0.300 |        2.396 |        6.175 |         ---- |
| last         |        0.849 |        0.359 |        0.417 |        2.637 |        6.742 |         ---- |
| argmax       |        2.507 |        0.602 |        0.885 |        6.424 |        7.165 |         ---- |
| argmin       |        4.829 |        0.569 |        0.823 |        6.168 |        7.088 |         ---- |
| nansum       |        4.042 |        0.449 |        1.758 |        6.239 |        8.955 |         ---- |
| nanprod      |        4.343 |        0.460 |        1.787 |        8.938 |        8.005 |         ---- |
| nanmin       |        4.826 |        0.947 |        1.998 |        6.240 |        8.124 |         ---- |
| nanmax       |        4.895 |        1.015 |        2.081 |        6.417 |        8.279 |         ---- |
| nanlen       |        2.471 |        0.428 |        1.707 |        6.512 |        7.175 |         ---- |
| nannunique   |       16.345 |        7.824 |         ---- |       13.253 |       29.234 |         ---- |
| nanall       |        4.892 |        1.047 |        3.276 |        6.191 |        8.430 |         ---- |
| nanany       |        5.700 |        0.648 |        2.663 |        6.229 |        8.725 |         ---- |
| nanmean      |        4.650 |        1.646 |        1.759 |        6.125 |        8.951 |         ---- |
| nanmedian    |       15.050 |        9.602 |         ---- |        7.601 |       14.772 |         ---- |
| nanquantile  |       17.460 |        9.513 |         ---- |        7.450 |       21.141 |         ---- |
| nantrapezoid |        6.532 |        1.761 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.097 |        1.856 |        1.907 |        6.198 |        9.006 |         ---- |
| nanstd       |        5.997 |        1.850 |        1.912 |        6.030 |        9.330 |         ---- |
| nanfirst     |        4.804 |        0.338 |        0.311 |        5.685 |        8.033 |         ---- |
| nanlast      |        4.098 |        0.438 |        1.602 |        5.588 |        8.131 |         ---- |
| nanargmin    |        6.053 |        0.743 |        1.932 |        6.340 |        8.494 |         ---- |
| nanargmax    |        3.863 |        0.777 |        1.996 |        6.459 |        8.463 |         ---- |
| cumsum       |       14.808 |        0.582 |         ---- |        8.518 |        7.454 |         ---- |
| cumprod      |         ---- |        0.572 |         ---- |        8.637 |        7.138 |         ---- |
| cummax       |         ---- |        0.711 |         ---- |        7.925 |        6.748 |         ---- |
| cummin       |         ---- |        0.734 |         ---- |        8.261 |        6.765 |         ---- |
| arbitrary    |      112.127 |        5.991 |         ---- |       79.784 |       94.001 |         ---- |
| sort         |       70.939 |       25.928 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.047 |        1.949 |         ---- |         ---- |         ---- |         ---- |

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
