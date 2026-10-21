# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better. The first call of a participant in a process starts the threads of its pool, so it is made once per participant, before the measurement, and not counted.
- The `polars` column uses all the cores of the machine (6 threads here), the `numpy` and `numba` columns are single-threaded - the Numba kernels of this package do not use `prange`. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 9.7ms for the quantile, against 3.8ms and 4.1ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.3ms).
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
| sum          |        0.895 |        0.432 |        0.457 |        2.613 |        7.085 |        0.863 |
| prod         |        0.868 |        0.429 |        0.425 |        4.949 |        6.410 |        0.864 |
| min          |        1.637 |        0.613 |        0.833 |        2.484 |        6.703 |        1.616 |
| max          |        1.649 |        0.618 |        0.861 |        2.871 |        6.716 |        1.633 |
| len          |        0.773 |        0.347 |        0.437 |        2.340 |        5.478 |        0.894 |
| all          |        1.726 |        0.800 |        1.579 |        2.280 |        7.087 |       32.822 |
| any          |        2.449 |        0.654 |        1.562 |        2.372 |        7.047 |       30.865 |
| anynan       |        0.850 |        0.616 |        2.680 |        2.915 |        6.832 |        5.227 |
| allnan       |        2.075 |        0.834 |        3.283 |        2.811 |        6.806 |        7.567 |
| mean         |        1.489 |        0.479 |        0.527 |        2.640 |        7.789 |         ---- |
| median       |       11.379 |        9.295 |         ---- |        4.149 |       15.801 |         ---- |
| quantile     |       13.801 |        9.273 |         ---- |        4.050 |       23.928 |         ---- |
| trapezoid    |        3.800 |        0.638 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.096 |        0.653 |        0.680 |        2.792 |        8.273 |         ---- |
| var          |        3.110 |        0.657 |        0.676 |        2.765 |        7.894 |         ---- |
| first        |        1.737 |        0.312 |        0.301 |        2.525 |        6.394 |         ---- |
| last         |        0.876 |        0.375 |        0.417 |        2.562 |        6.924 |         ---- |
| argmax       |        2.659 |        0.633 |        0.878 |        7.304 |        7.802 |         ---- |
| argmin       |        4.876 |        0.593 |        0.842 |        7.369 |        7.490 |         ---- |
| nansum       |        4.109 |        0.447 |        1.763 |        6.086 |        8.704 |         ---- |
| nanprod      |        4.176 |        0.457 |        1.745 |        8.366 |        8.057 |         ---- |
| nanmin       |        4.801 |        0.951 |        2.020 |        6.201 |        8.251 |         ---- |
| nanmax       |        4.865 |        1.001 |        2.073 |        6.360 |        8.315 |         ---- |
| nanlen       |        2.476 |        0.439 |        1.719 |        6.312 |        7.422 |         ---- |
| nanall       |        4.928 |        1.061 |        3.285 |        6.372 |        8.730 |         ---- |
| nanany       |        5.691 |        0.644 |        2.673 |        6.243 |        8.483 |         ---- |
| nanmean      |        4.661 |        1.637 |        1.755 |        6.278 |        9.025 |         ---- |
| nanmedian    |       13.848 |        9.493 |         ---- |        7.390 |       14.991 |         ---- |
| nanquantile  |       16.944 |        9.584 |         ---- |        7.701 |       21.416 |         ---- |
| nantrapezoid |        6.524 |        1.764 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.139 |        1.848 |        1.916 |        6.279 |        9.077 |         ---- |
| nanstd       |        6.136 |        1.878 |        1.949 |        6.448 |        9.212 |         ---- |
| nanfirst     |        4.822 |        0.363 |        0.318 |        5.936 |        8.239 |         ---- |
| nanlast      |        4.222 |        0.450 |        1.590 |        5.698 |        8.347 |         ---- |
| nanargmin    |        6.059 |        0.723 |        1.947 |        6.599 |        8.360 |         ---- |
| nanargmax    |        3.785 |        0.802 |        1.988 |        6.482 |        8.617 |         ---- |
| cumsum       |       14.403 |        0.567 |         ---- |        8.636 |        7.378 |         ---- |
| cumprod      |         ---- |        0.551 |         ---- |        8.634 |        7.316 |         ---- |
| cummax       |         ---- |        0.733 |         ---- |        8.154 |        6.850 |         ---- |
| cummin       |         ---- |        0.725 |         ---- |        8.420 |        6.633 |         ---- |
| arbitrary    |      110.980 |        6.010 |         ---- |       80.051 |       97.994 |         ---- |
| sort         |       70.855 |       25.910 |         ---- |         ---- |         ---- |         ---- |
| array        |        6.882 |        1.893 |         ---- |         ---- |         ---- |         ---- |

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
