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
| sum          |        0.886 |        0.428 |        0.459 |        2.253 |        7.133 |        0.892 |
| prod         |        0.862 |        0.433 |        0.430 |        5.364 |        6.432 |        0.879 |
| min          |        1.669 |        0.614 |        0.829 |        3.024 |        6.823 |        1.669 |
| max          |        1.622 |        0.614 |        0.844 |        3.155 |        6.786 |        1.661 |
| len          |        0.769 |        0.353 |        0.450 |        2.517 |        5.441 |        0.916 |
| nunique      |       14.594 |        8.208 |         ---- |       11.443 |       57.750 |         ---- |
| all          |        1.724 |        0.806 |        1.588 |        2.706 |        7.218 |       32.402 |
| any          |        2.466 |        0.664 |        1.580 |        2.331 |        7.056 |       30.645 |
| anynan       |        0.831 |        0.616 |        2.664 |        2.683 |        6.954 |        5.249 |
| allnan       |        2.059 |        0.806 |        3.263 |        2.844 |        6.942 |        7.545 |
| mean         |        1.520 |        0.496 |        0.529 |        2.382 |        7.455 |         ---- |
| median       |       12.723 |        9.277 |         ---- |        3.782 |       15.696 |         ---- |
| quantile     |       14.280 |        9.279 |         ---- |        4.229 |       23.648 |         ---- |
| trapezoid    |        3.872 |        0.647 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.131 |        0.656 |        0.693 |        2.407 |        8.142 |         ---- |
| var          |        3.170 |        0.650 |        0.666 |        2.846 |        8.189 |         ---- |
| skew         |        5.861 |        2.556 |         ---- |         ---- |        9.649 |         ---- |
| kurt         |        5.658 |        2.479 |         ---- |         ---- |       11.008 |         ---- |
| first        |        1.765 |        0.294 |        0.291 |        2.225 |        6.540 |         ---- |
| last         |        0.837 |        0.354 |        0.420 |        2.513 |        6.907 |         ---- |
| argmax       |        2.580 |        0.607 |        0.896 |        6.542 |        7.473 |         ---- |
| argmin       |        4.844 |        0.579 |        0.812 |        6.990 |        7.587 |         ---- |
| nansum       |        4.141 |        0.443 |        1.765 |        6.324 |        8.492 |         ---- |
| nanprod      |        4.100 |        0.441 |        1.766 |        8.454 |        7.664 |         ---- |
| nanmin       |        4.782 |        0.947 |        1.998 |        6.334 |        8.428 |         ---- |
| nanmax       |        4.824 |        1.012 |        2.073 |        6.074 |        8.221 |         ---- |
| nanlen       |        2.392 |        0.436 |        1.704 |        5.998 |        7.285 |         ---- |
| nannunique   |       16.558 |        7.816 |         ---- |       14.003 |       29.644 |         ---- |
| nanall       |        4.894 |        1.042 |        3.270 |        6.053 |        8.524 |         ---- |
| nanany       |        5.599 |        0.638 |        2.672 |        6.067 |        8.477 |         ---- |
| nanmean      |        4.616 |        1.629 |        1.767 |        6.322 |        8.762 |         ---- |
| nanmedian    |       14.705 |        9.551 |         ---- |        7.278 |       14.451 |         ---- |
| nanquantile  |       17.119 |        9.484 |         ---- |        7.130 |       21.050 |         ---- |
| nantrapezoid |        6.648 |        1.744 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.250 |        1.861 |        1.926 |        5.890 |        8.936 |         ---- |
| nanstd       |        6.121 |        1.860 |        1.916 |        5.959 |        8.725 |         ---- |
| nanskew      |        8.144 |        3.267 |         ---- |         ---- |        9.336 |         ---- |
| nankurt      |        8.263 |        3.286 |         ---- |         ---- |       10.123 |         ---- |
| nanfirst     |        4.865 |        0.343 |        0.317 |        5.964 |        7.855 |         ---- |
| nanlast      |        4.164 |        0.438 |        1.592 |        5.976 |        8.357 |         ---- |
| nanargmin    |        6.247 |        0.723 |        1.944 |        6.715 |        8.404 |         ---- |
| nanargmax    |        3.889 |        0.774 |        1.988 |        6.680 |        8.812 |         ---- |
| cumsum       |       14.926 |        0.561 |         ---- |        8.484 |        7.678 |         ---- |
| cumprod      |         ---- |        0.562 |         ---- |        8.120 |        7.356 |         ---- |
| cummax       |         ---- |        0.753 |         ---- |        8.227 |        6.745 |         ---- |
| cummin       |         ---- |        0.769 |         ---- |        8.100 |        6.675 |         ---- |
| arbitrary    |      111.000 |        5.961 |         ---- |       79.490 |       95.918 |         ---- |
| sort         |       69.947 |       24.451 |         ---- |         ---- |         ---- |         ---- |
| array        |        6.504 |        1.878 |         ---- |         ---- |         ---- |         ---- |

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
