# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed blocks of five executions, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better. The executions of a cell come in a block, and an untimed call precedes every block: the threads of a pool spin for a moment after their region and sleep afterwards, and a single call after that would measure the waking up instead of the work.
- The `polars` column uses all the cores of the machine (6 threads here). The `numba` column uses them too, for the operations that can be reduced in chunks - the sums, the products, the counts, the extremes and the mean: their data is split into chunks that are reduced separately and folded afterwards. The operations that need the values of a group together (the order statistics) and the ones that give an output per value (the cumulatives) stay single-threaded. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 9.7ms for the quantile, against 3.8ms and 4.1ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.3ms).
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number - 8.3ms for the Numba kernel against 14.5ms for the NumPy one, which has to sort the values of every group (numba's sort is about four times slower than numpy's, which vectorizes it).
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --polars --runs=3
```

The participants get the data as it is - they must not write into it, since that would change the results of the runs that follow. The benchmark checks the data after every function, undoes a change if there was one and reports the row as not measured.

`--funcs min,max` measures the named functions only, instead of all of them. `--threads N` sets the number of threads the numba kernels use - they split their data into as many chunks as there are threads. Without it they use the physical cores of the machine: the hardware threads that share a core bring much less than a core, and on some machines they cost. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`. The `numbagg`, `pandas` and `polars` columns are the group-by machinery of those libraries, the `ufunc` column is numpy's own `ufunc.at` - the last three are reference implementations for the comparison, not backends of this package.

## Results

| function     |        numpy |        numba |      numbagg |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.917 |        0.454 |        0.463 |        2.421 |        6.960 |        0.884 |
| prod         |        0.850 |        0.429 |        0.422 |        5.213 |        6.145 |        0.863 |
| min          |        1.635 |        0.607 |        0.821 |        3.015 |        6.424 |        1.672 |
| max          |        1.681 |        0.608 |        0.858 |        3.337 |        6.718 |        1.660 |
| len          |        0.771 |        0.353 |        0.449 |        2.197 |        5.294 |        0.906 |
| nunique      |       14.921 |        8.296 |         ---- |       11.896 |       58.062 |         ---- |
| all          |        1.681 |        0.797 |        1.553 |        2.661 |        6.855 |       32.417 |
| any          |        2.473 |        0.668 |        1.570 |        2.311 |        7.041 |       30.750 |
| anynan       |        0.816 |        0.613 |        2.678 |        2.534 |        6.896 |        5.146 |
| allnan       |        2.040 |        0.816 |        3.255 |        2.352 |        6.722 |        7.610 |
| mean         |        1.512 |        0.491 |        0.531 |        2.665 |        7.430 |         ---- |
| median       |       11.730 |        9.345 |         ---- |        3.675 |       16.229 |         ---- |
| quantile     |       14.050 |        9.469 |         ---- |        4.180 |       22.721 |         ---- |
| trapezoid    |        3.737 |        0.638 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.060 |        0.653 |        0.680 |        2.481 |        7.676 |         ---- |
| var          |        3.013 |        0.653 |        0.675 |        2.518 |        7.974 |         ---- |
| skew         |        5.578 |        2.577 |         ---- |        3.349 |        9.642 |         ---- |
| kurt         |        5.591 |        2.434 |         ---- |        4.125 |       10.717 |         ---- |
| first        |        1.731 |        0.299 |        0.302 |        2.375 |        6.412 |         ---- |
| last         |        0.826 |        0.349 |        0.416 |        2.494 |        6.620 |         ---- |
| argmax       |        2.475 |        0.597 |        0.883 |        7.249 |        7.481 |         ---- |
| argmin       |        4.822 |        0.589 |        0.834 |        7.020 |        7.253 |         ---- |
| nansum       |        4.099 |        0.429 |        1.754 |        5.915 |        8.288 |         ---- |
| nanprod      |        4.055 |        0.453 |        1.752 |        8.006 |        7.858 |         ---- |
| nanmin       |        4.699 |        0.953 |        1.996 |        6.015 |        8.129 |         ---- |
| nanmax       |        4.766 |        1.003 |        2.082 |        6.111 |        8.124 |         ---- |
| nanlen       |        2.425 |        0.435 |        1.706 |        5.712 |        7.326 |         ---- |
| nannunique   |       16.183 |        7.880 |         ---- |       13.522 |       28.775 |         ---- |
| nanall       |        4.812 |        1.040 |        3.269 |        5.713 |        8.453 |         ---- |
| nanany       |        5.515 |        0.645 |        2.647 |        6.137 |        8.585 |         ---- |
| nanmean      |        4.585 |        1.649 |        1.757 |        5.883 |        8.924 |         ---- |
| nanmedian    |       14.084 |        9.539 |         ---- |        6.598 |       14.731 |         ---- |
| nanquantile  |       16.588 |        9.404 |         ---- |        6.946 |       20.653 |         ---- |
| nantrapezoid |        6.424 |        1.761 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        5.967 |        1.852 |        1.902 |        5.999 |        9.045 |         ---- |
| nanstd       |        5.912 |        1.859 |        1.913 |        6.016 |        9.045 |         ---- |
| nanskew      |        8.190 |        3.248 |         ---- |        6.466 |        9.589 |         ---- |
| nankurt      |        8.282 |        3.293 |         ---- |        6.247 |       10.530 |         ---- |
| nanfirst     |        4.816 |        0.342 |        0.305 |        5.636 |        8.135 |         ---- |
| nanlast      |        4.107 |        0.443 |        1.601 |        5.932 |        8.266 |         ---- |
| nanargmin    |        6.110 |        0.740 |        1.939 |        6.433 |        8.465 |         ---- |
| nanargmax    |        3.831 |        0.769 |        1.908 |        6.384 |        8.624 |         ---- |
| cumsum       |       14.042 |        0.551 |         ---- |        8.483 |        7.283 |         ---- |
| cumprod      |         ---- |        0.588 |         ---- |        8.246 |        6.916 |         ---- |
| cummax       |         ---- |        0.730 |         ---- |        7.687 |        6.500 |         ---- |
| cummin       |         ---- |        0.730 |         ---- |        8.354 |        6.357 |         ---- |
| arbitrary    |      111.158 |        6.027 |         ---- |       77.851 |       93.918 |         ---- |
| sort         |       70.184 |       24.658 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.358 |        1.958 |         ---- |         ---- |         ---- |         ---- |

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
