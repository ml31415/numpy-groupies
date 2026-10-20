# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better. The first call of a participant in a process starts the threads of its pool, so it is made once per participant, before the measurement, and not counted.
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
| sum          |        0.908 |        0.428 |        0.452 |        2.669 |        7.223 |        0.908 |
| prod         |        0.918 |        0.472 |        0.440 |        5.717 |        6.408 |        0.868 |
| min          |        1.727 |        0.616 |        0.849 |        3.573 |        6.809 |        1.688 |
| max          |        1.731 |        0.612 |        0.865 |        3.416 |        7.092 |        1.693 |
| len          |        0.803 |        0.355 |        0.450 |        2.484 |        5.909 |        0.947 |
| all          |        1.809 |        0.814 |        1.590 |        2.757 |        7.215 |       32.773 |
| any          |        2.496 |        0.663 |        1.594 |        2.750 |        6.983 |       30.802 |
| anynan       |        0.861 |        0.616 |        2.680 |        2.647 |        6.776 |        5.335 |
| allnan       |        2.083 |        0.836 |        3.286 |        2.665 |        6.920 |        7.681 |
| mean         |        1.476 |        0.497 |        0.534 |        2.563 |        7.383 |         ---- |
| median       |       11.883 |        9.473 |         ---- |        4.149 |       16.084 |         ---- |
| quantile     |       14.548 |        9.305 |         ---- |        4.190 |       23.094 |         ---- |
| nanquantile  |       16.672 |        9.511 |         ---- |        7.394 |       21.027 |         ---- |
| trapezoid    |        3.811 |        0.628 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.091 |        0.672 |        0.688 |        2.970 |        8.282 |         ---- |
| var          |        3.250 |        0.675 |        0.667 |        3.067 |        8.034 |         ---- |
| first        |        1.737 |        0.308 |        0.322 |        2.570 |        6.373 |         ---- |
| last         |        0.873 |        0.365 |        0.419 |        2.729 |        7.027 |         ---- |
| argmax       |        2.666 |        0.624 |        0.893 |        7.558 |        7.331 |         ---- |
| argmin       |        4.905 |        0.590 |        0.822 |        7.408 |        7.401 |         ---- |
| nansum       |        4.158 |        0.437 |        1.763 |        6.204 |        8.554 |         ---- |
| nanprod      |        4.154 |        0.473 |        1.752 |        8.928 |        7.987 |         ---- |
| nanmin       |        4.690 |        0.939 |        1.993 |        6.168 |        8.554 |         ---- |
| nanmax       |        4.842 |        1.019 |        2.085 |        6.226 |        8.674 |         ---- |
| nanlen       |        2.447 |        0.454 |        1.724 |        6.038 |        7.293 |         ---- |
| nanall       |        4.888 |        1.062 |        3.277 |        6.124 |        8.921 |         ---- |
| nanany       |        5.625 |        0.664 |        2.672 |        6.109 |        8.858 |         ---- |
| nanmean      |        4.663 |        1.629 |        1.771 |        6.248 |        9.127 |         ---- |
| nanmedian    |       14.624 |        9.642 |         ---- |        7.194 |       15.814 |         ---- |
| nantrapezoid |        6.730 |        1.769 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.401 |        1.875 |        1.925 |        6.562 |        9.734 |         ---- |
| nanstd       |        6.156 |        1.886 |        1.949 |        6.513 |        9.397 |         ---- |
| nanfirst     |        4.849 |        0.361 |        0.313 |        5.880 |        8.741 |         ---- |
| nanlast      |        4.370 |        0.459 |        1.624 |        5.935 |        8.644 |         ---- |
| nanargmin    |        6.173 |        0.755 |        1.943 |        6.692 |        8.726 |         ---- |
| nanargmax    |        3.928 |        0.802 |        2.015 |        6.523 |        8.545 |         ---- |
| cumsum       |       14.506 |        0.553 |         ---- |        8.856 |        7.723 |         ---- |
| cumprod      |         ---- |        0.599 |         ---- |        8.759 |        7.272 |         ---- |
| cummax       |         ---- |        0.733 |         ---- |        8.805 |        6.591 |         ---- |
| cummin       |         ---- |        0.737 |         ---- |        8.605 |        7.004 |         ---- |
| arbitrary    |      110.928 |        5.971 |         ---- |       78.982 |       94.127 |         ---- |
| sort         |       70.602 |       25.783 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.137 |        1.887 |         ---- |         ---- |         ---- |         ---- |

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
