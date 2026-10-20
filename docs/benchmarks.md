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
| sum          |        0.899 |        0.446 |        0.451 |        2.735 |        6.703 |        0.853 |
| prod         |        0.908 |        0.444 |        0.434 |        5.766 |        5.662 |        0.875 |
| min          |        1.656 |        0.603 |        0.821 |        3.173 |        6.340 |        1.667 |
| max          |        1.642 |        0.596 |        0.844 |        3.490 |        6.419 |        1.728 |
| len          |        0.787 |        0.360 |        0.431 |        2.708 |        5.123 |        0.926 |
| all          |        1.751 |        0.833 |        1.594 |        3.099 |        6.511 |       32.621 |
| any          |        2.497 |        0.656 |        1.583 |        2.898 |        6.607 |       30.474 |
| anynan       |        0.838 |        0.624 |        2.671 |        2.897 |        6.671 |        5.334 |
| allnan       |        2.157 |        0.817 |        3.301 |        2.918 |        6.290 |        7.505 |
| mean         |        1.495 |        0.513 |        0.606 |        2.732 |        7.078 |         ---- |
| median       |       18.427 |       10.504 |         ---- |        3.793 |       15.726 |         ---- |
| trapezoid    |        3.738 |        0.632 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.124 |        0.664 |        0.673 |        2.963 |        7.333 |         ---- |
| var          |        3.091 |        0.667 |        0.696 |        2.686 |        7.012 |         ---- |
| first        |        1.816 |        0.298 |        0.290 |        2.849 |        6.425 |         ---- |
| last         |        0.931 |        0.377 |        0.470 |        2.566 |        6.031 |         ---- |
| argmax       |        2.602 |        0.602 |        0.887 |        7.779 |        7.410 |         ---- |
| argmin       |        4.939 |        0.602 |        0.859 |        7.010 |        6.884 |         ---- |
| nansum       |        4.162 |        0.429 |        1.782 |        6.291 |        8.607 |         ---- |
| nanprod      |        4.200 |        0.452 |        1.774 |        8.475 |        7.484 |         ---- |
| nanmin       |        4.736 |        0.969 |        2.013 |        6.262 |        7.857 |         ---- |
| nanmax       |        4.906 |        1.016 |        2.079 |        6.537 |        7.878 |         ---- |
| nanlen       |        2.502 |        0.454 |        1.705 |        6.763 |        6.893 |         ---- |
| nanall       |        5.016 |        1.043 |        3.244 |        6.564 |        8.381 |         ---- |
| nanany       |        5.697 |        0.648 |        2.674 |        5.934 |        8.304 |         ---- |
| nanmean      |        4.646 |        1.629 |        1.757 |        6.315 |        8.562 |         ---- |
| nanmedian    |       18.959 |       10.674 |         ---- |        7.178 |       15.062 |         ---- |
| nantrapezoid |        6.676 |        1.782 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.259 |        1.874 |        1.909 |        6.548 |        8.759 |         ---- |
| nanstd       |        6.027 |        1.849 |        1.926 |        6.226 |        8.778 |         ---- |
| nanfirst     |        4.811 |        0.356 |        0.320 |        5.803 |        7.961 |         ---- |
| nanlast      |        4.214 |        0.440 |        1.629 |        5.958 |        8.162 |         ---- |
| nanargmin    |        6.197 |        0.744 |        1.949 |        6.898 |        8.190 |         ---- |
| nanargmax    |        3.849 |        0.789 |        1.993 |        6.271 |        8.147 |         ---- |
| cumsum       |       15.320 |        0.552 |         ---- |        9.058 |        6.761 |         ---- |
| cumprod      |         ---- |        0.612 |         ---- |        9.527 |        7.386 |         ---- |
| cummax       |         ---- |        0.793 |         ---- |        9.203 |        6.538 |         ---- |
| cummin       |         ---- |        0.719 |         ---- |        8.886 |        6.225 |         ---- |
| arbitrary    |      111.830 |        6.069 |         ---- |       78.021 |       94.235 |         ---- |
| sort         |       70.663 |       25.811 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.313 |        1.908 |         ---- |         ---- |         ---- |         ---- |

The `array` row is the collecting function, which hands out the values of every group instead of reducing them to one number. The two numpy-based backends differ by a factor of four here: the NumPy implementation sorts the values by their group (a stable argsort of 500,000 labels), the Numba one counts the groups and places the values in their blocks - a counting sort, which is linear and needs no comparison at all. The other implementations have no such function.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks --sort --runs=3
```

| groups | values per group | numpy  | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 67.494 | 34.036 |
|   1000 |              500 | 70.983 | 25.770 |
|  10000 |               50 | 70.643 | 20.513 |
| 100000 |                5 | 161.413 | 11.896 |

NumPy slows down as the groups multiply, from 67 to 161 ms, and the last step costs the most: every group is ordered by a call of its own, and 100,000 groups mean 100,000 calls. The Numba column falls the whole way down, to 12 ms: with few large groups the comparison sort of a whole block is what costs, while many tiny groups are ordered by insertion sort in the flat array, without the per-group view that a library sort has to build - which is almost three times the speed of the five-thousand-value groups. The other implementations have no column here because none of them provides `sort`.
