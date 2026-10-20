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
| sum          |        0.917 |        0.445 |        0.480 |        2.676 |        6.889 |        0.857 |
| prod         |        0.856 |        0.431 |        0.442 |        5.352 |        6.100 |        0.869 |
| min          |        1.672 |        0.602 |        0.810 |        3.427 |        6.605 |        1.624 |
| max          |        1.639 |        0.598 |        0.856 |        2.812 |        6.611 |        1.658 |
| len          |        0.803 |        0.363 |        0.455 |        2.251 |        5.389 |        0.908 |
| all          |        1.723 |        0.805 |        1.559 |        2.577 |        7.060 |       32.785 |
| any          |        2.447 |        0.652 |        1.573 |        2.421 |        6.863 |       30.269 |
| anynan       |        0.859 |        0.611 |        2.678 |        2.807 |        6.646 |        5.186 |
| allnan       |        2.010 |        0.804 |        3.247 |        2.556 |        6.655 |        7.561 |
| mean         |        1.491 |        0.483 |        0.546 |        2.690 |        7.321 |         ---- |
| median       |       11.011 |        9.279 |         ---- |        3.717 |       15.598 |         ---- |
| trapezoid    |        3.707 |        0.637 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.028 |        0.659 |        0.685 |        2.169 |        7.644 |         ---- |
| var          |        3.076 |        0.647 |        0.672 |        2.732 |        7.910 |         ---- |
| first        |        1.735 |        0.292 |        0.301 |        2.466 |        6.420 |         ---- |
| last         |        0.853 |        0.371 |        0.430 |        2.468 |        7.087 |         ---- |
| argmax       |        2.520 |        0.602 |        0.899 |        6.532 |        7.407 |         ---- |
| argmin       |        4.840 |        0.570 |        0.835 |        5.809 |        7.372 |         ---- |
| nansum       |        4.054 |        0.441 |        1.752 |        5.880 |        8.247 |         ---- |
| nanprod      |        4.067 |        0.432 |        1.757 |        8.369 |        7.737 |         ---- |
| nanmin       |        4.697 |        0.942 |        2.000 |        6.144 |        8.258 |         ---- |
| nanmax       |        4.769 |        1.014 |        2.079 |        6.210 |        8.245 |         ---- |
| nanlen       |        2.461 |        0.460 |        1.727 |        5.949 |        7.075 |         ---- |
| nanall       |        4.920 |        1.066 |        3.252 |        5.857 |        8.812 |         ---- |
| nanany       |        5.576 |        0.637 |        2.664 |        6.222 |        8.605 |         ---- |
| nanmean      |        4.636 |        1.642 |        1.746 |        6.101 |        8.714 |         ---- |
| nanmedian    |       14.410 |        9.593 |         ---- |        7.779 |       15.024 |         ---- |
| nantrapezoid |        6.441 |        1.747 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        5.992 |        1.843 |        1.909 |        6.570 |        9.113 |         ---- |
| nanstd       |        6.045 |        1.868 |        1.922 |        6.281 |        9.042 |         ---- |
| nanfirst     |        4.793 |        0.335 |        0.309 |        5.884 |        8.129 |         ---- |
| nanlast      |        4.217 |        0.457 |        1.610 |        6.122 |        8.219 |         ---- |
| nanargmin    |        6.043 |        0.749 |        1.946 |        7.095 |        8.708 |         ---- |
| nanargmax    |        3.827 |        0.805 |        2.003 |        6.281 |        8.680 |         ---- |
| cumsum       |       14.476 |        0.600 |         ---- |        7.939 |        7.225 |         ---- |
| cumprod      |         ---- |        0.551 |         ---- |        7.993 |        7.222 |         ---- |
| cummax       |         ---- |        0.761 |         ---- |        8.096 |        6.978 |         ---- |
| cummin       |         ---- |        0.751 |         ---- |        8.628 |        6.345 |         ---- |
| arbitrary    |      111.390 |        5.898 |         ---- |       80.972 |       96.786 |         ---- |
| sort         |       70.287 |       25.720 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.537 |        1.898 |         ---- |         ---- |         ---- |         ---- |

The `array` row is the collecting function, which hands out the values of every group instead of reducing them to one number. The two numpy-based backends differ by a factor of four here: the NumPy implementation sorts the values by their group (a stable argsort of 500,000 labels), the Numba one counts the groups and places the values in their blocks - a counting sort, which is linear and needs no comparison at all. The other implementations have no such function.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks --sort --runs=3
```

| groups | values per group | numpy  | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 67.564 | 34.207 |
|   1000 |              500 | 70.504 | 25.737 |
|  10000 |               50 | 71.398 | 20.348 |
| 100000 |                5 | 161.912 | 11.464 |

NumPy slows down as the groups multiply, from 67 to 161 ms, and the last step costs the most: every group is ordered by a call of its own, and 100,000 groups mean 100,000 calls. The Numba column falls the whole way down, to 12 ms: with few large groups the comparison sort of a whole block is what costs, while many tiny groups are ordered by insertion sort in the flat array, without the per-group view that a library sort has to build - which is almost three times the speed of the five-thousand-value groups. The other implementations have no column here because none of them provides `sort`.
