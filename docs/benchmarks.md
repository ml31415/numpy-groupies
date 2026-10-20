# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions after one untimed warm-up call, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --polars --runs=3
```

`--funcs min,max` measures the named functions only, instead of all of them. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`. The `numbagg`, `pandas` and `polars` columns are the group-by machinery of those libraries, the `ufunc` column is numpy's own `ufunc.at` - the last three are reference implementations for the comparison, not backends of this package.

## Results

| function     |        numpy |        numba |      numbagg |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.931 |        0.438 |        0.463 |        4.076 |       15.744 |        0.886 |
| prod         |        0.882 |        0.438 |        0.432 |        6.114 |       15.007 |        0.866 |
| min          |        1.637 |        0.621 |        0.807 |        5.289 |       15.756 |        1.699 |
| max          |        1.692 |        0.614 |        0.857 |        5.535 |       15.709 |        1.686 |
| len          |        0.814 |        0.366 |        0.456 |        3.507 |       15.091 |        0.943 |
| all          |        1.785 |        0.800 |        1.584 |        3.651 |       15.746 |       32.497 |
| any          |        2.441 |        0.651 |        1.568 |        3.997 |       15.975 |       30.533 |
| anynan       |        0.861 |        0.616 |        2.661 |        3.822 |       14.996 |        5.149 |
| allnan       |        2.057 |        0.816 |        3.277 |        3.628 |       15.019 |        7.613 |
| mean         |        1.491 |        0.503 |        0.540 |        4.124 |       16.212 |         ---- |
| median       |       17.574 |       10.512 |         ---- |        5.505 |       27.962 |         ---- |
| trapezoid    |        3.796 |        0.636 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.104 |        0.662 |        0.683 |        4.578 |       16.649 |         ---- |
| var          |        3.091 |        0.671 |        0.673 |        4.340 |       17.238 |         ---- |
| first        |        1.772 |        0.300 |        0.311 |        3.797 |       15.071 |         ---- |
| last         |        0.859 |        0.371 |        0.421 |        3.149 |       15.239 |         ---- |
| argmax       |        2.620 |        0.617 |        0.905 |        7.262 |       14.924 |         ---- |
| argmin       |        4.865 |        0.577 |        0.840 |        7.246 |       15.047 |         ---- |
| nansum       |        4.184 |        0.464 |        1.751 |        7.008 |       21.679 |         ---- |
| nanprod      |        4.181 |        0.451 |        1.750 |        9.102 |       21.037 |         ---- |
| nanmin       |        4.889 |        0.959 |        2.005 |        7.192 |       21.168 |         ---- |
| nanmax       |        4.783 |        1.004 |        2.086 |        7.691 |       21.345 |         ---- |
| nanlen       |        2.486 |        0.424 |        1.708 |        6.672 |       20.457 |         ---- |
| nanall       |        4.886 |        1.055 |        3.297 |        7.071 |       22.022 |         ---- |
| nanany       |        5.608 |        0.639 |        2.657 |        7.037 |       22.364 |         ---- |
| nanmean      |        4.675 |        1.643 |        1.767 |        7.542 |       21.838 |         ---- |
| nanmedian    |       18.855 |       10.518 |         ---- |        8.321 |       29.702 |         ---- |
| nantrapezoid |        6.604 |        1.755 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        5.994 |        1.871 |        1.917 |        7.699 |       22.066 |         ---- |
| nanstd       |        6.058 |        1.879 |        1.912 |        7.812 |       22.133 |         ---- |
| nanfirst     |        4.931 |        0.348 |        0.319 |        6.856 |       21.550 |         ---- |
| nanlast      |        4.159 |        0.442 |        1.608 |        6.997 |       21.588 |         ---- |
| nanargmin    |        6.230 |        0.724 |        1.924 |        6.472 |       16.298 |         ---- |
| nanargmax    |        3.963 |        0.815 |        2.011 |        6.705 |       16.540 |         ---- |
| cumsum       |       15.223 |        0.562 |         ---- |        9.234 |        7.877 |         ---- |
| cumprod      |         ---- |        0.564 |         ---- |        8.990 |       13.375 |         ---- |
| cummax       |         ---- |        0.793 |         ---- |        8.627 |       12.815 |         ---- |
| cummin       |         ---- |        0.780 |         ---- |        8.379 |       12.747 |         ---- |
| arbitrary    |      112.915 |        5.972 |         ---- |       76.901 |      108.469 |         ---- |
| sort         |       69.880 |       25.492 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.132 |        1.882 |         ---- |         ---- |         ---- |         ---- |

The `array` row is the collecting function, which hands out the values of every group instead of reducing them to one number. The two numpy-based backends differ by a factor of four here: the NumPy implementation sorts the values by their group (a stable argsort of 500,000 labels), the Numba one counts the groups and places the values in their blocks - a counting sort, which is linear and needs no comparison at all. The other implementations have no such function.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks --sort --runs=3
```

| groups | values per group | numpy   | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 72.917 | 34.187 |
|   1000 |              500 | 74.087 | 25.573 |
|  10000 |               50 | 75.860 | 20.499 |
| 100000 |                5 | 165.801 | 11.557 |

NumPy slows down as the groups multiply, from 73 to 166 ms, and the last step costs the most: every group is ordered by a call of its own, and 100,000 groups mean 100,000 calls. The Numba column falls the whole way down, to 12 ms: with few large groups the comparison sort of a whole block is what costs, while many tiny groups are ordered by insertion sort in the flat array, without the per-group view that a library sort has to build - which is almost three times the speed of the five-thousand-value groups. The other implementations have no column here because none of them provides `sort`.
