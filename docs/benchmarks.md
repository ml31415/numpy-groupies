# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions after one untimed warm-up call, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --runs=3
```

See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     |        numpy |        numba |        ufunc |      numbagg |       pandas |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.929 |        0.447 |        0.894 |        0.458 |       15.274 |
| prod         |        0.882 |        0.451 |        0.881 |        0.436 |       15.047 |
| min          |        1.681 |        0.640 |        1.681 |        0.850 |       15.352 |
| max          |        1.677 |        0.621 |        1.658 |        0.855 |       15.252 |
| len          |        0.841 |        0.364 |        0.938 |        0.463 |       14.336 |
| all          |        1.785 |        0.829 |       32.841 |        1.580 |       15.698 |
| any          |        2.489 |        0.660 |       30.881 |        1.566 |       15.733 |
| anynan       |        0.883 |        0.608 |        5.301 |        2.659 |       15.342 |
| allnan       |        2.086 |        0.814 |        7.574 |        3.257 |       15.135 |
| mean         |        1.508 |        0.502 |         ---- |        0.565 |       16.321 |
| median       |       17.468 |       10.840 |         ---- |         ---- |       27.695 |
| trapezoid    |        3.809 |        0.653 |         ---- |         ---- |         ---- |
| std          |        3.175 |        0.671 |         ---- |        0.678 |       16.405 |
| var          |        3.105 |        0.679 |         ---- |        0.678 |       16.276 |
| first        |        1.792 |        0.299 |         ---- |        0.323 |       15.612 |
| last         |        0.857 |        0.374 |         ---- |        0.430 |       15.563 |
| argmax       |        2.667 |        0.617 |         ---- |        0.893 |       14.759 |
| argmin       |        4.923 |        0.572 |         ---- |        0.835 |       15.031 |
| nansum       |        4.197 |        0.457 |         ---- |        1.775 |       20.525 |
| nanprod      |        4.121 |        0.464 |         ---- |        1.779 |       19.728 |
| nanmin       |        4.827 |        0.949 |         ---- |        2.004 |       20.404 |
| nanmax       |        4.926 |        1.033 |         ---- |        2.078 |       20.864 |
| nanlen       |        2.461 |        0.434 |         ---- |        1.715 |       19.386 |
| nanall       |        4.954 |        1.062 |         ---- |        3.251 |       21.167 |
| nanany       |        5.737 |        0.654 |         ---- |        2.671 |       21.146 |
| nanmean      |        4.723 |        1.637 |         ---- |        1.755 |       20.405 |
| nanmedian    |       18.970 |       10.854 |         ---- |         ---- |       29.540 |
| nantrapezoid |        6.791 |        1.764 |         ---- |         ---- |         ---- |
| nanvar       |        5.951 |        1.874 |         ---- |        1.913 |       21.803 |
| nanstd       |        6.269 |        1.868 |         ---- |        1.929 |       21.341 |
| nanfirst     |        4.918 |        0.370 |         ---- |        0.319 |       20.174 |
| nanlast      |        4.357 |        0.457 |         ---- |        1.615 |       20.022 |
| nanargmin    |        6.151 |        0.762 |         ---- |        1.954 |       15.720 |
| nanargmax    |        3.906 |        0.804 |         ---- |        2.014 |       15.853 |
| cumsum       |       15.179 |        0.614 |         ---- |         ---- |        7.656 |
| cumprod      |         ---- |        0.630 |         ---- |         ---- |       15.745 |
| cummax       |         ---- |        0.809 |         ---- |         ---- |       14.784 |
| cummin       |         ---- |        0.778 |         ---- |         ---- |       14.752 |
| arbitrary    |      111.410 |        5.955 |         ---- |         ---- |      108.106 |
| sort         |       70.907 |       24.716 |         ---- |         ---- |         ---- |
| array        |        7.165 |        1.911 |         ---- |         ---- |         ---- |

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
