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
| sum          |        0.919 |        0.439 |        0.901 |        0.459 |       16.053 |
| prod         |        0.883 |        0.474 |        0.882 |        0.469 |       15.363 |
| min          |        1.719 |        0.607 |        1.714 |        0.803 |       15.224 |
| max          |        1.744 |        0.622 |        1.699 |        0.874 |       15.833 |
| len          |        0.830 |        0.362 |        0.959 |        0.477 |       15.086 |
| all          |        1.820 |        0.793 |       32.965 |        1.573 |       16.437 |
| any          |        2.609 |        0.666 |       31.230 |        1.565 |       16.608 |
| anynan       |        0.868 |        0.605 |        5.324 |        2.686 |       15.910 |
| allnan       |        2.160 |        0.809 |        7.862 |        3.293 |       15.802 |
| mean         |        1.588 |        0.577 |         ---- |        0.552 |       16.371 |
| median       |       48.304 |       11.541 |         ---- |         ---- |       27.979 |
| trapezoid    |        3.953 |        0.656 |         ---- |         ---- |         ---- |
| std          |        3.212 |        0.724 |         ---- |        0.698 |       16.670 |
| var          |        3.140 |        0.669 |         ---- |        0.686 |       16.458 |
| first        |        1.761 |        0.318 |         ---- |        0.300 |       15.679 |
| last         |        1.000 |        0.393 |         ---- |        0.474 |       15.768 |
| argmax       |        2.646 |        0.621 |         ---- |        0.897 |       14.987 |
| argmin       |        5.005 |        0.593 |         ---- |        0.820 |       14.562 |
| nansum       |        4.157 |        0.448 |         ---- |        1.760 |       21.438 |
| nanprod      |        4.319 |        0.452 |         ---- |        1.770 |       21.030 |
| nanmin       |        4.965 |        0.949 |         ---- |        2.014 |       21.308 |
| nanmax       |        4.855 |        0.999 |         ---- |        2.078 |       21.198 |
| nanlen       |        2.583 |        0.446 |         ---- |        1.726 |       20.939 |
| nanall       |        4.986 |        1.052 |         ---- |        3.294 |       22.110 |
| nanany       |        5.715 |        0.637 |         ---- |        2.675 |       21.507 |
| nanmean      |        4.879 |        1.652 |         ---- |        1.757 |       21.891 |
| nanmedian    |       43.631 |       10.752 |         ---- |         ---- |       30.445 |
| nantrapezoid |        6.525 |        1.751 |         ---- |         ---- |         ---- |
| nanvar       |        6.098 |        1.860 |         ---- |        1.901 |       21.978 |
| nanstd       |        6.103 |        1.850 |         ---- |        1.920 |       21.679 |
| nanfirst     |        4.893 |        0.370 |         ---- |        0.305 |       20.334 |
| nanlast      |        4.178 |        0.446 |         ---- |        1.598 |       21.074 |
| nanargmin    |        6.139 |        0.796 |         ---- |        1.950 |       15.376 |
| nanargmax    |        3.993 |        0.807 |         ---- |        2.052 |       16.154 |
| cumsum       |       44.652 |        0.616 |         ---- |         ---- |        7.265 |
| cumprod      |         ---- |        0.629 |         ---- |         ---- |       14.633 |
| cummax       |         ---- |        0.796 |         ---- |         ---- |       13.526 |
| cummin       |         ---- |        0.804 |         ---- |         ---- |       13.140 |
| arbitrary    |      143.205 |       42.987 |         ---- |         ---- |      109.294 |
| sort         |      137.691 |       26.210 |         ---- |         ---- |         ---- |
| array        |       36.524 |        2.074 |         ---- |         ---- |         ---- |

The `array` row is the collecting function, which hands out the values of every group instead of reducing them to one number. It is the one function where the two numpy-based backends differ by an order of magnitude: the NumPy implementation sorts the values by their group (a stable argsort of 500,000 labels), the Numba one counts the groups and places the values in their blocks - a counting sort, which is linear and needs no comparison at all. The other implementations have no such function.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks --sort --runs=3
```

| groups | values per group | numpy   | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 126.200 | 34.335 |
|   1000 |              500 | 143.293 | 26.167 |
|  10000 |               50 | 157.599 | 21.341 |
| 100000 |                5 | 167.970 | 12.095 |

NumPy slows down monotonically as the groups multiply, from 126 to 168 ms. The Numba column falls the whole way down, to 12 ms: with few large groups the comparison sort of a whole block is what costs, while many tiny groups are ordered by insertion sort in the flat array, without the per-group view that a library sort has to build - which is almost three times the speed of the five-thousand-value groups. The other implementations have no column here because none of them provides `sort`.
