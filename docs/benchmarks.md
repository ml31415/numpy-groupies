# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, taking the minimum over 5 timed runs after one untimed warm-up call (`repeat=5` in `numpy_groupies/benchmarks/generic.py`). Lower is better.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble - `--runs=3` takes the per-cell minimum of three complete runs. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks.generic --pandas --numbagg
```

Without `--pandas` the pandas column is omitted, and the pure-Python implementation only appears with `--purepy`. The optional numbagg column (from the `numbagg` package, included in the dev extras) appears with `--numbagg`. numbagg ships nan-guarded grouped reductions only, so its plain-func numbers run through the nan twins - on the benchmark's nan-free plain data the results coincide.

Columns: `ufunc` is the `ufunc.at`-based implementation (incomplete, for benchmarking only), `numpy` the default NumPy implementation, `numba` the Numba implementation, `numbagg` the numbagg grouped-reduction wrapper, and `pandas` the pandas `groupby` wrapper. `----` means the function is not available in that implementation. The row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     | ufunc  | numpy   | numba  | numbagg | pandas  |
| ------------ | -----: | ------: | -----: | ------: | -------: |
| sum          |  0.924 |   0.931 |  0.457 |   0.466 |  15.873 |
| prod         |  0.927 |   0.889 |  0.472 |   0.431 |  14.118 |
| min          |  1.753 |   1.730 |  0.631 |   0.825 |  15.100 |
| max          |  1.733 |   1.760 |  0.620 |   0.866 |  15.224 |
| len          |  0.944 |   0.846 |  0.377 |   0.473 |  14.434 |
| all          | 33.190 |   1.875 |  0.828 |   1.564 |  15.634 |
| any          | 31.127 |   2.536 |  0.671 |   1.570 |  15.978 |
| anynan       |  5.412 |   0.885 |  0.613 |   2.676 |  15.059 |
| allnan       |  7.598 |   2.120 |  0.809 |   3.255 |  14.899 |
| mean         |   ---- |   1.502 |  0.491 |   0.526 |  15.984 |
| median       |   ---- |  47.692 | 11.458 |    ---- |  27.532 |
| trapezoid    |   ---- |   3.846 |  0.638 |    ---- |    ---- |
| std          |   ---- |   3.093 |  0.700 |   0.697 |  16.036 |
| var          |   ---- |   3.161 |  0.678 |   0.678 |  16.058 |
| first        |   ---- |   1.773 |  0.325 |   0.327 |  15.042 |
| last         |   ---- |   0.930 |  0.388 |   0.428 |  15.352 |
| argmax       |   ---- |   2.652 |  0.639 |   0.895 |  14.738 |
| argmin       |   ---- |   5.026 |  0.588 |   0.824 |  14.332 |
| nansum       |   ---- |   4.118 |  0.439 |   1.764 |  20.537 |
| nanprod      |   ---- |   4.053 |  0.460 |   1.741 |  19.724 |
| nanmin       |   ---- |   4.809 |  0.950 |   1.998 |  19.939 |
| nanmax       |   ---- |   4.814 |  1.007 |   2.088 |  20.663 |
| nanlen       |   ---- |   2.444 |  0.432 |   1.708 |  19.672 |
| nanall       |   ---- |   4.852 |  1.041 |   3.270 |  20.570 |
| nanany       |   ---- |   5.580 |  0.643 |   2.648 |  20.615 |
| nanmean      |   ---- |   4.609 |  1.614 |   1.754 |  20.929 |
| nanmedian    |   ---- |  41.881 | 10.472 |    ---- |  29.542 |
| nantrapezoid |   ---- |   6.624 |  1.751 |    ---- |    ---- |
| nanvar       |   ---- |   6.052 |  1.861 |   1.918 |  21.300 |
| nanstd       |   ---- |   5.947 |  1.843 |   1.910 |  21.376 |
| nanfirst     |   ---- |   4.800 |  0.358 |   0.324 |  20.309 |
| nanlast      |   ---- |   4.156 |  0.444 |   1.605 |  20.358 |
| nanargmin    |   ---- |   6.390 |  0.739 |   1.937 |  15.614 |
| nanargmax    |   ---- |   3.995 |  0.801 |   2.027 |  15.807 |
| cumsum       |   ---- |  44.505 |  0.595 |    ---- |   7.171 |
| cumprod      |   ---- |    ---- |  0.607 |    ---- |  14.067 |
| cummax       |   ---- |    ---- |  0.752 |    ---- |  12.826 |
| cummin       |   ---- |    ---- |  0.743 |    ---- |  14.360 |
| arbitrary    |   ---- | 143.037 | 42.295 |    ---- | 107.738 |
| sort         |   ---- | 136.023 | 26.034 |    ---- |    ---- |

## Reading the table

- **numba vs numpy:** Numba is faster for every function both provide - by 1.4–13× across the reductions (`sum` 2.0×, `argmin` 8.5×, `nanfirst` 13×), by 5.2× on `sort` (26 ms against 136 ms), and by far the most on `cumsum` (about 75×).
- **numba vs numbagg:** par on the light scans (`sum` 0.46 vs 0.47 ms, `nanfirst` 0.36 vs 0.32 ms); numba leads wherever the nan handling branches per value (`nansum` 4.0×, `nanlen` 4.0×, `nanlast` 3.6×, `nanargmax` 2.5×), and is the only one of the two with `median`, the cumulatives, `trapezoid` and `sort`.
- **numpy vs pandas:** the NumPy implementation is about 5× (`std`, `var`) to 17× (`len`, `sum`; `prod` 16×) faster than pandas on common reductions. The exception is `median`, where pandas (28 ms) beats NumPy (48 ms) but not Numba (11 ms).
- **ufunc:** `ufunc.at` was historically slow; NumPy 1.25 narrowed the gap considerably, but this implementation remains incomplete and `all`/`any` are still very slow.
- **Expensive everywhere:** `median`, `sort` and custom callables.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks.generic --sort --runs=3
```

| groups | values per group | numpy   | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 125.849 | 34.331 |
|   1000 |              500 | 140.831 | 26.241 |
|  10000 |               50 | 156.002 | 22.529 |
| 100000 |                5 | 172.206 | 34.078 |

NumPy slows down monotonically as the groups multiply. The Numba column dips in the middle: with few large groups the comparison sort itself dominates, with many tiny ones the per-group bookkeeping does. The other implementations have no column here because none of them provides `sort`.
