# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, taking the minimum over 5 timed runs after one untimed warm-up call (`repeat=5` of `numpy_groupies.benchmarks`). Lower is better.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble - `--runs=3` takes the per-cell minimum of three complete runs, printing a counter while it works and the table once at the end. Re-run on your own hardware, from the repository root (with a single run the rows stream out as they are measured and nothing is repeated):

```sh
python -m numpy_groupies.benchmarks --pandas --numbagg
```

The table is Markdown, so it can be pasted into this file as it is.

Without `--pandas` the pandas column is omitted, and the pure-Python implementation only appears with `--purepy`. The optional numbagg column (from the `numbagg` package, included in the dev extras) appears with `--numbagg`. numbagg ships nan-guarded grouped reductions only, so its plain-func numbers run through the nan twins - on the benchmark's nan-free plain data the results coincide. `--grouploop` adds the same NumPy implementation with its fast path disabled (the function wrapped in a lambda), which is what a `func` the implementation does not recognise costs.

Columns: `ufunc` is the `ufunc.at`-based implementation (incomplete, for benchmarking only), `numpy` the default NumPy implementation, `numba` the Numba implementation, `numbagg` the numbagg grouped-reduction wrapper, and `pandas` the pandas `groupby` wrapper. `----` means the function is not available in that implementation. The row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     | ufunc  | numpy   | numba  | numbagg | pandas  |
| ------------ | -----: | ------: | -----: | ------: | ------: |
| sum          |  0.866 |   0.924 |  0.434 |   0.485 |  15.891 |
| prod         |  0.920 |   0.928 |  0.468 |   0.444 |  15.191 |
| min          |  1.730 |   1.687 |  0.611 |   0.815 |  15.018 |
| max          |  1.712 |   1.713 |  0.638 |   0.876 |  15.154 |
| len          |  0.942 |   0.830 |  0.373 |   0.471 |  14.100 |
| all          | 32.574 |   1.805 |  0.823 |   1.567 |  15.725 |
| any          | 30.603 |   2.506 |  0.644 |   1.568 |  15.675 |
| anynan       |  5.221 |   0.890 |  0.623 |   2.674 |  15.258 |
| allnan       |  7.607 |   2.122 |  0.826 |   3.255 |  15.315 |
| mean         |   ---- |   1.559 |  0.511 |   0.556 |  15.985 |
| median       |   ---- |  48.409 | 11.493 |    ---- |  27.700 |
| trapezoid    |   ---- |   3.895 |  0.667 |    ---- |    ---- |
| std          |   ---- |   3.094 |  0.673 |   0.677 |  16.823 |
| var          |   ---- |   3.187 |  0.675 |   0.675 |  16.696 |
| first        |   ---- |   1.765 |  0.362 |   0.313 |  15.390 |
| last         |   ---- |   0.952 |  0.373 |   0.431 |  15.573 |
| argmax       |   ---- |   2.671 |  0.606 |   0.888 |  15.260 |
| argmin       |   ---- |   4.992 |  0.586 |   0.838 |  14.827 |
| nansum       |   ---- |   4.169 |  0.443 |   1.755 |  22.259 |
| nanprod      |   ---- |   4.115 |  0.479 |   1.770 |  20.854 |
| nanmin       |   ---- |   4.875 |  0.977 |   2.011 |  21.029 |
| nanmax       |   ---- |   4.945 |  1.010 |   2.085 |  21.752 |
| nanlen       |   ---- |   2.477 |  0.465 |   1.712 |  20.736 |
| nanall       |   ---- |   4.985 |  1.058 |   3.249 |  21.618 |
| nanany       |   ---- |   5.664 |  0.634 |   2.659 |  21.525 |
| nanmean      |   ---- |   4.678 |  1.634 |   1.758 |  22.228 |
| nanmedian    |   ---- |  42.432 | 10.681 |    ---- |  30.505 |
| nantrapezoid |   ---- |   6.558 |  1.772 |    ---- |    ---- |
| nanvar       |   ---- |   6.255 |  1.860 |   1.921 |  22.432 |
| nanstd       |   ---- |   6.216 |  1.893 |   1.929 |  22.450 |
| nanfirst     |   ---- |   5.077 |  0.393 |   0.368 |  22.225 |
| nanlast      |   ---- |   4.370 |  0.434 |   1.627 |  21.577 |
| nanargmin    |   ---- |   6.385 |  0.774 |   1.958 |  16.259 |
| nanargmax    |   ---- |   3.998 |  0.807 |   2.028 |  16.571 |
| cumsum       |   ---- |  44.838 |  0.583 |    ---- |   7.322 |
| cumprod      |   ---- |    ---- |  0.627 |    ---- |  13.258 |
| cummax       |   ---- |    ---- |  0.758 |    ---- |  13.085 |
| cummin       |   ---- |    ---- |  0.786 |    ---- |  12.785 |
| arbitrary    |   ---- | 144.095 | 42.973 |    ---- | 108.594 |
| sort         |   ---- | 135.146 | 25.992 |    ---- |    ---- |

## Reading the table

- **numba vs numpy:** Numba is faster for every function both provide - by 1.4–13× across the reductions (`sum` 2.1×, `argmin` 8.5×, `nanfirst` 13×), by 5.2× on `sort` (26 ms against 135 ms), and by far the most on `cumsum` (about 77×).
- **numba vs numbagg:** par on the light scans (`sum` 0.43 vs 0.49 ms, `nanfirst` 0.39 vs 0.37 ms, and `first` even a sixth behind); numba leads wherever the nan handling branches per value (`anynan` 4.3×, `nansum` 4.0×, `nanlast` 3.8×, `nanargmax` 2.5×), and is the only one of the two with `median`, the cumulatives, `trapezoid` and `sort`.
- **numpy vs pandas:** the NumPy implementation is about 5× (`var`, `std`) to 17× (`sum`, `anynan`, `len`; `prod` 16×) faster than pandas on common reductions. The exceptions are `median`, where pandas (28 ms) beats NumPy (48 ms) but not Numba (11 ms), and the cumulatives, where pandas is six times faster than NumPy (`cumsum` 7.3 against 44.8 ms) and still twelve times slower than Numba (0.58 ms).
- **ufunc:** `ufunc.at` was historically slow; NumPy 1.25 narrowed the gap considerably, but this implementation remains incomplete and `all`/`any` are still very slow.
- **Expensive everywhere:** `median`, `sort` and custom callables.

## Sorting across group counts

`sort` emits one value per input item and orders every group on its own, so its cost turns on the group *size* - a few large blocks to order against many small ones. `--sort` measures that function alone over a hundredth, one, ten and a hundred times the group count of the table above, on the same 500,000 values:

```sh
python -m numpy_groupies.benchmarks --sort --runs=3
```

| groups | values per group | numpy   | numba  |
| -----: | ---------------: | ------: | -----: |
|    100 |             5000 | 124.895 | 34.421 |
|   1000 |              500 | 141.278 | 26.088 |
|  10000 |               50 | 159.103 | 21.794 |
| 100000 |                5 | 167.581 | 11.088 |

NumPy slows down monotonically as the groups multiply. The Numba column dips in the middle: with few large groups the comparison sort of a whole block is what costs, while many tiny groups are ordered by insertion sort in the flat array, without the per-group view that a library sort has to build - which is three times the speed it had before at five values per group. The other implementations have no column here because none of them provides `sort`.
