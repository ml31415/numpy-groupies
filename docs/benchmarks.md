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
| sum          |  0.859 |  0.878 |  0.429 |  0.441 | 15.451 |
| prod         |  0.865 |  0.883 |  0.426 |  0.430 | 14.811 |
| min          |  1.708 |  1.691 |  0.626 |  0.811 | 14.803 |
| max          |  1.684 |  1.686 |  0.617 |  0.863 | 15.012 |
| len          |  0.931 |  0.822 |  0.365 |  0.492 | 14.238 |
| all          | 35.409 |  1.798 |  0.813 |  1.582 | 15.629 |
| any          | 32.628 |  2.596 |  0.652 |  1.588 | 15.595 |
| anynan       |  5.238 |  0.829 |  0.607 |  2.664 | 15.289 |
| allnan       |  7.553 |  2.133 |  0.826 |  3.264 | 15.568 |
| mean         |   ---- |  1.533 |  0.497 |  0.538 | 16.113 |
| median       |   ---- | 47.333 | 11.702 |   ---- | 28.348 |
| trapezoid    |   ---- |  3.768 |  0.629 |   ---- |   ---- |
| std          |   ---- |  3.155 |  0.681 |  0.670 | 16.310 |
| var          |   ---- |  3.164 |  0.653 |  0.676 | 16.329 |
| first        |   ---- |  1.770 |  0.291 |  0.305 | 14.610 |
| last         |   ---- |  0.901 |  0.363 |  0.428 | 15.566 |
| argmax       |   ---- |  2.654 |  0.602 |  0.903 | 14.413 |
| argmin       |   ---- |  4.920 |  0.583 |  0.839 | 14.856 |
| nansum       |   ---- |  4.193 |  0.441 |  1.752 | 20.567 |
| nanprod      |   ---- |  4.221 |  0.456 |  1.759 | 19.578 |
| nanmin       |   ---- |  4.861 |  0.936 |  2.012 | 20.691 |
| nanmax       |   ---- |  4.798 |  1.010 |  2.072 | 20.260 |
| nanlen       |   ---- |  2.448 |  0.444 |  1.712 | 19.698 |
| nanall       |   ---- |  5.156 |  1.062 |  3.298 | 20.606 |
| nanany       |   ---- |  5.692 |  0.650 |  2.681 | 20.761 |
| nanmean      |   ---- |  4.694 |  1.609 |  1.756 | 22.667 |
| nanmedian    |   ---- | 42.818 | 10.858 |   ---- | 31.300 |
| nantrapezoid |   ---- |  6.769 |  1.784 |   ---- |   ---- |
| nanvar       |   ---- |  6.366 |  1.874 |  1.954 | 21.596 |
| nanstd       |   ---- |  6.526 |  1.868 |  1.916 | 21.983 |
| nanfirst     |   ---- |  4.822 |  0.331 |  0.334 | 20.891 |
| nanlast      |   ---- |  4.371 |  0.470 |  1.596 | 20.747 |
| nanargmin    |   ---- |  6.404 |  0.766 |  1.942 | 16.185 |
| nanargmax    |   ---- |  3.997 |  0.779 |  2.013 | 16.744 |
| cumsum       |   ---- | 45.385 |  0.619 |   ---- |  7.364 |
| cumprod      |   ---- |   ---- |  0.589 |   ---- | 14.302 |
| cummax       |   ---- |   ---- |  0.775 |   ---- | 13.378 |
| cummin       |   ---- |   ---- |  0.787 |   ---- | 12.651 |
| arbitrary    |   ---- | 141.830 | 43.556 |   ---- | 107.547 |
| sort         |   ---- | 135.021 |   ---- |   ---- |   ---- |

## Reading the table

- **numba vs numpy:** Numba is faster for every function both provide, typically by 1.7–6.3× on reductions (`argmin` 6.3×, `sum` 2.0×). The largest gap is `cumsum` (about 64×).
- **numba vs numbagg:** par on the light scans (`sum` 0.43 vs 0.44 ms, `nanfirst` 0.33 vs 0.33 ms); numba leads wherever the nan handling branches per value (`nansum` 4.0×, `nanlen` 3.9×, `nanlast` 3.4×, `nanargmax` 2.6×), and is the only one of the two with `median`, the cumulatives and `trapezoid`.
- **numpy vs pandas:** the NumPy implementation is about 5× (`std`, `var`) to 18× (`len`; `sum` ~17×, `prod` 16×) faster than pandas on common reductions. The exception is `median`, where pandas (29 ms) beats NumPy (48 ms) but not Numba (11 ms).
- **ufunc:** `ufunc.at` was historically slow; NumPy 1.25 narrowed the gap considerably, but this implementation remains incomplete and `all`/`any` are still very slow.
- **Expensive everywhere:** `median`, `sort` and custom callables.
