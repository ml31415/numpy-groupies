# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything below `0.2` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions another 20% of the values are set to `nan`, leaving the rest on the interval `[0.2, 0.8)`.
- Times in **milliseconds**, taking the minimum over 7 runs after discarding a warm-up run. Lower is better.
- Machine: Intel i7-7560U at 2.40 GHz, Linux (x86_64), Python 3.14.2, NumPy 2.5.3, Numba 0.67.0, pandas 3.0.6.

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. Re-run on your own hardware, from the repository root:

```
python -m numpy_groupies.benchmarks.generic
```

Columns: `ufunc` is the `ufunc.at`-based implementation (incomplete, for benchmarking only), `numpy` the default NumPy implementation, `numba` the Numba implementation, and `pandas` the pandas `groupby` wrapper. `----` means the function is not available in that implementation. The row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     | ufunc  | numpy   | numba  | pandas  |
| ------------ | -----: | ------: | -----: | ------: |
| sum          | 1.586  | 1.242   | 0.722  | 13.763  |
| prod         | 1.420  | 1.411   | 0.709  | 13.303  |
| min          | 2.746  | 2.735   | 0.864  | 12.792  |
| max          | 2.774  | 2.763   | 0.881  | 13.106  |
| len          | 1.494  | 1.032   | 0.526  | 12.174  |
| all          | 43.727 | 2.894   | 0.949  | 13.499  |
| any          | 42.965 | 3.301   | 1.272  | 13.555  |
| anynan       | 6.563  | 1.445   | 0.864  | 13.308  |
| allnan       | 9.487  | 3.554   | 0.785  | 13.284  |
| mean         | ----   | 1.823   | 0.985  | 13.913  |
| median       | ----   | 53.343  | 11.713 | 24.283  |
| trapezoid    | ----   | 4.486   | 0.996  | ----    |
| std          | ----   | 4.175   | 1.144  | 14.981  |
| var          | ----   | 4.085   | 1.154  | 14.942  |
| first        | ----   | 1.831   | 0.710  | 13.069  |
| last         | ----   | 1.570   | 0.589  | 13.247  |
| argmax       | ----   | 4.146   | 1.347  | 12.837  |
| argmin       | ----   | 6.576   | 1.297  | 12.567  |
| nansum       | ----   | 5.130   | 1.689  | 18.962  |
| nanprod      | ----   | 5.257   | 1.998  | 18.528  |
| nanmin       | ----   | 6.278   | 2.003  | 18.196  |
| nanmax       | ----   | 6.339   | 1.991  | 18.370  |
| nanlen       | ----   | 3.103   | 1.589  | 18.012  |
| nanall       | ----   | 6.293   | 1.768  | 18.895  |
| nanany       | ----   | 6.916   | 2.246  | 19.134  |
| nanmean      | ----   | 5.615   | 1.916  | 19.756  |
| nanmedian    | ----   | 55.496  | 10.082 | 26.411  |
| nantrapezoid | ----   | 7.718   | 2.122  | ----    |
| nanvar       | ----   | 7.476   | 2.079  | 20.148  |
| nanstd       | ----   | 7.679   | 2.052  | 20.417  |
| nanfirst     | ----   | 5.671   | 1.579  | 18.570  |
| nanlast      | ----   | 5.397   | 1.559  | 18.765  |
| nanargmin    | ----   | 8.633   | 2.006  | 13.773  |
| nanargmax    | ----   | 6.054   | 2.056  | 13.866  |
| cumsum       | ----   | 53.275  | 1.150  | 13.433  |
| cumprod      | ----   | ----    | 1.176  | 11.117  |
| cummax       | ----   | ----    | 1.498  | 11.597  |
| cummin       | ----   | ----    | 1.481  | 11.583  |
| arbitrary    | ----   | 161.542 | 50.944 | 131.864 |
| sort         | ----   | 143.335 | ----   | ----    |

## Reading the table

- **numba vs numpy:** Numba is faster for every function both provide, typically by 1.5–4.5× on reductions. The largest gap is `cumsum` (about 46×).
- **numpy vs pandas:** the NumPy implementation is about 4× (`std`, `var`) to 13× (`sum`, `prod`, `len`) faster than pandas on common reductions. The exception is `median`, where pandas (24 ms) beats NumPy (53 ms) but not Numba (12 ms).
- **ufunc:** `ufunc.at` was historically slow; NumPy 1.25 narrowed the gap considerably, but this implementation remains incomplete and `all`/`any` are still very slow.
- **Expensive everywhere:** `median`, `sort` and custom callables.
