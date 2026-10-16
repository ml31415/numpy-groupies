# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of five timed executions after one untimed warm-up call, over three complete passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`). Lower is better.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, numbagg 0.9.6 (Linux, x86_64) - every one of them the current release on PyPI at the time of the run.

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --runs=3
```

Which columns are measured by default, what the optional ones are and how the output arrives is what `--help` explains. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     |        numpy |        numba |        ufunc |      numbagg |       pandas |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.919 |        0.462 |        0.901 |        0.467 |       16.987 |
| prod         |        0.898 |        0.480 |        0.882 |        0.422 |       15.539 |
| min          |        1.713 |        0.629 |        1.714 |        0.822 |       16.106 |
| max          |        1.731 |        0.618 |        1.699 |        0.879 |       16.269 |
| len          |        0.832 |        0.372 |        0.959 |        0.449 |       15.239 |
| all          |        1.872 |        0.798 |       32.965 |        1.570 |       15.969 |
| any          |        2.499 |        0.659 |       31.230 |        1.573 |       17.145 |
| anynan       |        0.962 |        0.635 |        5.324 |        2.677 |       15.854 |
| allnan       |        2.244 |        0.831 |        7.862 |        3.293 |       15.933 |
| mean         |        1.525 |        0.546 |         ---- |        0.559 |       17.059 |
| median       |       48.486 |       11.704 |         ---- |         ---- |       28.602 |
| trapezoid    |        3.896 |        0.640 |         ---- |         ---- |         ---- |
| std          |        3.168 |        0.662 |         ---- |        0.689 |       16.629 |
| var          |        3.118 |        0.658 |         ---- |        0.696 |       17.090 |
| first        |        1.772 |        0.340 |         ---- |        0.319 |       15.289 |
| last         |        0.903 |        0.369 |         ---- |        0.426 |       15.872 |
| argmax       |        2.655 |        0.618 |         ---- |        0.876 |       15.049 |
| argmin       |        4.958 |        0.582 |         ---- |        0.844 |       15.437 |
| nansum       |        4.366 |        0.444 |         ---- |        1.768 |       21.314 |
| nanprod      |        4.287 |        0.487 |         ---- |        1.786 |       20.714 |
| nanmin       |        4.891 |        0.963 |         ---- |        2.006 |       21.252 |
| nanmax       |        5.132 |        1.024 |         ---- |        2.069 |       21.681 |
| nanlen       |        2.508 |        0.434 |         ---- |        1.717 |       20.142 |
| nanall       |        5.015 |        1.071 |         ---- |        3.294 |       21.713 |
| nanany       |        5.878 |        0.655 |         ---- |        2.694 |       21.965 |
| nanmean      |        4.782 |        1.645 |         ---- |        1.749 |       22.266 |
| nanmedian    |       43.424 |       10.772 |         ---- |         ---- |       30.061 |
| nantrapezoid |        6.890 |        1.766 |         ---- |         ---- |         ---- |
| nanvar       |        6.567 |        1.881 |         ---- |        1.934 |       21.847 |
| nanstd       |        6.128 |        1.868 |         ---- |        1.925 |       21.870 |
| nanfirst     |        4.949 |        0.386 |         ---- |        0.321 |       20.988 |
| nanlast      |        4.322 |        0.457 |         ---- |        1.610 |       21.455 |
| nanargmin    |        6.433 |        0.765 |         ---- |        1.954 |       17.108 |
| nanargmax    |        4.090 |        0.801 |         ---- |        1.989 |       17.112 |
| cumsum       |       45.697 |        0.567 |         ---- |         ---- |        7.951 |
| cumprod      |         ---- |        0.598 |         ---- |         ---- |       13.959 |
| cummax       |         ---- |        0.755 |         ---- |         ---- |       12.593 |
| cummin       |         ---- |        0.792 |         ---- |         ---- |       13.439 |
| arbitrary    |      141.598 |       43.168 |         ---- |         ---- |      109.821 |
| sort         |      138.175 |       26.226 |         ---- |         ---- |         ---- |

## Reading the table

- **numba vs numpy:** Numba is faster for every function both provide - by 1.5-8.5× across the plain reductions (`anynan` 1.5×, `sum` 2.0×, `argmin` 8.5×), by 2.9-12.8× on the nan ones (`nanmean` 2.9×, `nansum` 9.8×, `nanfirst` 12.8×), by 5.3× on `sort` (26 ms against 138 ms), and by far the most on `cumsum` (about 81×).
- **numba vs numbagg:** par on the light scans (`sum` 0.46 against 0.47 ms, `mean` 0.55 against 0.56 ms, while `prod`, `first` and `nanfirst` go to numbagg by a few per cent); numba leads wherever the nan handling branches per value (`anynan` 4.2×, `nanany` 4.1×, `nansum`, `nanlen` and `allnan` 4.0×), and is the only one of the two with `median`, the cumulatives, `trapezoid` and `sort`.
- **numba vs pandas:** Numba is the fastest column of the table on every function pandas provides, by 2.4× on `median` and 2.5× on a custom callable and by 12-54× on the rest (`nanvar` 11.6×, `nansum` 48×, `nanfirst` 54.4×).
- **numpy vs pandas:** the NumPy implementation is 5.2× (`std`) to 18.5× (`sum`) faster on the plain reductions, 2.7-8.0× on the nan ones (`nanlen` 8.0×) and 2.7-5.7× on the argreductions. The exceptions are `median` and `nanmedian`, where pandas (28.6 ms and 30.1 ms) beats NumPy (48.5 ms and 43.4 ms) but not Numba (11.7 ms and 10.8 ms), and the cumulatives, where pandas is about six times faster than NumPy (`cumsum` 8.0 against 45.7 ms) and still fourteen times slower than Numba (0.57 ms).
- **ufunc:** `ufunc.at` was historically slow; NumPy 1.25 brought [major speed improvements](https://numpy.org/doc/stable/release/1.25.0-notes.html) and on the arithmetic scans the column now stands level with NumPy (`sum` 0.90 against 0.92 ms, `prod`, `min` and `max` within a per cent, `len` 0.96 against 0.83 ms). It is eight to nine times behind on `anynan` and `allnan` and forty to forty-seven times on `all` and `any`, and the implementation remains incomplete.
- **Expensive everywhere:** `median`, `sort` and custom callables.

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
