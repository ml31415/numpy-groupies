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

Which columns are measured by default, what the optional ones are and how the output arrives is what `--help` explains. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     |        numpy |        numba |        ufunc |      numbagg |       pandas |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.953 |        0.444 |        0.903 |        0.472 |       15.961 |
| prod         |        0.886 |        0.456 |        0.909 |        0.443 |       15.274 |
| min          |        1.675 |        0.618 |        1.687 |        0.837 |       16.073 |
| max          |        1.684 |        0.632 |        1.767 |        0.849 |       16.259 |
| len          |        0.845 |        0.382 |        0.929 |        0.488 |       15.132 |
| all          |        1.756 |        0.819 |       32.878 |        1.575 |       16.028 |
| any          |        2.465 |        0.650 |       30.555 |        1.562 |       16.047 |
| anynan       |        0.920 |        0.614 |        5.350 |        2.658 |       15.452 |
| allnan       |        2.109 |        0.831 |        7.601 |        3.289 |       15.752 |
| mean         |        1.524 |        0.511 |         ---- |        0.545 |       16.978 |
| median       |       48.770 |       11.517 |         ---- |         ---- |       28.566 |
| trapezoid    |        3.814 |        0.653 |         ---- |         ---- |         ---- |
| std          |        3.093 |        0.656 |         ---- |        0.693 |       17.020 |
| var          |        3.133 |        0.667 |         ---- |        0.685 |       16.854 |
| first        |        1.790 |        0.293 |         ---- |        0.308 |       15.966 |
| last         |        0.938 |        0.366 |         ---- |        0.425 |       15.951 |
| argmax       |        2.625 |        0.624 |         ---- |        0.908 |       15.760 |
| argmin       |        5.136 |        0.583 |         ---- |        0.847 |       15.562 |
| nansum       |        4.237 |        0.446 |         ---- |        1.778 |       21.705 |
| nanprod      |        4.159 |        0.449 |         ---- |        1.741 |       21.129 |
| nanmin       |        4.844 |        0.936 |         ---- |        1.996 |       21.081 |
| nanmax       |        4.923 |        1.005 |         ---- |        2.083 |       21.219 |
| nanlen       |        2.495 |        0.448 |         ---- |        1.713 |       20.246 |
| nanall       |        4.929 |        1.057 |         ---- |        3.270 |       21.970 |
| nanany       |        5.698 |        0.642 |         ---- |        2.656 |       21.667 |
| nanmean      |        4.705 |        1.650 |         ---- |        1.797 |       21.271 |
| nanmedian    |       42.596 |       10.618 |         ---- |         ---- |       30.263 |
| nantrapezoid |        6.610 |        1.771 |         ---- |         ---- |         ---- |
| nanvar       |        6.111 |        1.860 |         ---- |        1.922 |       21.491 |
| nanstd       |        6.135 |        1.863 |         ---- |        1.934 |       21.645 |
| nanfirst     |        4.872 |        0.348 |         ---- |        0.317 |       21.347 |
| nanlast      |        4.224 |        0.449 |         ---- |        1.610 |       21.495 |
| nanargmin    |        6.409 |        0.758 |         ---- |        1.939 |       16.622 |
| nanargmax    |        3.947 |        0.794 |         ---- |        2.019 |       16.535 |
| cumsum       |       45.878 |        0.583 |         ---- |         ---- |        7.567 |
| cumprod      |         ---- |        0.563 |         ---- |         ---- |       13.840 |
| cummax       |         ---- |        0.750 |         ---- |         ---- |       14.388 |
| cummin       |         ---- |        0.761 |         ---- |         ---- |       13.083 |
| arbitrary    |      142.543 |       43.163 |         ---- |         ---- |      109.890 |
| sort         |      136.157 |       25.961 |         ---- |         ---- |         ---- |

## Reading the table

- **numba vs numpy:** Numba is faster for every function both provide - by 1.5-6.1× across the plain reductions (`sum` 2.1×, `first` 6.1×), by 2.9-14× on the nan ones (`nanmean` 2.9×, `nanfirst` 14×), by 4.2-8.8× on the argreductions (`argmin` 8.8×), by 5.2× on `sort` (26 ms against 136 ms), and by far the most on `cumsum` (about 79×).
- **numba vs numbagg:** par on the light scans (`sum` 0.44 against 0.47 ms, `first` 0.29 against 0.31 ms, while `prod` and `nanfirst` go to numbagg by a few per cent); numba leads wherever the nan handling branches per value (`anynan` 4.3×, `nansum` 4.0×, `nanlen` 3.8×, `nanlast` 3.6×, `nanargmax` 2.5×), and is the only one of the two with `median`, the cumulatives, `trapezoid` and `sort`.
- **numpy vs pandas:** the NumPy implementation is 5.4× (`var`) to 17.9× (`len`) faster on the plain reductions, up to 8.1× on the nan ones (`nanlen`) and 2.6-6.0× on the argreductions. The exceptions are `median` and `nanmedian`, where pandas (28.6 ms and 30.3 ms) beats NumPy (48.8 ms and 42.6 ms) but not Numba (11.5 ms and 10.6 ms), and the cumulatives, where pandas is six times faster than NumPy (`cumsum` 7.6 against 45.9 ms) and still thirteen times slower than Numba (0.58 ms).
- **ufunc:** `ufunc.at` was historically slow; NumPy 1.25 brought [major speed improvements](https://numpy.org/doc/stable/release/1.25.0-notes.html) and on the arithmetic scans the column now stands level with NumPy (`sum` 0.90 against 0.95 ms, `prod`, `min`, `max` and `len` within a few per cent). It is six times behind on `anynan` and twelve to nineteen times on `all` and `any`, and the implementation remains incomplete.
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
