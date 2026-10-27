# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of three passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`), every cell measured in one block of five executions. Lower is better. An untimed call precedes every block: the threads of a pool spin for a moment after their region and sleep afterwards, and the first call after that would measure the waking up instead of the work.
- The `numpy` and `numba` columns are the two backends of this package - the pure numpy one and the compiled one. The `numba` one is optimized as far as we could get it: it uses the threads of the machine where that pays for itself, and it falls back to a single thread where it does not. The other columns are the group-by machinery of the libraries of the same name and numpy's own `ufunc.at` - reference implementations for the comparison, not backends of this package.
- The conditions are meant to be the same for every participant: the same data, the same arguments, the same machine, and the threads of the machine are as much theirs as ours. What a participant has to build around the data before it can work on it is its own business, and it is small.
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number. The counts of the groups are cut into ranges for the threads, like the order statistics.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --polars --runs=3
```

The participants get the data as it is - they must not write into it, since that would change the results of the runs that follow. The benchmark checks the data after every function, undoes a change if there was one and reports the row as not measured.

## The calibration of the parallel path

Which calls of the numba kernels take the parallel path and which do not is
decided by a cost model, and the model is fitted on the machine it runs on:

```sh
python -m numpy_groupies.benchmarks.mt
```

The tool measures the parallel kernels against the sequential ones over a
grid of sizes, group counts and thread counts - the chunked path (the sum),
the group path (the median, the quantile, the sort and the number of the
distinct values) and the gather - fits the constants of the model to the
measurements and prints them together with the crossovers, the sizes from
which the parallel path is faster. It writes no file: its output is the
result, and the constants in `numpy_groupies/aggregate_numba/mt.py` are the
ones it prints.

Every kernel is measured twice: warm (the threads of the pool awake, which
is the state of a workload that is called in a loop) and cold (the pool
asleep, which is what a program that calls a kernel once pays). The
difference is the waking of the threads - about twenty times the cost of the
region itself, and the reason the floors of the model are what they are.

The machine has to be idle: the tool reports the spread between the fastest
call and the median of the calls, and a spread of two and more means that
the numbers of that run are worth little.

How the parallel path scales with the number of the threads is a question of
its own, and the tool answers it with `--scaling` - every thread count gets
a process of its own, because the threads of the pool spin after a region
and they disturb the measurement of the next one in the same process:

```sh
python -m numpy_groupies.benchmarks.mt --scaling
```

`--funcs min,max` measures the named functions only, instead of all of them. `--threads N` sets the number of threads the numba kernels use - they split their data into more chunks than that, so that the threads can take the chunks from the queue of the pool and a busy core does not hold up the whole call. Without it they use every hardware thread of the machine. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     |        numpy |        numba |      numbagg |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum         |      0.941 |      0.248 |      0.486 |      2.932 |      6.798 |      0.926 |
| prod        |      0.878 |      0.260 |      0.456 |      6.568 |      6.164 |      0.899 |
| min         |      1.724 |      0.284 |      0.839 |      3.370 |      6.580 |      1.741 |
| max         |      1.705 |      0.331 |      0.890 |      3.553 |      6.840 |      1.741 |
| len         |      0.801 |      0.196 |      0.475 |      2.728 |      5.309 |      0.927 |
| nunique     |     15.567 |      2.952 |       ---- |     12.481 |     56.930 |       ---- |
| all         |      1.803 |      0.309 |      1.604 |      2.838 |      6.775 |     32.673 |
| any         |      2.557 |      0.250 |      1.597 |      2.767 |      6.849 |     30.749 |
| anynan      |      0.911 |      0.255 |      2.670 |      2.806 |      6.720 |      5.224 |
| allnan      |      2.097 |      0.283 |      3.292 |      2.737 |      6.601 |      7.743 |
| mean        |      1.520 |      0.282 |      0.539 |      2.219 |      7.306 |       ---- |
| median      |     12.599 |      2.775 |       ---- |      4.145 |     15.699 |       ---- |
| quantile    |     15.341 |      2.756 |       ---- |      3.669 |     22.869 |       ---- |
| trapezoid   |      3.818 |      0.654 |       ---- |       ---- |       ---- |       ---- |
| std         |      3.147 |      0.359 |      0.689 |      2.841 |      7.556 |       ---- |
| var         |      3.137 |      0.377 |      0.684 |      2.435 |      7.429 |       ---- |
| skew        |      5.578 |      1.015 |       ---- |      3.160 |      9.080 |       ---- |
| kurt        |      5.655 |      0.989 |       ---- |      4.096 |     10.263 |       ---- |
| first       |      1.769 |      0.212 |      0.307 |      2.420 |      6.246 |       ---- |
| last        |      0.831 |      0.208 |      0.435 |      2.819 |      6.451 |       ---- |
| argmax      |      2.654 |      0.402 |      0.934 |      8.433 |      7.316 |       ---- |
| argmin      |      4.993 |      0.320 |      0.844 |      7.562 |      7.192 |       ---- |
| nansum      |      4.101 |      0.263 |      1.780 |      6.058 |      8.417 |       ---- |
| nanprod     |      4.216 |      0.255 |      1.791 |      9.174 |      7.716 |       ---- |
| nanmin      |      4.895 |      0.338 |      1.965 |      6.411 |      7.899 |       ---- |
| nanmax      |      4.961 |      0.405 |      2.043 |      6.321 |      7.855 |       ---- |
| nanlen      |      2.487 |      0.212 |      1.715 |      6.102 |      6.855 |       ---- |
| nannunique  |     16.678 |      2.673 |       ---- |     13.634 |     29.845 |       ---- |
| nanall      |      5.042 |      0.291 |      3.302 |      6.379 |      8.003 |       ---- |
| nanany      |      5.689 |      0.215 |      2.673 |      6.496 |      8.132 |       ---- |
| nanmean     |      4.713 |      0.415 |      1.787 |      6.392 |      8.392 |       ---- |
| nanmedian   |     14.168 |      2.906 |       ---- |      7.400 |     14.856 |       ---- |
| nanquantile |     17.279 |      2.790 |       ---- |      7.332 |     20.904 |       ---- |
| nantrapezoid |      6.610 |      1.904 |       ---- |       ---- |       ---- |       ---- |
| nanvar      |      5.972 |      0.480 |      1.953 |      6.558 |      8.921 |       ---- |
| nanstd      |      6.225 |      0.464 |      1.940 |      6.502 |      9.170 |       ---- |
| nanskew     |      8.430 |      1.076 |       ---- |      6.785 |      9.381 |       ---- |
| nankurt     |      8.445 |      1.055 |       ---- |      7.216 |     10.419 |       ---- |
| nanfirst    |      4.838 |      0.240 |      0.345 |      6.139 |      7.768 |       ---- |
| nanlast     |      4.300 |      0.214 |      1.603 |      6.131 |      8.137 |       ---- |
| nanargmin   |      6.228 |      0.523 |      1.968 |      8.332 |      8.416 |       ---- |
| nanargmax   |      3.899 |      0.570 |      2.032 |      7.200 |      8.465 |       ---- |
| cumsum      |     15.739 |      0.583 |       ---- |      9.976 |      7.241 |       ---- |
| cumprod     |       ---- |      0.570 |       ---- |      9.950 |      6.650 |       ---- |
| cummax      |       ---- |      0.766 |       ---- |      9.206 |      6.408 |       ---- |
| cummin      |       ---- |      0.768 |       ---- |      9.558 |      6.195 |       ---- |
| arbitrary   |    112.480 |      2.842 |       ---- |     81.065 |     98.810 |       ---- |
| sort        |     71.208 |      6.277 |       ---- |       ---- |       ---- |       ---- |
| array       |      7.562 |      1.952 |       ---- |       ---- |       ---- |       ---- |

*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
