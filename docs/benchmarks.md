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
its own, and `tools/mt_scaling.py` answers it - every thread count gets a
process of its own, because the threads of the pool spin after a region and
they disturb the measurement of the next one in the same process:

```sh
python -m tools.mt_scaling
```

`--funcs min,max` measures the named functions only, instead of all of them. `--threads N` sets the number of threads the numba kernels use - they split their data into more chunks than that, so that the threads can take the chunks from the queue of the pool and a busy core does not hold up the whole call. Without it they use every hardware thread of the machine. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`.

## Results

| function     |        numpy |        numba |      numbagg |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |       0.912 |       0.253 |       0.483 |       3.102 |       6.486 |       0.907 |
| prod         |       0.893 |       0.229 |       0.442 |       6.420 |       6.226 |       0.924 |
| min          |       1.715 |       0.295 |       0.844 |       3.219 |       6.106 |       1.750 |
| max          |       1.741 |       0.305 |       0.908 |       3.751 |       6.240 |       1.740 |
| len          |       0.863 |       0.189 |       0.485 |       3.160 |       5.127 |       0.949 |
| nunique      |      15.796 |       3.838 |        ---- |      14.059 |      59.548 |        ---- |
| all          |       1.834 |       0.270 |       1.629 |       2.872 |       6.877 |      32.791 |
| any          |       2.481 |       0.241 |       1.584 |       3.273 |       6.901 |      31.291 |
| anynan       |       0.898 |       0.281 |       2.690 |       3.011 |       6.833 |       5.244 |
| allnan       |       2.158 |       0.280 |       3.286 |       2.921 |       6.807 |       7.660 |
| mean         |       1.548 |       0.281 |       0.548 |       2.818 |       7.163 |        ---- |
| median       |      14.083 |       2.887 |        ---- |       4.460 |      15.915 |        ---- |
| quantile     |      15.486 |       2.750 |        ---- |       4.095 |      23.809 |        ---- |
| trapezoid    |       3.827 |       0.703 |        ---- |        ---- |        ---- |        ---- |
| std          |       3.168 |       0.418 |       0.741 |       2.357 |       7.778 |        ---- |
| var          |       3.128 |       0.371 |       0.679 |       2.458 |       7.544 |        ---- |
| skew         |       5.581 |       2.636 |        ---- |       4.265 |       9.668 |        ---- |
| kurt         |       5.638 |       2.562 |        ---- |       3.872 |      10.630 |        ---- |
| first        |       1.744 |       0.320 |       0.311 |       2.809 |       6.098 |        ---- |
| last         |       0.881 |       0.364 |       0.424 |       2.464 |       6.747 |        ---- |
| argmax       |       2.665 |       0.626 |       0.951 |       9.775 |       7.082 |        ---- |
| argmin       |       4.994 |       0.618 |       0.851 |       8.789 |       7.186 |        ---- |
| nansum       |       4.202 |       0.273 |       1.813 |       6.504 |       8.058 |        ---- |
| nanprod      |       4.291 |       0.260 |       1.800 |       9.484 |       7.702 |        ---- |
| nanmin       |       5.000 |       0.362 |       2.456 |       7.397 |       7.851 |        ---- |
| nanmax       |       4.891 |       0.393 |       2.105 |       6.336 |       7.895 |        ---- |
| nanlen       |       2.525 |       0.211 |       1.979 |       6.695 |       7.052 |        ---- |
| nannunique   |      16.865 |       2.601 |        ---- |      15.990 |      28.895 |        ---- |
| nanall       |       4.959 |       0.376 |       3.518 |       6.787 |       8.396 |        ---- |
| nanany       |       5.812 |       0.258 |       2.685 |       6.848 |       8.324 |        ---- |
| nanmean      |       4.795 |       0.549 |       2.033 |       6.376 |       8.733 |        ---- |
| nanmedian    |      14.695 |       4.299 |        ---- |       7.976 |      14.782 |        ---- |
| nanquantile  |      17.386 |       3.008 |        ---- |       7.857 |      20.958 |        ---- |
| nantrapezoid |       6.677 |       1.886 |        ---- |        ---- |        ---- |        ---- |
| nanvar       |       6.445 |       0.517 |       2.105 |       6.451 |       8.828 |        ---- |
| nanstd       |       6.224 |       0.493 |       2.455 |       6.899 |       8.795 |        ---- |
| nanskew      |       8.548 |       3.405 |        ---- |       6.794 |       9.304 |        ---- |
| nankurt      |       8.559 |       3.344 |        ---- |       8.200 |      10.728 |        ---- |
| nanfirst     |       4.949 |       0.343 |       0.339 |       6.717 |       7.912 |        ---- |
| nanlast      |       4.403 |       0.468 |       2.104 |       6.723 |       7.983 |        ---- |
| nanargmin    |       6.330 |       0.769 |       2.229 |       9.057 |       8.610 |        ---- |
| nanargmax    |       4.092 |       0.840 |       2.434 |       8.041 |       8.237 |        ---- |
| cumsum       |      15.371 |       0.572 |        ---- |      10.231 |       6.954 |        ---- |
| cumprod      |        ---- |       0.567 |        ---- |      10.129 |       7.325 |        ---- |
| cummax       |        ---- |       0.805 |        ---- |      11.120 |       6.293 |        ---- |
| cummin       |        ---- |       0.818 |        ---- |      10.219 |       6.322 |        ---- |
| arbitrary    |     114.729 |       6.023 |        ---- |      85.638 |      97.647 |        ---- |
| sort         |      72.128 |       7.192 |        ---- |        ---- |        ---- |        ---- |
| array        |       7.445 |       1.947 |        ---- |        ---- |        ---- |        ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
