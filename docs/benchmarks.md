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
| sum          |       0.926 |       0.256 |       0.477 |       2.707 |       6.480 |       0.886 |
| prod         |       0.883 |       0.240 |       0.433 |       7.303 |       5.385 |       0.882 |
| min          |       1.677 |       0.324 |       0.856 |       3.354 |       6.181 |       1.682 |
| max          |       1.683 |       0.313 |       0.872 |       3.588 |       6.349 |       1.683 |
| len          |       0.774 |       0.193 |       0.480 |       2.493 |       4.911 |       0.935 |
| nunique      |      15.398 |       2.988 |        ---- |      12.716 |      57.843 |        ---- |
| all          |       1.835 |       0.295 |       1.605 |       2.905 |       6.632 |      32.969 |
| any          |       2.521 |       0.305 |       1.606 |       2.967 |       6.847 |      31.541 |
| anynan       |       0.839 |       0.290 |       2.694 |       2.920 |       6.875 |       5.250 |
| allnan       |       2.126 |       0.273 |       3.275 |       2.915 |       6.632 |       7.780 |
| mean         |       1.530 |       0.297 |       0.540 |       2.731 |       7.189 |        ---- |
| median       |      12.614 |       2.813 |        ---- |       3.868 |      15.484 |        ---- |
| quantile     |      16.014 |       3.185 |        ---- |       4.409 |      23.320 |        ---- |
| trapezoid    |       3.806 |       0.650 |        ---- |        ---- |        ---- |        ---- |
| std          |       3.188 |       0.331 |       0.685 |       2.439 |       7.574 |        ---- |
| var          |       3.183 |       0.341 |       0.705 |       2.674 |       7.546 |        ---- |
| skew         |       5.735 |       2.594 |        ---- |       3.654 |       9.297 |        ---- |
| kurt         |       5.522 |       2.601 |        ---- |       4.116 |      10.396 |        ---- |
| first        |       1.754 |       0.309 |       0.326 |       2.631 |       5.995 |        ---- |
| last         |       0.861 |       0.399 |       0.436 |       2.964 |       6.365 |        ---- |
| argmax       |       2.619 |       0.639 |       0.938 |       8.173 |       7.218 |        ---- |
| argmin       |       4.904 |       0.606 |       0.944 |       7.262 |       6.990 |        ---- |
| nansum       |       4.104 |       0.285 |       1.795 |       6.221 |       8.152 |        ---- |
| nanprod      |       4.160 |       0.250 |       1.776 |       8.613 |       7.505 |        ---- |
| nanmin       |       4.860 |       0.320 |       1.951 |       6.198 |       8.035 |        ---- |
| nanmax       |       4.814 |       0.351 |       2.012 |       6.325 |       7.939 |        ---- |
| nanlen       |       2.430 |       0.266 |       1.705 |       5.976 |       7.139 |        ---- |
| nannunique   |      16.431 |       2.830 |        ---- |      14.613 |      29.423 |        ---- |
| nanall       |       4.979 |       0.298 |       3.370 |       6.421 |       8.602 |        ---- |
| nanany       |       5.736 |       0.317 |       2.708 |       6.440 |       8.567 |        ---- |
| nanmean      |       4.722 |       0.417 |       1.764 |       6.464 |       8.878 |        ---- |
| nanmedian    |      14.821 |       2.793 |        ---- |       7.662 |      14.909 |        ---- |
| nanquantile  |      17.034 |       2.805 |        ---- |       7.329 |      20.979 |        ---- |
| nantrapezoid |       6.542 |       1.900 |        ---- |        ---- |        ---- |        ---- |
| nanvar       |       6.216 |       0.489 |       1.951 |       6.263 |       9.038 |        ---- |
| nanstd       |       6.085 |       0.488 |       1.938 |       6.354 |       8.958 |        ---- |
| nanskew      |       8.434 |       3.288 |        ---- |       6.445 |       9.479 |        ---- |
| nankurt      |       8.313 |       3.282 |        ---- |       6.406 |      10.375 |        ---- |
| nanfirst     |       4.772 |       0.336 |       0.336 |       5.935 |       7.965 |        ---- |
| nanlast      |       4.139 |       0.454 |       1.648 |       6.066 |       8.274 |        ---- |
| nanargmin    |       6.227 |       0.762 |       1.951 |       8.237 |       8.004 |        ---- |
| nanargmax    |       3.931 |       0.806 |       2.019 |       6.491 |       8.397 |        ---- |
| cumsum       |      15.621 |       0.577 |        ---- |      11.228 |       7.045 |        ---- |
| cumprod      |        ---- |       0.623 |        ---- |       8.902 |       7.094 |        ---- |
| cummax       |        ---- |       0.749 |        ---- |       9.046 |       6.178 |        ---- |
| cummin       |        ---- |       0.780 |        ---- |       9.291 |       6.204 |        ---- |
| arbitrary    |     111.187 |       6.313 |        ---- |      82.065 |      96.429 |        ---- |
| sort         |      71.603 |       6.910 |        ---- |        ---- |        ---- |        ---- |
| array        |       7.183 |       1.952 |        ---- |        ---- |        ---- |        ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
