# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of three passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`), every cell measured in one block of five executions. Lower is better. An untimed call precedes every block: the threads of a pool spin for a moment after their region and sleep afterwards, and the first call after that would measure the waking up instead of the work.
- The `polars` column uses all the cores of the machine (6 threads here). The `numba` column uses them too, in two ways. The operations that can be reduced in chunks - the sums, the products, the counts, the extremes and the mean - split their data into chunks that are reduced separately and folded afterwards. The operations that cannot - the order statistics (the median, the quantile, the sort, the number of the distinct values) - cut their *groups* into ranges instead and let the threads work on the ranges one after another; every group belongs to one range, so nothing is folded afterwards and the result is the one of the sequential kernel bit for bit. The chunks and the ranges are more than the threads, so that a core that is busy with something else does not hold up the whole call - the threads that are done take the pieces that are left. The operations that give an output per value (the cumulatives) stay single-threaded. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 8.5ms for the quantile, against 4.6ms and 4.4ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.5ms).
- The `numbagg` column runs its calls with numba at one thread, where its grouped reductions are fastest on a machine with many cores; the `numbagg-mt` column runs them with all the threads of the machine. On the machine of the numbers below the two are within a few percent of each other - the thread sync of the parallel target eats what the cores gain.
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number - 3.3ms for the Numba kernel against 15.1ms for the NumPy one, which has to sort the values of every group (numba's sort is about four times slower than numpy's, which vectorizes it). The counts of the groups are cut into ranges for the threads, like the order statistics.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --numbagg-mt --pandas --polars --runs=3
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

`--funcs min,max` measures the named functions only, instead of all of them. `--threads N` sets the number of threads the numba kernels use - they split their data into more chunks than that, so that the threads can take the chunks from the queue of the pool and a busy core does not hold up the whole call. Without it they use every hardware thread of the machine. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`. The `numbagg` and `numbagg-mt` columns are the group-by machinery of numbagg, the first with one thread and the second with all the threads of the machine - the two show what the choice is worth, so that nobody has to take the one-threaded column for a handicap. The `pandas` and `polars` columns are the group-by machinery of those libraries, the `ufunc` column is numpy's own `ufunc.at` - the last three are reference implementations for the comparison, not backends of this package.

## Results

| function     |        numpy |        numba |      numbagg |   numbagg-mt |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |       0.919 |       0.230 |        0.472 |        0.466 |        2.931 |        6.903 |        0.910 |
| prod         |       0.949 |       0.231 |        0.448 |        0.438 |        7.633 |        6.237 |        0.913 |
| min          |       1.682 |       0.299 |        0.840 |        0.827 |        3.505 |        6.615 |        1.778 |
| max          |       1.699 |       0.307 |        0.864 |        0.868 |        3.470 |        6.559 |        1.766 |
| len          |       0.781 |       0.197 |        0.482 |        0.484 |        2.642 |        5.315 |        0.910 |
| nunique      |      15.141 |       3.276 |         ---- |         ---- |       13.413 |       56.520 |         ---- |
| all          |       1.790 |       0.274 |        1.641 |        1.595 |        2.859 |        7.151 |       32.917 |
| any          |       2.477 |       0.244 |        1.600 |        1.581 |        2.633 |        7.233 |       30.906 |
| anynan       |       0.832 |       0.276 |        2.717 |        2.709 |        2.317 |        6.844 |        5.266 |
| allnan       |       2.085 |       0.252 |        3.295 |        3.261 |        2.888 |        6.760 |        7.658 |
| mean         |       1.519 |       0.315 |        0.550 |        0.551 |        2.651 |        7.425 |         ---- |
| median       |      12.895 |       2.855 |         ---- |         ---- |        4.214 |       15.759 |         ---- |
| quantile     |      16.261 |       2.809 |         ---- |         ---- |        4.222 |       23.426 |         ---- |
| trapezoid    |       3.764 |       0.656 |         ---- |         ---- |         ---- |         ---- |         ---- |
| std          |       3.122 |       0.335 |        0.759 |        0.929 |        3.147 |        7.630 |         ---- |
| var          |       3.120 |       0.308 |        0.733 |        0.682 |        3.061 |        8.512 |         ---- |
| skew         |       5.757 |       2.790 |         ---- |         ---- |        3.739 |        9.477 |         ---- |
| kurt         |       5.726 |       2.755 |         ---- |         ---- |        4.129 |       10.473 |         ---- |
| first        |       1.739 |       0.297 |        0.323 |        0.321 |        2.691 |        6.249 |         ---- |
| last         |       0.836 |       0.368 |        0.464 |        0.602 |        2.839 |        6.867 |         ---- |
| argmax       |       2.630 |       0.642 |        0.916 |        0.936 |        8.425 |        7.365 |         ---- |
| argmin       |       4.954 |       0.601 |        0.873 |        0.869 |        8.276 |        7.699 |         ---- |
| nansum       |       4.185 |       0.242 |        1.803 |        1.774 |        6.503 |        9.124 |         ---- |
| nanprod      |       4.493 |       0.234 |        1.822 |        1.810 |        9.671 |        7.715 |         ---- |
| nanmin       |       4.892 |       0.347 |        2.034 |        2.011 |        6.626 |        8.054 |         ---- |
| nanmax       |       4.964 |       0.377 |        2.073 |        2.025 |        6.321 |        8.367 |         ---- |
| nanlen       |       2.465 |       0.250 |        1.752 |        1.726 |        6.076 |        7.664 |         ---- |
| nannunique   |      16.760 |       2.580 |         ---- |         ---- |       14.350 |       29.848 |         ---- |
| nanall       |       5.013 |       0.307 |        3.305 |        3.326 |        6.259 |        8.757 |         ---- |
| nanany       |       5.858 |       0.233 |        2.727 |        2.714 |        6.411 |        8.531 |         ---- |
| nanmean      |       4.769 |       0.438 |        1.788 |        1.771 |        6.099 |        9.215 |         ---- |
| nanmedian    |      15.000 |       2.868 |         ---- |         ---- |        8.036 |       15.220 |         ---- |
| nanquantile  |      17.494 |       2.878 |         ---- |         ---- |        7.734 |       21.325 |         ---- |
| nantrapezoid |       6.592 |       1.909 |         ---- |         ---- |         ---- |         ---- |         ---- |
| nanvar       |       6.214 |       0.473 |        1.936 |        1.959 |        6.733 |        9.411 |         ---- |
| nanstd       |       6.125 |       0.464 |        1.975 |        1.948 |        6.466 |        9.672 |         ---- |
| nanskew      |       8.350 |       3.289 |         ---- |         ---- |        6.689 |       10.046 |         ---- |
| nankurt      |       8.443 |       3.259 |         ---- |         ---- |        7.282 |       10.591 |         ---- |
| nanfirst     |       4.850 |       0.332 |        0.325 |        0.323 |        6.040 |        8.085 |         ---- |
| nanlast      |       4.235 |       0.442 |        1.623 |        1.611 |        5.946 |        8.253 |         ---- |
| nanargmin    |       6.130 |       0.760 |        1.975 |        1.949 |        8.299 |        8.411 |         ---- |
| nanargmax    |       3.847 |       0.812 |        2.017 |        2.011 |        6.947 |        8.699 |         ---- |
| cumsum       |      14.990 |       0.538 |         ---- |         ---- |        9.880 |        7.601 |         ---- |
| cumprod      |        ---- |       0.548 |         ---- |         ---- |        8.666 |        7.417 |         ---- |
| cummax       |        ---- |       0.751 |         ---- |         ---- |        9.250 |        6.604 |         ---- |
| cummin       |        ---- |       0.741 |         ---- |         ---- |        8.736 |        6.300 |         ---- |
| arbitrary    |     112.425 |       5.961 |         ---- |         ---- |       80.268 |       96.240 |         ---- |
| sort         |      70.669 |       6.673 |         ---- |         ---- |         ---- |         ---- |         ---- |
| array        |       7.564 |       1.941 |         ---- |         ---- |         ---- |         ---- |         ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
