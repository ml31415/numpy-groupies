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
| sum          |       0.935 |       0.291 |       0.519 |       2.934 |       7.136 |       0.917 |
| prod         |       0.867 |       0.250 |       0.514 |       7.103 |       6.043 |       0.892 |
| min          |       1.767 |       0.357 |       0.845 |       2.955 |       6.569 |       1.696 |
| max          |       1.708 |       0.311 |       0.856 |       3.595 |       6.575 |       1.697 |
| len          |       0.824 |       0.220 |       0.478 |       2.080 |       5.387 |       0.922 |
| nunique      |      14.755 |       2.605 |        ---- |      12.187 |      61.492 |        ---- |
| all          |       1.791 |       0.408 |       1.771 |       2.625 |       6.897 |      34.237 |
| any          |       2.598 |       0.332 |       1.682 |       2.433 |       7.479 |      30.973 |
| anynan       |       0.894 |       0.285 |       2.695 |       2.904 |       7.144 |       5.397 |
| allnan       |       2.132 |       0.313 |       3.665 |       3.027 |       6.835 |       7.709 |
| mean         |       1.577 |       0.407 |       0.560 |       2.794 |       7.765 |        ---- |
| median       |      13.850 |       2.998 |        ---- |       4.335 |      16.476 |        ---- |
| quantile     |      16.390 |       2.887 |        ---- |       4.492 |      23.434 |        ---- |
| trapezoid    |       4.052 |       0.677 |        ---- |        ---- |        ---- |        ---- |
| std          |       3.172 |       0.438 |       0.717 |       2.991 |       7.734 |        ---- |
| var          |       3.249 |       0.442 |       0.686 |       3.225 |       7.528 |        ---- |
| skew         |       5.746 |       2.593 |        ---- |       3.951 |       9.513 |        ---- |
| kurt         |       5.754 |       2.573 |        ---- |       4.327 |      10.727 |        ---- |
| first        |       1.826 |       0.362 |       0.339 |       2.659 |       6.971 |        ---- |
| last         |       0.916 |       0.436 |       0.458 |       2.565 |       7.021 |        ---- |
| argmax       |       2.680 |       0.656 |       0.910 |       8.629 |       7.522 |        ---- |
| argmin       |       4.897 |       0.615 |       0.952 |       7.598 |       7.200 |        ---- |
| nansum       |       4.262 |       0.273 |       1.831 |       6.098 |       8.656 |        ---- |
| nanprod      |       4.184 |       0.276 |       1.787 |       9.126 |       7.677 |        ---- |
| nanmin       |       4.993 |       0.353 |       2.017 |       6.320 |       8.201 |        ---- |
| nanmax       |       4.923 |       0.423 |       2.074 |       6.536 |       8.359 |        ---- |
| nanlen       |       2.474 |       0.289 |       1.760 |       6.499 |       7.515 |        ---- |
| nannunique   |      16.643 |       2.844 |        ---- |      15.938 |      30.305 |        ---- |
| nanall       |       4.989 |       0.355 |       3.346 |       7.143 |       8.565 |        ---- |
| nanany       |       5.837 |       0.286 |       2.751 |       6.026 |       8.724 |        ---- |
| nanmean      |       4.694 |       0.466 |       1.835 |       6.315 |       8.924 |        ---- |
| nanmedian    |      14.735 |       3.326 |        ---- |       7.245 |      15.063 |        ---- |
| nanquantile  |      17.232 |       3.095 |        ---- |       8.228 |      21.465 |        ---- |
| nantrapezoid |       6.682 |       1.928 |        ---- |        ---- |        ---- |        ---- |
| nanvar       |       6.106 |       0.519 |       1.946 |       6.631 |       9.310 |        ---- |
| nanstd       |       6.284 |       0.501 |       1.962 |       6.259 |       9.729 |        ---- |
| nanskew      |       8.327 |       3.315 |        ---- |       6.580 |       9.705 |        ---- |
| nankurt      |       8.533 |       3.313 |        ---- |       7.051 |      12.541 |        ---- |
| nanfirst     |       5.044 |       0.332 |       0.333 |       6.534 |       7.956 |        ---- |
| nanlast      |       4.471 |       0.463 |       1.615 |       6.472 |       8.454 |        ---- |
| nanargmin    |       6.284 |       0.769 |       1.964 |       8.155 |       8.528 |        ---- |
| nanargmax    |       3.911 |       0.823 |       2.042 |       7.209 |       8.372 |        ---- |
| cumsum       |      15.804 |       0.577 |        ---- |       9.176 |       7.844 |        ---- |
| cumprod      |        ---- |       0.571 |        ---- |       9.538 |       7.458 |        ---- |
| cummax       |        ---- |       0.757 |        ---- |       9.479 |       6.807 |        ---- |
| cummin       |        ---- |       0.749 |        ---- |       9.141 |       7.205 |        ---- |
| arbitrary    |     113.750 |       6.217 |        ---- |      86.194 |      97.678 |        ---- |
| sort         |      73.162 |       6.527 |        ---- |        ---- |        ---- |        ---- |
| array        |       7.361 |       1.964 |        ---- |        ---- |        ---- |        ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
