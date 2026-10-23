# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of three passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`), every cell measured in one block of five executions. Lower is better. An untimed call precedes every block: the threads of a pool spin for a moment after their region and sleep afterwards, and the first call after that would measure the waking up instead of the work.
- The `polars` column uses all the cores of the machine (6 threads here). The `numba` column uses them too, for the operations that can be reduced in chunks - the sums, the products, the counts, the extremes and the mean: their data is split into chunks that are reduced separately and folded afterwards. The chunks are more than the threads, so that a core that is busy with something else does not hold up the whole call - the threads that are done take the chunks that are left. The state is one per thread, not one per chunk, so the memory does not grow with the number of the chunks. The operations that need the values of a group together (the order statistics) and the ones that give an output per value (the cumulatives) stay single-threaded. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 8.5ms for the quantile, against 4.6ms and 4.4ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.5ms).
- The `numbagg` column runs its calls with numba at one thread, where its grouped reductions are fastest on a machine with many cores; the `numbagg-mt` column runs them with all the threads of the machine. On the machine of the numbers below the two are within a few percent of each other - the thread sync of the parallel target eats what the cores gain.
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number - 8.2ms for the Numba kernel against 15.3ms for the NumPy one, which has to sort the values of every group (numba's sort is about four times slower than numpy's, which vectorizes it).
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --numbagg-mt --pandas --polars --runs=3
```

The participants get the data as it is - they must not write into it, since that would change the results of the runs that follow. The benchmark checks the data after every function, undoes a change if there was one and reports the row as not measured.

`--funcs min,max` measures the named functions only, instead of all of them. `--threads N` sets the number of threads the numba kernels use - they split their data into more chunks than that, so that the threads can take the chunks from the queue of the pool and a busy core does not hold up the whole call. Without it they use every hardware thread of the machine. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`. The `numbagg` and `numbagg-mt` columns are the group-by machinery of numbagg, the first with one thread and the second with all the threads of the machine - the two show what the choice is worth, so that nobody has to take the one-threaded column for a handicap. The `pandas` and `polars` columns are the group-by machinery of those libraries, the `ufunc` column is numpy's own `ufunc.at` - the last three are reference implementations for the comparison, not backends of this package.

## Results

| function     |        numpy |        numba |      numbagg |   numbagg-mt |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.957 |        0.347 |        0.472 |        0.466 |        2.931 |        6.903 |        0.910 |
| prod         |        0.923 |        0.252 |        0.448 |        0.438 |        7.633 |        6.237 |        0.913 |
| min          |        1.710 |        0.288 |        0.840 |        0.827 |        3.505 |        6.615 |        1.778 |
| max          |        1.730 |        0.358 |        0.864 |        0.868 |        3.470 |        6.559 |        1.766 |
| len          |        0.823 |        0.182 |        0.482 |        0.484 |        2.642 |        5.315 |        0.910 |
| nunique      |       15.355 |        8.187 |         ---- |         ---- |       13.413 |       56.520 |         ---- |
| all          |        1.773 |        0.317 |        1.641 |        1.595 |        2.859 |        7.151 |       32.917 |
| any          |        2.490 |        0.221 |        1.600 |        1.581 |        2.633 |        7.233 |       30.906 |
| anynan       |        0.883 |        0.326 |        2.717 |        2.709 |        2.317 |        6.844 |        5.266 |
| allnan       |        2.097 |        0.281 |        3.295 |        3.261 |        2.888 |        6.760 |        7.658 |
| mean         |        1.553 |        0.342 |        0.550 |        0.551 |        2.651 |        7.425 |         ---- |
| median       |       12.522 |        9.527 |         ---- |         ---- |        4.214 |       15.759 |         ---- |
| quantile     |       14.649 |        9.496 |         ---- |         ---- |        4.222 |       23.426 |         ---- |
| trapezoid    |        3.882 |        0.663 |         ---- |         ---- |         ---- |         ---- |         ---- |
| std          |        3.128 |        0.303 |        0.759 |        0.929 |        3.147 |        7.630 |         ---- |
| var          |        3.126 |        0.300 |        0.733 |        0.682 |        3.061 |        8.512 |         ---- |
| skew         |        6.020 |        2.607 |         ---- |         ---- |        3.739 |        9.477 |         ---- |
| kurt         |        5.742 |        2.557 |         ---- |         ---- |        4.129 |       10.473 |         ---- |
| first        |        1.776 |        0.314 |        0.323 |        0.321 |        2.691 |        6.249 |         ---- |
| last         |        0.847 |        0.376 |        0.464 |        0.602 |        2.839 |        6.867 |         ---- |
| argmax       |        2.622 |        0.650 |        0.916 |        0.936 |        8.425 |        7.365 |         ---- |
| argmin       |        4.963 |        0.598 |        0.873 |        0.869 |        8.276 |        7.699 |         ---- |
| nansum       |        4.168 |        0.261 |        1.803 |        1.774 |        6.503 |        9.124 |         ---- |
| nanprod      |        4.306 |        0.261 |        1.822 |        1.810 |        9.671 |        7.715 |         ---- |
| nanmin       |        4.935 |        0.287 |        2.034 |        2.011 |        6.626 |        8.054 |         ---- |
| nanmax       |        4.818 |        0.360 |        2.073 |        2.025 |        6.321 |        8.367 |         ---- |
| nanlen       |        2.437 |        0.252 |        1.752 |        1.726 |        6.076 |        7.664 |         ---- |
| nannunique   |       16.765 |        7.724 |         ---- |         ---- |       14.350 |       29.848 |         ---- |
| nanall       |        4.887 |        0.341 |        3.305 |        3.326 |        6.259 |        8.757 |         ---- |
| nanany       |        5.766 |        0.234 |        2.727 |        2.714 |        6.411 |        8.531 |         ---- |
| nanmean      |        4.684 |        0.440 |        1.788 |        1.771 |        6.099 |        9.215 |         ---- |
| nanmedian    |       15.326 |        9.392 |         ---- |         ---- |        8.036 |       15.220 |         ---- |
| nanquantile  |       17.557 |        9.494 |         ---- |         ---- |        7.734 |       21.325 |         ---- |
| nantrapezoid |        6.582 |        1.890 |         ---- |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.110 |        0.510 |        1.936 |        1.959 |        6.733 |        9.411 |         ---- |
| nanstd       |        6.248 |        0.511 |        1.975 |        1.948 |        6.466 |        9.672 |         ---- |
| nanskew      |        8.478 |        3.301 |         ---- |         ---- |        6.689 |       10.046 |         ---- |
| nankurt      |        8.483 |        3.259 |         ---- |         ---- |        7.282 |       10.591 |         ---- |
| nanfirst     |        4.859 |        0.337 |        0.325 |        0.323 |        6.040 |        8.085 |         ---- |
| nanlast      |        4.228 |        0.449 |        1.623 |        1.611 |        5.946 |        8.253 |         ---- |
| nanargmin    |        6.057 |        0.757 |        1.975 |        1.949 |        8.299 |        8.411 |         ---- |
| nanargmax    |        3.870 |        0.801 |        2.017 |        2.011 |        6.947 |        8.699 |         ---- |
| cumsum       |       15.176 |        0.556 |         ---- |         ---- |        9.880 |        7.601 |         ---- |
| cumprod      |         ---- |        0.584 |         ---- |         ---- |        8.666 |        7.417 |         ---- |
| cummax       |         ---- |        0.790 |         ---- |         ---- |        9.250 |        6.604 |         ---- |
| cummin       |         ---- |        0.763 |         ---- |         ---- |        8.736 |        6.300 |         ---- |
| arbitrary    |      115.266 |        6.039 |         ---- |         ---- |       80.268 |       96.240 |         ---- |
| sort         |       70.556 |       24.860 |         ---- |         ---- |         ---- |         ---- |         ---- |
| array        |        7.575 |        1.935 |         ---- |         ---- |         ---- |         ---- |         ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
