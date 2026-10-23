# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of three passes of the table (`--repeat=5 --runs=3` of `numpy_groupies.benchmarks`), every cell measured in one block of five executions. Lower is better. An untimed call precedes every block: the threads of a pool spin for a moment after their region and sleep afterwards, and the first call after that would measure the waking up instead of the work.
- The `polars` column uses all the cores of the machine (6 threads here). The `numba` column uses them too, for the operations that can be reduced in chunks - the sums, the products, the counts, the extremes and the mean: their data is split into chunks that are reduced separately and folded afterwards. The chunks are more than the threads, so that a core that is busy with something else does not hold up the whole call - the threads that are done take the chunks that are left. The state is one per thread, not one per chunk, so the memory does not grow with the number of the chunks. The operations that need the values of a group together (the order statistics) and the ones that give an output per value (the cumulatives) stay single-threaded. The difference matters most for the order statistics: with one thread polars needs 8.9ms for the median and 8.5ms for the quantile, against 4.6ms and 4.4ms with six, which puts it level with the single-threaded Numba kernels (9.5ms and 9.5ms).
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number - 8.2ms for the Numba kernel against 15.3ms for the NumPy one, which has to sort the values of every group (numba's sort is about four times slower than numpy's, which vectorizes it).
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=3` measures everything three times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --polars --runs=3
```

The participants get the data as it is - they must not write into it, since that would change the results of the runs that follow. The benchmark checks the data after every function, undoes a change if there was one and reports the row as not measured.

`--funcs min,max` measures the named functions only, instead of all of them. `--threads N` sets the number of threads the numba kernels use - they split their data into more chunks than that, so that the threads can take the chunks from the queue of the pool and a busy core does not hold up the whole call. Without it they use every hardware thread of the machine. See `--help` for further options. `----` in the table below means the function is not available in that implementation, the row `arbitrary` is a custom Python callable passed as `func`. The `numbagg`, `pandas` and `polars` columns are the group-by machinery of those libraries, the `ufunc` column is numpy's own `ufunc.at` - the last three are reference implementations for the comparison, not backends of this package.

## Results

| function     |        numpy |        numba |      numbagg |       polars |       pandas |        ufunc |
| ------------ | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: |
| sum          |        0.958 |        0.347 |        0.496 |        2.793 |        7.131 |        0.932 |
| prod         |        0.896 |        0.332 |        0.503 |        6.928 |        6.005 |        0.893 |
| min          |        1.692 |        0.311 |        0.853 |        3.307 |        6.357 |        1.688 |
| max          |        1.672 |        0.392 |        0.888 |        3.400 |        6.447 |        1.668 |
| len          |        0.833 |        0.185 |        0.481 |        2.554 |        5.499 |        0.939 |
| nunique      |       15.537 |        8.274 |         ---- |       13.732 |       60.011 |         ---- |
| all          |        1.825 |        0.326 |        1.630 |        2.817 |        6.884 |       32.702 |
| any          |        2.518 |        0.240 |        1.632 |        2.862 |        6.608 |       30.892 |
| anynan       |        0.906 |        0.246 |        2.720 |        3.318 |        6.444 |        5.372 |
| allnan       |        2.168 |        0.264 |        3.315 |        3.147 |        6.864 |        7.771 |
| mean         |        1.554 |        0.368 |        0.574 |        2.811 |        7.547 |         ---- |
| median       |       13.331 |        9.807 |         ---- |        4.157 |       16.114 |         ---- |
| quantile     |       16.259 |        9.545 |         ---- |        4.308 |       23.815 |         ---- |
| trapezoid    |        3.860 |        0.731 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.252 |        0.318 |        0.707 |        2.508 |        7.787 |         ---- |
| var          |        3.102 |        0.305 |        0.702 |        2.951 |        7.859 |         ---- |
| skew         |        5.953 |        2.611 |         ---- |        4.270 |        9.278 |         ---- |
| kurt         |        5.800 |        2.671 |         ---- |        4.127 |       10.809 |         ---- |
| first        |        1.831 |        0.318 |        0.323 |        2.056 |        6.119 |         ---- |
| last         |        0.873 |        0.375 |        0.446 |        2.588 |        6.381 |         ---- |
| argmax       |        2.596 |        0.631 |        0.938 |        8.606 |        7.157 |         ---- |
| argmin       |        4.860 |        0.611 |        0.852 |        7.047 |        8.187 |         ---- |
| nansum       |        4.353 |        0.273 |        1.800 |        6.741 |        8.644 |         ---- |
| nanprod      |        4.357 |        0.303 |        1.804 |        8.903 |        7.557 |         ---- |
| nanmin       |        4.843 |        0.347 |        2.024 |        6.473 |        8.296 |         ---- |
| nanmax       |        5.193 |        0.443 |        2.113 |        6.648 |        8.260 |         ---- |
| nanlen       |        2.489 |        0.274 |        1.756 |        6.416 |        7.619 |         ---- |
| nannunique   |       16.941 |        7.705 |         ---- |       14.315 |       29.648 |         ---- |
| nanall       |        5.184 |        0.324 |        3.316 |        6.466 |        8.495 |         ---- |
| nanany       |        5.781 |        0.260 |        2.738 |        6.335 |        8.802 |         ---- |
| nanmean      |        4.725 |        0.438 |        1.810 |        6.326 |        8.949 |         ---- |
| nanmedian    |       14.845 |        9.584 |         ---- |        7.481 |       15.514 |         ---- |
| nanquantile  |       18.203 |        9.534 |         ---- |        7.378 |       21.166 |         ---- |
| nantrapezoid |        6.861 |        1.889 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.077 |        0.486 |        1.974 |        6.657 |        8.997 |         ---- |
| nanstd       |        6.163 |        0.543 |        1.996 |        6.739 |        9.194 |         ---- |
| nanskew      |        8.789 |        3.369 |         ---- |        7.282 |        9.420 |         ---- |
| nankurt      |        8.663 |        3.333 |         ---- |        7.428 |       10.436 |         ---- |
| nanfirst     |        4.918 |        0.358 |        0.344 |        6.229 |        7.735 |         ---- |
| nanlast      |        4.215 |        0.446 |        1.640 |        6.857 |        8.481 |         ---- |
| nanargmin    |        6.309 |        0.773 |        1.966 |        8.120 |        8.403 |         ---- |
| nanargmax    |        3.923 |        0.823 |        2.022 |        7.557 |        8.442 |         ---- |
| cumsum       |       16.318 |        0.573 |         ---- |        9.816 |        7.372 |         ---- |
| cumprod      |         ---- |        0.548 |         ---- |       11.132 |        8.321 |         ---- |
| cummax       |         ---- |        0.819 |         ---- |        9.607 |        6.530 |         ---- |
| cummin       |         ---- |        0.838 |         ---- |        9.214 |        6.382 |         ---- |
| arbitrary    |      110.769 |        5.977 |         ---- |       81.907 |       95.177 |         ---- |
| sort         |       70.692 |       24.493 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.210 |        2.014 |         ---- |         ---- |         ---- |         ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
