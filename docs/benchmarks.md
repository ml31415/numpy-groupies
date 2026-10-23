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
| sum          |        0.923 |        0.283 |        0.484 |        2.771 |        6.905 |        0.909 |
| prod         |        0.862 |        0.274 |        0.469 |        7.215 |        5.949 |        0.918 |
| min          |        1.690 |        0.288 |        0.849 |        3.611 |        6.539 |        1.736 |
| max          |        1.755 |        0.307 |        0.906 |        3.296 |        6.560 |        1.753 |
| len          |        0.838 |        0.220 |        0.504 |        2.696 |        5.372 |        0.966 |
| nunique      |       15.308 |        8.240 |         ---- |       12.366 |       61.704 |         ---- |
| all          |        1.874 |        0.276 |        1.615 |        2.872 |        7.141 |       34.435 |
| any          |        2.521 |        0.271 |        1.613 |        2.856 |        7.046 |       31.551 |
| anynan       |        0.899 |        0.228 |        2.719 |        2.583 |        7.179 |        5.395 |
| allnan       |        2.121 |        0.264 |        3.303 |        2.669 |        7.140 |        7.889 |
| mean         |        1.561 |        0.304 |        0.557 |        2.441 |        7.435 |         ---- |
| median       |       12.563 |        9.472 |         ---- |        4.247 |       16.301 |         ---- |
| quantile     |       15.770 |        9.460 |         ---- |        4.236 |       23.541 |         ---- |
| trapezoid    |        3.775 |        0.667 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.186 |        0.669 |        0.722 |        2.907 |        7.958 |         ---- |
| var          |        3.193 |        0.709 |        0.724 |        3.104 |        7.842 |         ---- |
| skew         |        5.827 |        2.592 |         ---- |        3.926 |        9.780 |         ---- |
| kurt         |        5.975 |        2.584 |         ---- |        4.590 |       11.139 |         ---- |
| first        |        1.780 |        0.302 |        0.319 |        2.516 |        6.486 |         ---- |
| last         |        0.884 |        0.417 |        0.463 |        2.616 |        6.844 |         ---- |
| argmax       |        2.673 |        0.659 |        0.915 |        7.811 |        7.169 |         ---- |
| argmin       |        4.952 |        0.626 |        0.857 |        7.863 |        7.177 |         ---- |
| nansum       |        4.226 |        0.233 |        1.799 |        6.272 |        8.267 |         ---- |
| nanprod      |        4.291 |        0.231 |        1.818 |        9.407 |        7.935 |         ---- |
| nanmin       |        4.861 |        0.303 |        2.046 |        6.296 |        8.288 |         ---- |
| nanmax       |        4.813 |        0.355 |        2.127 |        6.063 |        8.186 |         ---- |
| nanlen       |        2.508 |        0.264 |        1.742 |        5.990 |        7.173 |         ---- |
| nannunique   |       16.627 |        7.748 |         ---- |       13.775 |       30.639 |         ---- |
| nanall       |        5.165 |        0.330 |        3.351 |        6.267 |        9.075 |         ---- |
| nanany       |        5.796 |        0.233 |        2.713 |        6.594 |        8.420 |         ---- |
| nanmean      |        4.850 |        0.420 |        1.856 |        6.301 |        8.891 |         ---- |
| nanmedian    |       15.112 |        9.373 |         ---- |        7.263 |       15.322 |         ---- |
| nanquantile  |       17.358 |        9.453 |         ---- |        7.428 |       20.865 |         ---- |
| nantrapezoid |        6.627 |        1.904 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.202 |        1.846 |        1.955 |        6.467 |        9.142 |         ---- |
| nanstd       |        6.154 |        1.858 |        1.960 |        6.460 |        9.330 |         ---- |
| nanskew      |        8.464 |        3.303 |         ---- |        6.899 |        9.775 |         ---- |
| nankurt      |        8.414 |        3.322 |         ---- |        6.901 |       10.500 |         ---- |
| nanfirst     |        4.945 |        0.339 |        0.342 |        6.045 |        8.097 |         ---- |
| nanlast      |        4.259 |        0.463 |        1.637 |        6.172 |        8.388 |         ---- |
| nanargmin    |        6.313 |        0.769 |        1.976 |        8.118 |        8.366 |         ---- |
| nanargmax    |        3.903 |        0.822 |        2.027 |        7.573 |        8.546 |         ---- |
| cumsum       |       16.269 |        0.570 |         ---- |       10.154 |        7.524 |         ---- |
| cumprod      |         ---- |        0.573 |         ---- |        9.301 |        7.539 |         ---- |
| cummax       |         ---- |        0.798 |         ---- |        9.709 |        7.013 |         ---- |
| cummin       |         ---- |        0.803 |         ---- |        9.549 |        6.605 |         ---- |
| arbitrary    |      114.738 |        5.985 |         ---- |       82.364 |       98.066 |         ---- |
| sort         |       71.085 |       24.807 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.000 |        1.961 |         ---- |         ---- |         ---- |         ---- |
*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
