# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of ten passes of the table (`--repeat=5 --runs=10` of `numpy_groupies.benchmarks`), every cell measured in one block of five executions. Lower is better. An untimed call precedes every block: the threads of a pool spin for a moment after their region and sleep afterwards, and the first call after that would measure the waking up instead of the work.
- The `numpy` and `numba` columns are the two backends of this package - the pure numpy one and the compiled one. The `numba` one is optimized as far as we could get it: it uses the threads of the machine where that pays for itself, and it falls back to a single thread where it does not. The other columns are the group-by machinery of the libraries of the same name and numpy's own `ufunc.at` - reference implementations for the comparison, not backends of this package.
- The conditions are meant to be the same for every participant: the same data, the same arguments, the same machine, and the threads of the machine are as much theirs as ours. What a participant has to build around the data before it can work on it is its own business, and it is small.
- The `nunique` counts the distinct values of a group in a hash table, which needs no ordering of the values and so is linear in their number. The counts of the groups are cut into ranges for the threads, like the order statistics.
- Software: Python 3.14.7, NumPy 2.5.4, Numba 0.68.0, pandas 3.0.6, polars 2.0.0, numbagg 0.9.6 (Linux, x86_64).

Absolute numbers depend heavily on the machine, Python and library versions, so rely on the ratios between columns rather than on the values. On shared machines single runs wobble, so `--runs=10` measures everything ten times and reports the fastest measurement of each cell. Re-run on your own hardware, from the repository root:

```sh
python -m numpy_groupies.benchmarks --ufunc --numbagg --pandas --polars --runs=10
```

The participants get the data as it is - they must not write into it, since that would change the results of the runs that follow. The benchmark checks the data after every function, undoes a change if there was one and reports the row as not measured.

## The temporal table

`--temporal` measures the functions that take `datetime64` and `timedelta64` values instead of the numeric table, one block of rows per dtype:

- 500,000 indices as above; the values are `datetime64[ns]` and `timedelta64[ns]` ticks.
- The datetime values gather one day after the epoch, spread over a window as wide as the data is long: close enough that the mean of the ticks stays on the exact fast path of its kernel even with the falsy values in the group, and wide enough that `nunique` and `sort` see distinct values.
- The durations stay below a millisecond, so that the sums and the accumulations of every group stay inside `int64` - the table measures the work, not a wraparound.
- As in the numeric table, everything above a fifth of the window is set to the tick `0` (the falsy values of the boolean functions), and for the `nan…` functions about 20% of the remaining values are `NaT`.
- The functions a dtype refuses are left out of its block: the sum and the accumulations belong to the durations alone, the median does too - numpy adds two middles and has no addition of datetime64 values - while the quantile and the percentile numpy computes of both, so they are in the block of the datetime64 values as well. The moments belong to neither. `--funcs` narrows both blocks by name, and a name neither dtype knows is an error.

The result of a run on the machine of the numeric table (same conditions, `--runs=10`, milliseconds, lower is better):

| function                |        numpy |        numba |
| ----------------------- | -----------: | -----------: |
| min datetime64          |        1.645 |        0.386 |
| max datetime64          |        1.329 |        0.418 |
| nanmin datetime64       |        4.988 |        0.443 |
| nanmax datetime64       |        4.426 |        0.433 |
| first datetime64        |        1.795 |        0.327 |
| last datetime64         |        0.878 |        0.319 |
| nanfirst datetime64     |        5.064 |        0.306 |
| nanlast datetime64      |        4.308 |        0.317 |
| argmin datetime64       |        4.900 |        0.478 |
| argmax datetime64       |        2.905 |        0.484 |
| nanargmin datetime64    |        8.537 |        0.612 |
| nanargmax datetime64    |        6.401 |        0.605 |
| len datetime64          |        0.813 |        0.255 |
| nanlen datetime64       |        4.294 |        0.319 |
| nunique datetime64      |       15.062 |        3.242 |
| nannunique datetime64   |       16.343 |        2.732 |
| all datetime64          |        1.898 |        0.870 |
| any datetime64          |        2.884 |        0.838 |
| allnan datetime64       |        2.300 |        0.361 |
| anynan datetime64       |        1.027 |        0.341 |
| mean datetime64         |        5.303 |        0.683 |
| nanmean datetime64      |        7.950 |        0.745 |
| quantile datetime64     |       42.244 |        3.023 |
| nanquantile datetime64  |       37.977 |        3.552 |
| mode datetime64         |       26.661 |        3.569 |
| sort datetime64         |       53.316 |        5.861 |
| array datetime64        |        7.372 |        7.977 |
| topk datetime64         |       13.560 |        3.116 |
| min timedelta64         |        1.715 |        0.501 |
| max timedelta64         |        1.352 |        0.441 |
| nanmin timedelta64      |        4.916 |        0.427 |
| nanmax timedelta64      |        4.425 |        0.466 |
| first timedelta64       |        1.748 |        0.341 |
| last timedelta64        |        0.855 |        0.329 |
| nanfirst timedelta64    |        4.990 |        0.335 |
| nanlast timedelta64     |        4.342 |        0.332 |
| argmin timedelta64      |        4.877 |        0.512 |
| argmax timedelta64      |        2.874 |        0.519 |
| nanargmin timedelta64   |        8.285 |        0.608 |
| nanargmax timedelta64   |        6.224 |        0.622 |
| len timedelta64         |        0.810 |        0.260 |
| nanlen timedelta64      |        4.277 |        0.314 |
| nunique timedelta64     |       14.708 |        2.674 |
| nannunique timedelta64  |       16.179 |        2.757 |
| all timedelta64         |        1.851 |        0.863 |
| any timedelta64         |        2.829 |        0.829 |
| allnan timedelta64      |        2.248 |        0.363 |
| anynan timedelta64      |        1.009 |        0.320 |
| mean timedelta64        |        5.067 |        0.629 |
| nanmean timedelta64     |        7.547 |        0.755 |
| quantile timedelta64    |       42.240 |        2.977 |
| nanquantile timedelta64 |       37.621 |        2.885 |
| mode timedelta64        |       25.741 |        3.119 |
| sort timedelta64        |       53.505 |        6.445 |
| array timedelta64       |        6.973 |        7.212 |
| topk timedelta64        |       13.410 |        3.093 |
| sum timedelta64         |        1.591 |        0.354 |
| nansum timedelta64      |        4.742 |        0.343 |
| cumsum timedelta64      |       13.860 |        0.820 |
| nancumsum timedelta64   |       54.176 |        0.799 |
| cummax timedelta64      |         ---- |        1.073 |
| cummin timedelta64      |         ---- |        1.004 |
| median timedelta64      |       41.216 |        2.759 |
| nanmedian timedelta64   |       36.262 |        3.131 |

The kernels of the temporal rows are the ones of the numeric table working on the values or their ticks: the selection of the extrema, the positions, the counts and the truth of the values orders them and finds their `NaT` as it finds a nan, the sums and the accumulations of the durations add in the compiled code, the mean and the median of the ticks answer in exact int64 arithmetic, the topk of the ticks selects the wanted ones out of the ticks of every group, and the quantile of the ticks interpolates the way numpy interpolates it - the difference of the two ticks it falls between is a duration, its product with the fraction is truncated towards zero, and the answer is a tick again. The numba backend compiles all of it: the quantile of the ticks runs in a kernel of its own, with the same counting sort and the same selection the median uses, and all five methods of numpy are in it. The numpy backend is the reference implementation and computes the quantile of the ticks in python - that is why its quantile takes ten times the time of the median. The nan variants sit at the level of their float counterparts: the check of the NaT costs what the check of the nan costs.

## The calibration of the parallel path

Which calls of the numba kernels take the parallel path and which do not is
decided by a cost model, and the model is fitted on the machine it runs on:

```sh
python -m numpy_groupies.benchmarks.mt
```

The tool measures the parallel kernels against the sequential ones over a
grid of sizes, group counts and thread counts - the chunked path (the sum),
the group path (the median, the quantile, the sort, the topk, the mode and
the number of the distinct values) and the gather - fits the constants of the model to the
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
| sum          |        0.936 |        0.248 |        0.468 |        2.545 |        6.692 |        0.926 |
| prod         |        0.895 |        0.242 |        0.437 |        6.668 |        6.047 |        0.909 |
| min          |        1.716 |        0.277 |        0.854 |        3.114 |        6.566 |        1.705 |
| max          |        1.703 |        0.294 |        0.879 |        3.401 |        6.495 |        1.749 |
| len          |        0.812 |        0.163 |        0.492 |        2.811 |        5.262 |        0.938 |
| nunique      |       15.488 |        3.085 |         ---- |       12.259 |       56.783 |         ---- |
| all          |        1.813 |        0.292 |        1.613 |        2.949 |        6.933 |       32.799 |
| any          |        2.553 |        0.229 |        1.586 |        2.527 |        6.784 |       30.584 |
| anynan       |        0.902 |        0.229 |        2.725 |        2.497 |        6.852 |        5.544 |
| allnan       |        2.111 |        0.270 |        3.310 |        3.057 |        6.790 |        7.954 |
| mean         |        1.521 |        0.255 |        0.537 |        2.849 |        7.385 |         ---- |
| median       |       13.088 |        2.876 |         ---- |        3.983 |       16.008 |         ---- |
| quantile     |       14.847 |        3.038 |         ---- |        4.236 |       23.247 |         ---- |
| mode         |       26.215 |        3.181 |         ---- |         ---- |         ---- |         ---- |
| trapezoid    |        3.788 |        0.625 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.165 |        0.470 |        0.683 |        2.926 |        7.647 |         ---- |
| var          |        3.155 |        0.452 |        0.690 |        3.093 |        7.661 |         ---- |
| skew         |        8.020 |        1.165 |         ---- |        3.563 |        9.460 |         ---- |
| kurt         |        8.140 |        0.942 |         ---- |        4.009 |       10.532 |         ---- |
| first        |        1.784 |        0.219 |        0.333 |        2.643 |        6.133 |         ---- |
| last         |        0.855 |        0.213 |        0.447 |        2.389 |        6.727 |         ---- |
| argmax       |        2.625 |        0.371 |        0.907 |        7.579 |        7.066 |         ---- |
| argmin       |        4.947 |        0.312 |        0.854 |        7.672 |        7.148 |         ---- |
| sort         |       71.753 |        6.952 |         ---- |         ---- |         ---- |         ---- |
| topk         |       14.541 |        2.876 |         ---- |         ---- |         ---- |         ---- |
| array        |        7.183 |        1.973 |         ---- |         ---- |         ---- |         ---- |
| nansum       |        4.161 |        0.263 |        1.788 |        6.229 |        8.444 |         ---- |
| nanprod      |        4.191 |        0.257 |        1.809 |        9.500 |        7.777 |         ---- |
| nanmin       |        4.949 |        0.320 |        1.980 |        6.266 |        8.198 |         ---- |
| nanmax       |        4.902 |        0.401 |        2.042 |        6.438 |        8.355 |         ---- |
| nanlen       |        2.498 |        0.274 |        1.734 |        6.022 |        7.249 |         ---- |
| nannunique   |       16.610 |        2.867 |         ---- |       14.923 |       29.248 |         ---- |
| nanall       |        4.893 |        0.302 |        3.301 |        6.428 |        8.149 |         ---- |
| nanany       |        5.797 |        0.249 |        2.689 |        6.273 |        8.271 |         ---- |
| nanmean      |        4.691 |        0.412 |        1.795 |        6.150 |        9.032 |         ---- |
| nanmedian    |       15.056 |        3.062 |         ---- |        7.289 |       14.956 |         ---- |
| nanquantile  |       17.105 |        3.406 |         ---- |        8.151 |       21.318 |         ---- |
| nantrapezoid |        6.673 |        1.851 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.210 |        0.655 |        1.942 |        6.373 |        8.681 |         ---- |
| nanstd       |        6.233 |        0.622 |        1.963 |        6.542 |        9.164 |         ---- |
| nanskew      |       10.314 |        1.094 |         ---- |        6.890 |        9.673 |         ---- |
| nankurt      |       10.270 |        1.067 |         ---- |        6.846 |       10.470 |         ---- |
| nanfirst     |        4.912 |        0.206 |        0.313 |        6.494 |        8.044 |         ---- |
| nanlast      |        4.294 |        0.218 |        1.641 |        5.998 |        8.532 |         ---- |
| nanargmin    |        6.250 |        0.493 |        1.959 |        7.995 |        8.471 |         ---- |
| nanargmax    |        3.907 |        0.577 |        2.034 |        7.681 |        8.638 |         ---- |
| cumsum       |       15.876 |        0.565 |         ---- |        9.164 |        7.290 |         ---- |
| cumprod      |         ---- |        0.604 |         ---- |        9.660 |        7.086 |         ---- |
| cummax       |         ---- |        0.788 |         ---- |        9.378 |        6.328 |         ---- |
| cummin       |         ---- |        0.769 |         ---- |        9.444 |        6.402 |         ---- |
| arbitrary    |      111.997 |        2.355 |         ---- |       80.605 |       93.727 |         ---- |

*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
