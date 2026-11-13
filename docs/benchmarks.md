# Benchmarks

Complete benchmark results for `aggregate`. A short summary is in the [README](../README.md#performance).

## Setup

- 500,000 indices, uniformly drawn from `[0, 1000)`.
- The values of `a` are uniform on `[0, 1)`, and everything above `0.8` is then set to `0`, so that there are falsy values for the boolean functions.
- For the `nan…` functions, of the remaining values those below `0.2` are set to `nan` (about 20%), leaving the rest on the interval `[0.2, 0.8]`.
- Times in **milliseconds**, the fastest of ten passes of the table (`--repeat=5 --runs=10` of `numpy_groupies.benchmarks`), every cell measured five times and the fastest of the five counts. Lower is better. An untimed call precedes every measurement: the threads of a pool spin for a moment after their region and sleep afterwards, and the first call after that would measure the waking up instead of the work. A single call can also be slower than the others - the threads of a pool wake up, the scheduler moves the work to another core - and the fastest of them is the one that measures the work and not the noise.
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
| min datetime64          |        1.647 |        0.356 |
| max datetime64          |        1.315 |        0.424 |
| nanmin datetime64       |        4.922 |        0.473 |
| nanmax datetime64       |        4.347 |        0.474 |
| first datetime64        |        1.733 |        0.319 |
| last datetime64         |        0.827 |        0.284 |
| nanfirst datetime64     |        4.955 |        0.299 |
| nanlast datetime64      |        4.299 |        0.292 |
| argmin datetime64       |        4.871 |        0.450 |
| argmax datetime64       |        2.854 |        0.414 |
| nanargmin datetime64    |        8.278 |        0.593 |
| nanargmax datetime64    |        6.294 |        0.577 |
| len datetime64          |        0.799 |        0.247 |
| nanlen datetime64       |        4.236 |        0.291 |
| nunique datetime64      |       14.127 |        2.479 |
| nannunique datetime64   |       15.917 |        2.266 |
| all datetime64          |        1.801 |        0.766 |
| any datetime64          |        2.781 |        0.747 |
| allnan datetime64       |        2.217 |        0.330 |
| anynan datetime64       |        1.018 |        0.323 |
| mean datetime64         |        5.193 |        0.645 |
| nanmean datetime64      |        7.663 |        0.712 |
| quantile datetime64     |       41.348 |        2.583 |
| nanquantile datetime64  |       37.515 |        2.497 |
| mode datetime64         |       25.193 |        2.685 |
| sort datetime64         |       52.879 |        5.500 |
| array datetime64        |        6.930 |        7.125 |
| topk datetime64         |       12.982 |        2.850 |
| min timedelta64         |        1.636 |        0.340 |
| max timedelta64         |        1.312 |        0.382 |
| nanmin timedelta64      |        4.911 |        0.393 |
| nanmax timedelta64      |        4.399 |        0.444 |
| first timedelta64       |        1.733 |        0.337 |
| last timedelta64        |        0.836 |        0.301 |
| nanfirst timedelta64    |        4.916 |        0.293 |
| nanlast timedelta64     |        4.251 |        0.306 |
| argmin timedelta64      |        4.831 |        0.465 |
| argmax timedelta64      |        2.849 |        0.458 |
| nanargmin timedelta64   |        8.179 |        0.585 |
| nanargmax timedelta64   |        6.215 |        0.559 |
| len timedelta64         |        0.787 |        0.248 |
| nanlen timedelta64      |        4.186 |        0.287 |
| nunique timedelta64     |       13.884 |        2.550 |
| nannunique timedelta64  |       15.792 |        2.421 |
| all timedelta64         |        1.791 |        0.790 |
| any timedelta64         |        2.738 |        0.763 |
| allnan timedelta64      |        2.211 |        0.304 |
| anynan timedelta64      |        1.009 |        0.285 |
| mean timedelta64        |        5.061 |        0.611 |
| nanmean timedelta64     |        7.421 |        0.685 |
| quantile timedelta64    |       41.664 |        2.590 |
| nanquantile timedelta64 |       37.053 |        3.220 |
| mode timedelta64        |       24.819 |        2.827 |
| sort timedelta64        |       52.183 |        5.440 |
| array timedelta64       |        6.960 |        7.131 |
| topk timedelta64        |       12.509 |        2.865 |
| sum timedelta64         |        1.623 |        0.324 |
| nansum timedelta64      |        4.580 |        0.312 |
| cumsum timedelta64      |       12.711 |        0.785 |
| nancumsum timedelta64   |       53.773 |        0.784 |
| cummax timedelta64      |         ---- |        1.046 |
| cummin timedelta64      |         ---- |        0.995 |
| median timedelta64      |       40.171 |        2.715 |
| nanmedian timedelta64   |       35.740 |        2.495 |

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
result, and the constants in `numpy_groupies/npg_numba/mt.py` are the
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
| sum          |        0.953 |        0.223 |        0.477 |        2.407 |        6.468 |        0.903 |
| prod         |        0.884 |        0.247 |        0.422 |        5.419 |        5.533 |        0.863 |
| sumofsquares |        1.059 |        0.217 |        0.449 |         ---- |         ---- |         ---- |
| min          |        1.674 |        0.237 |        0.814 |        3.222 |        6.041 |        1.645 |
| max          |        1.661 |        0.262 |        0.852 |        3.263 |        6.012 |        1.662 |
| len          |        0.790 |        0.145 |        0.449 |        2.443 |        4.901 |        0.904 |
| nunique      |       15.340 |        2.660 |         ---- |       10.575 |       55.036 |         ---- |
| all          |        1.765 |        0.246 |        1.575 |        2.439 |        6.368 |       32.608 |
| any          |        2.480 |        0.258 |        1.583 |        2.502 |        6.308 |       30.589 |
| anynan       |        0.877 |        0.214 |        2.679 |        2.357 |        6.313 |        5.253 |
| allnan       |        2.085 |        0.247 |        3.278 |        2.526 |        6.487 |        7.539 |
| mean         |        1.492 |        0.230 |        0.527 |        2.181 |        6.841 |         ---- |
| median       |       12.133 |        2.844 |         ---- |        4.129 |       15.278 |         ---- |
| quantile     |       15.702 |        2.989 |         ---- |        3.821 |       22.842 |         ---- |
| mode         |       26.024 |        2.561 |         ---- |         ---- |         ---- |         ---- |
| trapezoid    |        3.792 |        0.683 |         ---- |         ---- |         ---- |         ---- |
| std          |        3.107 |        0.430 |        0.669 |        2.575 |        7.126 |         ---- |
| var          |        3.089 |        0.388 |        0.666 |        2.644 |        7.204 |         ---- |
| skew         |        7.841 |        0.980 |         ---- |        3.176 |        9.115 |         ---- |
| kurt         |        7.789 |        0.957 |         ---- |        3.789 |       10.056 |         ---- |
| first        |        1.727 |        0.208 |        0.293 |        2.506 |        5.756 |         ---- |
| last         |        0.845 |        0.227 |        0.427 |        2.560 |        6.080 |         ---- |
| argmax       |        2.556 |        0.337 |        0.886 |        6.752 |        6.693 |         ---- |
| argmin       |        4.812 |        0.289 |        0.824 |        6.585 |        6.615 |         ---- |
| sort         |       70.517 |        5.936 |         ---- |         ---- |         ---- |         ---- |
| topk         |       13.797 |        2.685 |         ---- |         ---- |         ---- |         ---- |
| array        |        6.716 |        1.927 |         ---- |         ---- |         ---- |         ---- |
| nansum       |        4.140 |        0.203 |        1.757 |        6.045 |        8.109 |         ---- |
| nanprod      |        4.114 |        0.214 |        1.756 |        8.167 |        7.266 |         ---- |
| nansumofsquares |     4.257 |        0.178 |        1.750 |         ---- |         ---- |         ---- |
| nanmin       |        4.796 |        0.275 |        1.941 |        6.135 |        7.701 |         ---- |
| nanmax       |        4.756 |        0.323 |        2.033 |        6.005 |        7.716 |         ---- |
| nanlen       |        2.413 |        0.209 |        1.702 |        5.794 |        6.824 |         ---- |
| nannunique   |       16.172 |        2.613 |         ---- |       12.945 |       28.645 |         ---- |
| nanall       |        4.837 |        0.252 |        3.288 |        5.836 |        8.215 |         ---- |
| nanany       |        5.618 |        0.226 |        2.668 |        6.028 |        8.136 |         ---- |
| nanmean      |        4.576 |        0.402 |        1.755 |        6.146 |        8.440 |         ---- |
| nanmedian    |       14.741 |        2.976 |         ---- |        7.278 |       14.460 |         ---- |
| nanquantile  |       16.882 |        2.937 |         ---- |        7.305 |       20.543 |         ---- |
| nantrapezoid |        6.497 |        1.861 |         ---- |         ---- |         ---- |         ---- |
| nanvar       |        6.019 |        0.655 |        1.908 |        6.003 |        8.632 |         ---- |
| nanstd       |        6.129 |        0.562 |        1.929 |        6.140 |        8.641 |         ---- |
| nanskew      |       10.272 |        1.050 |         ---- |        6.275 |        9.326 |         ---- |
| nankurt      |       10.028 |        0.983 |         ---- |        6.880 |       10.002 |         ---- |
| nanfirst     |        4.834 |        0.197 |        0.304 |        5.654 |        7.532 |         ---- |
| nanlast      |        4.159 |        0.204 |        1.596 |        5.813 |        7.650 |         ---- |
| nanargmin    |        6.154 |        0.463 |        1.900 |        5.788 |        7.866 |         ---- |
| nanargmax    |        3.836 |        0.517 |        1.997 |        6.416 |        7.992 |         ---- |
| cumsum       |       15.499 |        0.550 |         ---- |        8.693 |        6.780 |         ---- |
| cumprod      |         ---- |        0.540 |         ---- |        7.677 |        6.488 |         ---- |
| cummax       |         ---- |        0.763 |         ---- |        8.346 |        5.900 |         ---- |
| cummin       |         ---- |        0.761 |         ---- |        8.107 |        5.721 |         ---- |
| arbitrary    |      111.338 |        2.520 |         ---- |       78.176 |       92.887 |         ---- |

*Linux(x86_64), Python 3.14.7, Numpy 2.5.4, Numba 0.68.0, numbagg 0.9.6, Pandas 3.0.6, Polars 2.0.0 (6 threads max)*
