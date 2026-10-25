"""Calibration benchmark and cost model for the multithreaded kernels.

This module is a tool of its own, not a participant of the benchmark suite:
it measures the parallel variant of the kernels against the sequential one
and fits the cost models that decide between them.

Run it on every machine whose numbers should go into the model:

    python -m numpy_groupies.benchmarks.mt

It measures the parallel variant of the kernels against the sequential one
over a grid of data sizes, group counts and thread counts, fits the cost
models to the measurements and prints the constants and the crossovers they
imply.  No file is written - the output is the result.

What is measured:

  the chunked path   the sum: the sequential kernel of this script against
                     the parallel one, which splits the data into chunks and
                     gives every chunk its own partial sums.  Their ratio is
                     free of the machinery around the kernels and shows what
                     the machine can do.

  the group path     the median, the sort and the number of the distinct
                     values: the sequential kernel of the package against
                     its parallel one, which cuts the groups into ranges and
                     lets the threads work on the ranges.

  the gather         the gather of ``unpack_into`` against its parallel twin.

  the overhead       the fixed cost of entering the parallel path - warm
                     (the threads of the pool spinning) and cold (the pool
                     asleep, which is what a program that calls a kernel once
                     and waits pays).

  the scaling        how the parallel path scales with the number of the
                     threads, one process per thread count (--scaling).  The
                     threads of the pool spin after a region and disturb the
                     measurement of the next one in the same process, so the
                     question of how many threads are worth it is not asked
                     inside the grids above, where the thread counts follow
                     one another.

Every kernel is measured twice: warm and cold.  The threads of the pool spin
for a moment after a region and sleep afterwards - about ten milliseconds -
and waking them up costs twenty times what the region itself costs.  A
workload that is called in a loop finds the threads awake, a program that
calls a kernel once finds them asleep.  The floors of the model are fitted
on the cold measurement, and the model above the floors on the warm one.

The model of the chunked path:

    core(n)    = CORE0 * (working set / 1 MB) ** CORE_EXP
    sibling(n) = SMT_FRACTION * core(n)      # what an SMT thread adds
    peak(n)    = core(n) * PEAK_FACTOR       # what the machine delivers
    bandwidth(n, T) = min(peak(n),
                          core(n) * min(T, phys) + sibling(n) * max(0, T - phys))

The model has seven constants.  More of them did not help: the fit has
several solutions that explain the measurements equally well, and they all
come to the same decisions.  With seven, the constants that mean something
(the bandwidth of a core, the share of an SMT thread, the costs of the
groups) are the same, whichever start the search begins at.

    kernel_seq(n, g) = n * 16 / bandwidth(n, 1)  +  g * CG_SEQ
    kernel_mt(n, g, T) = n * 16 / bandwidth(n, T)
                          +  g * CG_MT + T * g * C_PART + OVERHEAD(T)

The bounds are structural: the SMT fraction lies between 0 and 1, the peak
factor is at least 1, the core bandwidth never exceeds the peak.  Nothing
can drift into the nonsense an unconstrained fit produces.

The model of the group path is of another shape: the values of a group have
to be together where they are worked on, so the groups are cut into ranges
and the threads work on the ranges - nothing is folded afterwards, and the
result is the one of the sequential kernel bit for bit.

    kernel_seq(n)        = n * NS_PER_VALUE
    kernel_mt(n, T)      = kernel_seq(n) * (SHARE + (1 - SHARE) / PARTS(T))
                           + OVERHEAD(T)
    PARTS(T)             = 1 + (T - 1) * PER_THREAD

``NS_PER_VALUE`` is what one value of the operation costs, ``SHARE`` the part
of it that stays single threaded (the counting sort that places the values
before the groups can be worked on) and ``PER_THREAD`` what a thread beyond
the first brings.

The script needs numpy and numba, nothing else; numpy_groupies is optional
for the chunked path and needed for the group path.  It checks that the
parallel result equals the sequential one before it reports a timing - a
wrong result is worse than no result.

Typical runtime: a few minutes with the default grid.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from multiprocessing.synchronize import SemLock

import numba as nb
import numpy as np

from numpy_groupies.aggregate_numba.mt import logical_core_count, physical_core_count

BYTES_PER_ELEMENT = 16.0  # a float64 value and an int64 index

# the thresholds of the decision
MT_MIN_VALUES = 100_000
MT_MAX_RATIO = 0.8

# the fixed cost of the parallel path, measured directly (an empty prange loop)
OVERHEAD_US = {1: 0.0, 2: 4.7, 3: 5.0, 4: 5.2, 6: 5.8, 8: 6.4, 16: 9.0}

# the pause that lets the threads of the pool go to sleep - it has to be
# longer than the time they spin after a region (about ten milliseconds on
# the machine of the fit)
COLD_PAUSE_S = 0.02

# the number of the calls of a cold measurement - enough that one disturbance
# of the machine does not decide it: the fastest of them is the one that was
# left alone, and one disturbance among four is not the fastest one
COLD_REPEAT = 4

# the operations that parallelize over the groups
#
# the quantile is the median with another index, and the model carries the
# two as one (one ns-per-value constant of the package) - the measurements of
# the two came out the same, so only the median is fitted and reported
GROUP_OPS = ("median", "sort", "nunique")

# the cost of a value of the group operations, as the package carries it -
# the tool measures its own and compares the two
_PACKAGE_NS_PER_VALUE: dict[str, float] = {}


def _package_ns_per_value() -> dict[str, float]:
    """The cost of a value of the group operations, as the package carries it."""
    global _PACKAGE_NS_PER_VALUE
    if _PACKAGE_NS_PER_VALUE:
        return _PACKAGE_NS_PER_VALUE
    try:
        from numpy_groupies.aggregate_numba import specials
    except ImportError:
        return {}
    _PACKAGE_NS_PER_VALUE = {
        "median": specials._NS_PER_VALUE_SELECT,
        "quantile": specials._NS_PER_VALUE_SELECT,
        "sort": specials._NS_PER_VALUE_SORT,
        "nunique": specials._NS_PER_VALUE_NUNIQUE,
    }
    return _PACKAGE_NS_PER_VALUE


def overhead_us(n_threads: int) -> float:
    """The fixed cost of entering the parallel path, in microseconds."""
    known = sorted(OVERHEAD_US)
    if n_threads in OVERHEAD_US:
        return OVERHEAD_US[n_threads]
    lower = max((t for t in known if t < n_threads), default=known[0])
    upper = min((t for t in known if t > n_threads), default=known[-1])
    if lower == upper:
        return OVERHEAD_US[lower]
    weight = (n_threads - lower) / (upper - lower)
    return OVERHEAD_US[lower] * (1 - weight) + OVERHEAD_US[upper] * weight


# ---------------------------------------------------------------------------
# the kernels
#
# the sequential one is the shape the numba backend uses: one pass over the
# data, the running value of every group in an output array.  the parallel
# one splits the data into chunks, gives every chunk its own partial sums,
# and adds them up afterwards.
#
# the original of the sequential kernel, for the record:
#
#     for i in range(group_idx.size):
#         out[group_idx[i]] += a[i]


@nb.njit(cache=True)
def sum_seq(group_idx, a, out):
    for i in range(group_idx.size):
        out[group_idx[i]] += a[i]
    return out


@nb.njit(parallel=True, cache=True)
def sum_mt(group_idx, a, partials, n_groups):
    """The parallel sum: one row of partials per thread.

    Every chunk accumulates into its own buffer and writes it to its own row
    afterwards, so that no two threads ever write to the same place.
    """
    n = group_idx.size
    n_threads = partials.shape[0]
    chunk = (n + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        start = t * chunk
        stop = min(start + chunk, n)
        acc = np.zeros(n_groups, a.dtype)
        for i in range(start, stop):
            acc[group_idx[i]] += a[i]
        partials[t] = acc
    return partials


@nb.njit(cache=True)
def gather_seq(ret, group_idx, out):
    for i in range(group_idx.size):
        out[i] = ret[group_idx[i]]
    return out


@nb.njit(parallel=True, cache=True)
def gather_mt(ret, group_idx, out):
    for i in nb.prange(group_idx.size):
        out[i] = ret[group_idx[i]]
    return out


@nb.njit(parallel=True, cache=True)
def empty_parallel(group_idx):
    """An empty parallel loop - it measures the fixed cost of the parallel path."""
    total = 0.0
    for i in nb.prange(group_idx.size):
        total += i
    return total


# ---------------------------------------------------------------------------
# the machine


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _l3_bytes() -> int:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10, check=False).stdout
        for line in out.splitlines():
            if "L3 cache" in line:
                parts = line.split(":", 1)[1].strip().split()
                number, unit = parts[0], parts[1]
                factor = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[unit]
                return int(number) * factor
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, IndexError):
        pass
    return 0


def _cgroup_quota() -> str:
    """The cpu quota of the container, if the process runs in one.

    A container whose quota is below the cores it sees answers the scaling
    question with numbers that no model can use - the quota has to be in the
    report for the numbers to be readable at all.
    """
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as handle:
                return handle.read().strip()
        except OSError:
            continue
    return "-"


def machine_info() -> dict:
    return {
        "cpu_model": _cpu_model(),
        "physical_cores": physical_core_count(),
        "hardware_threads": logical_core_count(),
        "cgroup_quota": _cgroup_quota(),
        "l3_bytes": _l3_bytes(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": nb.__version__,
        "numba_threads": nb.get_num_threads(),
        "numba_threading_layer": nb.threading_layer(),
    }


# ---------------------------------------------------------------------------
# the measurement


def measure(fn, repeat: int) -> tuple[float, float]:
    """The fastest and the median of the calls, in microseconds.

    The calls follow each other, so the threads of the pool are awake: this
    is the state of a workload that is called in a loop.

    The two numbers are the quality of the measurement: on a machine that is
    shared the fastest call is the one that was left alone, and the median
    is what the calls cost when the machine has something else to do.  A
    spread between them is the load of the machine, and a spread of a factor
    of two and more makes the numbers below worthless.
    """
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[0] * 1e6, times[len(times) // 2] * 1e6


def measure_pair(seq_fn, mt_fn, repeat: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """The times of two calls that are measured one after the other.

    The two calls are measured in the same round, one after the other: a
    hitch of the machine then hits both of them and not only one of them,
    and the fastest round is the one that counts.
    """
    seq_times = []
    mt_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        seq_fn()
        seq_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        mt_fn()
        mt_times.append(time.perf_counter() - start)
    seq_times.sort()
    mt_times.sort()
    middle = len(seq_times) // 2
    return (seq_times[0] * 1e6, seq_times[middle] * 1e6), (mt_times[0] * 1e6, mt_times[middle] * 1e6)


def measure_cold(fn, repeat: int, pause: float = COLD_PAUSE_S) -> tuple[float, float]:
    """The times of a call whose threads of the pool are asleep.

    Every call is preceded by a pause, so that the threads have gone to sleep
    - the case of a program that calls the kernel once and waits.  The pause
    has to be longer than the time the threads spin after a region.

    The *fastest* of the calls and not the median one: a call that follows a
    pause is the noisy one - what the machine did in the pause (the cache of
    the data, the pages of the kernel, the load of the other processes) is
    out of the caller's hands, and on a shared machine a call can be delayed
    by milliseconds.  The fastest of them is the one that was left alone,
    and it is the one the decision is about: a parallel path that does not
    pay in the calmest of the cold calls does not pay at all.

    The number of calls has to be big enough that one disturbance does not
    decide the measurement - a disturbance of ten milliseconds is nothing
    beside a call of half a millisecond, and it is gone after one call.
    """
    times = []
    for _ in range(repeat):
        time.sleep(pause)
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    middle = len(times) // 2
    return times[0] * 1e6, times[middle] * 1e6


def repeat_for(n_values: int) -> int:
    """Enough calls per measurement to get out of the noise, but not too many."""
    return int(min(100, max(3, 2e6 / max(n_values, 1))))


def _package_kernel():
    """The sequential kernel of numpy_groupies, if the package is there."""
    try:
        from numpy_groupies.aggregate_numba import sum as npg_sum
    except ImportError:  # the package is optional
        return None
    return npg_sum


def _group_handles():
    """The handles of the operations that parallelize over the groups."""
    try:
        from numpy_groupies.aggregate_numba import specials
    except ImportError:
        return {}
    handles = {}
    for name in GROUP_OPS:
        kernel = getattr(specials, name, None)
        if kernel is not None:
            handles[name] = kernel.plain
    return handles


def _group_callers(handle, group_idx, a, n_groups, out_dtype, aux_dtype, fill_value, n_threads):
    """The sequential and the parallel call of one group kernel.

    The two take the argument of their variant: the quantile its ``q``, the
    sort its ``reverse`` flag, the others nothing.
    """
    if handle.variant == "none":
        seq = lambda: handle.driver(group_idx, a, n_groups, out_dtype, aux_dtype, fill_value)
        par = lambda: handle.mt_driver(n_threads, group_idx, a, n_groups, out_dtype, aux_dtype, fill_value, 0)
    elif handle.variant == "q":
        seq = lambda: handle.driver(group_idx, a, n_groups, out_dtype, aux_dtype, fill_value, 0.5)
        par = lambda: handle.mt_driver(n_threads, group_idx, a, n_groups, out_dtype, aux_dtype, fill_value, 0.5)
    else:  # the reverse flag of the sort
        seq = lambda: handle.driver(group_idx, a, n_groups, out_dtype, aux_dtype, fill_value, False)
        par = lambda: handle.mt_driver(n_threads, group_idx, a, n_groups, out_dtype, aux_dtype, fill_value, False)
    return seq, par


# ---------------------------------------------------------------------------
# the grids


def run_chunked_grid(sizes, group_counts, thread_counts, rounds: int, cold: bool):
    """The measurements of the chunked path: the sum of every group.

    Two pairs of kernels are measured: the prototype of this script, whose
    ratio is free of the machinery around the kernels and shows what the
    machine can do (the bandwidth constants are fitted to it), and the
    kernels of the package, which are what the decisions are about.
    """
    rng = np.random.default_rng(100)
    rows = []
    pool = nb.get_num_threads()  # the size of the pool, before any set_num_threads

    npg_sum = _package_kernel()
    if npg_sum is None:
        print("numpy_groupies is not importable - the overhead of the package is unknown")

    points = len(sizes) * len(group_counts) * sum(1 for t in thread_counts if t <= pool)
    print(f"--- the chunked path, the prototype of this script: {points} points ---", flush=True)
    print("warming up the kernels ...", flush=True)
    warm = rng.integers(0, 10, 1000)
    warm_a = rng.random(1000)
    for t in thread_counts:
        if t <= pool:
            sum_mt(warm, warm_a, np.zeros((t, 10)), 10)
    empty_parallel(warm)
    sum_seq(warm, warm_a, np.zeros(10))
    if npg_sum is not None:
        npg_sum(warm, warm_a)
    print("         compiled, measuring", flush=True)

    for n_values in sizes:
        repeat = repeat_for(n_values)
        for n_groups in group_counts:
            group_idx = rng.integers(0, n_groups, n_values)
            a = rng.random(n_values)

            # the sequential kernel allocates its output and fills it - the
            # allocation is part of the cost, so it belongs into the timing
            def seq_call(group_idx=group_idx, a=a, n_groups=n_groups):
                out = np.zeros(n_groups)
                sum_seq(group_idx, a, out)
                return out

            if npg_sum is not None:

                def npg_call(group_idx=group_idx, a=a):
                    return npg_sum(group_idx, a)

            for n_threads in thread_counts:
                if n_threads > pool:
                    continue
                partials = np.zeros((n_threads, n_groups))

                def mt_call(group_idx=group_idx, a=a, partials=partials, n_groups=n_groups):
                    partials[...] = 0.0
                    sum_mt(group_idx, a, partials, n_groups)
                    return partials.sum(axis=0)

                # the result of the parallel kernel must be the one of the
                # sequential one - the order of the additions differs, so the
                # comparison is a close one, not an exact one
                want = seq_call()
                got = mt_call()
                if not np.allclose(got, want, rtol=1e-9, atol=1e-12):
                    raise SystemExit(
                        f"the parallel sum differs from the sequential one at "
                        f"n_values={n_values}, n_groups={n_groups}, n_threads={n_threads}"
                    )

                # the three are measured in the same round, one after the
                # other: a hitch of the machine then hits all of them and not
                # only one of them, and the fastest round is the one that
                # counts
                nb.set_num_threads(n_threads)
                seq = mt = npg = float("inf")
                seq_median = mt_median = 0.0
                for _ in range(rounds):
                    (seq, seq_median), (mt, mt_median) = measure_pair(seq_call, mt_call, repeat)
                    if npg_sum is not None:
                        npg, _ = measure(npg_call, repeat)
                nb.set_num_threads(pool)

                row = {
                    "path": "chunked",
                    "kernel": "prototype",
                    "n_values": n_values,
                    "n_groups": n_groups,
                    "n_threads": n_threads,
                    "seq_us": seq,
                    "mt_us": mt,
                    "npg_us": npg,
                    "seq_median_us": seq_median,
                    "mt_median_us": mt_median,
                }
                if cold:
                    # the same call with the threads of the pool asleep: the
                    # state of a program that calls the kernel once
                    row["seq_cold_us"], _ = measure_cold(seq_call, COLD_REPEAT)
                    row["mt_cold_us"], _ = measure_cold(mt_call, COLD_REPEAT)
                rows.append(row)
                print(
                    f"    chunked groups={n_groups:>8} threads={n_threads:>3}  "
                    f"seq={seq:10.1f} us  mt={mt:10.1f} us  ratio={mt / seq:.2f}",
                    flush=True,
                )

    # the kernels of the package: the decisions are about them, and the
    # prototype carries an overhead of its own (the numpy calls around the
    # kernel) that they do not have
    package_rows = run_package_chunked_grid(sizes, group_counts, thread_counts, rounds, cold)
    return rows + package_rows


def run_package_chunked_grid(sizes, group_counts, thread_counts, rounds: int, cold: bool):
    """The measurements of the chunked path with the kernels of the package."""
    try:
        from numpy_groupies.aggregate_numba import reductions
    except ImportError:
        print("numpy_groupies is not importable - the package kernels cannot be measured")
        return []

    rng = np.random.default_rng(100)
    rows = []
    pool = nb.get_num_threads()
    handle = reductions.sum.plain
    out_dtype = np.dtype(np.float64)
    aux_dtype = np.dtype(np.float64)

    points = len(sizes) * len(group_counts) * sum(1 for t in thread_counts if t <= pool)
    print(f"--- the chunked path, the kernels of the package: {points} points ---", flush=True)
    print("warming up the kernels of the package ...", flush=True)
    warm = rng.integers(0, 10, 1000)
    warm_a = rng.random(1000)
    handle.driver(warm, warm_a, 10, out_dtype, aux_dtype, 0.0)
    handle.mt_driver(2, warm, warm_a, 10, out_dtype, aux_dtype, 0.0)
    print("         compiled, measuring", flush=True)

    for n_values in sizes:
        repeat = repeat_for(n_values)
        for n_groups in group_counts:
            group_idx = rng.integers(0, n_groups, n_values)
            a = rng.random(n_values)
            for n_threads in thread_counts:
                if n_threads > pool:
                    continue

                def seq_call(group_idx=group_idx, a=a, n_groups=n_groups):
                    return handle.driver(group_idx, a, n_groups, out_dtype, aux_dtype, 0.0)

                def mt_call(group_idx=group_idx, a=a, n_groups=n_groups, n_threads=n_threads):
                    return handle.mt_driver(n_threads, group_idx, a, n_groups, out_dtype, aux_dtype, 0.0)

                nb.set_num_threads(n_threads)
                seq_us = mt_us = float("inf")
                seq_median = mt_median = 0.0
                for _ in range(rounds):
                    (seq_us, seq_median), (mt_us, mt_median) = measure_pair(seq_call, mt_call, repeat)
                nb.set_num_threads(pool)

                row = {
                    "path": "chunked",
                    "kernel": "package",
                    "n_values": n_values,
                    "n_groups": n_groups,
                    "n_threads": n_threads,
                    "seq_us": seq_us,
                    "mt_us": mt_us,
                    "npg_us": float("inf"),
                    "seq_median_us": seq_median,
                    "mt_median_us": mt_median,
                }
                if cold:
                    row["seq_cold_us"], _ = measure_cold(seq_call, COLD_REPEAT)
                    row["mt_cold_us"], _ = measure_cold(mt_call, COLD_REPEAT)
                rows.append(row)
                print(
                    f"    chunked (the package) groups={n_groups:>8} threads={n_threads:>3}  "
                    f"seq={seq_us:10.1f} us  mt={mt_us:10.1f} us  ratio={mt_us / seq_us:.2f}",
                    flush=True,
                )
    return rows


def run_group_grid(sizes, group_counts, thread_counts, rounds: int, cold: bool):
    """The measurements of the group path: the order statistics."""
    handles = _group_handles()
    if not handles:
        print("numpy_groupies is not importable - the group path cannot be measured")
        return []

    rng = np.random.default_rng(100)
    rows = []
    pool = nb.get_num_threads()
    out_dtype = np.dtype(np.float64)
    aux_dtype = np.dtype(np.float64)

    points = len(sizes) * len(group_counts) * sum(1 for t in thread_counts if t <= pool)
    print(f"--- the group path, {len(handles)} operations: {len(handles) * points} points ---", flush=True)
    print(f"warming up the group kernels ({len(handles)} operations, sequential and parallel each) ...", flush=True)
    warm = rng.integers(0, 10, 1000)
    warm_a = rng.random(1000)
    for name, handle in handles.items():
        seq, par = _group_callers(handle, warm, warm_a, 10, out_dtype, aux_dtype, np.nan, 2)
        seq()
        par()
    print("         compiled, measuring", flush=True)

    for name, handle in handles.items():
        for n_values in sizes:
            repeat = repeat_for(n_values)
            for n_groups in group_counts:
                group_idx = rng.integers(0, n_groups, n_values)
                a = rng.random(n_values)
                for n_threads in thread_counts:
                    if n_threads > pool:
                        continue
                    seq, par = _group_callers(handle, group_idx, a, n_groups, out_dtype, aux_dtype, np.nan, n_threads)
                    # the parallel path gives the result of the sequential one
                    # bit for bit - every group is worked on by one thread
                    want = seq()
                    got = par()
                    if not np.array_equal(want, got, equal_nan=True):
                        raise SystemExit(
                            f"the parallel {name} differs from the sequential one at "
                            f"n_values={n_values}, n_groups={n_groups}, n_threads={n_threads}"
                        )

                    nb.set_num_threads(n_threads)
                    seq_us = mt_us = float("inf")
                    seq_median = mt_median = 0.0
                    for _ in range(rounds):
                        (seq_us, seq_median), (mt_us, mt_median) = measure_pair(seq, par, repeat)
                    nb.set_num_threads(pool)

                    row = {
                        "path": "group",
                        "op": name,
                        "n_values": n_values,
                        "n_groups": n_groups,
                        "n_threads": n_threads,
                        "seq_us": seq_us,
                        "mt_us": mt_us,
                        "seq_median_us": seq_median,
                        "mt_median_us": mt_median,
                    }
                    if cold:
                        row["seq_cold_us"], _ = measure_cold(seq, COLD_REPEAT)
                        row["mt_cold_us"], _ = measure_cold(par, COLD_REPEAT)
                    rows.append(row)
                    print(
                        f"    {name:<8} groups={n_groups:>8} threads={n_threads:>3}  "
                        f"seq={seq_us:10.1f} us  mt={mt_us:10.1f} us  ratio={mt_us / seq_us:.2f}",
                        flush=True,
                    )
    return rows


def run_gather_grid(sizes, thread_counts, rounds: int, cold: bool):
    """The measurements of the gather: one value per value."""
    try:
        from numpy_groupies.aggregate_numba import gather as npg_gather
    except ImportError:
        print("numpy_groupies is not importable - the gather cannot be measured")
        return []

    rng = np.random.default_rng(100)
    rows = []
    pool = nb.get_num_threads()

    points = len(sizes) * sum(1 for t in thread_counts if t <= pool)
    print(f"--- the gather: {points} points ---", flush=True)
    print("warming up the gather ...", flush=True)
    warm = rng.integers(0, 10, 1000)
    npg_gather._gather_1d(rng.random(10), warm, np.empty(1000))
    npg_gather._gather_1d_mt(rng.random(10), warm, np.empty(1000), 2)
    print("         compiled, measuring", flush=True)

    for n_values in sizes:
        repeat = repeat_for(n_values)
        group_idx = rng.integers(0, 100, n_values)
        ret = rng.random(100)
        out = np.empty(n_values)

        def seq_call(group_idx=group_idx, ret=ret, out=out):
            return npg_gather._gather_1d(ret, group_idx, out)

        for n_threads in thread_counts:
            if n_threads > pool:
                continue

            def mt_call(group_idx=group_idx, ret=ret, out=out, n_threads=n_threads):
                return npg_gather._gather_1d_mt(ret, group_idx, out, n_threads)

            want = seq_call()
            got = mt_call()
            if not np.array_equal(want, got):
                raise SystemExit(f"the parallel gather differs from the sequential one at {n_values} values")

            nb.set_num_threads(n_threads)
            seq_us = mt_us = float("inf")
            seq_median = mt_median = 0.0
            for _ in range(rounds):
                (seq_us, seq_median), (mt_us, mt_median) = measure_pair(seq_call, mt_call, repeat)
            nb.set_num_threads(pool)

            row = {
                "path": "gather",
                "n_values": n_values,
                "n_groups": 1,
                "n_threads": n_threads,
                "seq_us": seq_us,
                "mt_us": mt_us,
                "seq_median_us": seq_median,
                "mt_median_us": mt_median,
            }
            if cold:
                row["seq_cold_us"], _ = measure_cold(seq_call, COLD_REPEAT)
                row["mt_cold_us"], _ = measure_cold(mt_call, COLD_REPEAT)
            rows.append(row)
            print(
                f"    gather  threads={n_threads:>3}  seq={seq_us:10.1f} us  "
                f"mt={mt_us:10.1f} us  ratio={mt_us / seq_us:.2f}",
                flush=True,
            )
    return rows


def run_overhead(thread_counts, rounds: int) -> list:
    """The fixed cost of a parallel region, warm and cold."""
    rng = np.random.default_rng(100)
    warm = rng.integers(0, 10, 1000)
    pool = nb.get_num_threads()
    rows = []
    print(f"--- the cost of the region: {len(thread_counts)} thread counts ---", flush=True)
    for n_threads in thread_counts:
        if n_threads > pool:
            continue
        nb.set_num_threads(n_threads)
        warm_us = cold_us = float("inf")
        warm_median = cold_median = 0.0
        for _ in range(rounds):
            warm_us, warm_median = measure(lambda: empty_parallel(warm), 200)
            cold_us, cold_median = measure_cold(lambda: empty_parallel(warm), COLD_REPEAT)
        nb.set_num_threads(pool)
        rows.append(
            {
                "n_threads": n_threads,
                "warm_us": warm_us,
                "cold_us": cold_us,
                "warm_median_us": warm_median,
                "cold_median_us": cold_median,
            }
        )
        print(f"    the region of {n_threads} threads: warm {warm_us:.2f} us  cold {cold_us:.2f} us", flush=True)
    return rows


def run_scaling(sizes, thread_counts) -> None:
    """How the parallel path scales with the number of the threads.

    Every thread count gets a process of its own: the threads of the pool
    spin after a parallel region, and they disturb the measurement of the
    next one in the same process - the question of how many threads are
    worth it is therefore not asked inside the grids above, where the
    thread counts follow one another.  The subprocess imports the package
    and finds its kernels in the on-disk cache, so a process of its own
    costs the interpreter start and little more.
    """
    script = (
        "import sys, timeit, numpy as np\n"
        "from numpy_groupies import aggregate_numba\n"
        "n = int(sys.argv[1])\n"
        "rng = np.random.default_rng(0)\n"
        "gi = rng.integers(0, 1000, n); a = rng.random(n)\n"
        "call = lambda: aggregate_numba.aggregate(gi, a, func=np.sum)\n"
        "for _ in range(3): call()\n"
        "times = sorted(timeit.Timer(call).repeat(repeat=20, number=10))\n"
        "print(min(times) * 1000 / 10, times[len(times) // 2] * 1000 / 10)\n"
    )
    pool = nb.get_num_threads()
    print(f"--- the scaling of the threads: {len(thread_counts)} thread counts x {len(sizes)} sizes ---", flush=True)
    print("    one process per thread count - the pool spins after a region and would disturb the next one", flush=True)
    for n_values in sizes:
        print(f"  n_values = {n_values:,}", flush=True)
        base = None
        for n_threads in thread_counts:
            if n_threads > pool:
                continue
            try:
                result = subprocess.run(
                    [sys.executable, "-c", script, str(n_values)],
                    capture_output=True,
                    text=True,
                    check=True,
                    env={**os.environ, "NUMBA_NUM_THREADS": str(n_threads)},
                )
            except subprocess.CalledProcessError as exc:
                tail = exc.stderr.strip().splitlines()[-1] if exc.stderr else "failed"
                print(f"    {n_threads:>2} threads: {tail}", flush=True)
                continue
            numbers = result.stdout.strip().splitlines()[-1].split()
            ms, median = float(numbers[0]), float(numbers[1])
            if base is None:
                base = ms
            bandwidth = n_values * BYTES_PER_ELEMENT / ms / 1e6
            print(
                f"    {n_threads:>2} threads: {ms:8.3f} ms   speedup {base / ms:5.2f}x   "
                f"{bandwidth:6.1f} GB/s   (median {median:8.3f} ms)",
                flush=True,
            )


# ---------------------------------------------------------------------------
# the model


class Model:
    """The cost model, with the bounds built into the parameters."""

    def __init__(self, params: list, phys: int):
        (self.core0, self.core_exp, self.smt_fraction, self.peak_factor, self.cg_seq, self.cg_mt, self.c_part) = params
        self.phys = phys

    def _megabytes(self, n_values: int) -> float:
        return max(n_values * BYTES_PER_ELEMENT / 1024 / 1024, 1e-6)

    def core_bandwidth(self, n_values: int) -> float:
        return self.core0 * self._megabytes(n_values) ** self.core_exp

    def bandwidth(self, n_values: int, n_threads: int) -> float:
        core = self.core_bandwidth(n_values)
        sibling = min(1.0, max(0.0, self.smt_fraction)) * core
        peak = max(core, max(1.0, self.peak_factor) * core)
        cores = min(n_threads, self.phys)
        siblings = max(0, n_threads - self.phys)
        return min(peak, core * cores + sibling * siblings)

    def kernel_seq_us(self, n_values: int, n_groups: int) -> float:
        data_us = n_values * BYTES_PER_ELEMENT / self.bandwidth(n_values, 1) * 1e6
        return data_us + n_groups * self.cg_seq / 1e3

    def kernel_mt_us(self, n_values: int, n_groups: int, n_threads: int) -> float:
        data_us = n_values * BYTES_PER_ELEMENT / self.bandwidth(n_values, n_threads) * 1e6
        return data_us + n_groups * self.cg_mt / 1e3 + n_threads * n_groups * self.c_part / 1e3 + overhead_us(n_threads)


class GroupModel:
    """The cost model of the path that cuts the groups into ranges.

    The cost of a value and the cost of a group are two different things:
    the counting sort that places the values walks over the values, and the
    selection of the middle of a block walks over the groups - and with many
    groups and few values the second is the larger of the two (a hundred
    nanoseconds a group against eighteen a value).

    The parallel path has a fixed cost of its own: the region of the pool,
    and with it the setting up of the ranges - the buffers they work on.  It
    does not grow with the data, and at the small sizes it is what decides
    whether the parallel path pays at all.
    """

    def __init__(
        self,
        ns_per_value: float,
        ns_per_group: float,
        share: float,
        per_thread: float,
        fixed_us: float,
    ):
        self.ns_per_value = ns_per_value
        self.ns_per_group = ns_per_group
        self.share = share
        self.per_thread = per_thread
        self.fixed_us = fixed_us

    def parts(self, n_threads: int) -> float:
        return 1.0 + (n_threads - 1) * self.per_thread

    def kernel_seq_us(self, n_values: int, n_groups: int) -> float:
        return (n_values * self.ns_per_value + n_groups * self.ns_per_group) / 1e3

    def kernel_mt_us(self, n_values: int, n_groups: int, n_threads: int, overhead: float) -> float:
        seq_us = self.kernel_seq_us(n_values, n_groups)
        par_us = seq_us * (self.share + (1.0 - self.share) / self.parts(n_threads))
        # the fixed cost of the region and of the setting up of the ranges
        return par_us + overhead + self.fixed_us


def fit(observations, phys: int) -> tuple[list, float, list]:
    """A pattern search over the constants - the loss is the squared
    logarithmic error of the times."""

    def bounded(p):
        return [
            max(1e6, p[0]),  # CORE0
            min(1.0, max(-1.0, p[1])),  # CORE_EXP
            min(1.0, max(0.0, p[2])),  # SMT_FRACTION
            max(1.0, p[3]),  # PEAK_FACTOR
            max(0.0, p[4]),  # CG_SEQ
            max(0.0, p[5]),  # CG_MT
            max(0.0, p[6]),  # C_PART
        ]

    def loss(p):
        p = bounded(p)
        model = Model(p, phys)
        total = 0.0
        for obs in observations:
            seq = model.kernel_seq_us(obs["n_values"], obs["n_groups"])
            mt = model.kernel_mt_us(obs["n_values"], obs["n_groups"], obs["n_threads"])
            if seq <= 0 or mt <= 0:
                return float("inf")
            total += (math.log(mt) - math.log(obs["mt_us"])) ** 2
            total += (math.log(seq) - math.log(obs["seq_us"])) ** 2
        return total

    # the search starts from several places: the landscape has more than one
    # solution that explains the measurements equally well, and the best one
    # is not always the one a single search finds
    starts = [
        [20e9, 0.0, 0.2, 4.0, 1.0, 3.0, 0.1],
        [8e9, -0.4, 0.05, 8.0, 0.5, 6.0, 0.0],
        [30e9, 0.3, 0.4, 2.0, 2.0, 1.0, 0.3],
    ]
    solutions = []
    for start in starts:
        params = bounded(start)
        step = [8e9, 0.3, 0.15, 1.5, 0.8, 1.5, 0.15]
        value = loss(params)
        for _ in range(3000):
            improved = False
            for i in range(len(params)):
                for sign in (1, -1):
                    candidate = params.copy()
                    candidate[i] = params[i] + sign * step[i]
                    trial = loss(candidate)
                    if trial < value - 1e-12:
                        value, params, improved = trial, candidate, True
            if not improved:
                step = [s * 0.5 for s in step]
                if abs(step[0]) < 1e5:
                    break
        solutions.append((value, bounded(params)))
    solutions.sort(key=lambda item: item[0])
    return solutions[0][1], solutions[0][0], [s for _, s in solutions]


def fit_group(observations, overhead_us_by_threads: dict) -> tuple[GroupModel, float]:
    """The constants of the group path: the cost of a value, the cost of a
    group, the share that stays single threaded and what a thread beyond the
    first brings.

    The loss is the squared logarithmic error of the times, the same measure
    the fit of the chunked path uses.
    """

    def bounded(p):
        return [
            max(0.05, p[0]),  # NS_PER_VALUE
            max(0.0, p[1]),  # NS_PER_GROUP
            min(0.99, max(0.01, p[2])),  # SHARE
            min(1.0, max(0.01, p[3])),  # PER_THREAD
            max(0.0, p[4]),  # FIXED_US
        ]

    def loss(p):
        p = bounded(p)
        model = GroupModel(*p)
        total = 0.0
        for obs in observations:
            seq = model.kernel_seq_us(obs["n_values"], obs["n_groups"])
            mt = model.kernel_mt_us(
                obs["n_values"], obs["n_groups"], obs["n_threads"], overhead_us_by_threads[obs["n_threads"]]
            )
            if seq <= 0 or mt <= 0:
                return float("inf")
            total += (math.log(mt) - math.log(obs["mt_us"])) ** 2
            total += (math.log(seq) - math.log(obs["seq_us"])) ** 2
        return total

    # the start of the search: the cost of a value and the cost of a group
    # are taken from the sequential times, the other two from the share of
    # the work that the threads can share
    starts = [
        [18.0, 100.0, 0.33, 0.4, 20.0],
        [18.0, 20.0, 0.1, 0.8, 50.0],
        [5.0, 50.0, 0.5, 0.2, 5.0],
    ]
    best = None
    for start in starts:
        params = bounded(start)
        step = [8.0, 40.0, 0.15, 0.2, 20.0]
        value = loss(params)
        for _ in range(3000):
            improved = False
            for i in range(len(params)):
                for sign in (1, -1):
                    candidate = params.copy()
                    candidate[i] = params[i] + sign * step[i]
                    trial = loss(candidate)
                    if trial < value - 1e-12:
                        value, params, improved = trial, candidate, True
            if not improved:
                step = [s * 0.5 for s in step]
                if step[0] < 1e-4:
                    break
        if best is None or value < best[0]:
            best = (value, bounded(params))
    value, params = best
    return GroupModel(*params), value


def call_overhead(rows: list) -> float:
    """The fixed cost per call that the package carries and the reference
    kernel does not: the difference of the two sequential kernels."""
    differences = []
    for row in rows:
        if row.get("npg_us", float("inf")) == float("inf"):
            continue
        differences.append(row["npg_us"] - row["seq_us"])
    if not differences:
        return 0.0
    # the smallest difference is the least contaminated one - where the kernel
    # itself takes the longest, the fixed cost matters least
    return max(0.0, min(differences))


# ---------------------------------------------------------------------------
# the crossovers


def _values(number: float) -> str:
    """A number of values, or the word for a crossover the grid does not reach."""
    if number == float("inf"):
        return "beyond the grid"
    return f"{number:,.0f} values"


def crossover(rows: list, key_seq: str, key_mt: str, margin: float) -> tuple[float, float]:
    """The size from which the parallel path is faster, and the one before.

    The first number is the size from which the measured ratio of the
    parallel path to the sequential one is below the margin - interpolated
    between the two sizes that straddle it.  The second is the size below
    which it is not: the floor of the model has to be at the first one, and
    a workload below the second one never uses the parallel path.

    The ratio of a size is the median of the measurements of that size: the
    threads and the groups are a few measurements of the same thing, and the
    median of them is the one that is not an outlier.
    """
    by_size: dict[int, list] = {}
    for row in rows:
        if row.get(key_seq) is None or row.get(key_mt) is None:
            continue
        ratio = row[key_mt] / row[key_seq]
        by_size.setdefault(row["n_values"], []).append(ratio)
    medians = {size: statistics.median(ratios) for size, ratios in by_size.items()}
    sizes = sorted(medians)
    first = None
    below = 0
    for size in sizes:
        if medians[size] <= margin:
            first = size
            break
        below = size
    if first is None:
        return float("inf"), float(sizes[-1]) if sizes else 0.0
    if below == 0 or below == first:
        return float(first), float(below)
    # the ratio between the two sizes is interpolated linearly - the grid is
    # coarse, and the crossover is where the model turns over
    ratio_below = medians[below]
    ratio_above = medians[first]
    if ratio_below == ratio_above:
        return float(first), float(below)
    fraction = (ratio_below - margin) / (ratio_below - ratio_above)
    return below + fraction * (first - below), float(below)


# ---------------------------------------------------------------------------
# the report


def _spread(rows: list) -> tuple[float, float]:
    """The spread of the measurements: the fastest against the median.

    The fastest call of a measurement is the one that was left alone by the
    machine, the median is what the calls cost when the machine has something
    else to do.  A spread of a factor of two and more means that the machine
    was busy and that the numbers below are worth nothing - the tool says so
    instead of reporting them as if they were good.
    """
    spreads = []
    for row in rows:
        fastest = row.get("seq_median_us")
        if not fastest:
            continue
        medians = row.get("mt_median_us") or 0.0
        for value in (row.get("seq_us"), row.get("mt_us")):
            if value:
                spreads.append(medians / value if value else 1.0)
        if row.get("seq_median_us"):
            spreads.append(row["seq_median_us"] / row["seq_us"] if row.get("seq_us") else 1.0)
    if not spreads:
        return 1.0, 1.0
    spreads.sort()
    return spreads[len(spreads) // 2], spreads[-1]


def report(rows: list, info: dict, overhead_rows: list, verbose: bool) -> None:
    phys = int(info["physical_cores"])
    chunked = [r for r in rows if r["path"] == "chunked" and r.get("kernel") == "prototype"]
    package = [r for r in rows if r["path"] == "chunked" and r.get("kernel") == "package"]
    group = [r for r in rows if r["path"] == "group"]
    gather = [r for r in rows if r["path"] == "gather"]
    try:
        from numpy_groupies.aggregate_numba import mt as package_mt

        GATHER_MT_MIN_VALUES = package_mt.GATHER_MT_MIN_VALUES
        MT_MIN_VALUES = package_mt.MT_MIN_VALUES
    except ImportError:
        GATHER_MT_MIN_VALUES = MT_MIN_VALUES = 0

    # the cost of the region, warm and cold
    warm_overhead = {r["n_threads"]: r["warm_us"] for r in overhead_rows}
    cold_overhead = {r["n_threads"]: r["cold_us"] for r in overhead_rows}
    print("\n=== the cost of the region (an empty parallel loop) ===")
    print(f"    {'threads':>8} {'warm us':>10} {'cold us':>10}")
    for r in overhead_rows:
        print(f"    {r['n_threads']:>8} {r['warm_us']:>10.2f} {r['cold_us']:>10.2f}")
    if cold_overhead:
        print(
            f"    the waking of the threads costs {max(cold_overhead.values()) / max(warm_overhead.values()):.0f} times"
            f" the region itself - the layer of OpenMP keeps them spinning for about ten milliseconds"
        )

    # the quality of the measurements: the fastest against the median
    print("\n=== the quality of the measurements ===")
    print("    the fastest call against the median of the calls - a spread of two and more")
    print("    means that the machine was busy and that the numbers below are worth little")
    for name, part in (
        ("the chunked path", [r for r in rows if r["path"] == "chunked" and r.get("kernel") == "package"]),
        ("the group path", [r for r in rows if r["path"] == "group"]),
        ("the gather", [r for r in rows if r["path"] == "gather"]),
    ):
        if not part:
            continue
        median, worst = _spread(part)
        mark = "  the machine was busy" if median > 2.0 else ""
        print(f"    {name:<18} the median of the spreads {median:5.2f}, the worst {worst:6.2f}{mark}")

    # the chunked path
    if chunked:
        print("\n=== the chunked path (the sum) ===")
        print("    the prototype of the script and the kernel of the package are both")
        print("    measured - the constants are fitted to the prototype, the crossovers")
        print("    and the decisions to the kernel of the package")
        params, loss_value, _solutions = fit(chunked, phys)
        model = Model(params, phys)
        print(f"    {len(chunked)} observations of the prototype, {phys} physical cores")
        print("    the constants:")
        names = [
            ("CORE0", "the bandwidth of one core at a working set of 1 MB"),
            ("CORE_EXP", "how it depends on the working set"),
            ("SMT_FRACTION", "what an SMT thread brings in relation to a core"),
            ("PEAK_FACTOR", "the peak of the machine, in cores"),
            ("CG_SEQ", "ns per group, sequential"),
            ("CG_MT", "ns per group, parallel"),
            ("C_PART", "ns per element of the partial matrix"),
        ]
        for i, (name, what) in enumerate(names):
            print(f"      {name:<13} = {params[i]:<12.6g} # {what}")
        print(f"      residual error (log RMS) = {math.sqrt(loss_value / (2 * len(chunked))):.4f}")
        warm_cross, warm_below = crossover(package, "seq_us", "mt_us", MT_MAX_RATIO)
        cold_cross, cold_below = crossover(package, "seq_cold_us", "mt_cold_us", MT_MAX_RATIO)
        print(f"    the crossover: warm at {_values(warm_cross)} (below {_values(warm_below)} it does not pay)")
        print(f"                   cold at {_values(cold_cross)} (below {_values(cold_below)} it does not pay)")
        print(f"    the floor of the package: {MT_MIN_VALUES:,} values")
        if verbose:
            print("\n    the times of the model against the measurement (sequential / parallel, us):")
            sizes = sorted({r["n_values"] for r in chunked})
            threads = sorted({r["n_threads"] for r in chunked})
            print("    " + " " * 12 + "".join(f"{'T=' + str(t):>22}" for t in threads))
            for n in sizes:
                cells = ""
                for t in threads:
                    measured = [r for r in chunked if r["n_values"] == n and r["n_threads"] == t]
                    if not measured:
                        cells += f"{'-':>22}"
                        continue
                    row = measured[0]
                    seq_model = model.kernel_seq_us(n, row["n_groups"])
                    mt_model = model.kernel_mt_us(n, row["n_groups"], t)
                    cells += f"{row['seq_us']:>7.0f}/{row['mt_us']:<6.0f}{seq_model:>5.0f}/{mt_model:<4.0f}"
                print(f"    {n:>10} MB{cells}")
            print("    (per cell: measured seq/mt, then model seq/mt)")

    # the group path
    if group:
        print("\n=== the group path (the order statistics) ===")
        by_op: dict[str, list] = {}
        for row in group:
            by_op.setdefault(row["op"], []).append(row)
        for op, observations in by_op.items():
            group_model, loss_value = fit_group(observations, warm_overhead)
            warm_cross, warm_below = crossover(observations, "seq_us", "mt_us", MT_MAX_RATIO)
            cold_cross, cold_below = crossover(observations, "seq_cold_us", "mt_cold_us", MT_MAX_RATIO)
            package_ns = _package_ns_per_value().get(op)
            print(f"    {op}:")
            print(f"      NS_PER_VALUE = {group_model.ns_per_value:<12.6g} # ns a value costs")
            print(f"      NS_PER_GROUP = {group_model.ns_per_group:<12.6g} # ns a group costs")
            if package_ns is not None:
                print(f"      the package carries {package_ns:.1f} ns a value")
            print(f"      SHARE        = {group_model.share:<12.6g} # the part that stays single threaded")
            print(f"      PER_THREAD   = {group_model.per_thread:<12.6g} # what a thread beyond the first brings")
            print(f"      FIXED_US     = {group_model.fixed_us:<12.6g} # the fixed cost of the parallel path")
            print(f"      residual error (log RMS) = {math.sqrt(loss_value / (2 * len(observations))):.4f}")
            print(f"      the crossover: warm at {_values(warm_cross)} (below {_values(warm_below)} it does not pay)")
            print(f"                     cold at {_values(cold_cross)} (below {_values(cold_below)} it does not pay)")
        try:
            from numpy_groupies.aggregate_numba import mt as package_mt

            print(f"    the floor of the package: {package_mt.GROUP_MT_MIN_VALUES:,} values")
        except ImportError:
            pass

    # the gather
    if gather:
        print("\n=== the gather ===")
        warm_cross, warm_below = crossover(gather, "seq_us", "mt_us", MT_MAX_RATIO)
        cold_cross, cold_below = crossover(gather, "seq_cold_us", "mt_cold_us", MT_MAX_RATIO)
        print("    the gather follows the bandwidth model of the chunked path - it moves the")
        print("    values and holds no state, so it needs no constants of its own")
        print(f"    the crossover: warm at {_values(warm_cross)} (below {_values(warm_below)} it does not pay)")
        print(f"                   cold at {_values(cold_cross)} (below {_values(cold_below)} it does not pay)")
        print(f"    the floor of the package: {GATHER_MT_MIN_VALUES:,} values")

    # the decisions of the package: what it takes against what the
    # measurement says
    print("\n=== the decisions of the package ===")
    disagreements = []
    for row in rows:
        n = row["n_values"]
        t = row["n_threads"]
        if t < 2:
            continue
        decision, modelled = _package_decision(row)
        if decision is None:
            continue
        for tag, seq_key, mt_key in (("warm", "seq_us", "mt_us"), ("cold", "seq_cold_us", "mt_cold_us")):
            if row.get(seq_key) is None or row.get(mt_key) is None:
                continue
            ratio = row[mt_key] / row[seq_key]
            measured_better = ratio <= MT_MAX_RATIO
            if decision != measured_better:
                disagreements.append((row, tag, ratio, decision, modelled))
    if disagreements:
        print(f"    {len(disagreements)} measurements disagree with the decision of the package:")
        print(
            f"    {'path':<8} {'op':<8} {'values':>10} {'groups':>8} {'T':>3} {'warm/cold':>10}"
            f" {'ratio':>6} {'the package':>11}"
        )
        for row, tag, ratio, decision, modelled in disagreements[:40]:
            print(
                f"    {row['path']:<8} {row.get('op', ''):<8} {row['n_values']:>10} {row['n_groups']:>8}"
                f" {row['n_threads']:>3} {tag:>10} {ratio:>6.2f} {('MT' if decision else 'seq'):>11}"
            )
        if len(disagreements) > 40:
            print(f"    ... and {len(disagreements) - 40} more")
    else:
        print("    every measurement agrees with the decision of the package")


def _package_decision(row: dict) -> tuple[bool | None, float]:
    """What the package decides for one measurement, and what its model says.

    The decision is the one of the cost model of the package, with the
    constants it carries now - the tool does not change them, it says what
    they do.
    """
    try:
        from numpy_groupies.aggregate_numba import mt as package_mt
    except ImportError:
        return None, float("nan")
    n, g, t = row["n_values"], row["n_groups"], row["n_threads"]
    if row["path"] == "chunked":
        decision = bool(package_mt.should_use_mt(g, t, n))
        modelled = package_mt.estimate_mt_ratio(g, t, n)
    elif row["path"] == "gather":
        decision = bool(
            n >= package_mt.GATHER_MT_MIN_VALUES and package_mt.estimate_mt_ratio(0, t, n, 0) <= package_mt.MT_MAX_RATIO
        )
        modelled = package_mt.estimate_mt_ratio(0, t, n, 0)
    else:
        ns = _PACKAGE_NS_PER_VALUE.get(row.get("op"))
        if ns is None:
            return None, float("nan")
        decision = bool(package_mt.should_use_group_mt(g, t, n, ns))
        modelled = package_mt.estimate_group_mt_ratio(g, t, n, ns)
    return decision, modelled


# the calibration of the machine is a run of minutes, and an interrupt has
# to end it cleanly - the mechanism for that lives in the ``__main__`` block
# at the end of this module, where the run is started.


def _release_numba_resources() -> None:
    """Release the semaphore the threading layer of numba leaves behind.

    The lock of the threading-layer initialization is a semaphore of
    multiprocessing, and the resource tracker of it reports the semaphore
    as leaked when the process leaves without releasing it - an interrupt
    leaves exactly like that.  The lock is dropped, which unregisters the
    semaphore from the tracker, and its name is unlinked from /dev/shm,
    which is what the shutdown of a normal exit does.
    """
    import gc

    from numba.np.ufunc import parallel as numba_parallel

    lock = numba_parallel._backend_init_process_lock
    if lock is None:
        return
    name = lock._semlock.name
    # dropping the reference of numba runs the dealloc of the semaphore,
    # and the dealloc is what unregisters it from the tracker
    numba_parallel._backend_init_process_lock = None
    gc.collect()
    # the name in /dev/shm is unlinked the way the shutdown of a normal
    # exit unlinks it
    SemLock._cleanup(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sizes",
        default="50000,500000,2000000,10000000",
        help="comma-separated numbers of values to measure (the chunked path)",
    )
    parser.add_argument(
        "--group-sizes",
        default="20000,100000,500000",
        help="comma-separated numbers of values to measure (the group path)",
    )
    parser.add_argument(
        "--gather-sizes",
        default="20000,200000,2000000",
        help="comma-separated numbers of values to measure (the gather)",
    )
    parser.add_argument(
        "--groups",
        default="1000,100000",
        help="comma-separated numbers of groups to measure",
    )
    parser.add_argument(
        "--threads",
        default="",
        help="comma-separated thread counts to measure (default: 1, the powers of two, and the pool size)",
    )
    parser.add_argument("--rounds", type=int, default=3, help="measurement rounds, the fastest counts (default: 3)")
    parser.add_argument("--no-cold", action="store_true", help="skip the measurements with the threads asleep")
    parser.add_argument("--no-gather", action="store_true", help="skip the gather measurements")
    parser.add_argument("--no-group", action="store_true", help="skip the group path measurements")
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="also measure how the parallel path scales with the threads, one process per thread count",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print the times of the model against the measurements in the report"
    )
    parser.add_argument("--seed", type=int, default=100, help="the seed of the data (default: 100)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    pool = nb.get_num_threads()
    if args.threads:
        thread_counts = [int(t) for t in args.threads.split(",")]
    else:
        # one thread, then the powers of two, the pool itself at the end:
        # the bandwidth of the threads saturates early, and a count in
        # between moves the numbers by less than the noise of the machine -
        # a pool of a hundred threads is then still a handful of points
        thread_counts = [1]
        while thread_counts[-1] * 2 < pool:
            thread_counts.append(thread_counts[-1] * 2)
        if thread_counts[-1] != pool:
            thread_counts.append(pool)
    # measured from the pool downwards: a pool that grows launches workers of
    # the openmp layer, and a launch ignores the interrupt for the whole
    # process - one that happened while a grid runs would swallow it until
    # the next measurement arms the handler again
    thread_counts.reverse()
    sizes = [int(s) for s in args.sizes.split(",")]
    group_sizes = [int(s) for s in args.group_sizes.split(",")]
    gather_sizes = [int(s) for s in args.gather_sizes.split(",")]
    group_counts = [int(g) for g in args.groups.split(",")]

    info = machine_info()
    print("=== the machine ===")
    for key, value in info.items():
        print(f"  {key:<24} {value}")
    print(f"  measured thread counts      {thread_counts}")
    print()

    package_ns = _package_ns_per_value()
    if package_ns:
        print("the cost of a value, as the package carries it:")
        for op, ns in package_ns.items():
            print(f"  {op:<10} {ns:6.1f} ns")
        print()

    cold = not args.no_cold
    rows = run_chunked_grid(sizes, group_counts, thread_counts, args.rounds, cold)
    if not args.no_group:
        rows += run_group_grid(group_sizes, group_counts, thread_counts, args.rounds, cold)
    if not args.no_gather:
        rows += run_gather_grid(gather_sizes, thread_counts, args.rounds, cold)
    overhead_rows = run_overhead(thread_counts, args.rounds)
    if args.scaling:
        run_scaling(sizes, thread_counts)
    report(rows, info, overhead_rows, args.verbose)
    return 0


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        # a reader that went away (``| head``) ends the run quietly, as it
        # does for any other command line filter; left alone, python prints
        # a traceback of the broken pipe on the way out
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    def _on_interrupt(signum: int, frame: object) -> None:
        """End the run at once, on an interrupt.

        A KeyboardInterrupt raised from the handler would not arrive: the
        interrupt reaches the compiled kernels, and the ctypes callbacks of
        numba swallow the exceptions raised inside them - the run would go
        on, and llvmlite would print a complaint about it.  Leaving the
        process is the only way out that works from within such a callback,
        and the exit code says what happened.

        The semaphore of the threading-layer initialization of numba is
        released before that, so that the resource tracker of
        multiprocessing has nothing to complain about on the way out.
        """
        with contextlib.suppress(Exception):
            _release_numba_resources()
        os._exit(130)

    signal.signal(signal.SIGINT, _on_interrupt)
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass
    except KeyboardInterrupt:
        # an interrupt that arrives outside of a compiled kernel
        pass
