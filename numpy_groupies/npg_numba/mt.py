"""The decision whether a kernel runs multithreaded.

The cost model is fitted to measurements of the sum kernel on the machine of
``docs/benchmarks.md`` (AMD EPYC, OpenMP); the constants are deliberately on
the pessimistic side - scaled down (the bandwidth) and up (the group costs) -
so the model errs towards the sequential path, and the ``MT_MAX_RATIO`` margin
adds to that.  The times are in microseconds, the constants in nanoseconds or
bytes per second::

    sequential:  n_values * bytes / BW(n_values, 1)  +  n_groups * cg_seq
    parallel:    n_values * bytes / BW(n_values, n_threads)
                 +  n_groups * cg_mt  +  n_threads * n_groups * c_part
                 +  overhead

The bandwidth saturates at the machine, not at the number of threads: a core
delivers its share and the machine about twice that - the model divides by the
bandwidth the threads use together, not by the thread count.  The parallel path
pays twice for the groups (the partial matrix and its fold), which is why the
group count enters the decision at all.  Nothing is measured at run time.
"""

import os

import numba as nb

from numpy_groupies.npg_numba.jit import njit_stable

# --- the fitted constants, on the pessimistic side ---
_BYTES_PER_ELEMENT = 16.0  # a float64 value and an int64 index
_CORE_BYTES_PER_S = 2.003e10  # what one core delivers at a working set of 1 MB
_CORE_BYTES_PER_DECADE = -0.04499  # and how that depends on the working set
_PEAK_CORES = 2  # the peak of the machine, in cores
_CG_SEQ = 1.2286  # nanoseconds per group, sequential
_CG_MT = 7.8473  # nanoseconds per group, parallel
_C_PART = 0.16679  # nanoseconds per element of the partial matrix

# the fixed cost of the parallel path, measured directly (an empty prange
# loop costs a few microseconds)
#
# The cost of the region is the cost of waking the pool's threads up - the
# threading layer of numba (OpenMP by default) spins them for a moment after a
# region and sleeps them afterwards (the pause in the milliseconds): an empty
# region costs a few microseconds with the threads awake and orders of
# magnitude more with them asleep.  That cost decides whether a parallel path
# is worth it at all - the reason the thresholds of this module are measured
# and not guessed.  The floors below are fitted on the *warm* cost of a
# workload called in a loop; the cold cost (first call, pool asleep) is many
# times the warm one and paid once - a floor fitted on it would keep such
# workloads out of the parallel path for good.
_OVERHEAD_US = 5.0

# below this many values the parallel path cannot pay for itself, whatever the
# number of groups: its fixed cost alone is a few percent of the sequential time
MT_MIN_VALUES = 100_000

# the parallel path must be this much faster before it is used - the margin
# absorbs the error of model and measurement, in the conservative direction
MT_MAX_RATIO = 0.8

# several chunks per thread: the pool's threads take the chunks from a queue,
# so a busy core delays its own chunk and not the whole call - one chunk per
# thread would wait for the slowest core (measured: one slow chunk held up a
# call by more than all its other chunks together were worth).  The states are
# one per thread, written by that thread alone - no lock needed, and that is
# the memory ``should_use_mt`` counts as the partial matrix.  A pool thread's
# own id is bounded by the thread count of the pool.
MT_CHUNKS_PER_THREAD = 2


# --- the operations that parallelize over the GROUPS instead of the values ---
#
# The order statistics (median, quantile, sort, distinct count) cannot be
# reduced in chunks: a group's values have to be together where they are
# worked on.  The counting sort that places the values stays single threaded,
# and the group loop - the expensive part - is cut into ranges of one thread
# per group: nothing folded, no partial matrix, the sequential result bit for
# bit.  That shape is not the one the constants above were fitted on, so it is
# not described by ``should_use_mt`` but by ``should_use_group_mt`` below.

# what one thread beyond the first brings to a group loop, measured with the
# full pool of the benchmarks machine: a fraction of a thread short of the
# linear share.  These loops do arithmetic on the values, not memory work, so
# they do not run into the bandwidth the model above is built around.
_GROUP_PER_THREAD = 0.40

# the share of these kernels the parallel path cannot take: the counting sort
# of the placement, measured at about a third of the whole call (median and
# sort alike) - the constant is the pessimistic side of that.
_GROUP_SERIAL_SHARE = 1.0 / 3.0

# below this many values a group loop is not worth a parallel region.  These
# operations cost orders of magnitude more per value than the sum kernel, so
# the same overhead of a few microseconds is a far smaller share of their time:
# the measured break even sits far below ``MT_MIN_VALUES``, which would keep
# them single threaded long past the point where the region pays handsomely.
# Fitted on the *warm* cost of the region (see ``_OVERHEAD_US``); the cold one
# puts the break even higher, and the floor is the conservative side of both.
GROUP_MT_MIN_VALUES = 20_000


GROUP_MT_CHUNKS_PER_THREAD = 2


# --- the gather, which copies one value per value and holds no state ---
#
# The gather has no groups and no state: every value writes its own position,
# the value loop is cut into chunks and the threads copy into their own - the
# sequential result bit for bit.  Memory bound, so the model above fits it
# (bandwidth plus the fixed cost of the region; the partial matrix drops out).
# Measured with ``python -m numpy_groupies.benchmarks.mt``, threads asleep (a
# single call) and awake (a workload in a loop): the region pays for itself
# from about the size of the warm chunked reductions, a few times above it
# cold - hence the same floor, whose one-off waking price a looping workload
# pays back after a few calls.
GATHER_MT_MIN_VALUES = MT_MIN_VALUES


@njit_stable
def _parallel_parts(n_threads: int) -> float:
    """The number of the threads worth of work the parallel path gets done in."""
    return 1.0 + (n_threads - 1) * _GROUP_PER_THREAD


@njit_stable
def group_chunk_count(n_threads: int, n_groups: int) -> int:
    """The number of the ranges the groups are cut into.

    More than the threads, for the same two reasons the reduction chunks have:
    a busy core delays its own range and not the whole call, and the ranges are
    the unit the balance is struck in.  Never more than there are groups, and
    a single thread is not split at all.
    """
    if n_threads < 2:
        return 1
    return min(n_threads * GROUP_MT_CHUNKS_PER_THREAD, n_groups)


@njit_stable
def estimate_group_mt_ratio(n_groups: int, n_threads: int, n_values: int, ns_per_value: float) -> float:
    """The group-parallel path's estimated time as a multiple of the sequential one.

    ``ns_per_value`` is the *total* sequential cost per value, counting sort of
    the placement included - which is why bandwidth and groups do not enter the
    estimate again.  The parallel time keeps the unshareable share and divides
    the rest by the threads' worth of work; the region overhead is the *warm*
    one - the estimate is what the path is worth in a loop, the floors what a
    single call is worth.
    """
    seq_us = n_values * ns_per_value / 1e3
    par_us = seq_us * (_GROUP_SERIAL_SHARE + (1.0 - _GROUP_SERIAL_SHARE) / _parallel_parts(n_threads))
    return (par_us + _OVERHEAD_US) / seq_us


@njit_stable
def should_use_group_mt(n_groups: int, n_threads: int, n_values: int, ns_per_value: float) -> bool:
    """Whether the loop over the groups of an operation is worth spreading.

    The arguments of ``should_use_mt`` plus ``ns_per_value``; ``n_groups`` is
    not in the estimate (no per-group cost of its own) but gates the call.
    """
    if n_threads < 2:
        return False
    if n_values < GROUP_MT_MIN_VALUES or n_groups < 2:
        return False
    return estimate_group_mt_ratio(n_groups, n_threads, n_values, ns_per_value) <= MT_MAX_RATIO


@njit_stable
def chunk_count(n_threads: int, n_values: int) -> int:
    """The number of the chunks the data is split into.

    Several per thread, so a busy core delays its own chunk and not the whole
    call.  Never more than there are values, and a single thread is not split
    at all - the result is the sequential kernel's, bit for bit.
    """
    if n_threads < 2:
        return 1
    chunks = n_threads * MT_CHUNKS_PER_THREAD
    return min(chunks, n_values)


@njit_stable
def _bandwidth_bytes_per_s(n_values: int, n_threads: int) -> float:
    """The bandwidth the threads use together for a working set of the given size:
    a core delivers its share, the machine about twice that, and the threads
    beyond the second bring little."""
    megabytes: float = max(n_values * _BYTES_PER_ELEMENT / 1024 / 1024, 1e-6)
    core: float = max(0.5e9, _CORE_BYTES_PER_S * megabytes**_CORE_BYTES_PER_DECADE)
    return core * min(n_threads, _PEAK_CORES)


@njit_stable
def estimate_seq_time_us(n_groups: int, n_values: int) -> float:
    """The estimated time of the sequential path, in microseconds."""
    data_us = n_values * _BYTES_PER_ELEMENT / _bandwidth_bytes_per_s(n_values, 1) * 1e6
    return data_us + n_groups * _CG_SEQ / 1e3


@njit_stable
def estimate_mt_time_us(n_groups: int, n_threads: int, n_values: int, state_size: int = 1) -> float:
    """The estimated time of the parallel path, in microseconds.

    The data arrives at the threads' joint bandwidth; the per-group cost of
    filling and folding the states scales with ``state_size``, the number of
    the state arrays against the two of the sum kernel.
    """
    data_us = n_values * _BYTES_PER_ELEMENT / _bandwidth_bytes_per_s(n_values, n_threads) * 1e6
    # the states are one per thread, and the threads are the ones of the pool
    partials_us = n_threads * n_groups * _C_PART * state_size / 1e3
    return data_us + n_groups * _CG_MT * state_size / 1e3 + partials_us + _OVERHEAD_US


@njit_stable
def estimate_mt_ratio(n_groups: int, n_threads: int, n_values: int, state_size: int = 1) -> float:
    """The estimated time of the parallel path as a multiple of the sequential
    one.  Below 1 means the parallel path is faster; 1 means they are equal."""
    return estimate_mt_time_us(n_groups, n_threads, n_values, state_size) / estimate_seq_time_us(n_groups, n_values)


@njit_stable
def should_use_mt(n_groups: int, n_threads: int, n_values: int, state_size: int = 1) -> bool:
    """Whether the multithreaded variant of a kernel is worth using.

    ``n_values`` is the input length, not the ``size`` argument of ``aggregate``
    (that one is the number of groups); ``state_size`` counts the state arrays
    against the two of the sum kernel.  False whenever the parallel path could
    be slower - too few values, or no clear gain promised.
    """
    if n_threads < 2:
        return False
    if n_values < MT_MIN_VALUES:
        return False
    return estimate_mt_ratio(n_groups, n_threads, n_values, state_size) <= MT_MAX_RATIO


# --- the machine, for the number of threads ---

_physical_cores: int | None = None


def mt_thread_count() -> int:
    """The number of the threads the parallel path uses.

    The pool size of numba - ``NUMBA_NUM_THREADS`` at process start,
    ``numba.set_num_threads`` at any time - read per call, since a number
    captured at the import would go stale.  One thread means the sequential
    path, with its result bit for bit.
    """
    return nb.get_num_threads()


def physical_core_count() -> int:
    """The number of physical cores the process may run on.

    Hardware threads sharing a core are counted once - measured, they brought
    between nothing and a third of a core, and where they brought nothing they
    cost up to 40% of the bandwidth.  A wrong VM topology yields a guess, but
    in the safe direction: fewer threads, not more.  Read once and kept; the
    file behind it does not change while the process runs.
    """
    global _physical_cores
    if _physical_cores is not None:
        return _physical_cores

    try:
        allowed = os.sched_getaffinity(0)
    except AttributeError:  # not on linux
        _physical_cores = os.cpu_count() or 1
        return _physical_cores

    cores: set[tuple[str, str]] = set()
    physical_id = core_id = ""
    try:
        with open("/proc/cpuinfo") as info:
            for line in info:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key == "processor":
                    if int(value) not in allowed:
                        physical_id = core_id = ""
                        continue
                elif key == "physical id":
                    physical_id = value
                elif key == "core id":
                    core_id = value
                    cores.add((physical_id, core_id))
    except OSError:
        cores = set()

    _physical_cores = len(cores) or len(allowed)
    return _physical_cores


def logical_core_count() -> int:
    """The number of hardware threads the process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on linux
        return os.cpu_count() or 1
