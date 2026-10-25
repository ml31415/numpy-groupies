"""The decision whether a kernel runs multithreaded.

The cost model is fitted to measurements of the sum kernel on the machine
documented in ``docs/benchmarks.md`` (AMD EPYC, OpenMP).  The constants are
deliberately on the pessimistic side: they were scaled down (the bandwidth)
and up (the costs of the groups) after the fit, so that the model errs
towards the sequential path.  The margin of ``MT_MAX_RATIO`` adds to that:
the parallel path is only used when the model promises it a clear gain.

The times are in microseconds, the constants in nanoseconds or bytes per
second.

    sequential:  n_values * bytes / BW(n_values, 1)  +  n_groups * cg_seq
    parallel:    n_values * bytes / BW(n_values, n_threads)
                 +  n_groups * cg_mt  +  n_threads * n_groups * c_part
                 +  overhead

The bandwidth saturates at the machine, not at the number of threads: a
core delivers its share and the machine about twice that - the constants
below say how much.  The model does not divide the time by the number of
threads, but by the bandwidth the threads use together.

The parallel path pays twice for the groups: once for the partial matrix it
accumulates into, and once for reducing it.  That is why the number of
groups enters the decision at all - with as many groups as values the
partial matrix costs more than the data.

Nothing is measured at run time: the decision must be able to run without
knowing anything about the machine it runs on.
"""

import os

import numba as nb

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
# The cost of the region is the cost of waking the threads of the pool up -
# the threading layer of numba (OpenMP by default) keeps them asleep between
# the calls, and every region wakes them up again.  The layer spins the
# threads for a moment after a region and sleeps them afterwards: an empty
# region costs a few microseconds with the threads awake and orders of
# magnitude more with them asleep, and the pause that puts them to sleep is
# in the milliseconds.  On a machine whose threads spin (and the layer of
# OpenMP can be told to), the cost of the region is what decides whether a
# parallel path is worth it at all, and it is the reason the thresholds of
# this module are measured and not guessed.  A call that does not pay for
# the region does not make one - the sequential path is a few microseconds
# away and it is exact.
#
# The floors below are fitted on the *warm* cost - the state of a workload
# that is called in a loop.  The cold cost (the pool asleep, which is what a
# program that calls a kernel once pays) is many times the warm one, and it
# is paid once by the first call of a workload: the calls after it find the
# threads awake.  A floor fitted on the cold cost would keep a whole band of
# workloads out of the parallel path for good - and a workload that is
# called in a loop would never get there, because it would never make the
# region that wakes the threads up.
_OVERHEAD_US = 5.0

# below this many values the parallel path cannot pay for itself, whatever
# the number of groups: its fixed cost alone is a few percent of the
# sequential time, and the machines that run it are not exactly this one
MT_MIN_VALUES = 100_000

# the parallel path must be this much faster than the sequential one before
# it is used - the margin absorbs the error of the model and of the
# measurement, and the error goes in the conservative direction
MT_MAX_RATIO = 0.8

# The chunks are not one per thread but several: the threads of the pool take
# the chunks from a queue, so a thread whose core is busy with something else
# does not hold up the whole call - the threads that are done with their
# chunks take the ones that are left.  With one chunk per thread there is
# nothing to take, and the call waits for the slowest core (measured: one
# slow chunk held up a call by more than all its other chunks together were
# worth).
#
# The states are one per thread, and the thread is what the state belongs to:
# the state of a thread is written by that thread alone, so the states need no
# lock - and the memory is one state per thread, which is what
# ``should_use_mt`` counts as the partial matrix.
#
# The number of the threads is the number of the threads of the pool of numba,
# and the number a thread reports for itself is bounded by it: a thread of
# the pool is one of the threads the call counted.
MT_CHUNKS_PER_THREAD = 2


# --- the operations that parallelize over the GROUPS instead of the values ---
#
# The order statistics (the median, the quantile, the sort, the number of the
# distinct values) cannot be reduced in chunks: the values of a group have to
# be together where they are worked on.  They parallelize the other way round -
# the counting sort that places the values stays single threaded, and the loop
# over the groups, which is the expensive part, is cut into ranges that the
# threads work on one after another.  Every group belongs to one range and is
# worked on by one thread, so nothing is folded afterwards, no partial matrix
# is accumulated and the result is the one of the sequential kernel bit for bit,
# whatever the number of the threads.
#
# That shape is not the one the constants above were fitted on, so it is not
# described by ``should_use_mt`` but by ``should_use_group_mt`` below.

# what one thread beyond the first brings to a loop over the groups.  Measured
# with the full pool of the machine of ``docs/benchmarks.md``: the median, the
# quantile, the sort and the distinct count run a few times faster than on one
# thread, a fraction of a thread short of the linear share - the constant is
# the measured one.  The work these loops do is arithmetic on the values and
# not the memory, so it does not run into the bandwidth the model above is
# built around.
_GROUP_PER_THREAD = 0.40

# the share of these kernels that the parallel path cannot take: the counting
# sort that places the values before the groups can be worked on.  Measured on
# this machine, it takes about a third of the whole call, for the median as
# for the sort - the constant is the pessimistic side of that.
_GROUP_SERIAL_SHARE = 1.0 / 3.0

# below this many values the loop over the groups is not worth a parallel
# region.  The operations of this shape cost orders of magnitude more per
# value than the sum the model above was fitted on, so the same overhead of
# a few microseconds is a far smaller share of their time: the measured
# break even sits far below the floor of ``MT_MIN_VALUES``, which would
# keep these operations single threaded long past the point where the
# region pays handsomely.  They get a floor of their own for that reason.
#
# The floor is fitted on the *warm* cost of the region (see the note at
# ``_OVERHEAD_US``) - the cold one puts the break even higher, and the floor
# below is the conservative side of both: a workload below it would not have
# paid for the region even with the threads awake.
GROUP_MT_MIN_VALUES = 20_000


GROUP_MT_CHUNKS_PER_THREAD = 2


# --- the gather, which copies one value per value and holds no state ---
#
# The gather is the other shape of parallelism: it has no groups at all, and
# every value writes the one position of its own.  There is nothing to fold
# and nothing to accumulate - the loop over the values is cut into chunks and
# the threads copy into their own, and the result is the one of the sequential
# loop bit for bit, whatever the number of the threads and of the chunks.
#
# The gather is bound by the memory and not by the arithmetic, so the model
# above is the one that fits it: the bandwidth of the machine and the fixed
# cost of the parallel region.  The number of the groups is zero and the
# state is empty, so what is left is the transfer of the values - and the
# estimate of the chunked path, with the partial matrix out of it.
#
# Measured with ``python -m numpy_groupies.benchmarks.mt``, with the threads
# of the pool asleep (a pause before every timed call - the case of a program
# that calls the gather once) and with them awake (the case of a workload
# that is called in a loop): the gather pays for the region from about the
# size the chunked reductions do in the warm state, and from a few times
# above it in the cold one.
#
# The floor is therefore the one of the chunked reductions: the workload
# below it is one of a few hundred microseconds - the waking of the threads
# costs about as much as the gather itself, and a workload that is called
# once pays it without ever getting it back.
#
# A workload above the floor that is called once pays the waking of the
# threads once, and every call after it is the warm one: that is the price
# of the floor, and it is a price that a workload pays back after a few
# calls.
GATHER_MT_MIN_VALUES = MT_MIN_VALUES


@nb.njit(cache=True)
def _parallel_parts(n_threads: int) -> float:
    """The number of the threads worth of work the parallel path gets done in."""
    return 1.0 + (n_threads - 1) * _GROUP_PER_THREAD


@nb.njit(cache=True)
def group_chunk_count(n_threads: int, n_groups: int) -> int:
    """The number of the ranges the groups are cut into.

    More than the threads, for the same two reasons the chunks of the reductions
    are: the threads take the ranges from the queue of the pool, so a core that
    is busy with something else delays its own range and not the whole call, and
    the ranges are the unit the balance is struck in - twice as many of them
    halves the granularity of that balance as well.

    Never more than there are groups: a range without a group is no work at all,
    and with a single thread there is nothing to balance.
    """
    if n_threads < 2:
        return 1
    return min(n_threads * GROUP_MT_CHUNKS_PER_THREAD, n_groups)


@nb.njit(cache=True)
def estimate_group_mt_ratio(n_groups: int, n_threads: int, n_values: int, ns_per_value: float) -> float:
    """The estimated time of the group parallel path as a multiple of the sequential
    one, for an operation that works on the groups one after another.

    ``ns_per_value`` is what one value of the operation costs *in total*, measured
    on the sequential kernel of the operation, the counting sort of the
    placement included - which is why the bandwidth and the groups do not
    enter the estimate again: they are in that number.

    The sequential time is then ``n_values * ns_per_value``; the parallel one
    keeps the part that cannot be shared and divides the rest by the number of
    the threads worth of work the pool gets through with.

    The overhead of the region is the *warm* one (the threads of the pool
    spinning): the estimate is what the path is worth in a loop, and the
    floors of this module are what a single call is worth.
    """
    seq_us = n_values * ns_per_value / 1e3
    par_us = seq_us * (_GROUP_SERIAL_SHARE + (1.0 - _GROUP_SERIAL_SHARE) / _parallel_parts(n_threads))
    return (par_us + _OVERHEAD_US) / seq_us


@nb.njit(cache=True)
def should_use_group_mt(n_groups: int, n_threads: int, n_values: int, ns_per_value: float) -> bool:
    """Whether the loop over the groups of an operation is worth spreading.

    The arguments are those of ``should_use_mt``, plus ``ns_per_value``, the
    measured cost of one value of the operation.  ``n_groups`` is not in the
    estimate - the loops these operations spread have no per-group cost of
    their own - but a call with no groups to work on has no work at all.
    """
    if n_threads < 2:
        return False
    if n_values < GROUP_MT_MIN_VALUES or n_groups < 2:
        return False
    return estimate_group_mt_ratio(n_groups, n_threads, n_values, ns_per_value) <= MT_MAX_RATIO


@nb.njit(cache=True)
def chunk_count(n_threads: int, n_values: int) -> int:
    """The number of the chunks the data is split into.

    Several per thread, so that the threads can take the chunks from the
    queue of the pool and a busy core does not hold up the whole call.  Never
    more than there are values: an empty chunk is no work at all.

    With a single thread there is nothing to balance, and the data is not
    split at all - the result is then the one of the sequential kernel, bit
    for bit.
    """
    if n_threads < 2:
        return 1
    chunks = n_threads * MT_CHUNKS_PER_THREAD
    return min(chunks, n_values)


@nb.njit(cache=True)
def _bandwidth_bytes_per_s(n_values: int, n_threads: int) -> float:
    """The bandwidth the threads use together for a working set of the given size.

    A core delivers what it delivers; the machine delivers about twice that,
    and the threads beyond the second bring little.
    """
    megabytes: float = max(n_values * _BYTES_PER_ELEMENT / 1024 / 1024, 1e-6)
    core: float = max(0.5e9, _CORE_BYTES_PER_S * megabytes**_CORE_BYTES_PER_DECADE)
    return core * min(n_threads, _PEAK_CORES)


@nb.njit(cache=True)
def estimate_seq_time_us(n_groups: int, n_values: int) -> float:
    """The estimated time of the sequential path, in microseconds."""
    data_us = n_values * _BYTES_PER_ELEMENT / _bandwidth_bytes_per_s(n_values, 1) * 1e6
    return data_us + n_groups * _CG_SEQ / 1e3


@nb.njit(cache=True)
def estimate_mt_time_us(n_groups: int, n_threads: int, n_values: int, state_size: int = 1) -> float:
    """The estimated time of the parallel path, in microseconds.

    The chunks run in parallel, so the data arrives at the speed the threads
    use together; the states have to be filled and folded again, and that
    costs with the number of the groups.  ``state_size`` is the number of
    the arrays the state of the operation holds, counted against the two of
    the sum the constants were fitted on: a state of more arrays costs that
    much more to fill and to fold.
    """
    data_us = n_values * _BYTES_PER_ELEMENT / _bandwidth_bytes_per_s(n_values, n_threads) * 1e6
    # the states are one per thread, and the threads are the ones of the pool
    partials_us = n_threads * n_groups * _C_PART * state_size / 1e3
    return data_us + n_groups * _CG_MT * state_size / 1e3 + partials_us + _OVERHEAD_US


@nb.njit(cache=True)
def estimate_mt_ratio(n_groups: int, n_threads: int, n_values: int, state_size: int = 1) -> float:
    """The estimated time of the parallel path as a multiple of the sequential
    one.  Below 1 means the parallel path is faster; 1 means they are equal."""
    return estimate_mt_time_us(n_groups, n_threads, n_values, state_size) / estimate_seq_time_us(n_groups, n_values)


@nb.njit(cache=True)
def should_use_mt(n_groups: int, n_threads: int, n_values: int, state_size: int = 1) -> bool:
    """Whether the multithreaded variant of a kernel is worth using.

    ``n_groups`` is the number of groups, ``n_threads`` the number of threads
    the parallel path would use, ``n_values`` the number of values in the
    input.  Note that ``n_values`` is not the ``size`` argument of
    ``aggregate`` - that one is the number of groups.  ``state_size`` is the
    number of the arrays the state of the operation holds, against the two
    of the sum the constants were fitted on.

    The answer is False whenever the parallel path could be slower than the
    sequential one: with too few values, and whenever the estimate does not
    promise a clear gain.
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

    The threads of the pool of numba are the ones that work on the chunks,
    and numba decides how many they are: the environment variable
    ``NUMBA_NUM_THREADS`` at the start of the process, ``numba.set_num_threads``
    at any time.  This package asks no questions of its own - it uses what it
    finds.

    Read of every call and not once at the import: a number captured at the
    import would be the number the process started with, whatever the caller
    does with ``numba.set_num_threads`` afterwards.

    The number cuts the data into chunks - twice as many, see
    ``MT_CHUNKS_PER_THREAD`` - and the chunks are what the threads take from
    the queue of the pool.  One thread is the sequential path: the call goes
    the sequential way instead, with its result bit for bit.
    """
    return nb.get_num_threads()


def physical_core_count() -> int:
    """The number of physical cores the process may run on.

    Hardware threads that share a core are not counted twice: they share the
    load and store units of the core and its share of the memory bandwidth,
    so a second thread on the same core brings much less than a second core.
    On the machines that were measured, the threads beyond the physical
    cores brought between nothing and a third - and where they brought
    nothing, they cost up to 40% of the bandwidth.

    In a virtual machine the topology can be wrong (a guest sees the
    threads of the host as siblings of a core that is not there).  The count
    is then a guess - but a guess in the safe direction: the parallel path
    gets fewer threads, not more.

    The answer is read once and kept: the file behind it does not change
    while the process runs, and reading it per call would cost more than
    the kernel it decides about.
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
