"""The decision whether a kernel runs multithreaded.

The cost model is fitted to measurements of the sum kernel on a machine with
3 physical cores and 2 hardware threads each (AMD EPYC, 6 threads, numba
0.68, OpenMP).  The constants are deliberately on the pessimistic side: they
were scaled down (the bandwidth) and up (the costs of the groups) after the
fit, so that the model errs towards the sequential path.  The margin of 0.8
adds to that: the parallel path is only used when the model promises it at
least 20% less time.

The times are in microseconds, the constants in nanoseconds or bytes per
second.

    sequential:  n_values * bytes / BW(n_values, 1)  +  n_groups * cg_seq
    parallel:    n_values * bytes / BW(n_values, n_threads)
                 +  n_groups * cg_mt  +  n_threads * n_groups * c_part
                 +  overhead

The bandwidth saturates at the machine, not at the number of threads: a
core delivers about 17 GB/s and the machine delivers about twice that.  The
model does not divide the time by the number of threads, but by the
bandwidth the threads use together.

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

# the fixed cost of the parallel path, measured directly (an empty prange loop
# costs 4.7us with two threads and 5.8us with six)
#
# The cost of the region is the cost of waking the threads of the pool up -
# the threading layer of numba (OpenMP by default) keeps them asleep between
# the calls, and every region wakes them up again.  It is a cost of a few
# microseconds here, but it is not the only one of its kind: on a machine
# whose threads spin (and the layer of OpenMP can be told to), the cost of
# the region is what decides whether a parallel path is worth it at all, and
# it is the reason the thresholds of this module are measured and not guessed.
# A call that does not pay for the region does not make one - the sequential
# path is a few microseconds away and it is exact.
#
# The floors below are fitted on the *warm* cost - the state of a workload
# that is called in a loop.  The cold cost (the pool asleep, which is what a
# program that calls a kernel once pays) is about twenty times the warm one,
# and it is paid once by the first call of a workload: the calls after it
# find the threads awake.  A floor fitted on the cold cost would keep a
# whole band of workloads out of the parallel path for good - and a workload
# that is called in a loop would never get there, because it would never
# make the region that wakes the threads up.
_OVERHEAD_US = 5.0

# The cost of the region whose threads are asleep.  The layer spins the
# threads for a moment after a region and sleeps them afterwards: measured on
# this machine, an empty region costs 4us with the threads awake and 85us
# with them asleep - and the pause that puts them to sleep is about ten
# milliseconds.
#
# The two costs are not alternatives: a program that calls a kernel in a loop
# pays the warm one every time, and a program that calls it once and waits
# pays the cold one.  The floors of this module are fitted on the warm cost
# (see the note at ``_OVERHEAD_US``), and the cold one is what a single call
# pays on top of them.
_OVERHEAD_COLD_US = 90.0

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
# nothing to take, and the call waits for the slowest core (measured: a
# kernel with one slow chunk measured 2.3ms with six chunks against 0.4ms
# with twelve).
#
# The chunks are not the states, though: the states are one per thread of the
# pool, and the chunks are aggregated into the state of the thread that takes
# them.  The state of a thread is written by that thread only, so the chunks
# need no lock - and the memory stays at one state per thread of the pool.
#
# The pool and not the number the call asked for: the number a thread reports
# for itself is bounded by the pool, and a call for two threads on a machine
# whose pool has six still hands its chunks to those six - a state per thread
# of the call would be read out of its range, and the chunks that fell off the
# end of it would be lost.
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
# on six threads (three physical cores): 3.0 to 3.4 times the sequential time
# for the median, the quantile, the sort and the distinct count, which is 0.4
# per thread.  The constant is the measured one, and the work these loops do is
# arithmetic on the values and not the memory, so it does not run into the
# bandwidth the model above is built around.
_GROUP_PER_THREAD = 0.40

# the share of these kernels that the parallel path cannot take: the counting
# sort that places the values before the groups can be worked on.  Measured on
# 2 million float64 values in 100 groups: 5.5 ms of the 34 ms of the median and
# 6 ms of the 204 ms of the sort.  A third is the pessimistic side of that.
_GROUP_SERIAL_SHARE = 1.0 / 3.0

# below this many values the loop over the groups is not worth a parallel
# region.  The operations of this shape cost 18 to 83 nanoseconds a value
# against the 0.5 of the sum the model above was fitted on, so the same
# overhead of a few microseconds is a few hundred times smaller a share of
# their time: the measured break even of the median is at 5 000 values (1.9
# times faster) and 3.4 times faster from 20 000 on.  The floor of
# ``MT_MIN_VALUES`` would keep these operations single threaded up to twenty
# times past the point where it costs them a factor of three.
#
# The floor is fitted on the *warm* cost of the region - the state of a
# workload that is called in a loop.  The cold cost (the threads of the pool
# asleep) puts the break even of the median at 10 000 values and of the
# sort at 50 000; the warm one puts it at 8 000 and at 16 000.  The floor
# below is the conservative side of both, and a workload below it would not
# have paid for the region even with the threads awake.
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
# Measured with ``python -m numpy_groupies.benchmarks.mt`` (six threads,
# three cores), every timed call preceded by a pause, so that the threads
# of the pool are asleep when the region starts - the case of a program
# that calls the gather once:
#
#     values      sequential    parallel
#      60 000        68.8us      103.1us     0.67 times
#     100 000       101.1us      122.3us     0.83 times
#     150 000       154.8us      148.8us     1.04 times
#     200 000       201.1us      159.2us     1.26 times
#
# and with the threads awake - the state of a workload that is called in a
# loop:
#
#     values      sequential    parallel
#      20 000        10.7us       9.7us      1.10 times
#      50 000        26.5us      22.3us      1.19 times
#     100 000        49.5us      36.7us      1.35 times
#
# The floor is the one of the chunked reductions (100 000 values): that is
# where the gather pays in the warm state, and the workload that is below it
# is a workload of a few hundred microseconds - the waking of the threads
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
    on the sequential kernel (the median 18, the sort 49), the counting sort of
    the placement included - which is why the bandwidth and the groups do not
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
    use together; the state of every thread has to be filled and folded
    again, and that costs with the number of the groups.  ``state_size`` is
    the number of the arrays the state of the operation holds, counted
    against the state of the sum the constants were fitted on - the
    variance holds six arrays against the two of the sum, and its state
    costs three times as much to fill and to fold.
    """
    data_us = n_values * _BYTES_PER_ELEMENT / _bandwidth_bytes_per_s(n_values, n_threads) * 1e6
    # the states are one per thread of the pool - the number the call asked
    # for is what the chunks are cut after, and the states are one per thread
    # that takes a chunk
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

# The number of threads the parallel path uses.  None hands the decision to
# the pool of numba - every hardware thread it offers.  A number overrides
# it: the chunks are cut after it (twice as many, see MT_CHUNKS_PER_THREAD),
# so it is the number of the threads that are asked to do the work.
_thread_count: int | None = None

# The size of the pool, read once: the pool does not grow or shrink while
# the process runs, and the code that wants another size says so through
# ``set_thread_count``.
_pool_threads = max(1, nb.get_num_threads())

# The number of the threads the pool was started with - the bound of the
# numbers a thread reports for itself.  ``numba.set_num_threads`` masks the
# threads that are not to be used, so the threads of a parallel region are
# fewer than this, but never more: a kernel that keeps one state per thread
# of the pool needs this many of them, and not the number the call asked for
# (a call for two threads on a pool of six still hands its chunks to six).
#
# Read from the configuration and not from ``numba.get_num_threads()``: that
# one is a dynamic global, and a driver that reads one never reaches the
# on-disk cache - which is worth 30 seconds of compiling at every start.  The
# number is handed to the kernels as an argument for the same reason.
_pool_upper_bound = max(1, int(getattr(nb.config, "NUMBA_NUM_THREADS", _pool_threads)))


def mt_thread_count() -> int:
    """The number of the threads the parallel path should be cut for.

    Every hardware thread of the machine by default: the threads that share
    a core bring little on some machines and a lot on others, and the reports
    disagree - on a guest with six processors the parallel path measured 1.8
    times faster with six threads than with three, on another machine the
    two were within the noise of the measurement.  The hardware threads are
    free to use, and the parallel path is faster than the sequential one
    with any number of them.

    ``set_thread_count`` overrides the number, and so does the environment
    variable ``NUMPY_GROUPIES_MT_THREADS`` at the start of the process -
    for the machines where the threads that share a core do cost.

    The number cuts the data into chunks; which of them runs the pieces is the
    pool's say, and a machine whose pool is smaller than the number delivers
    the smaller of the two.  One thread is the sequential path: the call goes
    the sequential way instead, with its result bit for bit.
    """
    if _thread_count is not None:
        return _thread_count
    return _pool_threads


def set_thread_count(n: int | None) -> None:
    """The number of threads the parallel path uses.

    None gives the pool of numba the say (every hardware thread of the
    machine), a number of at least one takes it over.  The data is cut into
    twice as many chunks as that (``MT_CHUNKS_PER_THREAD``), so that the
    threads have something to take from the queue - and a machine whose pool
    is smaller than the number delivers the smaller of the two.
    """
    global _thread_count
    _thread_count = None if n is None else max(1, int(n))


_override = os.environ.get("NUMPY_GROUPIES_MT_THREADS")
if _override:
    set_thread_count(int(_override))


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
