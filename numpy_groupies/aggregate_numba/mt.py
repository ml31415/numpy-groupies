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
_CORE_BYTES_PER_S = 1.746e10  # what one core delivers at a working set of 1 MB
_CORE_BYTES_PER_DECADE = -0.09133  # and how that depends on the working set
_PEAK_CORES = 2  # the peak of the machine, in cores
_CG_SEQ = 0.9504  # nanoseconds per group, sequential
_CG_MT = 8.412  # nanoseconds per group, parallel
_C_PART = 0.1181  # nanoseconds per element of the partial matrix

# the fixed cost of the parallel path, measured directly (an empty prange loop
# costs 4.7us with two threads and 5.8us with six)
_OVERHEAD_US = 5.0

# below this many values the parallel path cannot pay for itself, whatever
# the number of groups: its fixed cost alone is a few percent of the
# sequential time, and the machines that run it are not exactly this one
MT_MIN_VALUES = 100_000

# the parallel path must be this much faster than the sequential one before
# it is used - the margin absorbs the error of the model and of the
# measurement, and the error goes in the conservative direction
MT_MAX_RATIO = 0.8


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
def estimate_mt_time_us(n_groups: int, n_threads: int, n_values: int) -> float:
    """The estimated time of the parallel path, in microseconds.

    The chunks run in parallel, so the data arrives at the speed the threads
    use together; the partial matrix of every chunk has to be written and
    read again, and that costs with the number of the groups.
    """
    data_us = n_values * _BYTES_PER_ELEMENT / _bandwidth_bytes_per_s(n_values, n_threads) * 1e6
    partials_us = n_threads * n_groups * _C_PART / 1e3
    return data_us + n_groups * _CG_MT / 1e3 + partials_us + _OVERHEAD_US


@nb.njit(cache=True)
def estimate_mt_ratio(n_groups: int, n_threads: int, n_values: int) -> float:
    """The estimated time of the parallel path as a multiple of the sequential
    one.  Below 1 means the parallel path is faster; 1 means they are equal."""
    return estimate_mt_time_us(n_groups, n_threads, n_values) / estimate_seq_time_us(n_groups, n_values)


@nb.njit(cache=True)
def should_use_mt(n_groups: int, n_threads: int, n_values: int) -> bool:
    """Whether the multithreaded variant of a kernel is worth using.

    ``n_groups`` is the number of groups, ``n_threads`` the number of threads
    the parallel path would use, ``n_values`` the number of values in the
    input.  Note that ``n_values`` is not the ``size`` argument of
    ``aggregate`` - that one is the number of groups.

    The answer is False whenever the parallel path could be slower than the
    sequential one: with too few values, and whenever the estimate does not
    promise a clear gain.
    """
    if n_threads < 2:
        return False
    if n_values < MT_MIN_VALUES:
        return False
    return estimate_mt_ratio(n_groups, n_threads, n_values) <= MT_MAX_RATIO


# --- the machine, for the number of threads ---

_physical_cores: int | None = None

# The number of threads the parallel path uses.  None hands the decision to
# the pool of numba - every hardware thread it offers.  A number overrides
# it: the chunks are as many as the threads, so the number is also the
# number of the threads that do the work.
_thread_count: int | None = None

# The size of the pool, read once: the pool does not grow or shrink while
# the process runs, and the code that wants another size says so through
# ``set_thread_count``.
_pool_threads = max(1, nb.get_num_threads())


def mt_thread_count() -> int:
    """The number of threads the parallel path should use.

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
    """
    if _thread_count is not None:
        return _thread_count
    return _pool_threads


def set_thread_count(n: int | None) -> None:
    """The number of threads the parallel path uses.

    None gives the pool of numba the say (every hardware thread of the
    machine), a number of at least one takes it over.  The number of the
    threads is also the number of the chunks the data is split into, so it
    is also the number of the threads that do the work.
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
