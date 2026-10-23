"""Calibration benchmark and cost model for the multithreaded kernels.

This module is a tool of its own, not a participant of the benchmark suite:
it measures the parallel variant of the kernels against the sequential one
and fits the cost model that decides between them.

Run it on every machine whose numbers should go into the model:

    python -m numpy_groupies.benchmarks.mt

It measures the sequential kernel against the parallel one over a grid of
data sizes, group counts and thread counts, fits the cost model to the
measurements and prints the constants and the decisions the model takes.
No file is written - the output is the result.

What is measured:

  the sum        the sequential kernel of this script - a plain single pass -
                 against the parallel one, which splits the data into chunks
                 and gives every chunk its own partial sums.  Their ratio is
                 free of the machinery around the kernels and shows what the
                 machine can do.

  the gather     the gather of ``unpack_into`` against its parallel twin.

  the overhead   the fixed cost of entering the parallel path.

If numpy_groupies is importable, its sequential kernel is measured as well:
the package carries a fixed cost per call that the reference kernel does
not have, and that cost sits on both sides of the decision.

The model of the kernel:

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

The script needs numpy and numba, nothing else; numpy_groupies is optional.
It checks that the parallel result equals the sequential one before it
reports a timing - a wrong result is worse than no result.

Typical runtime: a few minutes with the default grid.
"""

from __future__ import annotations

import argparse
import math
import platform
import subprocess
import sys
import time

import numba as nb
import numpy as np

from numpy_groupies.aggregate_numba.mt import logical_core_count, physical_core_count

BYTES_PER_ELEMENT = 16.0  # a float64 value and an int64 index

# the thresholds of the decision
MT_MIN_VALUES = 100_000
MT_MAX_RATIO = 0.9

# the fixed cost of the parallel path, measured directly (an empty prange loop)
OVERHEAD_US = {1: 0.0, 2: 4.7, 3: 5.0, 4: 5.2, 6: 5.8, 8: 6.4, 16: 9.0}


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


def machine_info() -> dict:
    return {
        "cpu_model": _cpu_model(),
        "physical_cores": physical_core_count(),
        "hardware_threads": logical_core_count(),
        "l3_bytes": _l3_bytes(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": nb.__version__,
        "numba_threads": nb.get_num_threads(),
        "numba_threading_layer": nb.threading_layer(),
    }


# ---------------------------------------------------------------------------
# the measurement


def measure(fn, repeat: int) -> float:
    """The time of one call in microseconds, the fastest of ``repeat`` calls."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1e6


def repeat_for(n_values: int) -> int:
    """Enough calls per measurement to get out of the noise, but not too many."""
    return int(min(200, max(3, 4e6 / max(n_values, 1))))


def _package_kernel():
    """The sequential kernel of numpy_groupies, if the package is there."""
    try:
        from numpy_groupies.aggregate_numba import sum as npg_sum
    except ImportError:  # the package is optional
        return None
    return npg_sum


def run_grid(sizes, group_counts, thread_counts, rounds: int, with_gather: bool):
    """The measurements: the times of the sequential and the parallel kernel."""
    rng = np.random.default_rng(100)
    rows = []
    pool = nb.get_num_threads()  # the size of the pool, before any set_num_threads

    npg_sum = _package_kernel()
    if npg_sum is None:
        print("numpy_groupies is not importable - the overhead of the package is unknown")

    print("warming up the kernels ...", flush=True)
    warm = rng.integers(0, 10, 1000)
    warm_a = rng.random(1000)
    for t in thread_counts:
        if t <= pool:
            sum_mt(warm, warm_a, np.zeros((t, 10)), 10)
            if with_gather:
                gather_mt(np.zeros(10), warm, np.zeros(1000))
    empty_parallel(warm)
    sum_seq(warm, warm_a, np.zeros(10))
    gather_seq(np.zeros(10), warm, np.zeros(1000))
    if npg_sum is not None:
        npg_sum(warm, warm_a)

    for n_values in sizes:
        repeat = repeat_for(n_values)
        print(f"n_values = {n_values:,}", flush=True)
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
                    print(f"    skipping {n_threads} threads - the pool has {pool}", flush=True)
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
                for _ in range(rounds):
                    seq = min(seq, measure(seq_call, repeat))
                    mt = min(mt, measure(mt_call, repeat))
                    if npg_sum is not None:
                        npg = min(npg, measure(npg_call, repeat))
                nb.set_num_threads(pool)

                rows.append(
                    {
                        "n_values": n_values,
                        "n_groups": n_groups,
                        "n_threads": n_threads,
                        "seq_us": seq,
                        "mt_us": mt,
                        "npg_us": npg,
                    }
                )
                print(
                    f"    groups={n_groups:>8}  threads={n_threads:>3}  "
                    f"seq={seq:10.1f} us  mt={mt:10.1f} us  ratio={mt / seq:.2f}"
                    + (f"  npg={npg:10.1f} us" if npg_sum is not None else ""),
                    flush=True,
                )

        if with_gather:
            ret = rng.random(max(group_counts))
            out = np.empty(n_values)
            gather_seq(ret, group_idx, out)
            seq = float("inf")
            for _ in range(rounds):
                seq = min(
                    seq, measure(lambda ret=ret, group_idx=group_idx, out=out: gather_seq(ret, group_idx, out), repeat)
                )
            for n_threads in thread_counts:
                if n_threads > pool:
                    continue
                gather_mt(ret, group_idx, out)
                nb.set_num_threads(n_threads)
                mt = float("inf")
                for _ in range(rounds):
                    mt = min(
                        mt,
                        measure(lambda ret=ret, group_idx=group_idx, out=out: gather_mt(ret, group_idx, out), repeat),
                    )
                nb.set_num_threads(pool)
                print(
                    f"    gather        threads={n_threads:>3}  "
                    f"seq={seq:10.1f} us  mt={mt:10.1f} us  ratio={mt / seq:.2f}",
                    flush=True,
                )

    # the fixed cost of the parallel path
    for n_threads in thread_counts:
        if n_threads > pool:
            continue
        nb.set_num_threads(n_threads)
        overhead = float("inf")
        for _ in range(rounds):
            overhead = min(overhead, measure(lambda: empty_parallel(warm), 200))
        nb.set_num_threads(pool)
        print(f"    overhead of {n_threads} threads: {overhead:.2f} us", flush=True)

    return rows


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


def call_overhead(rows: list) -> float:
    """The fixed cost per call that the package carries and the reference
    kernel does not: the difference of the two sequential kernels."""
    differences = []
    for row in rows:
        if row["npg_us"] == float("inf"):
            continue
        differences.append(row["npg_us"] - row["seq_us"])
    if not differences:
        return 0.0
    # the smallest difference is the least contaminated one - where the kernel
    # itself takes the longest, the fixed cost matters least
    return max(0.0, min(differences))


# ---------------------------------------------------------------------------
# the report


def report(rows: list, info: dict) -> None:
    phys = int(info["physical_cores"])
    overhead = call_overhead(rows)
    params, loss_value, solutions = fit(rows, phys)
    model = Model(params, phys)

    print("\n" + "=" * 72)
    print("=== the fit ===")
    print(f"  {len(rows)} observations, {phys} physical cores, overhead per call in the package: {overhead:.1f} us")
    print("\n  the constants:")
    print(f"    CORE0        = {params[0]:.6g}   # the bandwidth of one core at a working set of 1 MB")
    print(f"    CORE_EXP     = {params[1]:.6g}   # how it depends on the working set")
    print(f"    SMT_FRACTION = {params[2]:.6g}   # what an SMT thread brings in relation to a core")
    print(f"    PEAK_FACTOR  = {params[3]:.6g}   # the peak of the machine, in cores")
    print(f"    CG_SEQ       = {params[4]:.6g}   # ns per group, sequential")
    print(f"    CG_MT        = {params[5]:.6g}   # ns per group, parallel")
    print(f"    C_PART       = {params[6]:.6g}   # ns per element of the partial matrix")
    print(f"    residual error (log RMS) = {math.sqrt(loss_value / (2 * len(rows))):.4f}")

    # the spread of the parameters: a constant that lies far apart over the
    # start points is not identifiable - the measurements do not pin it down,
    # and its value is an artifact of the search
    names = ["CORE0", "CORE_EXP", "SMT_FRACTION", "PEAK_FACTOR", "CG_SEQ", "CG_MT", "C_PART"]
    print("\n  the spread of the constants over the start points of the search:")
    print(f"    {'constant':<14}{'best':>12}{'lowest':>12}{'highest':>12}   identifiable")
    for i, name in enumerate(names):
        values = [s[i] for s in solutions]
        low, high = min(values), max(values)
        spread = abs(high - low) / abs(values[0]) if values[0] else (0.0 if high == low else float("inf"))
        mark = "yes" if spread < 0.25 else ("no" if spread > 1.0 else "weak")
        print(f"    {name:<14}{values[0]:>12.4g}{low:>12.4g}{high:>12.4g}   {mark}")

    sizes = sorted({r["n_values"] for r in rows})
    threads = sorted({r["n_threads"] for r in rows})
    print("\n  the times of the model against the measurement (sequential / parallel, us):")
    print("    " + " " * 12 + "".join(f"{'T=' + str(t):>22}" for t in threads))
    for n in sizes:
        cells = ""
        for t in threads:
            measured = [r for r in rows if r["n_values"] == n and r["n_threads"] == t]
            if not measured:
                cells += f"{'-':>22}"
                continue
            row = measured[0]
            seq_model = model.kernel_seq_us(n, row["n_groups"])
            mt_model = model.kernel_mt_us(n, row["n_groups"], t)
            cells += f"{row['seq_us']:>7.0f}/{row['mt_us']:<6.0f}{seq_model:>5.0f}/{mt_model:<4.0f}"
        print(f"    {n:>10} MB{cells}")
    print("    (per cell: measured seq/mt, then model seq/mt)")

    print("\n  the decisions (with the overhead of the package on both sides):")
    print(f"    {'n_values':>10} {'groups':>8} {'T':>3} {'npg-ratio':>10} {'model':>8} {'decision':>13}  ok")
    ok = True
    missed = 0
    dangerous = 0
    for obs in rows:
        n, g, t = obs["n_values"], obs["n_groups"], obs["n_threads"]
        seq = model.kernel_seq_us(n, g) + overhead
        mt = model.kernel_mt_us(n, g, t) + overhead
        ratio = mt / seq
        measured = obs["mt_us"] / obs["npg_us"] if obs["npg_us"] != float("inf") else obs["mt_us"] / obs["seq_us"]
        if t < 2 or n < MT_MIN_VALUES:
            decision, right = False, True
        else:
            decision = ratio <= MT_MAX_RATIO
            # the measured npg ratio is the most optimistic view: the
            # prototype does not carry the overhead of the package
            if measured >= 0.99:
                right = not decision
                if decision:
                    dangerous += 1
            else:
                right = decision
                if not decision:
                    missed += 1
        ok &= right
        print(
            f"    {n:>10} {g:>8} {t:>3} {measured:>10.2f} {ratio:>8.2f} "
            f"{'MT' if decision else 'seq':>13}  {'ok' if right else 'FEHLER'}"
        )
    print(f"\n  all decisions correct: {ok}   missed gains: {missed}   dangerous cases: {dangerous}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sizes",
        default="100000,200000,500000,1000000,2000000,5000000,10000000",
        help="comma-separated numbers of values to measure",
    )
    parser.add_argument(
        "--groups",
        default="1000,10000,100000",
        help="comma-separated numbers of groups to measure",
    )
    parser.add_argument(
        "--threads",
        default="",
        help="comma-separated thread counts to measure (default: 1 up to the pool size)",
    )
    parser.add_argument("--rounds", type=int, default=5, help="measurement rounds, the fastest counts (default: 5)")
    parser.add_argument("--no-gather", action="store_true", help="skip the gather measurements")
    parser.add_argument("--seed", type=int, default=100, help="the seed of the data (default: 100)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    pool = nb.get_num_threads()
    if args.threads:
        thread_counts = [int(t) for t in args.threads.split(",")]
    else:
        thread_counts = list(range(1, pool + 1))
    sizes = [int(s) for s in args.sizes.split(",")]
    group_counts = [int(g) for g in args.groups.split(",")]

    info = machine_info()
    print("=== machine ===")
    for key, value in info.items():
        print(f"  {key:<24} {value}")
    print(f"  measured thread counts      {thread_counts}")
    print()

    rows = run_grid(sizes, group_counts, thread_counts, args.rounds, not args.no_gather)
    report(rows, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
