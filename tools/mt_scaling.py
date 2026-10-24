"""How the parallel path of the numba kernels scales with the threads.

    python -m tools.mt_scaling [threads ...]

Every thread count gets a process of its own: the threads of the pool spin
after a parallel region, and they disturb the measurement of the next one in
the same process.  The times are the fastest of five blocks of ten calls.

The numbers say how many threads the parallel path is worth: it splits its
data into as many chunks as there are threads, and the chunks are reduced
separately.
"""

import os
import subprocess
import sys


def measure(threads: int, n_values: int) -> tuple[float, float]:
    """The time of one call in milliseconds, in a process of its own.

    The fastest and the median of the blocks - the spread between them is
    the load of the machine, which on a shared one is larger than the
    difference between the thread counts.
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
    result = subprocess.run(
        [sys.executable, "-c", script, str(n_values)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "NUMBA_NUM_THREADS": str(threads)},
    )
    numbers = result.stdout.strip().splitlines()[-1].split()
    return float(numbers[0]), float(numbers[1])


def machine_facts() -> dict:
    """What the machine offers the threads - and what it really has."""
    import numba as nb

    facts = {
        "cpu_count": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "numba_threads": nb.get_num_threads(),
        "cgroup_quota": "-",
    }
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as handle:
                facts["cgroup_quota"] = handle.read().strip()
        except OSError:
            continue
        else:
            break
    try:
        with open("/proc/cpuinfo") as info:
            for line in info:
                if line.startswith("model name"):
                    facts["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return facts


def main(argv: list) -> int:
    threads = [int(t) for t in argv[1:]] or [1, 2, 3, 4, 5, 6, 7, 8]
    sizes = [100_000, 1_000_000, 5_000_000]
    print("the time of one call of the sum, in milliseconds")
    print("\nthe machine:")
    for key, value in machine_facts().items():
        print(f"  {key:<16} {value}")
    for n_values in sizes:
        print(f"\nn_values = {n_values:,}")
        base = None
        for t in threads:
            try:
                ms, median = measure(t, n_values)
            except subprocess.CalledProcessError as exc:
                print(f"  {t} threads: {exc.stderr.strip().splitlines()[-1] if exc.stderr else 'fehlgeschlagen'}")
                continue
            if base is None:
                base = ms
            speedup = base / ms
            bandwidth = n_values * 16 / ms / 1e3
            print(
                f"  {t:>2} threads: {ms:8.3f} ms   Speedup {speedup:5.2f}x   {bandwidth:6.1f} GB/s"
                f"   (Mittel {median:8.3f} ms)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
