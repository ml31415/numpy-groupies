"""The benchmark suite of the aggregation implementations.

Run it with ``python -m numpy_groupies.benchmarks`` - the module measures every
function of ``func_list`` once per implementation and prints the milliseconds
per call.  ``--help`` lists the optional columns (purepy, pandas, numbagg and
the numpy group-and-loop path) and the data size.
"""

import argparse
import platform
import sys
import timeit

import numpy as np

from numpy_groupies import aggregate_numpy, aggregate_purepy
from numpy_groupies.benchmarks import _grouploop, _ufunc

try:
    from numpy_groupies import aggregate_numba
except ImportError:
    aggregate_numba = None
try:
    from numpy_groupies.benchmarks import _numbagg
except ImportError:
    _numbagg = None
try:
    from numpy_groupies.benchmarks import _pandas
except ImportError:
    _pandas = None


def arbitrary(iterator):
    tmp = 0
    for i, x in enumerate(iterator, 1):
        tmp += x**i
    return tmp


func_list = (
    np.sum,
    np.prod,
    np.min,
    np.max,
    len,
    np.all,
    np.any,
    "anynan",
    "allnan",
    np.mean,
    np.median,
    np.trapezoid,
    np.std,
    np.var,
    "first",
    "last",
    "argmax",
    "argmin",
    np.nansum,
    np.nanprod,
    np.nanmin,
    np.nanmax,
    "nanlen",
    "nanall",
    "nanany",
    np.nanmean,
    np.nanmedian,
    "nantrapezoid",
    np.nanvar,
    np.nanstd,
    "nanfirst",
    "nanlast",
    "nanargmin",
    "nanargmax",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    arbitrary,
    "sort",
)


def benchmark_data(size=5e5, seed=100, groups=1000):
    rnd = np.random.RandomState(seed=seed)
    group_idx = rnd.randint(0, int(groups), int(size))
    a = rnd.random_sample(group_idx.size)
    a[a > 0.8] = 0
    nana = a.copy()
    nana[(nana < 0.2) & (nana != 0)] = np.nan
    nan_share = np.mean(np.isnan(nana))
    assert 0.15 < nan_share < 0.25, f"{nan_share * 100:3f}% nans"
    return a, nana, group_idx


def single_run(implementations, group_idx, a, nana, *, repeat, raise_errors, verbose, funcs=func_list):
    """One complete measurement pass - returns the table as
    ``{func_name: [cell, ...]}``, where a cell is a float in
    milliseconds, None for unavailable combinations or a marker string
    ("ERROR"/"FAIL")."""
    table = {}
    for func in funcs:
        func_name = getattr(func, "__name__", func)
        used_a = nana if "nan" in func_name else a
        cells = []
        results = []
        if verbose:
            print(func_name.ljust(13), end="", file=sys.stderr)
        for name, aggregate_fn in implementations:
            try:
                res = aggregate_fn(group_idx, used_a, func=func)
            except NotImplementedError:
                cells.append(None)
                if verbose:
                    print("----".rjust(14), end="", file=sys.stderr)
                continue
            except Exception:
                if raise_errors:
                    raise
                cells.append("ERROR")
                if verbose:
                    print("ERROR".rjust(14), end="", file=sys.stderr)
            else:
                results.append(res)
                try:
                    np.testing.assert_array_almost_equal(res, results[0])
                except AssertionError:
                    cells.append("FAIL")
                    if verbose:
                        print("FAIL".rjust(14), end="", file=sys.stderr)
                else:
                    t0 = min(
                        timeit.Timer(lambda: aggregate_fn(group_idx, used_a, func=func)).repeat(repeat=repeat, number=1)
                    )
                    cells.append(t0 * 1000)
                    if verbose:
                        print(f"{t0 * 1000:.3f}".rjust(14), end="", file=sys.stderr)
            if verbose:
                sys.stderr.flush()
        if verbose:
            print(file=sys.stderr)
        table[func_name] = cells
    return table


def _merge_runs(tables):
    """Per-cell minimum over the complete runs - filters the system noise
    (cpu frequency, background load) that makes single runs wobble."""
    if len(tables) == 1:
        return tables[0]
    table = {}
    for func_name in tables[0]:
        merged = []
        for col in zip(*(t[func_name] for t in tables)):
            numbers = [c for c in col if isinstance(c, float)]
            if numbers:
                merged.append(min(numbers))
            else:
                merged.append(next((c for c in col if c is not None), None))
        table[func_name] = merged
    return table


def _print_table(header, rows, implementations):
    """``rows`` are (row label, cells) pairs as produced by ``single_run``."""
    print(f"{header:13}" + "".join(name.rjust(14) for name, _ in implementations))
    print("-" * (13 + 14 * len(implementations)))
    for label, cells in rows:
        print(label.ljust(13), end="")
        for cell in cells:
            text = "----" if cell is None else (f"{cell:.3f}" if isinstance(cell, float) else cell)
            print(text.rjust(14), end="")
        print()


def _print_versions(implementations):
    implementation_names = [name for name, _ in implementations]
    postfix = ""
    if "numba" in implementation_names:
        import numba

        postfix += f", Numba {numba.__version__}"
    if "numbagg" in implementation_names:
        import numbagg

        postfix += f", numbagg {numbagg.__version__}"
    if "pandas" in implementation_names:
        import pandas

        postfix += f", Pandas {pandas.__version__}"
    print(
        f"{platform.system()}({platform.machine()}), Python {sys.version.split()[0]}, Numpy {np.version.version}"
        f"{postfix}"
    )


def benchmark(implementations, repeat=5, size=5e5, seed=100, raise_errors=False, runs=1):
    a, nana, group_idx = benchmark_data(size=size, seed=seed)

    tables = []
    for r in range(runs):
        if runs > 1:
            print(f"run {r + 1}/{runs}", file=sys.stderr)
        tables.append(
            single_run(implementations, group_idx, a, nana, repeat=repeat, raise_errors=raise_errors, verbose=runs == 1)
        )

    table = _merge_runs(tables)
    _print_table(
        "function",
        [(name, table[name]) for name in (getattr(func, "__name__", func) for func in func_list)],
        implementations,
    )
    _print_versions(implementations)


# The group counts of the ``--sort`` series - the default of the main table and
# a hundredth and a hundred times it.  ``sort`` orders every group on its own,
# so its cost turns on the group SIZE (a few large blocks against many small
# ones) and the main table would only ever show one point of that curve.
SORT_GROUPS = (100, 1_000, 10_000, 100_000)


def sort_benchmark(implementations, *, repeat=5, size=5e5, seed=100, raise_errors=False, runs=1):
    """Measure ``func='sort'`` alone, over the group counts of ``SORT_GROUPS``."""
    rows = []
    for groups in SORT_GROUPS:
        a, nana, group_idx = benchmark_data(size=size, seed=seed, groups=min(groups, int(size)))
        tables = []
        for r in range(runs):
            if runs > 1:
                print(f"groups {groups}: run {r + 1}/{runs}", file=sys.stderr)
            tables.append(
                single_run(
                    implementations,
                    group_idx,
                    a,
                    nana,
                    repeat=repeat,
                    raise_errors=raise_errors,
                    verbose=False,
                    funcs=("sort",),
                )
            )
        rows.append((str(groups), _merge_runs(tables)["sort"]))

    _print_table("groups", rows, implementations)
    _print_versions(implementations)


def _build_implementations(args):
    """the participants as (column name, aggregate callable) pairs, in
    column order - purepy first, numbagg before the right-most pandas"""
    if aggregate_numba is None:
        raise SystemExit("this benchmark needs the numba package installed")
    impls = []
    if args.purepy:
        impls.append(("purepy", aggregate_purepy.aggregate))
    impls.extend(
        (
            ("ufunc", _ufunc.aggregate),
            ("numpy", aggregate_numpy.aggregate),
        )
    )
    if args.grouploop:
        impls.append(("grouploop", _grouploop.aggregate))
    impls.append(("numba", aggregate_numba.aggregate))
    if args.numbagg:
        if _numbagg is None:
            raise SystemExit("--numbagg needs the optional numbagg package installed")
        impls.append(("numbagg", _numbagg.aggregate))
    if args.pandas:
        if _pandas is None:
            raise SystemExit("--pandas needs the optional pandas package installed")
        impls.append(("pandas", _pandas.aggregate))
    return impls


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m numpy_groupies.benchmarks",
        description="Benchmark the aggregate implementations against each other "
        "(default columns: ufunc, numpy, numba).",
    )
    parser.add_argument("--purepy", action="store_true", help="include the pure-Python implementation")
    parser.add_argument("--pandas", action="store_true", help="include the pandas implementation")
    parser.add_argument(
        "--grouploop",
        action="store_true",
        help="include numpy's group-and-loop path (what an unknown callable costs)",
    )
    parser.add_argument(
        "--numbagg", action="store_true", help="include the numbagg implementation (optional dependency, dev extras)"
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help=f"measure 'sort' only, over the group counts {'-'.join(str(g) for g in SORT_GROUPS)} instead of the full table",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="complete measurement passes, the per-cell minimum is reported (default: 1)"
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="timed executions per cell, the fastest counts (default: 5)"
    )
    parser.add_argument("--size", type=float, default=5e5, help="number of values in the benchmark data (default: 5e5)")
    parser.add_argument(
        "--raise-errors",
        action="store_true",
        help="abort on the first failing implementation instead of printing ERROR",
    )
    args = parser.parse_args(argv)
    measure = sort_benchmark if args.sort else benchmark
    measure(
        _build_implementations(args),
        repeat=args.repeat,
        size=args.size,
        raise_errors=args.raise_errors,
        runs=max(1, args.runs),
    )


if __name__ == "__main__":
    main()
