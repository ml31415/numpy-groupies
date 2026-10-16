"""The benchmark suite of the aggregation implementations.

Run it with ``python -m numpy_groupies.benchmarks`` - the module measures every
function of ``func_list`` once per implementation and prints the milliseconds
per call as a Markdown table.  With the default single run the rows arrive as
they are measured, ``--runs=3`` instead reports a counter and prints the merged
table once.  ``--help`` lists the optional columns (purepy, pandas, numbagg and
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


def single_run(implementations, group_idx, a, nana, *, repeat, raise_errors, funcs=func_list, on_row=None):
    """One complete measurement pass - returns the table as
    ``{func_name: [cell, ...]}``, where a cell is a float in
    milliseconds, None for unavailable combinations or a marker string
    ("ERROR"/"FAIL").

    ``on_row(label, cells)`` is called after every function, so the caller can
    report the row as it is measured instead of waiting for the pass."""
    table = {}
    for func in funcs:
        func_name = getattr(func, "__name__", func)
        used_a = nana if "nan" in func_name else a
        cells = []
        results = []
        for name, aggregate_fn in implementations:
            try:
                res = aggregate_fn(group_idx, used_a, func=func)
            except NotImplementedError:
                cells.append(None)
                continue
            except Exception:
                if raise_errors:
                    raise
                cells.append("ERROR")
            else:
                results.append(res)
                try:
                    np.testing.assert_array_almost_equal(res, results[0])
                except AssertionError:
                    cells.append("FAIL")
                else:
                    t0 = min(
                        timeit.Timer(lambda: aggregate_fn(group_idx, used_a, func=func)).repeat(repeat=repeat, number=1)
                    )
                    cells.append(t0 * 1000)
        table[func_name] = cells
        if on_row is not None:
            on_row(func_name, cells)
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


# Width of every measured column: a Markdown table reads just as well in a
# terminal when the cells are padded, and padding does not hurt the rendering.
CELL_WIDTH = 12
NO_IMPL = "----"


def _cell_text(cell):
    """The cell as text: the milliseconds, ``NO_IMPL`` for a combination the
    implementation does not offer, or the ERROR/FAIL marker."""
    if cell is None:
        return NO_IMPL
    if isinstance(cell, float):
        return f"{cell:.3f}"
    return str(cell)


def _table_row(label, cells, label_width):
    return "| " + " | ".join([label.ljust(label_width)] + [_cell_text(cell).rjust(CELL_WIDTH) for cell in cells]) + " |"


def _table_rule(header, implementations, label_width):
    """The two Markdown lines that introduce a table - the numeric columns
    carry a trailing colon, which is what right-aligns them."""
    names = [name for name, _ in implementations]
    return [
        "| " + " | ".join([header.ljust(label_width)] + [name.rjust(CELL_WIDTH) for name in names]) + " |",
        "| " + " | ".join(["-" * label_width] + ["-" * (CELL_WIDTH - 1) + ":" for _ in names]) + " |",
    ]


def _table_lines(header, rows, implementations, label_width):
    """``rows`` are (row label, cells) pairs - the complete Markdown table."""
    return [
        *_table_rule(header, implementations, label_width),
        *(_table_row(label, cells, label_width) for label, cells in rows),
    ]


def _open_report(header, implementations, labels, runs):
    """Where the rows go while measuring, and how wide the label column is.

    A single run is the report itself: its rows stream to stdout as Markdown
    and nothing is repeated at the end.  Several runs would print the table
    once per run, so those rows go to stderr behind a counter and the merged
    minimum is what stdout gets."""
    label_width = max(len(header), *(len(label) for label in labels))
    out = sys.stdout if runs == 1 else sys.stderr
    print(*_table_rule(header, implementations, label_width), sep="\n", file=out, flush=True)
    return out, label_width


def _versions_line(implementations):
    """The setup of this run as an italic Markdown paragraph."""
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
    return (
        f"*{platform.system()}({platform.machine()}), Python {sys.version.split()[0]}, "
        f"Numpy {np.version.version}{postfix}*"
    )


def benchmark(implementations, repeat=5, size=5e5, seed=100, raise_errors=False, runs=1):
    a, nana, group_idx = benchmark_data(size=size, seed=seed)
    labels = [getattr(func, "__name__", func) for func in func_list]
    out, label_width = _open_report("function", implementations, labels, runs)

    tables = []
    for run in range(runs):
        if runs > 1:
            print(f"run {run + 1}/{runs}", file=sys.stderr, flush=True)
        tables.append(
            single_run(
                implementations,
                group_idx,
                a,
                nana,
                repeat=repeat,
                raise_errors=raise_errors,
                on_row=lambda label, cells: print(_table_row(label, cells, label_width), file=out, flush=True),
            )
        )

    if runs > 1:
        table = _merge_runs(tables)
        print(
            *_table_lines("function", [(label, table[label]) for label in labels], implementations, label_width),
            _versions_line(implementations),
            sep="\n",
        )
    else:
        print(_versions_line(implementations))


# The group counts of the ``--sort`` series - the default of the main table and
# a hundredth and a hundred times it.  ``sort`` orders every group on its own,
# so its cost turns on the group SIZE (a few large blocks against many small
# ones) and the main table would only ever show one point of that curve.
SORT_GROUPS = (100, 1_000, 10_000, 100_000)


def sort_benchmark(implementations, *, repeat=5, size=5e5, seed=100, raise_errors=False, runs=1):
    """Measure ``func='sort'`` alone, over the group counts of ``SORT_GROUPS``."""
    labels = [str(groups) for groups in SORT_GROUPS]
    out, label_width = _open_report("groups", implementations, labels, runs)

    rows = []
    for groups, label in zip(SORT_GROUPS, labels):
        a, nana, group_idx = benchmark_data(size=size, seed=seed, groups=min(groups, int(size)))
        tables = []
        for run in range(runs):
            if runs > 1:
                print(f"groups {label}: run {run + 1}/{runs}", file=sys.stderr, flush=True)
            tables.append(
                single_run(
                    implementations,
                    group_idx,
                    a,
                    nana,
                    repeat=repeat,
                    raise_errors=raise_errors,
                    funcs=("sort",),
                    on_row=lambda label_, cells: print(_table_row(label, cells, label_width), file=out, flush=True),
                )
            )
        rows.append((label, _merge_runs(tables)["sort"]))

    if runs > 1:
        print(*_table_lines("groups", rows, implementations, label_width), _versions_line(implementations), sep="\n")
    else:
        print(_versions_line(implementations))


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
