"""The benchmark suite of the aggregation implementations.

Run it with ``python -m numpy_groupies.benchmarks`` - the module measures every
function of ``func_list`` once per implementation and prints the milliseconds
per call as a Markdown table.  NumPy and Numba are always measured, ``--help``
explains the columns to ask for, ``--all`` takes them all.  A single run, the
default, prints its rows as they are measured; a repeated one reports a counter
and then the fastest measurement of all runs.
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
    report the row as it is measured instead of waiting for the pass.  ``repeat``
    is the number of timed executions of a cell, capped per column by
    ``MAX_REPEAT``."""
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
                    timer = timeit.Timer(lambda: aggregate_fn(group_idx, used_a, func=func))
                    t0 = min(timer.repeat(repeat=min(repeat, MAX_REPEAT.get(name, repeat)), number=1))
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
# The slowest columns are timed once whatever --repeat asks for: several
# seconds of waiting would buy a marginally better minimum, and the whole
# measurement would take a multiple of its own runtime for one such column.
MAX_REPEAT = {"purepy": 1}


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
    """How the measurement reports itself while it works.

    A single run is its own report: the rows stream out as Markdown, header
    first, and nothing is repeated at the end.  Every further run would print
    the same table again, so a repeated measurement announces itself with a
    counter alone and the merged minimum is the only output."""
    label_width = max(len(header), *(len(label) for label in labels))
    if runs > 1:
        return None, label_width
    print(*_table_rule(header, implementations, label_width), sep="\n", flush=True)
    return sys.stdout, label_width


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

    def report(label, cells):
        if out is not None:
            print(_table_row(label, cells, label_width), file=out, flush=True)

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
                on_row=report,
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

    def report(label, cells):
        if out is not None:
            print(_table_row(label, cells, label_width), file=out, flush=True)

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
                    on_row=lambda _func_name, cells: report(label, cells),
                )
            )
        rows.append((label, _merge_runs(tables)["sort"]))

    if runs > 1:
        print(*_table_lines("groups", rows, implementations, label_width), _versions_line(implementations), sep="\n")
    else:
        print(_versions_line(implementations))


# The columns in the order they appear in the table.
COLUMN_ORDER = ("numpy", "numba", "purepy", "grouploop", "ufunc", "numbagg", "pandas")
# of those, the ones that have to be asked for
OPTIONAL_COLUMNS = COLUMN_ORDER[2:]


def _build_implementations(args):
    """the participants as (column name, aggregate callable) pairs, in
    ``COLUMN_ORDER``, the optional ones only when they were asked for"""
    if aggregate_numba is None:
        raise SystemExit("this benchmark needs the numba package installed")
    aggregates = {
        "numpy": aggregate_numpy.aggregate,
        "numba": aggregate_numba.aggregate,
        "purepy": aggregate_purepy.aggregate,
        "grouploop": _grouploop.aggregate,
        "ufunc": _ufunc.aggregate,
        # the optional dependencies are None when they are not installed
        "numbagg": _numbagg.aggregate if _numbagg is not None else None,
        "pandas": _pandas.aggregate if _pandas is not None else None,
    }
    impls = []
    for name in COLUMN_ORDER:
        if name in OPTIONAL_COLUMNS and not getattr(args, name):
            continue
        aggregate = aggregates[name]
        if aggregate is None:
            raise SystemExit(f"--{name} needs the optional {name} package installed")
        impls.append((name, aggregate))
    return impls


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m numpy_groupies.benchmarks",
        description="Benchmark the aggregate implementations against each other.\n"
        "The numpy and numba columns are always measured, the rest is asked for.",
        epilog="""The table is Markdown: milliseconds per call, '----' where an implementation
has no such function, and a line naming the versions it was measured with.  A
single run prints its rows as they are measured; a repeated run replaces them
with a counter and prints the fastest measurement of all runs once at the end.

numbagg ships nan-guarded reductions only, so its numbers for the plain
functions are those of the nan twins - on the nan-free data of this benchmark
the two coincide.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="include every optional column listed below")
    parser.add_argument(
        "--purepy",
        action="store_true",
        help="the pure-Python implementation, the reference the others are checked against",
    )
    parser.add_argument(
        "--grouploop",
        action="store_true",
        help="numpy with its fast path disabled, what a func it does not recognise costs",
    )
    parser.add_argument(
        "--ufunc", action="store_true", help="the ufunc.at-based implementation, incomplete and for comparison only"
    )
    parser.add_argument(
        "--numbagg",
        action="store_true",
        help="the numbagg grouped reductions (optional dependency, part of the dev extras)",
    )
    parser.add_argument("--pandas", action="store_true", help="the pandas groupby wrapper (optional dependency)")
    parser.add_argument(
        "--sort",
        action="store_true",
        help=f"measure 'sort' only, over the group counts {'-'.join(str(g) for g in SORT_GROUPS)} instead of the full table",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="complete measurement passes, the per-cell minimum is reported (default: 1)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="timed executions per cell, the fastest counts, the pure-Python column excepted (default: 5)",
    )
    parser.add_argument("--size", type=float, default=5e5, help="number of values in the benchmark data (default: 5e5)")
    parser.add_argument(
        "--raise-errors",
        action="store_true",
        help="abort on the first failing implementation instead of printing ERROR",
    )
    args = parser.parse_args(argv)
    if args.all:
        for column in OPTIONAL_COLUMNS:
            setattr(args, column, True)
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
