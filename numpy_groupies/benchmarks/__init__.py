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
from typing import Any

import numpy as np

from numpy_groupies import aggregate_numpy, aggregate_purepy
from numpy_groupies.aggregate_common import funcs_collecting
from numpy_groupies.benchmarks import _ufunc
from numpy_groupies.tests.reference import aggregate_ref

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
try:
    from numpy_groupies.benchmarks import _polars
except ImportError:
    _polars = None


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
    "array",
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


# the sizes of the temporaries the participants ask for are of this order (an
# arange over the input, the cast of the weights of bincount), so the ladder
# reaches every one of them
_ALLOCATOR_WARMUP_MIN = 1 << 16
_ALLOCATOR_WARMUP_MAX = 1 << 26


def _warm_up_allocator() -> None:
    """Let the memory of the process arrive before anything is measured.

    The library allocator answers the first requests for a block size with
    freshly mapped pages and keeps them only after a round or two: every one of
    those pages has to be faulted in and zeroed, measured 2.2ms for the 4MB an
    implementation fills per call, of a function that takes 0.5ms otherwise.
    A measurement that starts before that has happened reports the kernel and
    not the implementation, and it reports it differently for every row - the
    first run of a table came out a third slower than the runs after it.

    Walking the sizes once pays the faulting up front, for 18ms.
    """
    itemsize = np.dtype(np.float64).itemsize
    size = _ALLOCATOR_WARMUP_MIN
    while size <= _ALLOCATOR_WARMUP_MAX:
        block = np.empty(size // itemsize)
        # every page of the block, not only its two ends
        block[:: 4096 // itemsize] = 1.0
        del block
        size <<= 1


def _byte_snapshot(*arrays: np.ndarray) -> tuple[bytes, ...]:
    """The bytes of the arrays, to see whether a call changed them.

    The bytes themselves and not a digest of them, though the name suggests a
    fingerprint: hashing measured 24x slower than the comparison it feeds (6ms
    for the 4MB of one array against 0.3ms of the copy and the byte-by-byte
    compare), and it is the memory the comparison has to cross either way that
    costs the time.  Nor is it ``np.array_equal``, which compares NaN by value
    and so reports a change for the untouched nan-data of this benchmark.

    The participants must not write into the data: one that does would change
    the results of the runs that follow, silently.  The snapshot is taken once
    per run - the data is the same for every function and every participant -
    and compared after every function, so the copy is paid once and not once
    per cell of the table.
    """
    return tuple(array.tobytes() for array in arrays)


def _restore(arrays: tuple[np.ndarray, ...], snapshot: tuple[bytes, ...]) -> bool:
    """Put the data back as it was, and say whether it had changed.

    The snapshot holds the bytes of the data as they were at the start of the
    run, so the original values are still in it: a participant that wrote into
    the data is undone, and the rest of the table is measured on the data the
    benchmark meant to measure.
    """
    changed = False
    for array, data in zip(arrays, snapshot):
        if array.tobytes() != data:
            array[...] = np.frombuffer(data, dtype=array.dtype).reshape(array.shape)
            changed = True
    return changed


def _assert_same_result(got, want, func_name):
    """Whether two results of the same function agree.

    The collecting function hands out a sequence per group, which numpy's own
    comparison cannot take in one go - the groups are compared one by one.
    """
    if func_name in funcs_collecting:
        groups = list(got) if isinstance(got, list) else np.ravel(got)
        expected = list(want) if isinstance(want, list) else np.ravel(want)
        assert len(groups) == len(expected), f"{len(groups)} groups against {len(expected)}"
        for one, other in zip(groups, expected):
            if one is None or other is None:
                assert one is None and other is None, f"{one!r} != {other!r}"
            else:
                np.testing.assert_array_equal(one, other)
        return
    np.testing.assert_array_almost_equal(got, want)


# the participants whose pool has been started already: the first call of a
# participant in a process starts the threads of its pool and lets each of them
# map the arena it allocates from, measured 2700 page faults for polars - none
# of which belongs into a measurement, and the first timed call would show
# nothing else.  The pool starts once per process, so once per participant is
# enough - not once per cell of the table.
_warmed_up: set[str] = set()


def single_run(
    implementations,
    group_idx,
    a,
    nana,
    *,
    repeat,
    raise_errors,
    funcs=func_list,
    on_row=None,
    snapshot: tuple[bytes, ...] | None = None,
):
    """One complete measurement pass - returns the table as
    ``{func_name: [cell, ...]}``, where a cell is a float in
    milliseconds, None for unavailable combinations or a marker string
    ("ERROR"/"FAIL").

    ``on_row(label, cells)`` is called after every function, so the caller can
    report the row as it is measured instead of waiting for the pass.  ``repeat``
    is the number of timed executions of a cell, capped per column by
    ``MAX_REPEAT``.

    ``snapshot`` is the bytes of the data as they were at the start of the run.
    It is compared with the data after every function: a participant that wrote
    into the data is undone, and the row it did it in is not reported, since
    its numbers were measured on data that changed under it.
    """
    table = {}
    for func in funcs:
        func_name = getattr(func, "__name__", func)
        used_a = nana if "nan" in func_name else a
        cells: list[Any] = []
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
                # the first result is the one the others are compared to
                results.append(res)
                try:
                    _assert_same_result(res, results[0], func_name)
                except AssertionError:
                    cells.append("FAIL")
                else:
                    if name not in _warmed_up:
                        # an untimed call that starts the pool of the participant
                        aggregate_fn(group_idx, used_a, func=func)
                        _warmed_up.add(name)
                    timer = timeit.Timer(lambda: aggregate_fn(group_idx, used_a, func=func))
                    t0 = min(timer.repeat(repeat=min(repeat, MAX_REPEAT.get(name, repeat)), number=1))
                    cells.append(t0 * 1000)
        if snapshot is not None and _restore((group_idx, a, nana), snapshot):
            # a participant wrote into the data: the numbers of this row were
            # measured on data that changed under it, so they are not reported
            print(
                f"{func_name}: a participant changed the data - the row is not reported",
                file=sys.stderr,
                flush=True,
            )
            cells = ["ERROR"] * len(cells)
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
            failures = [c for c in col if c in ("ERROR", "FAIL")]
            if failures:
                # a run that failed speaks for all of them: the numbers of the
                # other runs were measured against data that was not the same
                merged.append(failures[0])
                continue
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
# The reference paths are timed once whatever --repeat asks for: their number
# is a ratio rather than a result this package competes on, and every further
# execution only makes the whole measurement wait for them.
MAX_REPEAT = {"purepy": 1, "grouploop": 1}


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
    if "polars" in implementation_names:
        import polars

        postfix += f", Polars {polars.__version__}"
    return (
        f"*{platform.system()}({platform.machine()}), Python {sys.version.split()[0]}, "
        f"Numpy {np.version.version}{postfix}*"
    )


def select_funcs(names: str | None) -> tuple[Any, ...]:
    """The functions to measure, picked by name.

    Without a selection every function of the table is measured.  The names
    are the ones the table shows, so ``--funcs min,max`` measures the two
    extremes and nothing else.
    """
    if not names:
        return func_list
    known = {getattr(func, "__name__", func): func for func in func_list}
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown function(s): {', '.join(unknown)} - known are: {', '.join(known)}")
    return tuple(known[name] for name in wanted)


def benchmark(implementations, repeat=5, size=5e5, seed=100, raise_errors=False, runs=1, funcs=None):
    _warm_up_allocator()
    a, nana, group_idx = benchmark_data(size=size, seed=seed)
    # the data as it is now: the participants must not write into it, and the
    # snapshot is the one they are compared against after every function
    snapshot = _byte_snapshot(group_idx, a, nana)
    funcs = func_list if funcs is None else funcs
    labels = [getattr(func, "__name__", func) for func in funcs]
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
                funcs=funcs,
                on_row=report,
                snapshot=snapshot,
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
    _warm_up_allocator()
    labels = [str(groups) for groups in SORT_GROUPS]
    out, label_width = _open_report("groups", implementations, labels, runs)

    def report(label, cells):
        if out is not None:
            print(_table_row(label, cells, label_width), file=out, flush=True)

    rows = []
    for groups, label in zip(SORT_GROUPS, labels):
        a, nana, group_idx = benchmark_data(size=size, seed=seed, groups=min(groups, int(size)))
        snapshot = _byte_snapshot(group_idx, a, nana)
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
                    snapshot=snapshot,
                )
            )
        rows.append((label, _merge_runs(tables)["sort"]))

    if runs > 1:
        print(*_table_lines("groups", rows, implementations, label_width), _versions_line(implementations), sep="\n")
    else:
        print(_versions_line(implementations))


# The columns in the order they appear in the table.
COLUMN_ORDER = ("numpy", "numba", "purepy", "grouploop", "numbagg", "polars", "pandas", "ufunc")
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
        "grouploop": aggregate_ref,
        "ufunc": _ufunc.aggregate,
        # the optional dependencies are None when they are not installed
        "numbagg": _numbagg.aggregate if _numbagg is not None else None,
        "pandas": _pandas.aggregate if _pandas is not None else None,
        "polars": _polars.aggregate if _polars is not None else None,
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
        help="the reference implementation, the group loop the backends are measured against",
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
    parser.add_argument("--polars", action="store_true", help="the polars group_by wrapper (optional dependency)")
    parser.add_argument(
        "--sort",
        action="store_true",
        help=f"measure 'sort' only, over the group counts {'-'.join(str(g) for g in SORT_GROUPS)} instead of the full table",
    )
    parser.add_argument(
        "--funcs",
        help="comma-separated names of the functions to measure, e.g. 'min,max' (default: all of them)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="complete measurement passes, the per-cell minimum is reported (default: 1)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help=(
            "timed executions per cell, the fastest counts, except for the pure-Python and "
            "the group-loop column (default: 5)"
        ),
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
    if args.sort:
        # the sort is measured on its own, over the group counts
        sort_benchmark(
            _build_implementations(args),
            repeat=args.repeat,
            size=args.size,
            raise_errors=args.raise_errors,
            runs=max(1, args.runs),
        )
        return
    benchmark(
        _build_implementations(args),
        repeat=args.repeat,
        size=args.size,
        raise_errors=args.raise_errors,
        runs=max(1, args.runs),
        funcs=select_funcs(args.funcs),
    )


if __name__ == "__main__":
    main()
