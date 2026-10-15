#!/usr/bin/python -B

import platform
import sys
import timeit
import types
from operator import itemgetter

import numpy as np

from numpy_groupies import aggregate_numpy
from numpy_groupies.tests.helpers import _implementations
from numpy_groupies.utils import allnan, anynan, nanfirst, nanlast


def aggregate_grouploop(*args, **kwargs):
    """wraps func in lambda which prevents aggregate_numpy from
    recognising and optimising it. Instead it groups and loops."""
    extrafuncs = {
        "allnan": allnan,
        "anynan": anynan,
        "first": itemgetter(0),
        "last": itemgetter(-1),
        "nanfirst": nanfirst,
        "nanlast": nanlast,
    }
    func = kwargs.pop("func")
    func = extrafuncs.get(func, func)
    if isinstance(func, str):
        raise NotImplementedError("Grouploop needs to be called with a function")
    return aggregate_numpy.aggregate(*args, func=lambda x: func(x), **kwargs)


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


def _numbagg_implementation():
    """The numbagg backend as a benchmark participant, included with
    ``--numbagg`` (lazy import - numbagg is an optional dependency).

    numbagg 0.9 ships nan-guarded grouped reductions only, so the plain
    funcs run through their nan twins - the benchmark's plain data is
    nan-free, where the results coincide.  ``ddof`` defaults to 0 like in
    the rest of this package (numbagg's own default is 1).  Operations
    without a numbagg counterpart raise NotImplementedError and are
    reported as "----".  ``size``/``fill_value`` are accepted but ignored:
    numbagg infers the group count and fills empty groups itself (the
    benchmark data leaves no group empty).
    """
    try:
        import numbagg
    except ImportError as e:
        raise SystemExit("--numbagg needs the optional numbagg package installed") from e

    funcs = {
        "sum": numbagg.group_nansum,
        "prod": numbagg.group_nanprod,
        "len": numbagg.group_nancount,
        "all": numbagg.group_nanall,
        "any": numbagg.group_nanany,
        "min": numbagg.group_nanmin,
        "max": numbagg.group_nanmax,
        "mean": numbagg.group_nanmean,
        "std": numbagg.group_nanstd,
        "var": numbagg.group_nanvar,
        "first": numbagg.group_nanfirst,
        "last": numbagg.group_nanlast,
        "argmin": numbagg.group_nanargmin,
        "argmax": numbagg.group_nanargmax,
        "anynan": numbagg.group_nanany,
        "allnan": numbagg.group_nanall,
        "nansum": numbagg.group_nansum,
        "nanprod": numbagg.group_nanprod,
        "nanlen": numbagg.group_nancount,
        "nanall": numbagg.group_nanall,
        "nanany": numbagg.group_nanany,
        "nanmin": numbagg.group_nanmin,
        "nanmax": numbagg.group_nanmax,
        "nanmean": numbagg.group_nanmean,
        "nanstd": numbagg.group_nanstd,
        "nanvar": numbagg.group_nanvar,
        "nanfirst": numbagg.group_nanfirst,
        "nanlast": numbagg.group_nanlast,
        "nanargmin": numbagg.group_nanargmin,
        "nanargmax": numbagg.group_nanargmax,
    }
    ddof_funcs = frozenset(("std", "var", "nanstd", "nanvar"))

    def aggregate(group_idx, a, func="sum", size=None, fill_value=None, order="C", dtype=None, axis=None, ddof=0):
        name = func if isinstance(func, str) else getattr(func, "__name__", None)
        fn = funcs.get(name)
        if fn is None:
            raise NotImplementedError(f"numbagg has no grouped '{name}'")
        if name in ddof_funcs:
            return fn(a, group_idx, ddof=ddof)
        return fn(a, group_idx)

    mod = types.ModuleType("aggregate_numbagg")
    mod.aggregate = aggregate
    return mod


def benchmark_data(size=5e5, seed=100):
    rnd = np.random.RandomState(seed=seed)
    group_idx = rnd.randint(0, int(1e3), int(size))
    a = rnd.random_sample(group_idx.size)
    a[a > 0.8] = 0
    nana = a.copy()
    nana[(nana < 0.2) & (nana != 0)] = np.nan
    nan_share = np.mean(np.isnan(nana))
    assert 0.15 < nan_share < 0.25, f"{nan_share * 100:3f}% nans"
    return a, nana, group_idx


def benchmark(implementations, repeat=5, size=5e5, seed=100, raise_errors=False):
    a, nana, group_idx = benchmark_data(size=size, seed=seed)

    print(f"{'function':13}" + "".join(impl.__name__.rsplit("_", 1)[1].rjust(14) for impl in implementations))
    print("-" * (13 + 14 * len(implementations)))
    for func in func_list:
        func_name = getattr(func, "__name__", func)
        print(func_name.ljust(13), end="")
        results = []
        used_a = nana if "nan" in func_name else a

        for impl in implementations:
            if impl is None:
                print("----".rjust(14), end="")
                continue
            aggregatefunc = impl.aggregate

            try:
                res = aggregatefunc(group_idx, used_a, func=func)
            except NotImplementedError:
                print("----".rjust(14), end="")
                continue
            except Exception:
                if raise_errors:
                    raise
                print("ERROR".rjust(14), end="")
            else:
                results.append(res)
                try:
                    np.testing.assert_array_almost_equal(res, results[0])
                except AssertionError:
                    print("FAIL".rjust(14), end="")
                else:
                    t0 = min(
                        timeit.Timer(lambda: aggregatefunc(group_idx, used_a, func=func)).repeat(
                            repeat=repeat, number=1
                        )
                    )
                    print(f"{t0 * 1000:.3f}".rjust(14), end="")
            sys.stdout.flush()
        print()

    implementation_names = [impl.__name__.rsplit("_", 1)[1] for impl in implementations]
    postfix = ""
    if "numba" in implementation_names:
        import numba

        postfix += f", Numba {numba.__version__}"
    if "pandas" in implementation_names:
        import pandas

        postfix += f", Pandas {pandas.__version__}"
    print(
        f"{platform.system()}({platform.machine()}), Python {sys.version.split()[0]}, Numpy {np.version.version}"
        f"{postfix}"
    )


if __name__ == "__main__":
    implementations = _implementations if "--purepy" in sys.argv else _implementations[1:]
    with_pandas = "--pandas" in sys.argv
    if not with_pandas:
        implementations = implementations[:-1]
    if "--numbagg" in sys.argv:
        # numbagg joins before pandas, which stays the right-most column
        if with_pandas:
            implementations.insert(-1, _numbagg_implementation())
        else:
            implementations.append(_numbagg_implementation())
    benchmark(implementations, raise_errors=False)
