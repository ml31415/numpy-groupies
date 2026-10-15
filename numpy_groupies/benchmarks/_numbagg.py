"""The numbagg backend as a benchmark participant, included with
``--numbagg`` in :mod:`numpy_groupies.benchmarks.generic` (lazy import -
numbagg is an optional dependency, part of the dev extras)."""

import types


def implementation():
    """the numbagg aggregate wrapped as a benchmark participant module.

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

    import numba

    # numbagg compiles its grouped reductions to parallel-target gufuncs -
    # for a scatter reduction the per-call thread sync of that target
    # dominates at this data size (measured 3.6ms against 0.5ms at one
    # thread on a 6-core machine).  every other participant of this
    # benchmark runs single threaded, so numbagg is held there too
    numba.set_num_threads(1)

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
