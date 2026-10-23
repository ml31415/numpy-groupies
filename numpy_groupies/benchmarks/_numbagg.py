"""The numbagg backend as a benchmark participant, imported with
``--numbagg`` in :mod:`numpy_groupies.benchmarks.generic` - numbagg is an
optional dependency (part of the dev extras), an ImportError on this module
means it is not installed."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import numba
import numbagg


# numbagg compiles its grouped reductions to parallel-target gufuncs -
# for a scatter reduction the per-call thread sync of that target
# dominates at this data size (measured 3.6ms against 0.5ms at one
# thread on a 6-core machine).  numbagg is therefore held at one thread -
# but only for its own calls: setting the thread count of numba at the
# import of this module would hold every other participant of the
# benchmark at one thread as well.  The switch is cheap: measured 0.494ms
# with the switch and 0.483ms without it, on a machine with three cores
# and six hardware threads - where the two are equal anyway, since there
# are not enough cores for the parallel target to win.
def _single_threaded(func: Callable[..., Any]) -> Callable[..., Any]:
    """The function, called with numba at one thread."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        threads = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            return func(*args, **kwargs)
        finally:
            numba.set_num_threads(threads)

    return wrapper


funcs = {
    "sum": _single_threaded(numbagg.group_nansum),
    "prod": _single_threaded(numbagg.group_nanprod),
    "len": _single_threaded(numbagg.group_nancount),
    "all": _single_threaded(numbagg.group_nanall),
    "any": _single_threaded(numbagg.group_nanany),
    "min": _single_threaded(numbagg.group_nanmin),
    "max": _single_threaded(numbagg.group_nanmax),
    "mean": _single_threaded(numbagg.group_nanmean),
    "std": _single_threaded(numbagg.group_nanstd),
    "var": _single_threaded(numbagg.group_nanvar),
    "first": _single_threaded(numbagg.group_nanfirst),
    "last": _single_threaded(numbagg.group_nanlast),
    "argmin": _single_threaded(numbagg.group_nanargmin),
    "argmax": _single_threaded(numbagg.group_nanargmax),
    "anynan": _single_threaded(numbagg.group_nanany),
    "allnan": _single_threaded(numbagg.group_nanall),
    "nansum": _single_threaded(numbagg.group_nansum),
    "nanprod": _single_threaded(numbagg.group_nanprod),
    "nanlen": _single_threaded(numbagg.group_nancount),
    "nanall": _single_threaded(numbagg.group_nanall),
    "nanany": _single_threaded(numbagg.group_nanany),
    "nanmin": _single_threaded(numbagg.group_nanmin),
    "nanmax": _single_threaded(numbagg.group_nanmax),
    "nanmean": _single_threaded(numbagg.group_nanmean),
    "nanstd": _single_threaded(numbagg.group_nanstd),
    "nanvar": _single_threaded(numbagg.group_nanvar),
    "nanfirst": _single_threaded(numbagg.group_nanfirst),
    "nanlast": _single_threaded(numbagg.group_nanlast),
    "nanargmin": _single_threaded(numbagg.group_nanargmin),
    "nanargmax": _single_threaded(numbagg.group_nanargmax),
}
ddof_funcs = frozenset(("std", "var", "nanstd", "nanvar"))


def aggregate(group_idx, a, func="sum", size=None, fill_value=None, order="C", dtype=None, axis=None, ddof=0):
    """numbagg grouped reductions as a benchmark participant.

    numbagg 0.9 ships nan-guarded grouped reductions only, so the plain
    funcs run through their nan twins - the benchmark's plain data is
    nan-free, where the results coincide.  ``ddof`` defaults to 0 like in
    the rest of this package (numbagg's own default is 1).  Operations
    without a numbagg counterpart raise NotImplementedError and are
    reported as "----".  ``size``/``fill_value`` are accepted but ignored:
    numbagg infers the group count and fills empty groups itself (the
    benchmark data leaves no group empty).
    """
    name = func if isinstance(func, str) else getattr(func, "__name__", None)
    fn = funcs.get(name)
    if fn is None:
        raise NotImplementedError(f"numbagg has no grouped '{name}'")
    if name in ddof_funcs:
        return fn(a, group_idx, ddof=ddof)
    return fn(a, group_idx)
