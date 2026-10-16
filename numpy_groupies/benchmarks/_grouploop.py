"""The numpy implementation with its fast path disabled, imported with
``--grouploop`` in :mod:`numpy_groupies.benchmarks`.

Wrapping the function in a lambda keeps ``aggregate_numpy`` from recognising it
as one of its own, so it groups the values and loops over the groups instead -
the column therefore measures what a ``func`` that the implementation does not
know costs, which is the price of the recognition itself."""

from operator import itemgetter

from .. import aggregate_numpy
from ..utils import allnan, anynan, nanfirst, nanlast

# the string names of func_list that a per-group callable can stand in for -
# everything else (the cumulatives, the argreductions, sort) needs the
# implementation's own handling and has no group-loop equivalent
extrafuncs = {
    "allnan": allnan,
    "anynan": anynan,
    "first": itemgetter(0),
    "last": itemgetter(-1),
    "nanfirst": nanfirst,
    "nanlast": nanlast,
}


def aggregate(*args, **kwargs):
    """``aggregate_numpy.aggregate`` on a function it cannot recognise."""
    func = kwargs.pop("func")
    func = extrafuncs.get(func, func)
    if isinstance(func, str):
        raise NotImplementedError("Grouploop needs to be called with a function")
    return aggregate_numpy.aggregate(*args, func=lambda x: func(x), **kwargs)
