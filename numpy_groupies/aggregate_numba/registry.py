"""The operation registry: names to python-level wrappers, plus alias dispatch."""

from ..aggregate_common import build_dispatch, funcs_no_separate_nan
from ..utils import aliasing
from .argreductions import argmax, argmin
from .averaging import mean, std, var
from .cumulative import cummax, cummin, cumprod, cumsum
from .ops import AggregateOp
from .reductions import all, allnan, any, anynan, first, last, len, max, min, prod, sum, sumofsquares
from .specials import MedianOp, TrapezoidOp, median, trapezoid

# (name, kernel pair, wrapper class) - the order is the registration order
_TABLE = (
    ("sum", sum, AggregateOp),
    ("prod", prod, AggregateOp),
    ("len", len, AggregateOp),
    ("all", all, AggregateOp),
    ("any", any, AggregateOp),
    ("last", last, AggregateOp),
    ("first", first, AggregateOp),
    ("allnan", allnan, AggregateOp),
    ("anynan", anynan, AggregateOp),
    ("min", min, AggregateOp),
    ("max", max, AggregateOp),
    ("argmin", argmin, AggregateOp),
    ("argmax", argmax, AggregateOp),
    ("mean", mean, AggregateOp),
    ("median", median, MedianOp),
    ("trapezoid", trapezoid, TrapezoidOp),
    ("std", std, AggregateOp),
    ("var", var, AggregateOp),
    ("sumofsquares", sumofsquares, AggregateOp),
    ("cumsum", cumsum, AggregateOp),
    ("cumprod", cumprod, AggregateOp),
    ("cummax", cummax, AggregateOp),
    ("cummin", cummin, AggregateOp),
)


def get_funcs():
    funcs = {}
    for funcname, pair, wrapper in _TABLE:
        funcs[funcname] = wrapper(funcname, pair.plain)
        if funcname not in funcs_no_separate_nan:
            nan_name = "nan" + funcname
            funcs[nan_name] = wrapper(nan_name, pair.nan, nans=True)
    return funcs


_impl_dict = get_funcs()

# alias (string or callable) -> wrapper instance, so that the common call
# path resolves with a single dict lookup instead of the two-step aliasing +
# implementation resolution in get_func
_dispatch = build_dispatch(_impl_dict, aliasing)
