"""The operation registry: names to compiled op instances, plus alias dispatch."""

from ..aggregate_common import build_dispatch, funcs_no_separate_nan
from ..utils import aliasing
from .argreductions import ArgMax, ArgMin
from .averaging import Mean, Std, Var
from .cumulative import CumMax, CumMin, CumProd, CumSum
from .reductions import All, AllNan, Any, AnyNan, First, Last, Len, Max, Min, Prod, Sum, SumOfSquares
from .specials import Median, Trapezoid


def get_funcs():
    funcs = {}
    for op in (
        Sum,
        Prod,
        Len,
        All,
        Any,
        Last,
        First,
        AllNan,
        AnyNan,
        Min,
        Max,
        ArgMin,
        ArgMax,
        Mean,
        Median,
        Trapezoid,
        Std,
        Var,
        SumOfSquares,
        CumSum,
        CumProd,
        CumMax,
        CumMin,
    ):
        funcname = op.__name__.lower()
        funcs[funcname] = op(funcname)
        if funcname not in funcs_no_separate_nan:
            funcname = "nan" + funcname
            funcs[funcname] = op(funcname, nans=True)
    return funcs


_impl_dict = get_funcs()

# alias (string or callable) -> compiled op instance, so that the common call
# path resolves with a single dict lookup instead of the two-step aliasing +
# implementation resolution in get_func
_dispatch = build_dispatch(_impl_dict, aliasing)

# input dtypes for which the 'sum' output dtype depends on the input length
