"""Cumulative (N-to-N) operations: the running value lives in ``outer``."""

from .base import AggregateNtoN
from .reductions import Max, Min, Prod, Sum


class CumSum(AggregateNtoN, Sum):
    pass


class CumProd(AggregateNtoN, Prod):
    pass


class CumMax(AggregateNtoN, Max):
    pass


class CumMin(AggregateNtoN, Min):
    pass
