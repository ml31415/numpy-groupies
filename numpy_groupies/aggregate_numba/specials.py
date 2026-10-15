"""Operations with fully custom kernels.

``Median`` is an order statistic: values are placed group by group via a
counting sort and the middle is selected by partition.  ``Trapezoid``
integrates consecutive sample pairs within each group in a single pass.
"""

import numpy as np

from ..aggregate_common import DEFAULT_FILL_VALUE
from ..aggregate_numpy import _aggregate_base
from .base import AggregateOp
from .jit import njit_stable


class Median(AggregateOp):
    """The median is an order statistic, so unlike the streaming reductions
    it needs group-ordered data: values are placed group by group via
    counting sort (O(n)) and the middle of each group is then *selected*
    via partition (O(n) on average) instead of sorting within the groups."""

    def __call__(
        self,
        group_idx,
        a,
        size=None,
        fill_value=DEFAULT_FILL_VALUE,
        order="C",
        dtype=None,
        axis=None,
        ddof=0,
    ):
        if not np.isscalar(a) and np.issubdtype(a.dtype, np.complexfloating):
            # numba cannot partition complex values - the numpy implementation
            # selects the middle of the lexicographically ordered values
            return _aggregate_base(
                group_idx, a, self.func, size=size, fill_value=fill_value, order=order, dtype=dtype, axis=axis
            )
        return super().__call__(group_idx, a, size, fill_value, order, dtype, axis, ddof)

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # median is order-insensitive, so reverse is ignored
        _count = njit_stable(cls._count)
        _starts = njit_stable(cls._starts)
        _place = njit_stable(cls._place)
        _valid = njit_stable(cls._valid)
        _middle = njit_stable(cls._middle)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # signature kept identical to AggregateOp.callable
            size = len(ret)
            counts = _count(group_idx, size)
            starts = _starts(counts)
            # counting-sort placement into contiguous group-blocks (the
            # order within a block is arbitrary, which the median ignores)
            a_sorted = np.empty(group_idx.size, a.dtype)
            _place(group_idx, a, starts.copy(), a_sorted)
            buf = np.empty(group_idx.size, a.dtype)
            for g in range(size):
                if counts[g] == 0:
                    continue  # untouched group, keeps the fill value
                m = _valid(g, a_sorted, counts, starts, buf, nans)
                if m == 0:
                    # all-nan group for nanmedian, or a nan-poisoned group
                    # for plain median (where any nan is contagious)
                    ret[g] = fill_value if nans else np.nan
                else:
                    ret[g] = _middle(buf[:m])
                    counter[g] = True

        return loop

    @staticmethod
    def _count(group_idx, size):
        """bounds-checked per-group element counts"""
        counts = np.zeros(size, np.int64)
        for i in range(len(group_idx)):
            g = group_idx[i]
            if g < 0:
                raise ValueError("negative indices not supported")
            if g >= size:
                raise ValueError("one or more indices in group_idx are too large")
            counts[g] += 1
        return counts

    @staticmethod
    def _starts(counts):
        """first index of each group-block"""
        return np.cumsum(counts) - counts

    @staticmethod
    def _place(group_idx, a, cursor, a_sorted):
        """scatter the values into their group-blocks"""
        for i in range(len(group_idx)):
            g = group_idx[i]
            a_sorted[cursor[g]] = a[i]
            cursor[g] += 1

    @staticmethod
    def _valid(g, a_sorted, counts, starts, buf, nans):
        """copy group g into buf, returning the number of usable values;
        groups with nans yield 0 for plain median (nan is contagious),
        while nanmedian keeps the group with the nans filtered out"""
        begin, size_g = starts[g], counts[g]
        if not nans:
            buf[:size_g] = a_sorted[begin : begin + size_g]
            if np.sum(buf[:size_g] != buf[:size_g]) > 0:
                return 0
            return size_g
        m = 0
        for i in range(size_g):
            v = a_sorted[begin + i]
            if v == v:
                buf[m] = v
                m += 1
        return m

    @staticmethod
    def _middle(values):
        """select the middle value(s) of a group - no sorting needed"""
        m = len(values)
        part = np.partition(values, m // 2)
        if m % 2 == 1:
            return part[m // 2]
        # the lower middle is the maximum of the unsorted left half; the 1.0
        # promotes integral input to float64, just like np.median promotes
        # before averaging - summing the two middles in their own dtype
        # overflows (int64 2**62 + 2**62) and adds bool as a logical or
        return (np.max(part[: m // 2]) * 1.0 + part[m // 2]) / 2


class Trapezoid(AggregateOp):
    """Trapezoidal integration of each group, in the order the samples appear.
    The area accumulates in ret, the previous sample is kept in mean and
    counter doubles as the "group started" flag, so one pass over the data
    is enough."""

    forced_fill_value = 0
    mean_fill_value = 0
    mean_dtype = None

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # the integral follows the order of the input, so reverse is ignored
        valgetter = njit_stable(cls._valgetter_scalar if scalar else cls._valgetter)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            # ddof carries the sample spacing dx, see Trapezoid.__call__
            size = len(ret)
            for i in range(len(group_idx)):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                val = valgetter(a, i)
                if nans and val != val:
                    continue
                if counter[ri]:
                    counter[ri] = False
                else:
                    ret[ri] += 0.5 * ddof * (mean[ri] + val)
                mean[ri] = val

        return loop

    def __call__(
        self,
        group_idx,
        a,
        size=None,
        fill_value=DEFAULT_FILL_VALUE,
        order="C",
        dtype=None,
        axis=None,
        dx=1.0,
    ):
        # the one float argument of this operation is the sample spacing,
        # which the kernels expect in the position otherwise used for ddof
        return super().__call__(group_idx, a, size, fill_value, order, dtype, axis, dx)
