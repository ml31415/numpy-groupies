"""Streaming one-pass reductions: sums, products, lengths, logic, min/max."""

from .base import AggregateOp, complex_less


class Sum(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += val


class Prod(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] *= val


class Len(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] += 1


class All(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] &= bool(val)


class Any(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] |= bool(val)


class Last(AggregateOp):
    counter_fill_value = None

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        ret[ri] = val


class First(Last):
    reverse = True


class AllNan(AggregateOp):
    forced_fill_value = 1

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] &= val != val


class AnyNan(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        ret[ri] |= val != val


class Max(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        # select-form on purpose: the branchy equivalent (see below) compiles
        # ~15x slower under numba >= 0.66 when reached as a separate njit call.
        # Note: ret[ri] must be read into a local first - repeating the
        # subscript expression keeps the slow codegen (aliasing).
        # val != val has to be tested explicitly - a nan compares neither
        # smaller nor greater, in numpy's order for complex values as well as
        # for floats, so `complex_less(cur, val)` alone would leave a group
        # whose nan arrives after its first value at the finite value, unlike
        # np.max (and unlike this module's own arg kernels, which do report
        # such a group as invalid).  The nan variants are unaffected, they
        # filter the nans out before the kernel runs.  The extra comparison is
        # the cheapest way to get there: np.maximum in the non-first branch
        # (which propagates nans on its own, but does not order complex values
        # the way numpy does) and the branchy form below both measured ~45%
        # slower than this, while for integer dtypes LLVM folds the test away
        # entirely.
        # if counter[ri]:
        #     ret[ri] = val
        #     counter[ri] = 0
        # elif complex_less(ret[ri], val):
        #     ret[ri] = val
        first = counter[ri]
        counter[ri] = 0
        cur = ret[ri]
        ret[ri] = val if (first or val != val or complex_less(cur, val)) else cur


class Min(AggregateOp):
    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        # select-form on purpose, see Max._inner (including why val != val is
        # part of the condition and what it costs)
        # if counter[ri]:
        #     ret[ri] = val
        #     counter[ri] = 0
        # elif complex_less(val, ret[ri]):
        #     ret[ri] = val
        first = counter[ri]
        counter[ri] = 0
        cur = ret[ri]
        ret[ri] = val if (first or val != val or complex_less(val, cur)) else cur


class SumOfSquares(AggregateOp):
    forced_fill_value = 0

    @staticmethod
    def _inner(ri, val, ret, counter, mean, fill_value):
        counter[ri] = 0
        # the squared magnitude of val - complex values give a real, sqrt-free
        # result like np.var; bool has no .real in numba and keeps the plain
        # val * val product (isinstance resolves at compile time per dtype)
        if isinstance(val, bool):
            ret[ri] += val * val
        else:
            ret[ri] += val.real * val.real + val.imag * val.imag
