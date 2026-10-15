"""Index-returning reductions: argmax and argmin."""

from .base import AggregateOp, complex_less
from .jit import njit_stable


class ArgMax(AggregateOp):
    mean_fill_value = 0
    mean_dtype = None

    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # The loop body is inlined on purpose: with numba >= 0.67 the generic
        # loop plus a separate njit `_inner` call compiles ~20x slower for this
        # reduction. The inlined logic is equivalent to the former _inner:
        #   first value of a group seeds ret/mean, nans reset the group to
        #   fill_value, and for nans=True nans are skipped entirely.
        if scalar or reverse:
            # scalar/reverse argreductions are not meaningfully supported;
            # keep the generic (lazy) path for unchanged behaviour.
            return AggregateOp.callable(nans=nans, reverse=reverse, scalar=scalar)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            size = len(ret)
            for i in range(len(group_idx)):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                cmp_val = a[i]
                if cmp_val != cmp_val:
                    if nans:
                        continue
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = fill_value
                    continue
                if counter[ri]:
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = i
                elif complex_less(mean[ri], cmp_val):
                    mean[ri] = cmp_val
                    ret[ri] = i

        return loop


class ArgMin(ArgMax):
    @classmethod
    def callable(cls, nans=False, reverse=False, scalar=False):
        # inlined on purpose, see ArgMax.callable
        if scalar or reverse:
            return AggregateOp.callable(nans=nans, reverse=reverse, scalar=scalar)

        @njit_stable
        def loop(group_idx, a, ret, counter, mean, outer, fill_value, ddof):
            size = len(ret)
            for i in range(len(group_idx)):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                cmp_val = a[i]
                if cmp_val != cmp_val:
                    if nans:
                        continue
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = fill_value
                    continue
                if counter[ri]:
                    counter[ri] = 0
                    mean[ri] = cmp_val
                    ret[ri] = i
                elif complex_less(cmp_val, mean[ri]):
                    mean[ri] = cmp_val
                    ret[ri] = i

        return loop
