"""The jitted callers of the numba kernels, shared by the kernel tests.

A caller that reaches a kernel from jitted code is a dispatcher of its own,
and a dispatcher compiles the first time it is called.  Built per test, every
test would compile its own copy of the very same wrapper - the most expensive
thing the test suite does, and the reason the kernel tests took minutes
although they hold a few thousand assertions.

The wrappers are therefore built once per kernel and shared, and they are
compiled through ``njit_stable``: the name of the kernel keeps the symbols and
the cache files of the wrappers apart, and the pinned uuid keeps their
compiled form valid from one process to the next.  They live in a file of
their own, because numba keys the cache of a function by the file it is
written in - a wrapper in the test module itself would lose its compiled form
with every change of the tests.

Every wrapper calls the kernel the way a caller of the package calls it: with
the labels, the values and the size of the groups, and with a fill value of
``None`` - the default of the kernel itself.
"""

from typing import Any

from numpy_groupies.aggregate_numba.jit import njit_stable


def jitted_caller(kernel: Any, name: str) -> Any:
    """A jitted caller of one kernel, with the name the kernel has."""

    def caller(group_idx, values, size):
        return kernel(group_idx, values, size=size, fill_value=None)

    return njit_stable(caller, name=name)


def jitted_caller_with_extra(kernel: Any, extra: Any, name: str) -> Any:
    """A jitted caller of one kernel, with the argument of its own baked in.

    The kernel of a function that has an argument of its own takes it behind
    the two arrays, and the value of it is a constant of the caller - the
    kernel is called with it, not with a parameter of its own.
    """
    if extra is None:

        def caller(group_idx, a, size, fill_value):
            return kernel(group_idx, a, size=size, fill_value=None)

    else:
        value = extra

        def caller(group_idx, a, size, fill_value):
            return kernel(group_idx, a, value, size=size, fill_value=None)

    return njit_stable(caller, name=name)
