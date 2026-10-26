"""The jitted callers of the numba kernels, shared by the kernel tests.

A caller that reaches a kernel from jitted code is a dispatcher of its own,
and a dispatcher compiles the first time it is called.  Built per test, every
test would compile its own copy of the very same wrapper - the most expensive
thing the test suite does, and the reason the kernel tests took minutes
although they hold a few thousand assertions.

The wrappers are therefore built once per kernel and shared by the tests.

Every wrapper carries the name of the kernel it calls.  Two wrappers that
differ only in their closure share the symbol numba mangles their code into,
and one of them may bind the calls of the other - the same types and the
wrong kernel.  The name keeps the symbols of the wrappers apart.
"""

from typing import Any

import numba as nb


def jitted_caller(kernel: Any, name: str) -> Any:
    """A jitted caller of one kernel, with the name the kernel has."""

    def caller(group_idx, values, size):
        return kernel(group_idx, values, size=size, fill_value=None)

    # the name is set before numba sees the function: it is the name the
    # symbol of the compiled code is built from
    caller.__qualname__ = caller.__name__ = name
    return nb.njit(caller)


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

    caller.__qualname__ = caller.__name__ = name
    return nb.njit(caller)
