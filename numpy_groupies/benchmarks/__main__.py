"""Entry point of ``python -m numpy_groupies.benchmarks``."""

import contextlib
import os
import signal

from numpy_groupies.benchmarks import _release_numba_resources, main

if hasattr(signal, "SIGPIPE"):
    # a reader that went away (``| head``) ends the run quietly, as it does for
    # any other command line filter; left alone, python prints a traceback
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def _on_interrupt(signum: int, frame: object) -> None:
    """End the run at once, on an interrupt.

    A KeyboardInterrupt raised from the handler would not arrive: the
    interrupt reaches the ctypes callbacks of the compiled kernels (numba,
    numbagg), and ctypes swallows the exception that is raised in one of
    them - the run would go on, and llvmlite would print a complaint about
    it.  Leaving the process is the only way out that works from within
    such a callback, and the exit code says what happened.

    The semaphore of the threading-layer initialization of numba is
    released before that, so that the resource tracker of multiprocessing
    has nothing to complain about on the way out.
    """
    with contextlib.suppress(Exception):
        _release_numba_resources()
    os._exit(130)


signal.signal(signal.SIGINT, _on_interrupt)

try:
    main()
except BrokenPipeError:
    pass
except KeyboardInterrupt:
    # an interrupt that arrives outside of a callback of a compiled kernel
    pass
