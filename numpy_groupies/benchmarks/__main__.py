"""Entry point of ``python -m numpy_groupies.benchmarks``."""

import contextlib
import os

# The resource tracker of multiprocessing complains about the semaphore of
# the thread pool of numba when the process leaves without the cleanup - and
# the interrupt leaves exactly like that.  The tracker is a child process,
# and it reads its warning filters at its start, so the silence has to be
# asked for before anything is imported.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import signal

from numpy_groupies.benchmarks import main

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

    The pool of the threads of numba is closed before that, so that the
    semaphore of it does not leak - the resource tracker of multiprocessing
    would complain about it on the way out.
    """
    with contextlib.suppress(Exception):
        # the pool of the threads is closed before the exit, so that the
        # semaphore of it does not leak - the resource tracker of
        # multiprocessing would complain about it on the way out
        import numba

        numba.close()
    os._exit(130)


signal.signal(signal.SIGINT, _on_interrupt)

try:
    main()
except BrokenPipeError:
    pass
except KeyboardInterrupt:
    # an interrupt that arrives outside of a callback of a compiled kernel
    pass
