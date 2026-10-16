"""Entry point of ``python -m numpy_groupies.benchmarks``."""

import signal

from . import main

if hasattr(signal, "SIGPIPE"):
    # a reader that went away (``| head``) ends the run quietly, as it does for
    # any other command line filter; left alone, python prints a traceback
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

try:
    main()
except BrokenPipeError:
    pass
