"""The cache of the compiled kernels of the numba backend.

The compiled kernels are cached on the disk, and the cache of numba has two
properties that the package has to answer for: it is never cleaned up (numba
has no garbage collection in its cache directory - every renamed kernel leaves
its files behind forever), and it lives next to the source of the function it
belongs to - in the installation directory of an installed package, where it
survives the uninstallation and leaves a broken installation behind (numba
#9312).

The cache of this package therefore lives in a directory of its own: the one
``NUMBA_CACHE_DIR`` names, or the user-wide cache directory of numba otherwise.
``purge()`` removes it - and the leftovers of the versions that cached next to
their source.

The module imports nothing but the standard library: the package has to be
importable without numpy (the pure python implementation needs no numpy), and
the cache of the numba backend must be reachable from there.
"""

import os
import shutil
import sys


def numba_cache_dir() -> str:
    """The directory the cached compiles of this package live in.

    ``NUMBA_CACHE_DIR`` if the environment names one, the user-wide cache
    directory of numba (``~/.cache/numba`` on Linux) otherwise - with a
    directory of the package in it, so that the cache of this package can be
    removed without touching the one of another.

    The variable is read at every call, but the cache of a kernel is written
    when the kernel is compiled - which happens at the import of the package:
    the variable has to be set before that.
    """
    root = os.environ.get("NUMBA_CACHE_DIR") or _user_cache_dir()
    return os.path.join(root, "numpy_groupies")


def _user_cache_dir() -> str:
    """The user-wide cache directory, the way numba and pip place it.

    ``$XDG_CACHE_HOME`` (or ``~/.cache``) on Linux, ``~/Library/Caches`` on
    macOS and ``%LOCALAPPDATA%`` on Windows - with the name of numba in it,
    the way the cache locator of numba places its own.
    """
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "numba")


def purge() -> int:
    """Remove the cached compiles of this package.

    Both the cache of the package - the directory ``NUMBA_CACHE_DIR`` names or
    the user-wide cache directory - and the leftovers in the source tree are
    removed: the versions that cached next to their source left ``.nbi`` and
    ``.nbc`` files in the ``__pycache__`` directories of the package, and
    numba never cleans them up.

    The byte code of the package (the ``.pyc`` files) and the caches of the
    other packages are left alone.

    Returns the number of the files removed.
    """
    removed = 0
    cache = numba_cache_dir()
    if os.path.isdir(cache):
        for _root, _dirs, files in os.walk(cache):
            removed += len(files)
        shutil.rmtree(cache, ignore_errors=True)
    package = os.path.dirname(os.path.abspath(__file__))
    for root, _dirs, files in os.walk(package):
        for name in files:
            if name.endswith((".nbi", ".nbc", ".nbi.lock")):
                try:
                    os.remove(os.path.join(root, name))
                except OSError:  # pragma: no cover - the file is gone
                    continue
                removed += 1
    return removed
