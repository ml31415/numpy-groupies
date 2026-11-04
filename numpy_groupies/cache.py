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
``purge_cache()`` removes it - and the leftovers of the versions that cached next to
their source.

The module imports nothing but the standard library: the package has to be
importable without numpy (the pure python implementation needs no numpy), and
the cache of the numba backend must be reachable from there.
"""

import os
import shutil
import sys
from pathlib import Path


def numba_cache_dir() -> Path:
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
    return Path(root) / "numpy_groupies"


def _user_cache_dir() -> Path:
    """The user-wide cache directory, the way numba and pip place it.

    ``$XDG_CACHE_HOME`` (or ``~/.cache``) on Linux, ``~/Library/Caches`` on
    macOS and ``%LOCALAPPDATA%`` on Windows - with the name of numba in it,
    the way the cache locator of numba places its own.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "numba"


def _count_files(directory: Path) -> int:
    """The number of the files below a directory."""
    return sum(1 for path in directory.rglob("*") if path.is_file())


def package_dir() -> Path:
    """The directory of the package - its source tree with the caches in it.

    The leftovers of the versions that cached next to their source are in the
    ``__pycache__`` directories of the whole package, and the caller of
    ``purge_cache()`` has to know where the package sits.

    The modules of the package live next to each other, so the directory of
    this file is the directory of the package.
    """
    return Path(__file__).parent


def purge_cache() -> int:
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
    if cache.is_dir():
        before = _count_files(cache)
        shutil.rmtree(cache, ignore_errors=True)
        # the files the removal did not reach are not counted: the counter is
        # the number of the files that are really gone
        removed += before - _count_files(cache)
    for path in package_dir().rglob("*"):
        if not path.name.endswith((".nbi", ".nbc", ".nbi.lock")):
            continue
        try:
            path.unlink()
        except OSError:  # pragma: no cover - the file is gone
            continue
        removed += 1
    return removed
