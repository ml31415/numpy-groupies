"""The cache of the compiled kernels of the numba backend.

Numba's own cache is never cleaned up and lives next to the source - in an
installation directory, where it survives uninstallation and leaves a broken
installation behind (numba #9312). This package therefore caches in a
directory of its own (``NUMBA_CACHE_DIR``, else numba's user-wide cache
directory); ``purge_cache()`` removes it and the next-to-source leftovers.

The module imports nothing but the standard library: the package has to be
importable without numpy.
"""

import os
import shutil
import sys
from pathlib import Path


def numba_cache_dir() -> Path:
    """The directory the cached compiles of this package live in:
    ``NUMBA_CACHE_DIR`` if set, numba's user-wide cache directory otherwise,
    with a ``numpy_groupies`` subdirectory. Must be set before the import of
    the package, when the kernels are compiled.
    """
    root = os.environ.get("NUMBA_CACHE_DIR") or _user_cache_dir()
    return Path(root) / "numpy_groupies"


def _user_cache_dir() -> Path:
    """The user-wide cache directory, the way numba and pip place it:
    ``$XDG_CACHE_HOME``/``~/.cache`` on Linux, ``~/Library/Caches`` on macOS,
    ``%LOCALAPPDATA%`` on Windows - plus ``numba``.
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
    """The directory of the package - its source tree, which holds the
    ``__pycache__`` leftovers of versions that cached next to their source.
    """
    return Path(__file__).parent


def purge_cache() -> int:
    """Remove the cached compiles of this package - the package cache directory
    and the ``.nbi``/``.nbc`` leftovers in the source tree - leaving ``.pyc``
    files and other packages' caches alone. Returns the number of files
    removed.
    """
    removed = 0
    cache = numba_cache_dir()
    if cache.is_dir():
        before = _count_files(cache)
        shutil.rmtree(cache, ignore_errors=True)
        # count only the files that are really gone
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
