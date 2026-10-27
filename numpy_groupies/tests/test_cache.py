"""The cache of the compiled kernels of the numba backend.

The cache lives in a directory of its own - the one ``NUMBA_CACHE_DIR``
names, or the user-wide cache directory of numba otherwise - and not next to
the source of the kernels: the cache of numba is never cleaned up, and the one
of an installed package would sit in the installation directory, where it
survives the uninstallation and leaves a broken installation behind
(numba #9312).

``purge_cache()`` removes both that directory and the leftovers of the versions that
cached next to their source.
"""

import os
import pathlib
import subprocess
import sys

from numba.core.caching import NullCache

import numpy_groupies
from numpy_groupies.tests.test_numba import _walk_dispatchers
from numpy_groupies.utils_cache import numba_cache_dir, purge_cache

PACKAGE = pathlib.Path(numpy_groupies.__file__).parent


def test_the_cache_directory_is_the_one_of_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    assert numba_cache_dir() == str(tmp_path / "numpy_groupies")


def test_the_cache_directory_is_the_user_wide_one_without_the_environment(monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    path = numba_cache_dir()
    # the directory of the package under the cache directory of the user
    assert path.endswith(os.path.join("numba", "numpy_groupies"))
    assert os.path.isabs(path)


def test_purge_cache_removes_the_cache_and_the_leftovers_in_the_source_tree(monkeypatch, tmp_path):
    # both parts in one test: they touch the same files, and two tests
    # that do so get in each other's way under xdist
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    # the cache of the package
    cache = pathlib.Path(numba_cache_dir())
    cache.mkdir(parents=True)
    (cache / "a.nbi").write_text("index")
    (cache / "a.1.nbc").write_text("code")
    # a __pycache__ directory of the package, as the versions that
    # cached next to their source left it
    tree = PACKAGE / "__pycache__"
    tree.mkdir(exist_ok=True)
    (tree / "old.nbi").write_text("index")
    # the data files carry the number before the extension
    (tree / "old.1.nbc").write_text("code")
    (tree / "old.nbi.lock").write_text("lock")
    # the byte code of the package is left alone
    bytecode = tree / "keep.pyc"
    bytecode.write_bytes(b"\x00")
    try:
        removed = purge_cache()
        assert removed >= 5
        assert not cache.exists()
        assert not (tree / "old.nbi").exists()
        assert not (tree / "old.1.nbc").exists()
        assert not (tree / "old.nbi.lock").exists()
        assert bytecode.exists()
    finally:
        bytecode.unlink(missing_ok=True)
        for name in ("old.nbi", "old.1.nbc", "old.nbi.lock"):
            (tree / name).unlink(missing_ok=True)


def test_the_package_hands_out_the_purge_cache():
    assert numpy_groupies.purge_cache is purge_cache


def test_every_dispatcher_caches_in_the_directory_of_the_package():
    """The kernel of every operation caches in the directory of the package.

    The cache of numba lives next to the source of the kernel - in the
    installation directory of an installed package, where it survives the
    uninstallation and leaves a broken installation behind (numba #9312).
    """
    wanted = numba_cache_dir()
    checked = 0
    for dispatcher in _walk_dispatchers():
        cache = dispatcher._cache
        if isinstance(cache, NullCache):
            # the cache is switched off - there is nothing to check then
            continue
        assert cache._cache_path == wanted
        assert cache._cache_file._cache_path == wanted
        checked += 1
    assert checked > 0


def test_the_cache_of_a_kernel_lands_in_the_directory_and_is_used(tmp_path):
    """A kernel compiles into the directory, and the next process finds it there."""
    script = (
        "import numpy as np\n"
        "from numpy_groupies.aggregate_numba import aggregate\n"
        "aggregate(np.arange(1000), np.random.random(1000), func='sum')\n"
    )
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}

    def run() -> None:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PACKAGE.parent),
            env=env,
        )
        assert result.returncode == 0, result.stderr

    run()
    cache = tmp_path / "numpy_groupies"
    assert list(cache.glob("*.nbi")), "der Cache des Kernels liegt nicht im Verzeichnis"
    assert list(cache.glob("*.nbc")), "der Cache des Kernels liegt nicht im Verzeichnis"
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    run()
    after = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    # the second run hits the cache: no file is added and none is written
    # anew
    assert after == before
