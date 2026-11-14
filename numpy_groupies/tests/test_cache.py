"""The cache of the compiled kernels of the numba backend.

The cache lives in a directory of its own - the one ``NUMBA_CACHE_DIR``
names, or the user-wide cache directory of numba otherwise - and not next to
the source of the kernels: the cache of numba is never cleaned up, and the one
of an installed package would sit in the installation directory, where it
survives the uninstallation and leaves a broken installation behind
(numba #9312).

``purge_cache()`` removes both that directory and the leftovers of the versions that
cached next to their source.

The module of the cache imports nothing but the standard library, and the root
imports it on the module level: the sweep also answers where the backend whose
files it cleans up after is gone - which is when it is asked for most.

Every test that asks the sweep to remove something asks it of directories of
its own under ``tmp_path``, which :func:`numpy_groupies.cache.purge_cache` takes
as arguments: a sweep of the real ones would take the compiles of a run in
flight away from under it - the cache directory is the one every worker of a
parallel run writes into, and the tree of the package holds the files of that
run.
"""

import os
import subprocess
import sys
from collections.abc import Sequence
from inspect import signature
from pathlib import Path

from numba.core.caching import NullCache

import numpy_groupies
from numpy_groupies.cache import numba_cache_dir, package_dir, purge_cache
from numpy_groupies.tests.test_numba import _walk_dispatchers
from numpy_groupies.tests.test_purepy import _BLOCKER

PACKAGE = package_dir()


def _source_tree(root: Path, leftovers: Sequence[str] = ()) -> Path:
    """A package tree of a test, with leftovers of the versions that cached
    next to their source in its ``__pycache__``.

    The tree a sweep is asked of through the ``source_tree`` argument - the tree
    of the package under test holds the files of the run itself.
    """
    tree = root / "numpy_groupies"
    cache = tree / "__pycache__"
    cache.mkdir(parents=True)
    for name in leftovers:
        (cache / name).write_text("leftover")
    return tree


def test_cache_directory_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    assert numba_cache_dir() == tmp_path / "numpy_groupies"


def test_cache_directory_is_user_wide_without_environment(monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    path = numba_cache_dir()
    # the directory of the package under the cache directory of the user
    assert path.parts[-2:] == ("numba", "numpy_groupies")
    assert path.is_absolute()


def test_purge_cache_removes_cache_and_leftovers_in_source_tree(tmp_path):
    """The sweep removes the cache directory and the leftovers of a tree.

    Both parts in one test: they are the one call of the sweep.  Neither of the
    two directories it removes is one of the package - the cache directory is
    the one the workers of a run write their compiles into, and the tree of the
    package holds the files of that run - so the call is asked of two
    directories under ``tmp_path``.
    """
    # the cache of the package
    cache = tmp_path / "caches" / "numpy_groupies"
    cache.mkdir(parents=True)
    (cache / "a.nbi").write_text("index")
    (cache / "a.1.nbc").write_text("code")
    # a __pycache__ directory of the tree, as the versions that cached next to
    # their source left it
    tree = _source_tree(tmp_path / "tree")
    leftovers = tree / "__pycache__"
    (leftovers / "old.nbi").write_text("index")
    # the data files carry the number before the extension
    (leftovers / "old.1.nbc").write_text("code")
    (leftovers / "old.nbi.lock").write_text("lock")
    # the byte code of the package is left alone
    (leftovers / "keep.pyc").write_bytes(b"\x00")
    # and one in a subpackage: the leftovers are in the whole tree, not only
    # next to the modules of the top level
    sub = tree / "npg_numpy" / "__pycache__"
    sub.mkdir(parents=True)
    (sub / "old.nbi").write_text("index")

    assert purge_cache(cache_dir=cache, source_tree=tree) == 6
    assert not cache.exists()
    assert not (leftovers / "old.nbi").exists()
    assert not (leftovers / "old.1.nbc").exists()
    assert not (leftovers / "old.nbi.lock").exists()
    assert not (sub / "old.nbi").exists()
    assert (leftovers / "keep.pyc").read_bytes() == b"\x00"


def test_package_hands_out_purge_cache():
    assert numpy_groupies.purge_cache is purge_cache
    # the two directories the sweep removes are the ones of the package, unless
    # the caller names others
    parameters = signature(purge_cache).parameters
    assert parameters["cache_dir"].default is None
    assert parameters["source_tree"].default is None


def test_sweep_answers_where_backend_is_gone(tmp_path):
    """The sweep works without the backend whose files it sweeps.

    A caller may just have uninstalled numba - its cache and the leftovers of
    the versions that cached next to their source are still there.  The module
    of the cache needs nothing but the standard library, so the sweep runs and
    removes them even then: blocked numpy brings no numba with it, and the
    backend of the package is gone - the cache is not.  Both directories the
    child removes are of this test, which it names in the call itself.
    """
    (tmp_path / "sitecustomize.py").write_text(_BLOCKER)
    cache = tmp_path / "caches" / "numpy_groupies"
    cache.mkdir(parents=True)
    (cache / "a.nbi").write_text("index")
    tree = _source_tree(tmp_path / "tree", ("old.nbi",))
    script = (
        "from pathlib import Path\n"
        "import numpy_groupies as npg\n"
        "assert npg.npg_numba is None, 'the backend should not be there'\n"
        f"assert npg.purge_cache(cache_dir=Path({str(cache)!r}), source_tree=Path({str(tree)!r})) == 2\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PACKAGE.parent),
        env={**os.environ, "PYTHONPATH": str(tmp_path), "NUMBA_CACHE_DIR": str(tmp_path / "caches")},
    )
    assert result.returncode == 0, result.stderr
    assert not cache.exists(), "the cache is not removed"


def test_every_dispatcher_caches_in_directory_of_package():
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


def test_cache_of_kernel_lands_in_directory_and_is_used(tmp_path):
    """A kernel compiles into the directory, and the next process finds it there."""
    script = (
        "import numpy as np\n"
        "from numpy_groupies.npg_numba import aggregate\n"
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
    assert list(cache.glob("*.nbi")), "the cache of the kernel is not in the directory"
    assert list(cache.glob("*.nbc")), "the cache of the kernel is not in the directory"
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    run()
    after = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    # the second run hits the cache: no file is added and none is written
    # anew
    assert after == before
