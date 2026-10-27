"""The cache of the compiled kernels of the numba backend.

The cache lives in a directory of its own - the one ``NUMBA_CACHE_DIR``
names, or the user-wide cache directory of numba otherwise - and not next to
the source of the kernels: the cache of numba is never cleaned up, and the one
of an installed package would sit in the installation directory, where it
survives the uninstallation and leaves a broken installation behind
(numba #9312).

``purge()`` removes both that directory and the leftovers of the versions that
cached next to their source.
"""

import os
import pathlib

import numpy_groupies
from numpy_groupies.utils_cache import numba_cache_dir, purge

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


def test_purge_removes_the_cache_and_the_leftovers_in_the_source_tree(monkeypatch, tmp_path):
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
        removed = purge()
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


def test_the_package_hands_out_the_purge():
    assert numpy_groupies.purge is purge
