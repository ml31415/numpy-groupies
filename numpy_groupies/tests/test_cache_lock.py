"""The on-disk numba cache is shared by the workers of a parallel test run.

Numba writes the index of a cache file before the data file it announces, so
concurrent writers of one cache can leave an index naming the machine code of
another signature.  ``_ProcessSafeCacheFile`` excludes those writers from each
other, which is only observable under contention - so this is the one place
the suite writes one cache from several flows of control at once.  Threads
rather than processes: ``flock`` excludes separate open file descriptions,
which is what a writer holds, and the interpreter of a pytest worker is not a
safe thing to fork.
"""

import os
from threading import Thread

from numpy_groupies.aggregate_numba.jit import _ProcessSafeCacheFile

_workers = min(4, os.cpu_count())
_per_worker = 25


def _write(worker, count, cache_path):
    """Save ``count`` distinct signatures of one cache file, from one flow."""
    cache_file = _ProcessSafeCacheFile(
        cache_path=cache_path,
        filename_base="probe",
        source_stamp="stamp",
    )
    for index in range(count):
        cache_file.save(f"key-{worker}-{index}", (worker, index))


def test_every_key_of_a_contended_cache_names_its_own_machine(tmp_path):
    """No key of a raced cache may name another writer's machine code."""
    cache_path = str(tmp_path)
    threads = [Thread(target=_write, args=(worker, _per_worker, cache_path)) for worker in range(_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    cache_file = _ProcessSafeCacheFile(cache_path=cache_path, filename_base="probe", source_stamp="stamp")
    assert len(cache_file._load_index()) == _workers * _per_worker
    for worker in range(_workers):
        for index in range(_per_worker):
            assert cache_file.load(f"key-{worker}-{index}") == (worker, index)
