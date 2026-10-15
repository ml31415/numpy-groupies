"""JIT plumbing: an njit wrapper with a deterministic on-disk cache identity.

Numba draws a dispatcher's uuid lazily from ``uuid.uuid4()`` and embeds it in
the cloudpickle representation of the dispatcher, so closures over dispatchers
hash to new bytes in every interpreter process and the on-disk cache of the
enclosing function never hits (numba #6522).  Every jitted callable of this
package therefore goes through ``njit_stable``.
"""

import hashlib
import inspect

import numba as nb
from numba.core import caching

# Cache revision of this package - folded into the on-disk cache file names
# below.  Numba keys its cache files by module and qualname of the compiled
# function and pickles the index payload (which embeds the compiled-over
# signatures and their types) BEFORE checking the source stamp - renaming a
# class that appears in a signature therefore crashes the cache read with an
# AttributeError on the dangling global.  Bumping the revision renames every
# cache file of this package, orphaning the stale indexes (they keep lying
# around, numba never reads them again).
_cache_revision = "3"


def njit_stable(py_func=None, cache=True, **options):
    """``nb.njit`` plus a deterministic dispatcher identity.

    Numba draws a dispatcher's uuid lazily from ``uuid.uuid4()`` and embeds
    it in the cloudpickle representation of the dispatcher.  Closures over
    dispatchers therefore hash to new bytes in every interpreter process,
    so numba's on-disk cache key of the enclosing function never hits and
    every run appends a new cache file instead (see numba #6522).  Pinning
    a content-derived uuid keeps the closure hash stable across processes,
    which makes the on-disk cache effective for the factory-nested loops
    of this module.  Measured against numba 0.68: the cache index of a
    cached *caller* now survives unpinned closures, but the dispatcher-level
    keys do not (direct driver calls recompile in every process) and the
    cache directory keeps growing with duplicate entries - the pinning is
    still required everywhere in a closure chain.

    The fingerprint covers module, qualname, source, options and the
    closure contents (dispatchers contribute their already pinned uuid),
    so dispatchers compiled from the same factory lines for different
    operations receive distinct identities.  Usable as
    ``njit_stable(func, **options)`` or ``@njit_stable`` /
    ``@njit_stable(**options)``.  Disk caching is enabled by default
    (``cache=True``), which is the main point of this wrapper.
    """

    def wrap(func):
        disp = nb.njit(func, cache=cache, **options)
        try:
            cells = []
            if func.__closure__ is not None:
                for cell in func.__closure__:
                    try:
                        content = cell.cell_contents
                    except ValueError:  # pragma: no cover - empty cell
                        content = "<empty cell>"
                    # dispatchers were wrapped earlier and already carry
                    # their pinned uuid, closures are created in order
                    cells.append(getattr(content, "_uuid", repr(content)))
            fingerprint = "\n".join(
                [
                    func.__module__,
                    func.__qualname__,
                    inspect.getsource(func),
                    repr(sorted(options.items())),
                    repr(cells),
                ]
            )
            disp._set_uuid(hashlib.sha256(fingerprint.encode()).hexdigest()[:16])
            # move the on-disk cache of this dispatcher to revisioned file
            # names - the numba defaults key by module and qualname only, so
            # refactoring signature-relevant globals would crash the read of
            # the stale index (see the comment on _cache_revision)
            impl = disp._cache._impl
            impl._filename_base = f"{impl._filename_base}-r{_cache_revision}"
            disp._cache._cache_file = caching.IndexDataCacheFile(
                cache_path=disp._cache._cache_path,
                filename_base=impl.filename_base,
                source_stamp=impl.locator.get_source_stamp(),
            )
        except Exception:  # noqa: BLE001 - any numba internals change must degrade to no caching
            # numba internals changed: without a stable uuid the closure
            # hash differs in every process, the cache could never hit and
            # would grow without bound - disable caching for this
            # dispatcher instead of polluting the disk (numba #6522)
            disp._cache = caching.NullCache()
        return disp

    if py_func is None:
        return wrap
    return wrap(py_func)
