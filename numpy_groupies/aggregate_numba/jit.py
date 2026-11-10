"""JIT plumbing: an njit wrapper with a deterministic on-disk cache identity.

Numba draws a dispatcher's uuid lazily from ``uuid.uuid4()`` and embeds it in
the cloudpickle representation of the dispatcher, so closures over dispatchers
hash to new bytes in every interpreter process and the on-disk cache of the
enclosing function never hits (numba #6522).  Every jitted callable of this
package therefore goes through ``njit_stable``.
"""

import contextlib
import hashlib
import inspect
import os
from collections.abc import Callable, Generator
from typing import Any, TypeVar, overload

import numba as nb
from numba.core import caching

from numpy_groupies.cache import numba_cache_dir

# The overload implementations of the package, keyed by the function they
# overload.  Numba compiles the implementation into the caller - the body of
# the implementation is what ends up in the machine code - and the source of
# the caller says nothing about it.  The identity of a dispatcher (and with it
# the key of its on-disk cache) would therefore not change when only the
# implementation changes, and the stale artifact would be served: a
# correctly typed, silently wrong kernel.  The registry lets the fingerprint
# of a dispatcher cover the implementations it may reach.
_OVERLOADS: dict[Any, Any] = {}

# the decorated function is a dispatcher that keeps the signature of its
# python function - the overloads below hand that signature through
# ``njit_stable`` for the type checker, the way ``nb.njit`` does
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])

# windows has no flock - the name is an Optional so the no-op fallback below is
# an honest annotation and not a type error
flock: Callable[[int, int], None] | None

try:
    from fcntl import LOCK_EX, LOCK_SH, LOCK_UN, flock
except ImportError:  # pragma: no cover - windows
    flock = None

# Cache revision of this package - folded into the on-disk cache file names
# below.  Numba keys its cache files by module and qualname of the compiled
# function and pickles the index payload (which embeds the compiled-over
# signatures and their types) BEFORE checking the source stamp - renaming a
# class that appears in a signature therefore crashes the cache read with an
# AttributeError on the dangling global.  Bumping the revision renames every
# cache file of this package, orphaning the stale indexes (they keep lying
# around, numba never reads them again).
_cache_revision = "10"


@contextlib.contextmanager
def _advisory_lock(path: str, shared: bool) -> Generator[None, None, None]:
    """An advisory whole-file lock on *path*, held for the duration of the block.

    Without ``fcntl`` (windows) this is a no-op: the platform has no ``flock``
    and numba's own cache is not safe there either.
    """
    if flock is None:  # pragma: no cover - windows
        yield
        return
    # ``a+`` creates the lock file without truncating an existing one
    with open(path, "a+") as handle:
        flock(handle.fileno(), LOCK_SH if shared else LOCK_EX)
        try:
            yield
        finally:
            flock(handle.fileno(), LOCK_UN)


class _ProcessSafeCacheFile(caching.IndexDataCacheFile):
    """Numba's index-and-data cache file with the writers of one machine excluded.

    ``IndexDataCacheFile.save`` reloads the index, takes the first data file
    number the reloaded index does not mention, writes the index announcing
    that number and only then writes the data file it announces.  Two
    processes that miss the same signature therefore both pick number 1, both
    write an index mapping their own key onto it, and both write their machine
    code into ``.1.nbc``: whichever index write came last names a key whose
    data file now holds the *other* process' machine.  Loading that is a
    correctly typed, silently wrong kernel - it showed up as one or two
    mismatching cells in the numba column of the test suite, and as an
    exception when the machine did not fit the arguments at all.

    The lock covers the whole of ``save`` so index and data file are written
    as one transaction, and ``load`` shared so no writer can recycle a data
    file number (a source edit empties the index and makes every number look
    free) while a reader is still reading it.
    """

    def _lock_path(self) -> str:
        return os.path.join(self._cache_path, f"{self._index_name}.lock")

    def save(self, key: Any, data: Any) -> None:
        # The directory of the cache may be gone: ``purge_cache()`` removes it, and
        # a process that compiles while the purge runs would write into a
        # directory that is not there any more.
        os.makedirs(self._cache_path, exist_ok=True)
        with _advisory_lock(self._lock_path(), shared=False):
            # the caching of numba is not annotated, so the checker knows no
            # type of the answer of the superclass - it saves and returns
            # nothing, and the call is made for its effect
            super().save(key, data)

    def load(self, key: Any) -> Any:
        with _advisory_lock(self._lock_path(), shared=True):
            return super().load(key)

    def flush(self) -> None:
        with _advisory_lock(self._lock_path(), shared=False):
            # the caching of numba is not annotated - see the save above
            super().flush()


def register_overload(func: Any) -> Callable[[Any], Any]:
    """``nb.extending.overload``, with the implementation remembered.

    The registration itself is numba's - the decorator returns the
    implementation, and numba binds the overloaded function to a template
    that carries it.  The package remembers the implementation as well, so
    that the identity of the dispatchers that may reach it covers its body:
    without that, a change of the implementation alone leaves the on-disk
    cache key of every dispatcher that uses it untouched, and the stale
    artifact is served (see :data:`_OVERLOADS`).
    """

    def register(implementation: Any) -> Any:
        _OVERLOADS[func] = implementation
        return nb.extending.overload(func)(implementation)

    return register


def _stable_content(content: Any) -> str:
    """A description of a closure cell that is the same in every process.

    The fingerprint of a dispatcher covers the contents of its closure, and
    the fingerprint decides the name of the on-disk cache file.  ``repr`` of a
    python function carries its memory address, which differs in every
    process: the fingerprint would differ, the cache file would get a new
    name in every run and the cache would never hit - the dispatcher would
    be compiled again and again, in every process, at the cost of the time
    and the memory of a compilation.

    A dispatcher contributes its pinned uuid, a python function its module,
    its qualified name and the hash of its source (a change of the body must
    invalidate the cache), and everything else its repr.
    """

    uuid = getattr(content, "_uuid", None)
    if isinstance(uuid, str):
        return uuid
    module = getattr(content, "__module__", None)
    qualname = getattr(content, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        try:
            source = inspect.getsource(content)
        except (OSError, TypeError):  # pragma: no cover - a source that cannot be read
            source = ""
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        return f"{module}.{qualname}:{digest}"
    return repr(content)


def _overload_sources(func: Callable[..., Any]) -> list[str]:
    """The sources of the overload implementations the function may reach.

    A jitted function reaches an overloaded function through its global name
    or through a closure cell - and the implementation of the overload is
    what numba compiles into it.  The closure is walked one level down into
    the python function of a dispatcher, since that is where the name of the
    overloaded function is looked up.

    An implementation of its own may reach an overloaded function as well -
    numba inlines that one into the very same machine code, so the walk
    follows the implementations it finds (the accumulate of the dispersion
    tracks the extrema of the group through an overload of its own).  The
    walk stops at every function it has already seen, so a cycle cannot
    spin it forever.
    """
    found: dict[str, str] = {}
    pending = [func]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        # The names the one at hand calls through its globals - and the ones
        # the functions inside it call: the branches of an overload template,
        # the bodies an implementation picks between, sit in the constants of
        # the enclosing function, one code object per def.  A name is looked
        # up in the globals of the enclosing one, which is where numba looks
        # when it compiles the body.
        names: set[str] = set()
        pending_codes = [current.__code__]
        seen_codes: set[int] = set()
        while pending_codes:
            code = pending_codes.pop()
            if id(code) in seen_codes:
                continue
            seen_codes.add(id(code))
            names.update(code.co_names)
            for const in code.co_consts:
                inner = getattr(const, "__code__", None)
                if inner is None and isinstance(const, type(code)):
                    inner = const
                if inner is not None:
                    pending_codes.append(inner)
        for cell in current.__closure__ or ():
            try:
                content = cell.cell_contents
            except ValueError:  # pragma: no cover - empty cell
                continue
            # a dispatcher in the closure contributes the name of its python
            # function - the overload it reaches is looked up in the globals
            # of that function
            py_func = getattr(content, "py_func", None)
            if py_func is not None:
                pending.append(py_func)
            name = getattr(content, "__name__", None)
            if isinstance(name, str):
                names.add(name)
        globals_ = getattr(current, "__globals__", {})
        for name in names:
            stub = globals_.get(name)
            implementation = _OVERLOADS.get(stub) if stub is not None else None
            if implementation is not None:
                try:
                    found[name] = inspect.getsource(implementation)
                except (OSError, TypeError):  # pragma: no cover - a source that cannot be read
                    found[name] = ""
                # the body of the implementation may reach an overload of its
                # own - numba inlines it, so it belongs to the identity too
                pending.append(implementation)
    return [found[name] for name in sorted(found)]


@overload
def njit_stable(py_func: _FuncT, cache: bool = True, name: str | None = None, **options: Any) -> _FuncT: ...
@overload
def njit_stable(
    py_func: None = None, cache: bool = True, name: str | None = None, **options: Any
) -> Callable[[_FuncT], _FuncT]: ...
def njit_stable(
    py_func: Callable[..., Any] | None = None,
    cache: bool = True,
    name: str | None = None,
    **options: Any,
) -> Any:
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
    operations receive distinct identities - a block that is one and the same
    function in every operation shares one identity instead, which is the
    point of the unnamed no-ops of ``compose.build_kernel``.  Usable as
    ``njit_stable(func, **options)`` or ``@njit_stable`` /
    ``@njit_stable(**options)``.  Disk caching is enabled by default
    (``cache=True``), which is the main point of this wrapper.

    A ``name`` keyword is not passed on to numba: it becomes the qualname of
    the function, before it is fingerprinted and before numba mangles its
    symbol.  Numba mangles that symbol from module, qualname and argument
    signature, so two functions that differ only in their closure share a
    symbol - and a dispatcher loaded from the on-disk cache then binds its
    calls into the environment of the other function, which is correct types
    and wrong results.  Blocks that differ from operation to operation
    therefore pass the name of their operation; ``compose.build_kernel`` tells
    the case in full.
    """

    def wrap(func: Callable[..., Any]) -> Any:
        if name is not None:
            # both the fingerprint and the mangled symbol read the qualname,
            # so the rename has to happen before nb.njit is reached
            func.__qualname__ = func.__name__ = name
        disp: Any = nb.njit(func, cache=cache, **options)
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
                    cells.append(_stable_content(content))
            fingerprint = "\n".join(
                [
                    func.__module__,
                    func.__qualname__,
                    inspect.getsource(func),
                    repr(sorted(options.items())),
                    repr(cells),
                    *_overload_sources(func),
                ]
            )
            uuid = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
            disp._set_uuid(uuid)
            # move the on-disk cache of this dispatcher to revisioned file
            # names - the numba defaults key by module and qualname only, so
            # refactoring signature-relevant globals would crash the read of
            # the stale index (see the comment on _cache_revision).  The uuid
            # additionally separates same-named functions whose sources
            # changed: numba's own index key only covers the bytecode and the
            # closure bytes, and module-level functions have no closure - two
            # versions of one function differing only in a constant (which
            # lives in co_consts, not co_code) would otherwise share an index
            # and could load each other's compiled artifacts.
            cache_impl = disp._cache._impl
            cache_impl._filename_base = f"{cache_impl._filename_base}-r{_cache_revision}-{uuid}"
            # The cache of the kernel does not live next to its source but in
            # a directory of the package: the cache of numba is never cleaned
            # up, and the one of an installed package sits in the installation
            # directory - where it survives the uninstallation and leaves a
            # broken installation behind (numba #9312).
            cache_path = numba_cache_dir()
            cache_path.mkdir(parents=True, exist_ok=True)
            disp._cache._cache_path = cache_path
            disp._cache._cache_file = _ProcessSafeCacheFile(
                cache_path=cache_path,
                filename_base=cache_impl.filename_base,
                source_stamp=cache_impl.locator.get_source_stamp(),
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
