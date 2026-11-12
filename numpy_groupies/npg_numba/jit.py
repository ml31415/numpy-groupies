"""JIT plumbing: an njit wrapper with a deterministic on-disk cache identity.

Numba draws a dispatcher's uuid lazily from ``uuid.uuid4()`` and embeds it in its
cloudpickle representation, so closures over dispatchers hash to new bytes in
every process and the on-disk cache of the enclosing function never hits
(numba #6522).  Every jitted callable of this package goes through ``njit_stable``.
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

# Overload implementations, keyed by the function they overload.  Numba compiles
# the implementation body into the caller, so a dispatcher's identity (and cache
# key) would not change when only the implementation does - the stale, correctly
# typed but wrong kernel would be served.  The registry lets the fingerprint of
# a dispatcher cover the implementations it may reach.
_OVERLOADS: dict[Any, Any] = {}

# the decorated function is a dispatcher keeping the python signature - the
# overloads below hand it through ``njit_stable``, like ``nb.njit`` does
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])

# windows has no flock - Optional so the no-op fallback below is an honest annotation
flock: Callable[[int, int], None] | None

try:
    from fcntl import LOCK_EX, LOCK_SH, LOCK_UN, flock
except ImportError:  # pragma: no cover - windows
    flock = None

# Cache revision, folded into the on-disk cache file names below.  Numba pickles
# the index payload (embedding the compiled-over types) BEFORE checking the
# source stamp, so renaming a class in a signature crashes the cache read on the
# dangling global - bumping the revision renames every cache file, orphaning the
# stale indexes (numba never reads them again).
_cache_revision = "10"


@contextlib.contextmanager
def _advisory_lock(path: str, shared: bool) -> Generator[None, None, None]:
    """An advisory whole-file lock on *path*; a no-op without ``fcntl`` (windows)."""
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

    ``IndexDataCacheFile.save`` writes the index announcing a data file number
    and only then the data file, so two processes missing the same signature
    both pick number 1, and whichever index write came last names a key whose
    ``.1.nbc`` holds the *other* process' machine - a correctly typed, silently
    wrong kernel (it showed up as one or two mismatching cells in the numba
    column of the test suite).  The lock covers all of ``save`` so index and
    data are written as one transaction, and ``load`` shared so no writer
    recycles a data file number (a source edit empties the index and makes
    every number look free) while a reader is still reading it.
    """

    def _lock_path(self) -> str:
        return os.path.join(self._cache_path, f"{self._index_name}.lock")

    def save(self, key: Any, data: Any) -> None:
        # the cache dir may be gone: ``purge_cache()`` removes it while another
        # process is still compiling into it
        os.makedirs(self._cache_path, exist_ok=True)
        with _advisory_lock(self._lock_path(), shared=False):
            # numba's caching is not annotated: the call is made for its effect
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

    The registration itself is numba's; the package keeps the implementation so
    the identity of the dispatchers that may reach it covers its body - without
    that, a change of the implementation alone leaves the cache key untouched
    and the stale artifact is served (see :data:`_OVERLOADS`).
    """

    def register(implementation: Any) -> Any:
        _OVERLOADS[func] = implementation
        return nb.extending.overload(func)(implementation)

    return register


def _stable_content(content: Any) -> str:
    """A description of a closure cell that is the same in every process.

    ``repr`` of a python function carries its memory address, so the fingerprint
    would differ per process, the cache file would get a new name in every run
    and never hit.  A dispatcher contributes its pinned uuid, a python function
    its module, qualname and source hash (a body change must invalidate), and
    everything else its repr.
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

    Reached through global names or closure cells (walked one level down into a
    dispatcher's python function, where the names are looked up), and followed
    into the implementations themselves - numba inlines them into the same
    machine code (the dispersion accumulate reaches an overload of its own).
    Functions already seen stop the walk, so a cycle cannot spin it forever.
    """
    found: dict[str, str] = {}
    pending = [func]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        # the names the current function calls through its globals, including
        # the nested code objects (one per def, in co_consts) - a name is looked
        # up in the globals of the enclosing function, which is where numba looks
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
            # a dispatcher contributes its python function's name - the overload
            # it reaches is looked up in that function's globals
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
                # the body may reach an overload of its own - numba inlines it,
                # so it belongs to the identity too
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

    A pinned content-derived uuid keeps closures over dispatchers hashing to the
    same bytes in every process, so the on-disk cache of the enclosing function
    hits instead of growing with duplicate entries (numba #6522; measured against
    numba 0.68: a cached caller's index survives unpinned closures, but the
    dispatcher-level keys do not - direct driver calls recompile in every
    process).  The fingerprint covers module, qualname, source, options and
    closure contents, so dispatchers from the same factory lines for different
    operations get distinct identities, while the unnamed no-ops of
    ``compose.build_kernel`` share one.  Usable as ``njit_stable(func,
    **options)`` or ``@njit_stable`` / ``@njit_stable(**options)``; disk caching
    on by default (``cache=True``), which is the point of this wrapper.

    A ``name`` keyword is not passed on to numba - it becomes the qualname,
    before fingerprinting and before numba mangles the symbol from module,
    qualname and signature.  Two functions that differ only in their closure
    share a symbol, and a dispatcher loaded from the on-disk cache would then
    bind its calls into the environment of the other - correct types, wrong
    results.  Blocks that differ per operation pass their name; see
    ``compose.build_kernel``.
    """

    def wrap(func: Callable[..., Any]) -> Any:
        if name is not None:
            # both fingerprint and mangled symbol read the qualname - the rename
            # has to happen before nb.njit is reached
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
                    # dispatchers were wrapped earlier and carry their pinned
                    # uuid, closures are created in order
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
            # revisioned cache file names: the numba defaults key by module and
            # qualname only, so refactoring signature-relevant globals would
            # crash the read of the stale index (see _cache_revision).  The uuid
            # also separates same-named functions whose sources changed - the
            # index key only covers the bytecode and closure bytes, and two
            # versions differing only in a constant (which lives in co_consts,
            # not co_code) would otherwise share an index and load each other's
            # compiled artifacts.
            cache_impl = disp._cache._impl
            cache_impl._filename_base = f"{cache_impl._filename_base}-r{_cache_revision}-{uuid}"
            # the cache lives in a package directory, not next to the source:
            # numba never cleans its cache up, and beside an installed package
            # it would survive uninstallation and break it (numba #9312).
            cache_path = numba_cache_dir()
            cache_path.mkdir(parents=True, exist_ok=True)
            disp._cache._cache_path = cache_path
            disp._cache._cache_file = _ProcessSafeCacheFile(
                cache_path=cache_path,
                filename_base=cache_impl.filename_base,
                source_stamp=cache_impl.locator.get_source_stamp(),
            )
        except Exception:  # noqa: BLE001 - any numba internals change must degrade to no caching
            # without a stable uuid the closure hash differs in every process,
            # the cache could never hit and would grow unboundedly (numba #6522)
            disp._cache = caching.NullCache()
        return disp

    if py_func is None:
        return wrap
    return wrap(py_func)
