"""Tests for the numba backend: the jitted fast path, the slow-path
fallback, the callable cache, the ``out=`` parameter, the lock around the
shared on-disk compilation cache and the pinned identity that keeps that
cache from growing from run to run.
"""

import importlib
import inspect
import os
import pkgutil
import re
import subprocess
import sys
from threading import Thread
from types import ModuleType

import numba as nb
import numpy as np
import pytest
from numba.core.dispatcher import Dispatcher
from numba.core.serialize import dumps

from numpy_groupies import npg_numba, uaggregate
from numpy_groupies.dtypes import resolve_output_dtype
from numpy_groupies.npg_numba import aggregate, averaging, ops
from numpy_groupies.npg_numba.gather import _gather_1d
from numpy_groupies.npg_numba.jit import _OVERLOADS, _ProcessSafeCacheFile, njit_stable
from numpy_groupies.npg_numba.ops import _dispatch


@pytest.fixture(
    params=[np.float64, np.float32, np.float16, np.int64, np.bool_],
    ids=lambda t: np.dtype(t).name,
)
def adtype(request):
    return request.param


@pytest.fixture(
    params=[
        "sum",
        "prod",
        "len",
        "min",
        "max",
        "first",
        "last",
        "mean",
        "median",
        "std",
        "var",
        "sumofsquares",
        "any",
        "anynan",
        "allnan",
        "trapezoid",
    ]
)
def func(request):
    return request.param


def _make_inputs(func, adtype, rng):
    group_idx = rng.integers(0, 6, 300)
    if adtype is np.bool_:
        if func in ("mean", "median", "std", "var", "trapezoid"):
            pytest.skip("float functions do not accept bool input")
        a = rng.integers(0, 2, 300).astype(adtype)
    elif np.issubdtype(adtype, np.integer):
        if func in ("mean", "median", "std", "var", "trapezoid"):
            a = rng.random(300).astype(np.float64)
            group_idx = group_idx.astype(np.int32)
            return group_idx, a
        a = rng.integers(0, 4, 300).astype(adtype)
    else:
        a = (rng.random(300) * 2 - 1).astype(adtype)
    return group_idx, a


def test_matches_slow_path(rng, func, adtype):
    group_idx, a = _make_inputs(func, adtype, rng)
    # a numpy-integer size bypasses the fast path and runs the full machinery
    ref = aggregate(group_idx, a, func=func, size=np.int64(6))
    for size in (6, None):
        fast = aggregate(group_idx, a, func=func, size=size)
        assert fast.dtype == ref.dtype
        np.testing.assert_array_equal(fast, ref)


def test_nan_variants(rng, func, adtype):
    if func in ("len", "prod", "sumofsquares", "trapezoid", "anynan", "allnan"):
        pytest.skip("no separate nan variant or covered elsewhere")
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    a[group_idx == 2] = np.nan
    nfunc = "nan" + func
    ref = aggregate(group_idx, a, func=nfunc, size=np.int64(6))
    for size in (6, None):
        fast = aggregate(group_idx, a, func=nfunc, size=size)
        assert fast.dtype == ref.dtype
        np.testing.assert_array_equal(fast, ref)


def test_argreductions(rng):
    group_idx = rng.integers(0, 8, 400)
    a = rng.random(400)
    for func in ("argmax", "argmin", "nanargmax", "nanargmin"):
        ref = aggregate(group_idx, a, func=func, size=np.int64(8))
        np.testing.assert_array_equal(aggregate(group_idx, a, func=func), ref)
        np.testing.assert_array_equal(
            aggregate(group_idx, a, func=func, size=10), aggregate(group_idx, a, func=func, size=np.int64(10))
        )


def test_size_detection_matches_explicit(rng):
    group_idx = rng.integers(0, 9, 500)
    a = rng.random(500)
    ref = aggregate(group_idx, a, func="sum", size=int(group_idx.max()) + 1)
    np.testing.assert_array_equal(aggregate(group_idx, a, func="sum"), ref)


@pytest.mark.parametrize("func", ["argmin", "argmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("size", [3, np.int64(3)])
def test_integer_argreductions_preserve_comparisons(func, dtype, size):
    base = int(np.iinfo(dtype).max) - 8
    a = np.array([base + 1, base + 2, base + 2, base + 4, base + 3], dtype=dtype)
    group_idx = np.array([0, 0, 0, 1, 1])
    expected = [0, 4, -1] if "min" in func else [1, 3, -1]
    np.testing.assert_array_equal(aggregate(group_idx, a, func=func, size=size), expected)


@pytest.mark.parametrize("func", ["argmin", "argmax", "nanargmin", "nanargmax"])
def test_integer_argreductions_along_axis(rng, func):
    a = np.array([[2**53 + 1, 2**53, 2**53 + 2], [-(2**53), -(2**53) - 1, -(2**53) - 2]])
    expected = [[1, 2, -1], [1, 2, -1]] if "min" in func else [[0, 2, -1], [0, 2, -1]]
    result = aggregate(np.array([0, 0, 1]), a, func=func, axis=1, size=3)
    np.testing.assert_array_equal(result, expected)


# the input the fast path of the wrapper refuses to take and hands to the slow
# one, which names it: the shape and the dtype are what a call may be refused
# for without looking at a single value, and the size is what the labels of the
# call have to fit in.  A multidimensional group_idx is not one of them - that
# is the indexed input the wrapper documents, and the deviation of the kernels
# from it.  The values of group_idx the wrapper never looks at, and the two
# lengths it only compares for routing: the tests of the kernels ask those of
# both entry points of the one driver, so this one does not ask twice
@pytest.mark.parametrize(
    "args,kwargs",
    [
        # 1D group_idx and 2D a with mismatched trailing dim
        ((np.array([0, 1, 2]), np.array([[10.0, 20.0, 30.0]])), {"func": "sum"}),
        # non-integer group_idx dtype
        ((np.array([0.5, 1.2]), np.array([10.0, 20.0])), {"func": "sum"}),
        # datetime64 values
        ((np.array([0, 1]), np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]")), {"func": "sum"}),
        # object values
        ((np.array([0, 1]), np.array(["a", "b"])), {"func": "sum"}),
        # labels that do not fit the size the caller asked for, python int and
        # numpy int64 alike
        ((np.arange(5), np.arange(5.0)), {"func": "sum", "size": 3}),
        ((np.arange(5), np.arange(5.0)), {"func": "sum", "size": np.int64(3)}),
    ],
)
def test_errors_match_slow_path(args, kwargs):
    """Both paths of the wrapper answer a refused input in the same words.

    The fast path is a shortcut around the checks of the slow one, not around
    their outcome: an input it cannot take it hands on, and the answer that comes
    back is the one the slow path would have written - not one less, and never
    two.
    """
    group_idx, a = args
    func = kwargs.get("func", "sum")
    kwargs = {**kwargs, "func": func}

    with pytest.raises(Exception) as fast_exc:
        aggregate(group_idx, a, **kwargs)
    with pytest.raises(Exception) as slow_exc:
        npg_numba.aggregate(group_idx, a, **kwargs)

    assert type(fast_exc.value) is type(slow_exc.value)
    assert str(fast_exc.value) == str(slow_exc.value)


def test_custom_callable_checks_indices():
    """The custom callable groups the values with the counting sort, which
    checks the group indices on the way."""
    a = np.arange(4.0)
    with pytest.raises(ValueError, match="negative"):
        aggregate(np.array([0, -1, 0, 1]), a, func=np.sum)
    with pytest.raises(ValueError, match="too large"):
        aggregate(np.array([0, 1, 0, 1]), a, func=np.sum, size=1)


def test_integer_sum_dtype_guess(rng):
    # the 'sum' overflow guess is length dependent, so check several lengths
    for adtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32, np.bool_):
        for n in (10, 1000):
            group_idx = rng.integers(0, 4, n)
            a = (rng.random(n) > 0.5).astype(adtype)
            expected_dtype = resolve_output_dtype(None, "sum", a.dtype, n)
            ref = aggregate(group_idx, a, func="sum", size=np.int64(4))
            fast = aggregate(group_idx, a, func="sum", size=4)
            assert fast.dtype == ref.dtype == expected_dtype
            np.testing.assert_array_equal(fast, ref)


def test_slow_path_reachable(rng):
    """Inputs the fast path does not take must keep their behaviour."""
    group_idx = rng.integers(0, 4, 100)
    a = rng.random(100)
    # scalar input
    np.testing.assert_array_equal(
        aggregate(group_idx, 2.0, func="sum"), [2.0 * np.sum(group_idx == i) for i in range(4)]
    )
    # custom fill value forces the finalize machinery
    ref = aggregate(group_idx, a, func="sum", fill_value=-1.0, size=np.int64(5))
    np.testing.assert_array_equal(aggregate(group_idx, a, func="sum", fill_value=-1.0, size=5), ref)
    # 2d group_idx / axis handling
    group_idx_2d = rng.integers(0, 3, (2, 40))
    a2 = rng.random(40)
    ref2 = aggregate(group_idx_2d, a2, func="sum", size=[3, 3])
    np.testing.assert_array_equal(aggregate(group_idx_2d, a2, func="sum", size=None), ref2)
    # axis aggregation
    a3 = rng.random((100, 3))
    ref3 = aggregate(group_idx, a3, func="sum", axis=0, size=np.int64(4))
    np.testing.assert_array_equal(aggregate(group_idx, a3, func="sum", axis=0, size=4), ref3)


def test_numba_callable_cache_construction(monkeypatch):
    group_idx = np.array([0, 1, 1, 2])
    a = np.arange(4.0)
    func = lambda x: np.sum(x)

    constructed = []
    orig_build = ops.build_generic_aggregate

    def counting(func):
        constructed.append(func)
        return orig_build(func)

    monkeypatch.setattr(ops, "build_generic_aggregate", counting)

    for _ in range(3):
        npg_numba.aggregate(group_idx, a, func=func)

    # the loop is built once and reused - the cache of the backend is bounded
    # and holds it (dict.setdefault used to build it on every call)
    assert len(constructed) == 1


def test_aggregate_out(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    ref = aggregate(group_idx, a, func="sum")
    out = np.empty(6)
    ret = aggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # wrong shape or dtype is rejected
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="sum", out=np.empty(5))
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="sum", out=np.empty(6, dtype=np.float32))
    # custom callables do not support out
    with pytest.raises(NotImplementedError, match="out="):
        aggregate(group_idx, a, func=lambda x: np.sum(x), out=out)


def test_every_known_function_is_implemented():
    """A function the aliasing knows and this backend does not have.

    The dispatch table holds such a name too, with no implementation behind it,
    so the answer is about a function that is not available here rather than
    about a func that named nothing at all.  This backend knows them all - the
    collecting function included, which has no compiled kernel but a python
    implementation of its own.
    """
    missing = sorted({resolved.funcname for resolved in _dispatch.values() if resolved.op is None})
    assert missing == []


def test_func_named_nothing():
    """A func that is no key of the dispatch table at all.

    It is neither a function anybody knows nor a callable of the caller's own,
    so it says so - the error of a callable is a different one.
    """
    group_idx = np.array([0, 0, 1, 1])
    a = np.array([1.0, 3.0, 7.0, 2.0])
    with pytest.raises(ValueError, match="neither a valid function string"):
        aggregate(group_idx, a, func="no_such_function", size=2)


def test_gather_is_jitted():
    """The gather of the results must stay jitted.

    A plain python loop needs 220ms for a million values, the jitted one
    0.8ms - and the docstring of ``unpack_into`` promises the latter.
    """
    assert isinstance(_gather_1d, Dispatcher)


def test_uaggregate_out(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300)
    ref = uaggregate(group_idx, a, func="sum")
    out = np.empty(300)
    ret = uaggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # shape of ret[group_idx] is enforced
    with pytest.raises(ValueError, match="expected shape"):
        uaggregate(group_idx, a, func="sum", out=np.empty(299))
    with pytest.raises(TypeError, match="result dtype"):
        uaggregate(group_idx, a, func="sum", out=np.empty(300, dtype=np.int64))
    # multidimensional group_idx falls back to fancy indexing assignment
    group_idx_2d = rng.integers(0, 3, (2, 40))
    a2 = rng.random(40)
    ref2 = uaggregate(group_idx_2d, a2, func="sum")
    out2 = np.empty(ref2.shape)
    ret2 = uaggregate(group_idx_2d, a2, func="sum", out=out2)
    assert ret2 is out2
    np.testing.assert_array_equal(out2, ref2)


def test_aggregate_out_complex(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300) + 1j * rng.random(300)
    ref = aggregate(group_idx, a, func="sum")
    out = np.empty(6, dtype=np.complex128)
    ret = aggregate(group_idx, a, func="sum", out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    # a real out= would throw the imaginary part away, so it is refused
    with pytest.raises(TypeError, match="dtype complex128"):
        aggregate(group_idx, a, func="sum", out=np.empty(6))
    # var measures a real squared magnitude, so a real out= is the right shape
    ref_var = aggregate(group_idx, a, func="var")
    out_var = np.empty(6)
    aggregate(group_idx, a, func="var", out=out_var)
    np.testing.assert_allclose(out_var, ref_var)


def test_uaggregate_out_complex(rng):
    group_idx = rng.integers(0, 6, 300)
    a = rng.random(300) + 1j * rng.random(300)
    ref = uaggregate(group_idx, a, func="sum")
    out = np.empty(300, dtype=np.complex128)
    ret = uaggregate(group_idx, a, out=out, func="sum")
    assert ret is out
    np.testing.assert_array_equal(out, ref)
    with pytest.raises(TypeError, match="does not match the result dtype"):
        uaggregate(group_idx, a, out=np.empty(300), func="sum")


def test_float16_follows_float32_plan():
    # numba has no float16: the input is widened to float32, the kernel runs
    # in the float32 plan and an inexact result is rounded back once.  The
    # numpy twin instead rounds after every step in float16, so on values
    # whose partial sums a float16 cannot represent the two legitimately
    # differ - this pins the float32 plan as the numba behaviour.
    group_idx = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float16)

    ref = aggregate(group_idx, a.astype(np.float32), func="cumsum").astype(np.float16)
    np.testing.assert_array_equal(aggregate(group_idx, a, func="cumsum"), ref)


# the collecting function - the one function whose result is a sequence per
# group; its semantics are tested against every backend in test_funcs.py, what
# is left here is what only this backend offers


def test_collecting_out():
    # the result is an object array, so it can be gathered into
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    out = np.empty(2, dtype=object)
    ret = aggregate(group_idx, a, func="array", out=out)
    assert ret is out
    np.testing.assert_array_equal(out[1], [1.0, 2.0])
    with pytest.raises(TypeError, match="out must have shape"):
        aggregate(group_idx, a, func="array", out=np.empty(3, dtype=object))


def test_collecting_refuses_index_beyond_size():
    group_idx = np.array([0, 1, 1])
    a = np.arange(3.0)
    with pytest.raises(ValueError, match="too large"):
        aggregate(group_idx, a, func="array", size=1)


# The on-disk cache is shared by the workers of a parallel test run: numba
# writes the index of a cache file before the data file it announces, so
# concurrent writers of one cache can leave an index naming the machine code of
# another signature.  ``_ProcessSafeCacheFile`` excludes those writers from each
# other, which is only observable under contention - so this is the one place
# the suite writes one cache from several flows of control at once.  Threads
# rather than processes: ``flock`` excludes separate open file descriptions,
# which is what a writer holds, and a pytest worker is not safe to fork.

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


def test_every_key_of_contended_cache_names_machine(tmp_path):
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


# ---------------------------------------------------------------------------
# The identity of the on-disk cache.  Numba keys its cache index by the
# signature, the machine and two hashes: of the bytecode and of the
# cloudpickled closure of the compiled function.  A dispatcher that reaches
# into such a closure with a uuid drawn fresh in every process moves the key
# of an already cached signature in every process - the cache gains one
# duplicate per signature and parallel kernel per run, without bound.
# What pins the uuid is ``njit_stable``; the tests below keep the pinning
# from decaying one module at a time.

_uuid4 = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_pinned = re.compile(r"\A[0-9a-f]{16}\Z")


def _walk_dispatchers():
    """Every dispatcher of the numba backend, the ones only reachable
    through a closure included - the pickle of a closure descends into all
    of them and every one of them needs its uuid pinned."""
    seen: set[int] = set()
    found: list[Dispatcher] = []

    def push(obj):
        # tuples of dispatchers (the pass structures) and dispatchers kept
        # in dicts count as reachable, the pickle of a cell follows them
        if isinstance(obj, Dispatcher):
            if id(obj) not in seen:
                seen.add(id(obj))
                found.append(obj)
        elif isinstance(obj, (tuple, list, set, frozenset)):
            for item in obj:
                push(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                push(item)

    queue = [npg_numba] + [
        importlib.import_module(f"numpy_groupies.npg_numba.{name}")
        for _, name, _ in sorted(pkgutil.iter_modules(npg_numba.__path__))
    ]
    while queue:
        item = queue.pop(0)
        if isinstance(item, ModuleType):
            # only the modules of this backend - the names a module imports
            # are followed as values, they are not walked as namespaces
            if not item.__name__.startswith("numpy_groupies.npg_numba"):
                continue
            queue.extend(value for _, value in sorted(vars(item).items()))
            continue
        push(item)
        if isinstance(item, Dispatcher):
            for cell in item.py_func.__closure__ or ():
                try:
                    push(cell.cell_contents)
                except ValueError:  # pragma: no cover - empty cell
                    pass
    return found


def _cell_contents(disp):
    """The contents of the non-empty closure cells of one dispatcher."""
    cells = []
    for cell in disp.py_func.__closure__ or ():
        try:
            cells.append(cell.cell_contents)
        except ValueError:  # pragma: no cover - empty cell
            pass
    return tuple(cells)


def test_jitted_callables_of_backend_are_pinned():
    """Every dispatcher of the backend carries the 16 hex digits of a pinned
    uuid - the 36 of a ``uuid4`` say it was drawn fresh in this process."""
    unpinned = sorted(
        f"{d.py_func.__module__}.{d.py_func.__qualname__}"
        for d in _walk_dispatchers()
        if not _pinned.match(str(d._uuid))
    )
    assert unpinned == []


def test_no_process_local_uuid_reaches_index_key_of_cache():
    """The cloudpickled closure of every cached dispatcher - the bytes the
    second hash of the index key covers - is free of a uuid of the draw."""
    leaks = sorted(
        f"{d.py_func.__module__}.{d.py_func.__qualname__}"
        for d in _walk_dispatchers()
        if d.py_func.__closure__ is not None
        and type(d._cache).__name__ != "NullCache"
        and _uuid4.search(dumps(_cell_contents(d)))
    )
    assert leaks == []


def test_dispatcher_without_pin_would_be_caught():
    """The two checks above are not vacuous: a dispatcher jitted without
    ``njit_stable`` right here leaks its draw into both of them."""

    @nb.njit
    def inner(x):
        return x + 1

    outer = njit_stable(lambda x: inner(x), name="probe_outer")
    assert not _pinned.match(str(inner._uuid))
    assert _uuid4.search(dumps(_cell_contents(outer))) is not None
    assert outer._uuid != str(inner._uuid)


def test_identity_of_dispatcher_is_same_in_every_process(tmp_path):
    """The identity of a dispatcher decides the name of its cache file.

    The identity covers the contents of the closure of the dispatcher, and
    its description has to be the same in every process: a python function
    carries its memory address, which differs from process to process - the
    cache file would get a new name in every run, the cache would never hit.

    The dispatcher is built from a file, in this process and in another one:
    the fingerprint covers the source of the function as well, and the
    source of a function given on the command line cannot be read.
    """
    script = tmp_path / "probe_identity.py"
    script.write_text(
        "from numpy_groupies.npg_numba.jit import njit_stable\n"
        "\n"
        "def build():\n"
        "    def helper(x):\n"
        "        return x + 1\n"
        "\n"
        "    def caller(x):\n"
        "        return helper(x)\n"
        "\n"
        "    return njit_stable(caller, name='probe_stable')\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("probe_identity")
        here = str(module.build()._uuid)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("probe_identity", None)
    # the module is imported in the other process as well: the name of the
    # module is part of the identity, and a script run as __main__ carries
    # that name instead
    there = subprocess.run(
        [sys.executable, "-c", "import probe_identity; print(probe_identity.build()._uuid)"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),
    ).stdout.strip()
    assert here == there


# The identity of a dispatcher - and with it the key of its on-disk cache -
# has to cover the body of the overload implementation it reaches.  Numba
# compiles the implementation into the caller, and the source of the caller
# says nothing about it: a change of the implementation alone would leave the
# key untouched, and the stale artifact would be served - a correctly typed,
# silently wrong kernel.


def test_overload_implementations_are_registered():
    """Every overload of the backend is remembered by the package.

    The registry is what lets the identity of a dispatcher cover the body of
    the implementation it reaches.
    """

    for stub in (averaging._track_extrema, averaging._mean_limit_check, averaging._span_limit_check):
        assert stub in _OVERLOADS, stub
    assert averaging._fold_std in _OVERLOADS


def test_dispatcher_identity_covers_overload_body(monkeypatch):
    """The body of an overload implementation is part of the identity.

    The fold of the dispersion is reached by the dispatcher of its caller -
    and the caller says nothing of the body of the implementation in its own
    source, so the identity has to cover it separately.
    """
    plain = njit_stable(averaging._std_combine)._uuid
    original = inspect.getsource

    def changed(func):
        source = original(func)
        if func is averaging._fold_std_overload:
            return source + "\n# a change of the body\n"
        return source

    monkeypatch.setattr(inspect, "getsource", changed)
    assert njit_stable(averaging._std_combine)._uuid != plain


def test_identity_of_dispatcher_covers_nested_overload(monkeypatch):
    """The body of an overload behind an overload is part of the identity.

    The accumulate of the dispersion reaches the extrema through the
    overload of ``_accumulate_std`` - and numba inlines that implementation
    into the caller, with the body of ``_track_extrema`` in it.  The update
    itself says nothing of either body in its own source, so the walk of the
    identity has to follow the implementations it finds.
    """
    plain = njit_stable(averaging._std_update)._uuid
    original = inspect.getsource

    def changed(func):
        source = original(func)
        if func is averaging._track_extrema_overload:
            return source + "\n# a change of the body\n"
        return source

    monkeypatch.setattr(inspect, "getsource", changed)
    assert njit_stable(averaging._std_update)._uuid != plain
