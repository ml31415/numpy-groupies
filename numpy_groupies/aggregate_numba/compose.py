"""Kernel composition for the numba implementation.

Every operation is described by a ``State`` namedtuple - its named per-group
scratch arrays - and a handful of tiny jitted building blocks:

``init(size, n_input, out_dtype, aux_dtype, in_dtype, fill_value,
forced_fill) -> state``
    allocates the scratch arrays, including the ``out`` result array.  The
    auxiliary dtype carries the widened accumulation dtype of the dispersion
    kernels (the running deviation sum); every other operation ignores it.
``seed(state, a)``
    optional pre-loop initialization that needs input data (the min/max
    scratch of the integer exactness gates)
``update(state, ri, i, val, fill_value, ddof)``
    folds one value into the state (``i`` is the input position, only the
    argreductions use it; ``ddof`` carries the sample spacing of trapezoid)
``record(state, ri, i)``
    optional per-position side effect of the cumulative kernels
``gate(state, n_values)``
    optional exactness check between the accumulation and the finishing pass
``finish(state, g, fill_value, ddof) -> value``
    optional per-group second pass (mean, std, var); groups that received
    too few values return the fill value
``untouched(state) -> bool array``
    the groups that received no value, queried to apply a custom fill value
    (empty whenever the fill value equals the forced default - the ops
    then skip their untouched bookkeeping entirely)

The factory assembles these into ONE compiled driver per operation that owns
the group loop, bounds checks and pass structure, and exposes it as a plain
function callable from jitted code (see the ``kernels`` subpackage).  The
python-level ``aggregate()`` calls the same driver with fully resolved
arguments, so there is exactly one compiled artifact per operation.  Unused
building blocks are replaced by synthesized no-ops - the driver always runs
every stage, and LLVM folds the empty calls away.
"""

import typing

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.jit import njit_stable


@njit_stable
def complex_less(x: typing.Any, y: typing.Any) -> bool:
    """The ordering every reduction kernel of this package compares with.

    For real values this is plain ``<``.  numpy orders complex values
    lexicographically - the real part first, the imaginary part only on a tie -
    and compares the parts with ordinary float comparisons rather than with a
    total order, so a nan in either part compares neither smaller nor greater,
    just like a nan of a real dtype.  Note that numpy does not look at the
    imaginary part at all once the real parts differ, but it does require both
    imaginary parts to be comparable for the real part to decide the order -
    which makes ``1+nanj < 2+1j`` False instead of True.

    Written as a jitted function with an ``isinstance`` branch rather than as
    an overload - measured, the call is not what the group loop costs (1.1ms
    of the reduction with it, 1.3ms with an overload numba inlines).  The test
    names what numba can name: the python class ``complex`` is complex128
    there, so ``np.complex64`` has to be spelled out as well, and the abstract
    ``np.complexfloating`` cannot be named at all - numba types the classes of
    an ``isinstance`` tuple as constants and numpy refuses that one as a dtype,
    which would take the whole reduction down with it.

    Disk cached like its callers - numba cannot resolve an uncached callee
    from cached caller code at all (the loaded module dies on a NULL ``descr``
    in ``NRT_adapt_ndarray_to_python``).  It is therefore not inlined, which
    the kernels account for by resolving the comparison into a local variable
    before the select expression instead of calling inside it.
    """
    if isinstance(x, (complex, np.complex64)) and isinstance(y, (complex, np.complex64)):
        if x.real < y.real:
            return not (np.isnan(x.imag) or np.isnan(y.imag))
        if x.real == y.real:
            # numba's compare is already a bool, only numpy's stub answers
            # np.bool_ - the call is a no-op in compiled code
            return bool(x.imag < y.imag)
        return False
    if isinstance(x, (complex, np.complex64)) or isinstance(y, (complex, np.complex64)):
        raise TypeError("can not compare float with complex")
    less: bool = x < y
    return less


# ---------------------------------------------------------------------------
# compile-time dtype rules - run inside the overload builders on numba types,
# returning the (out_dtype, aux_dtype) pair of np.dtypes


def _as_np_dtype(a_type: nb.types.Type) -> np.dtype:
    """The np.dtype of the numba array type an overload builder sees."""
    # every rule is handed the numba type of the value array and nothing else -
    # the type lattice has no named ancestor of ArrayCTypes that carries dtype
    array_type = typing.cast("nb.types.ArrayCTypes", a_type)
    return nb.np.numpy_support.as_dtype(array_type.dtype)


def rule_same(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """min/max/first/last: the input dtype passes through."""
    dt = _as_np_dtype(a_type)
    return dt, dt


def rule_sum(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """sum: integers and bools widen to int64.

    This is the njit-kernel approximation - the python-level ``aggregate()``
    keeps the exact narrow-dtype overflow guesses, which depend on the value
    count and therefore cannot be resolved at compile time.
    """
    dt = _as_np_dtype(a_type)
    if dt.kind in "iub":
        return np.dtype(np.int64), np.dtype(np.int64)
    return dt, dt


def rule_widen_int(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """prod and the cumulatives: integer input widens to int64, everything
    else (bool included) passes through - matching the python-level rules."""
    dt = _as_np_dtype(a_type)
    if dt.kind in "iu":
        return np.dtype(np.int64), np.dtype(np.int64)
    return dt, dt


def rule_float(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """mean/median/trapezoid: inexact input keeps its dtype, otherwise
    everything widens to float64."""
    dt = _as_np_dtype(a_type)
    out = dt if dt.kind in "fc" else np.dtype(np.float64)
    return out, out


def rule_real_float(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """var/std: real float passes through, complex input yields the real
    counterpart (complex64 -> float32), integers/bools widen to float64.
    The auxiliary dtype is the deviation-sum accumulator, which stays complex
    for complex input and widens like the running mean."""
    dt = _as_np_dtype(a_type)
    if dt.kind == "f":
        out = dt
    elif dt.kind == "c":
        out = np.dtype(dt.char.lower())
    else:
        out = np.dtype(np.float64)
    return out, np.result_type(dt, np.float64)


def rule_sumofsquares(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """sumofsquares: like var/std, except integers/bools widen to int64 - the
    exact squares of integers are integral."""
    dt = _as_np_dtype(a_type)
    if dt.kind == "f":
        out = dt
    elif dt.kind == "c":
        out = np.dtype(dt.char.lower())
    else:
        out = np.dtype(np.int64)
    return out, out


def rule_bool(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    out = np.dtype(np.bool_)
    return out, out


def rule_int64(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    out = np.dtype(np.int64)
    return out, out


def _resolve_default_fill(fill_default: typing.Any, out_dtype: np.dtype) -> typing.Any:
    """The njit-kernel twin of the DEFAULT_FILL_VALUE resolution: a nan
    default only fits inexact outputs (a complex output needs the nan in both
    parts), integer and bool outputs fall back to 0."""
    if fill_default != fill_default:  # nan
        if out_dtype.kind == "c":
            return complex(fill_default, fill_default)
        if out_dtype.kind not in "fc":
            return 0
    return fill_default


@njit_stable
def output_size(group_idx: npt.NDArray[np.integer], size_arg: int) -> int:
    """The resolved output size of one aggregation: one past the largest
    group index, capped by an explicitly requested size.  Once this returns,
    every index in ``group_idx`` is within ``[0, size)`` - the drivers use it
    once per call instead of repeating the logic inline."""
    size: int = np.max(group_idx) + 1
    if size_arg >= 0:
        # an explicit output size may be larger than the largest index, but
        # never smaller
        if size > size_arg:
            raise ValueError("one or more indices in group_idx are too large")
        size = size_arg
    return size


def _make_stub(
    stub_name: str,
    driver_getter: typing.Callable[[], typing.Any],
    rule: typing.Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: typing.Any,
    doc: str | None,
    ddof_default: typing.Any = 0,
) -> typing.Callable[..., typing.Any]:
    """Create the njit-callable ``aggregate_*`` function of one variant.

    The stub is a plain python function: calling it from python raises, from
    jitted code the ``@overload`` resolves the dtype rule at compile time and
    forwards to the compiled driver.  ``fill_value=None`` asks for the
    function default, which - like the python-level DEFAULT_FILL_VALUE -
    depends on the output dtype (nan only fits inexact results); the
    resolution happens in the overload builder, the choice itself is a
    runtime branch on the typed-none argument.  The third argument is named
    ``ddof`` for every operation - trapezoid receives the sample spacing
    through it.
    """

    # the two shims below speak the numba overload protocol: numba hands them
    # its own types at compile time (or the python defaults of the omitted
    # arguments), so their parameters are honestly Any.  The three signatures
    # below are one protocol: numba compares the typing function with the
    # implementation it returns, argument names *and* annotations included, so
    # they have to be spelled alike - see the note in ``averaging``
    def stub(
        group_idx: typing.Any,
        a: typing.Any,
        ddof: int = ddof_default,
        size: int = -1,
        fill_value: float | None = None,
    ) -> typing.Any:
        raise NotImplementedError("only callable from jitted code - use aggregate() otherwise")

    stub.__name__ = stub_name
    stub.__qualname__ = stub_name
    stub.__doc__ = f"{doc or f'Njit-callable {stub_name} kernel'} - see ``numpy_groupies.aggregate_numba.kernels``."

    # no jit_options cache here: the impl shim is compiled by numba's
    # overload machinery OUTSIDE the revisioned ``njit_stable`` namespace,
    # and its cache key does not reliably capture the identity of the
    # captured driver - stale impls of earlier refactors then keep being
    # served (observed as a nanfirst running plain-first machine code).
    # The shim itself compiles in microseconds, the expensive compilation
    # (the driver) stays cached through ``njit_stable``.
    @nb.extending.overload(stub)
    def _overload(
        group_idx: typing.Any,
        a: typing.Any,
        ddof: int = ddof_default,
        size: int = -1,
        fill_value: float | None = None,
    ) -> typing.Callable[..., typing.Any]:
        driver = driver_getter()
        # here ``a`` is the numba *type* of the values (an ``Array`` of the
        # input dtype), not the values - those only arrive in the impl below,
        # which is why the shared signature can only call the parameter ``a``
        a_type: nb.types.Type = a
        out_dtype, aux_dtype = rule(a_type)
        default_fill = _resolve_default_fill(fill_default, out_dtype)

        def impl(
            group_idx: typing.Any,
            a: typing.Any,
            ddof: int = ddof_default,
            size: int = -1,
            fill_value: float | None = None,
        ) -> typing.Any:
            if fill_value is None:
                return driver(group_idx, a, size, out_dtype, aux_dtype, default_fill, ddof)
            return driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, ddof)

        return impl

    return stub


class KernelHandle(typing.NamedTuple):
    """The compiled artifact of one operation variant.

    ``stub``    the njit-callable function, exported via the ``kernels``
                module
    ``driver``  the raw compiled driver - the python-level ``aggregate()``
                calls it with fully resolved arguments
    ``forced_fill`` the seeding default of the output (or None when the
                output is seeded with the fill value itself)
    ``aux_widen``   whether the auxiliary dtype is the input dtype widened
                with float64 (the dispersion accumulators)
    ``nton``    whether the output has the input length (the cumulatives)
    ``fallback``    python-level callable for input dtypes the compiled
                driver cannot handle (complex medians go to the numpy
                implementation - numba cannot partition complex values)
    """

    name: str
    nans: bool
    stub: typing.Any
    driver: typing.Any
    forced_fill: float | int | None
    aux_widen: bool
    nton: bool
    fallback: typing.Callable[..., typing.Any] | None = None


class KernelPair(typing.NamedTuple):
    """Plain and nan-variant handles of one operation - ``nan`` is None for
    the operations without a separate nan variant."""

    plain: KernelHandle
    nan: KernelHandle | None


def nan_handle(pair: KernelPair, name: str) -> KernelHandle:
    """The nan variant of a pair that the caller knows has one.

    Callers reach for it behind the ``funcs_no_separate_nan`` gate (or the
    kernel table), where ``nan`` is never None - this only states that
    promise for the type checker and keeps the message honest if it breaks.
    """
    handle = pair.nan
    if handle is None:
        raise ValueError(f"{name} has no nan variant")
    return handle


def build_kernel(
    name: str,
    *,
    init: typing.Callable[..., typing.Any],
    update: typing.Callable[..., typing.Any],
    rule: typing.Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: typing.Any = 0,
    forced_fill: float | None = None,
    seed: typing.Callable[..., typing.Any] | None = None,
    record: typing.Callable[..., typing.Any] | None = None,
    gate: typing.Callable[..., typing.Any] | None = None,
    finish: typing.Callable[..., typing.Any] | None = None,
    untouched: typing.Callable[..., typing.Any] | None = None,
    update_nan: typing.Callable[..., typing.Any] | None = None,
    nans: bool = True,
    ddof_default: float = 0,
    aux_widen: bool = False,
    nton: bool = False,
    fallback: typing.Callable[..., typing.Any] | None = None,
    doc: str | None = None,
) -> KernelPair:
    """Compose the kernels of one operation from jitted building blocks.

    See the module docstring for the building-block signatures.  Both the
    plain and (unless ``nans=False``) the nan-guarded variant are built.  The
    nan variant uses ``update_nan`` when the operation provides one - a
    branchless select-form body that folds the nan handling into its stores
    so that LLVM keeps the loop vectorized - and otherwise falls back to the
    generic guard that skips nan values before they reach ``update``.
    """
    init_c = njit_stable(init)
    update_c = njit_stable(update)
    update_nan_c = njit_stable(update_nan) if update_nan is not None else None

    # synthesized no-ops for the unused building blocks - all of them get
    # inlined and folded away, the driver never branches on their presence.
    # They carry no operation in their name: the four of them are one and the
    # same function in every operation - same source, no closure over
    # anything operation specific - so symbol, cache entry and specialization
    # name the same function everywhere and the operations share one compiled
    # artifact instead of each carrying a copy of it.  Sharing is only sound
    # while that holds, and these four once did carry the operation: numba
    # 0.68 corrupted the boxing of the second driver's return array when the
    # nan variants of sum and sumofsquares ran in that order (one state
    # struct, one signature between them), which #129 answered with a name
    # per operation.  The corruption does not reproduce in either order, cold
    # or warm cache, so they are shared again - should the error ("In
    # 'NRT_adapt_ndarray_to_python', 'descr' is NULL") ever come back, the
    # name is the knob.  The building blocks that *are* used are different
    # functions per operation, so they keep the name of their operation (see
    # below) - and so do the drivers and the kernels.
    # the building blocks below are jitted: numba compiles them from the
    # argument types and ignores these annotations, the type checker needs them
    def _seed_none(st: typing.Any, a: typing.Any) -> None:
        pass

    def _record_none(st: typing.Any, ri: int, i: int) -> None:
        pass

    def _gate_none(st: typing.Any, n_values: int) -> None:
        pass

    def _finish_none(st: typing.Any, g: int, fill_value: float, ddof: int) -> typing.Any:
        return st.out[g]

    # the same story in the other direction: the qualname is part of numba's
    # mangled symbol, so nested blocks of this factory need the operation in
    # their name to keep their environments apart (see njit_stable) - two
    # different functions that share module, qualname and signature share one
    # symbol, and a driver loaded from the on-disk cache then binds its
    # callees into the environment of the other function, correct types and
    # wrong results (caught for ``last`` against ``nanlast``, both taking an
    # int64 and a float64 array)
    seed_c = njit_stable(seed) if seed is not None else njit_stable(_seed_none)
    record_c = njit_stable(record) if record is not None else njit_stable(_record_none)
    gate_c = njit_stable(gate) if gate is not None else njit_stable(_gate_none)
    finish_c = njit_stable(finish) if finish is not None else njit_stable(_finish_none)
    if untouched is not None:
        untouched_c = njit_stable(untouched)

        def fill_untouched(st: typing.Any, size: int, fill_value: float) -> None:
            # a custom fill value also belongs to the groups that never
            # received a value - their output was seeded with the forced
            # default of the operation
            if fill_value != forced_fill:
                un = untouched_c(st)
                for g in range(size):
                    if un[g]:
                        st.out[g] = fill_value

    else:

        def fill_untouched(st: typing.Any, size: int, fill_value: float) -> None:
            pass

    fill_untouched = njit_stable(fill_untouched, name=f"{name}_fill_untouched")

    def build(nan_variant: bool) -> KernelHandle:
        # plain and nan variant are two compiled artifacts of the same source
        # block - without the prefix in the name their drivers share a symbol
        # too, which is what ``last`` and ``nanlast`` were caught doing
        op_name = name if not nan_variant else f"nan{name}"
        if nan_variant:
            if update_nan_c is not None:
                update_ = update_nan_c
            else:

                def guarded_update(
                    st: typing.Any,
                    ri: int,
                    i: int,
                    val: float,
                    fill_value: float,
                    ddof: int,
                ) -> None:
                    if val == val:
                        update_c(st, ri, i, val, fill_value, ddof)

                update_ = njit_stable(guarded_update, name=f"{op_name}_guarded_update")
        else:
            update_ = update_c

        def kernel(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            st: typing.Any,
            fill_value: float,
            ddof: int,
            size: int,
        ) -> None:
            # a function of its own on purpose: with the arrays living in the
            # caller's frame (allocated there, reached through the state) the
            # same loop measures ~12% slower - numba has to reload the data
            # pointers around every store instead of keeping them in
            # registers.  measured against numba 0.68 on argmax
            for i in range(group_idx.shape[0]):
                ri = group_idx[i]
                if ri < 0:
                    raise ValueError("negative indices not supported")
                if ri >= size:
                    raise ValueError("one or more indices in group_idx are too large")
                update_(st, ri, i, a[i], fill_value, ddof)
                record_c(st, ri, i)

        kernel = njit_stable(kernel, name=f"{op_name}_kernel")

        def driver(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: float,
            ddof: int,
        ) -> typing.Any:
            # numba's np.max reduction is several times faster than a scalar
            # loop here; negative indices slip through and are caught by the
            # bounds check inside the kernel below
            size = np.max(group_idx) + 1
            if size_arg >= 0:
                # an explicit output size may be larger than the largest
                # index, but never smaller
                if size > size_arg:
                    raise ValueError("one or more indices in group_idx are too large")
                size = size_arg
            st = init_c(size, len(group_idx), out_dtype, aux_dtype, a.dtype, fill_value, forced_fill)
            seed_c(st, a)
            kernel(group_idx, a, st, fill_value, ddof, size)
            gate_c(st, len(group_idx))
            for g in range(size):
                st.out[g] = finish_c(st, g, fill_value, ddof)
            fill_untouched(st, size, fill_value)
            return st.out

        driver = njit_stable(driver, name=f"{op_name}_driver")

        stub = _make_stub(
            op_name,
            lambda: driver,
            rule,
            fill_default,
            doc,
            ddof_default,
        )
        return KernelHandle(
            name=op_name,
            nans=nan_variant,
            stub=stub,
            driver=driver,
            forced_fill=forced_fill,
            aux_widen=aux_widen,
            nton=nton,
            fallback=fallback,
        )

    plain = build(False)
    nan_variant = build(True) if nans else None
    return KernelPair(plain, nan_variant)


def build_custom_kernel(
    name: str,
    *,
    driver_factory: typing.Callable[[bool], typing.Any],
    rule: typing.Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: typing.Any = 0,
    forced_fill: float | None = None,
    nans: bool = True,
    ddof_default: float = 0,
    aux_widen: bool = False,
    nton: bool = False,
    fallback: typing.Callable[..., typing.Any] | None = None,
    doc: str | None = None,
) -> KernelPair:
    """Compose the kernels of an operation with a hand-written driver.

    ``driver_factory(nans)`` returns the compiled driver with the same
    signature as the composed ones - used by operations whose pass structure
    does not fit the accumulate/finish pattern (median groups the data with a
    counting sort before selecting, sort groups it before ordering).
    """

    def build(nan_variant: bool) -> KernelHandle:
        driver = driver_factory(nan_variant)
        stub_name_variant = name if not nan_variant else "nan" + name
        stub = _make_stub(
            stub_name_variant,
            lambda: driver,
            rule,
            fill_default,
            doc,
            ddof_default,
        )
        return KernelHandle(
            name=stub_name_variant,
            nans=nan_variant,
            stub=stub,
            driver=driver,
            forced_fill=forced_fill,
            aux_widen=aux_widen,
            nton=nton,
            fallback=fallback,
        )

    plain = build(False)
    nan_variant = build(True) if nans else None
    return KernelPair(plain, nan_variant)
