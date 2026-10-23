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
``update(state, ri, i, val, fill_value, extra)``
    folds one value into the state (``i`` is the input position, only the
    argreductions use it; the last argument is the one the operation has of its
    own - the ``ddof`` of the variance, the sample spacing ``dx`` of the
    trapezoid - and the zero an operation without an argument of its own is
    called with.  The driver reaches the building blocks positionally, so a
    block that uses the value names it after the operation and one that ignores
    it leaves it as ``extra``)
``record(state, ri, i)``
    optional per-position side effect of the cumulative kernels
``gate(state, n_values)``
    optional exactness check between the accumulation and the finishing pass
``finish(state, g, fill_value, extra) -> value``
    optional per-group second pass (mean, std, var); groups that received
    too few values return the fill value
``untouched(state) -> bool array``
    the groups that received no value, queried to apply a custom fill value
    (empty whenever the fill value equals the forced default - the ops
    then skip their untouched bookkeeping entirely)

The factory assembles these into ONE compiled driver per operation that owns
the group loop, bounds checks and pass structure, and exposes it as a function
callable from jitted code and from python alike (the kernels of
``numpy_groupies.aggregate_numba``).  The driver carries the argument the
operation has of its own under the name the caller uses - ``build_driver`` and
its siblings below spell the signatures out, so that the python function, the
typing function of its overload and the jitted driver of an operation all
carry the same one.  The python-level ``aggregate()`` calls
the same driver with fully resolved arguments, so there is exactly one
compiled artifact per operation.  Unused
building blocks are replaced by synthesized no-ops - the driver always runs
every stage, and LLVM folds the empty calls away.
"""

import functools
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, cast

import numba as nb
import numpy as np
import numpy.typing as npt

from numpy_groupies.aggregate_numba.jit import njit_stable
from numpy_groupies.utils_np import AResult, AScalar, check_fill_value_for_dtype
from numpy_groupies.validation import (
    EMPTY_GROUP_IDX,
    FILL_NOT_BOOLEAN,
    FILL_NOT_CONVERTIBLE,
    LENGTH_MISMATCH,
    NEGATIVE_INDICES,
    NON_INTEGER_GROUP_IDX,
    TOO_LARGE,
    TOO_LARGE_SIZE_PREFIX,
    is_integer_dtype,
    is_numeric_dtype,
)


@njit_stable
def complex_less(x: Any, y: Any) -> bool:
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
    array_type = cast("nb.types.ArrayCTypes", a_type)
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


def rule_object(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    """array: the result is a sequence per group.

    The sequences carry the dtype of the values, the collection of them is of
    the object dtype - it holds the sequences themselves, not numbers.
    """
    out = np.dtype(object)
    return out, out


def rule_int64(a_type: nb.types.Type) -> tuple[np.dtype, np.dtype]:
    out = np.dtype(np.int64)
    return out, out


def _resolve_default_fill(fill_default: AScalar, out_dtype: np.dtype) -> AScalar:
    """The njit-kernel twin of the DEFAULT_FILL_VALUE resolution: a nan
    default only fits inexact outputs (a complex output needs the nan in both
    parts), integer and bool outputs fall back to 0."""
    if fill_default != fill_default:  # nan
        if out_dtype.kind == "c":
            return complex(fill_default, fill_default)
        if out_dtype.kind not in "fc":
            return 0
    return fill_default


# What the drivers of this package are written for and never look at twice:
# two flat 1d arrays, of integer labels and of numeric values, of the same
# length and not empty.  The shape and the dtype are what a type knows,
# jitted code is answered for them by the typing function of the kernel and a
# python call by its body, and the two remaining value checks go into both of
# those entry points as well (see ``_make_stub``).  ``aggregate()`` is the entry
# point that knows an axis, an indexed ``group_idx`` and a single value
# repeated over every label, and it validates all of that in
# ``utils_np.input_validation`` before a driver sees anything - no call of a
# driver ever validates twice.
#
# The words of the complaints live in ``numpy_groupies.validation``: the
# python-level call and the kernels say the same things about the same
# faults.  What is left here is what only a kernel has to say - the flatness
# of its input and the numeric dtypes of its values.
_FLAT_INPUT = "the kernels take flat one-dimensional input"
_SCALAR_INPUT = "the kernels take one value per label - a scalar is not supported"
_NUMERIC_VALUES = "the kernels take numeric values"
# what only the kernel of an operation that cannot place a fill value has to
# say - the collecting one, whose absent group holds an empty sequence
_REFUSAL = "The {name} kernel has no fill_value - an absent group holds an empty sequence"


def _ndim(x: Any) -> int:
    """The number of dimensions of an argument, before it is converted.

    A scalar would be promoted to a one-element array by the conversion the
    kernel does, and the complaint about it would be the wrong one - so the
    shape is asked of the argument as it was handed over.
    """
    ndim = getattr(x, "ndim", None)
    if ndim is None:
        return int(np.ndim(x))
    return int(ndim)


def _kernel_inputs(group_idx: Any, a: Any) -> tuple[Any, Any]:
    """The two arrays of a python call of a kernel, contiguous and checked.

    What the typing function of the kernel answers for jitted code - a shape it
    can see and a dtype it can name - the python call answers here, where the
    arrays are values again and the answer can name the entry point that does
    know an axis and an indexed ``group_idx``.
    """
    ndim_group_idx = _ndim(group_idx)
    ndim_a = _ndim(a)
    if ndim_a == 0:
        raise ValueError(_SCALAR_INPUT)
    if ndim_group_idx != 1 or ndim_a != 1:
        raise ValueError(f"{_FLAT_INPUT} - use aggregate() for an axis or an indexed group_idx")
    group_idx = np.ascontiguousarray(group_idx)
    a = np.ascontiguousarray(a)
    if not is_integer_dtype(group_idx.dtype):
        # the same complaint the python-level call makes about the labels
        raise TypeError(NON_INTEGER_GROUP_IDX)
    if not is_numeric_dtype(a.dtype):
        raise ValueError(f"{_NUMERIC_VALUES}, not {a.dtype}")
    return group_idx, a


def _numpy_dtype(dtype: Any) -> np.dtype | None:
    """The numpy dtype of a dtype of either world.

    The python path has numpy dtypes, the typing function of a kernel has the
    numba types of the elements of its array - numba knows how to translate
    them, and the predicates of ``numpy_groupies.validation`` ask the
    translated one.
    """
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype
    return cast("np.dtype", nb.np.numpy_support.as_dtype(dtype))


def _fill_type_ok(fill_type: Any) -> bool:
    """Whether the type of the fill value can hold a number.

    The typing function of a kernel sees the type of the fill value, not its
    value: a fill that is not a number (a string, say) cannot be put into any
    output, and saying so is better than the numba error of the driver.  A
    None is the "give me the default" of the signature and an optional value
    may hold it at runtime - both are fine.
    """
    if fill_type is None:
        # the default of the signature: numba hands the python None of an
        # argument the caller left out to the typing function
        return True
    if isinstance(fill_type, (nb.types.NoneType, nb.types.Number, nb.types.Boolean)):
        return True
    if isinstance(fill_type, nb.types.Omitted):
        # an argument the caller left out - it carries the default of the
        # signature, which is the None that asks for the default of the
        # operation
        return fill_type.value is None
    if isinstance(fill_type, nb.types.Optional):
        return isinstance(fill_type.type, (nb.types.Number, nb.types.Boolean))
    return False


def _fill_fits(fill_value: Any, low: Any, high: Any, high_exclusive: Any) -> bool:
    """Whether the fill value fits the output it fills.

    The overload below knows the type of the value and picks the comparison
    that is exact for it: a signed integer is compared with the integer
    bounds of the output, everything else with the bounds a float holds
    exactly (and a nan fits nothing).  An unsigned integer is the one case
    the float comparison cannot serve exactly - a value at the very top of
    the range of the output is refused, where numpy would take it.

    The function itself is never called - jitted code reaches the
    implementation of the overload.
    """
    raise NotImplementedError("the overload of _fill_fits provides the implementation")


@nb.extending.overload(_fill_fits)
def _fill_fits_overload(fill_value: Any, low: Any, high: Any, high_exclusive: Any) -> Any:
    if isinstance(fill_value, nb.types.Integer) and getattr(fill_value, "signed", True):
        # a signed integer is compared with the integer bounds of the output -
        # the largest one fits exactly, which a float comparison would not say
        def impl(fill_value: Any, low: Any, high: Any, high_exclusive: Any) -> bool:
            return bool(low <= fill_value <= high)
    else:
        # everything else is compared with the bounds a float holds exactly -
        # and a nan fits nothing
        def impl(fill_value: Any, low: Any, high: Any, high_exclusive: Any) -> bool:
            return bool(fill_value == fill_value and low <= fill_value < high_exclusive)

    return impl


@njit_stable
def _require_inputs(group_idx: Any, a: Any) -> None:
    """The two value checks a call of a kernel cannot be trusted to skip.

    An empty ``group_idx`` leaves the driver with nothing to reduce, and a
    short ``a`` has it read a value past the end of the array, which numba
    never catches.  The python call of a kernel asks the same two questions in
    ``_python_prologue``, where the arrays are values again and the answer can
    name the entry point that does know an axis.
    """
    if len(group_idx) == 0:
        raise ValueError(EMPTY_GROUP_IDX)
    if len(a) != len(group_idx):
        raise ValueError(LENGTH_MISMATCH)


@njit_stable
def _require_fittable(
    fill_value: Any,
    is_bool: bool,
    is_exact: bool,
    low: int,
    high: int,
    high_exclusive: float,
    message: str,
) -> None:
    """Whether the fill value fits the output it fills, as compiled code
    sees it: a boolean output takes the two booleans, an integer output an
    exact number of its own range, and the inexact ones anything.

    The bounds and the complaint are resolved by the typing function of the
    kernel once per compiled signature (``_typing_prologue``) - jitted code
    cannot ask numpy, and the message is the one numpy would have said.
    """
    if is_bool and fill_value != 0 and fill_value != 1:
        raise ValueError(FILL_NOT_BOOLEAN)
    if is_exact and not _fill_fits(fill_value, low, high, high_exclusive):
        raise ValueError(message)


@functools.cache
def _array_type(dtype: np.dtype) -> nb.types.Type:
    """The numba type of a contiguous 1d array of *dtype*.

    The dtype rules of the kernels are written against the numba type an
    overload builder is handed, so the python call of a kernel reads them
    through this type - the two calls of one kernel cannot disagree about the
    output dtype that way.  The rules look at nothing but the dtype, so one
    type per dtype is enough - and the cache is worth having, since ``typeof``
    of an array costs a dozen microseconds, more than the whole call of a
    small kernel.
    """
    return cast("nb.types.Type", nb.typeof(np.empty(0, dtype=dtype)))


@njit_stable
def _raise_too_large_for_size(size_arg: int) -> None:
    """The complaint of an explicit size that is too small.

    The message is built in a function of its own: the string machinery of
    numba is big, and inlined into the caller it costs the hot loop of every
    operation more than the size in the message is worth - the drivers
    measured ~15% slower with the concatenation in them.
    """
    raise ValueError(TOO_LARGE_SIZE_PREFIX + str(size_arg))


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
            _raise_too_large_for_size(size_arg)
        size = size_arg
    return size


# ---------------------------------------------------------------------------
# the argument an operation has of its own
#
# Beside the two arrays, the size, the dtype pair and the fill value a driver
# takes nothing (``none``), the ``ddof`` of the variance, the quantile ``q``,
# the sample spacing ``dx`` of the trapezoid or the ``reverse`` flag of the
# sort.  The name of the argument is the one the caller uses, all the way from
# the signature of the kernel to the signature of the jitted driver: a slot of
# a driver that carries something named after something else is what the
# kernels of an earlier version of this module had to write by hand, and what
# the explicit signatures below make impossible.
Variant = Literal["none", "ddof", "q", "dx", "reverse"]

# the names the composed kernels of ``build_kernel`` take - the quantile and
# the reverse flag belong to operations whose pass structure is written by
# hand (see ``numpy_groupies.aggregate_numba.specials``)
ComposedVariant = Literal["none", "ddof", "dx"]

# what the python half of a kernel call hands the driver: the two checked
# arrays, the resolved dtype pair and the resolved fill value
_PythonCall = tuple[Any, Any, np.dtype, np.dtype, AScalar]

# what the typing function of a kernel resolves once per compiled signature:
# the dtype pair, the default fill value, the complaint about a fill value
# that does not fit and the rules that decide whether one fits
_TypingCall = tuple[np.dtype, np.dtype, AScalar, str, bool, bool, int, int, float]

# the two halves of a kernel call, and the pair of functions one variant of an
# operation is made of: the python entry point and the typing function of the
# overload that answers the same call for jitted code
_PythonPrologue = Callable[[Any, Any, int, AScalar | None], _PythonCall]
_TypingPrologue = Callable[[Any, Any, Any], _TypingCall]
_Kernel = tuple[Callable[..., Any], Callable[..., Any]]


def _no_fill_value(fill_type: Any) -> bool:
    """Whether the argument of the signature asks for the default fill value.

    The None of the default reaches the typing function of a kernel either as
    the default of the signature itself, as the numba type of a None the caller
    passed, or as the value of an argument the caller left out entirely.
    """
    if fill_type is None or isinstance(fill_type, nb.types.NoneType):
        return True
    return bool(isinstance(fill_type, nb.types.Omitted) and fill_type.value is None)


def _python_prologue(
    rule: Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: AScalar,
    reject_fill: bool,
    stub_name: str,
) -> _PythonPrologue:
    """The python half of a call of a kernel: everything between the arguments
    of the signature and the arguments of the driver.

    What the typing function of the overload answers by rule once per compiled
    signature, this answers by hand once per call - the same dtype rule, read
    from the numpy dtype behind the numba type of the values, and the same
    rules for the fill value, asked of numpy itself (the very function the
    python-level ``aggregate()`` uses).  The two cannot disagree, since they
    read the one rule.
    """

    def resolve(group_idx: Any, a: Any, size: int, fill_value: AScalar | None) -> _PythonCall:
        group_idx, a = _kernel_inputs(group_idx, a)
        if reject_fill and fill_value is not None:
            # the driver of this operation cannot place a fill value - saying
            # so beats ignoring the argument silently
            raise ValueError(_REFUSAL.format(name=stub_name))
        if len(group_idx) == 0:
            raise ValueError(EMPTY_GROUP_IDX)
        if len(a) != len(group_idx):
            raise ValueError(LENGTH_MISMATCH)
        out_dtype, aux_dtype = rule(_array_type(a.dtype))
        if fill_value is None:
            fill_value = _resolve_default_fill(fill_default, out_dtype)
        else:
            fill_value = check_fill_value_for_dtype(fill_value, out_dtype)
        return group_idx, a, out_dtype, aux_dtype, fill_value

    return resolve


def _typing_prologue(
    rule: Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: AScalar,
    reject_fill: bool,
    stub_name: str,
) -> _TypingPrologue:
    """The compile-time half of a call of a kernel: what the typing function of
    the overload answers before it returns the implementation.

    The shape and the dtype of an argument are what a type knows, so the flat
    input, the numeric values and the integer labels are what the typing
    function says no to; the two value checks that need a value (an empty
    ``group_idx``, an ``a`` of another length) are left to the implementation,
    which runs ``_require_inputs``.

    The fill value is decided here as far as a type reaches: a fill value that
    is not a number cannot be put into any output, and a driver that cannot
    place one refuses it while the argument still is a type.  What fits the
    output *dtype* is resolved for the compiled code into the handful of
    numbers and the one message that ``_require_fittable`` compares against.
    """

    def prepare(group_idx: Any, a: Any, fill_value: Any) -> _TypingCall:
        # the drivers read one label per value: a multidimensional group_idx
        # they walk by row and answer with a result that is quietly wrong
        if getattr(group_idx, "ndim", -1) != 1 or getattr(a, "ndim", -1) != 1:
            raise nb.TypingError(_FLAT_INPUT)
        # an index numba does not know as an index: a float one is rounded by
        # the caller, a bool one is not an index at all
        if not is_integer_dtype(_numpy_dtype(getattr(group_idx, "dtype", None))):
            raise nb.TypingError(NON_INTEGER_GROUP_IDX)
        if not is_numeric_dtype(_numpy_dtype(getattr(a, "dtype", None))):
            raise nb.TypingError(_NUMERIC_VALUES)
        out_dtype, aux_dtype = rule(a)
        if reject_fill:
            if not _no_fill_value(fill_value):
                raise nb.TypingError(_REFUSAL.format(name=stub_name))
        elif not _fill_type_ok(fill_value):
            raise nb.TypingError(FILL_NOT_CONVERTIBLE.format(dtype=out_dtype.type.__name__))
        default_fill = _resolve_default_fill(fill_default, out_dtype)
        # the rules of the fill value, as the compiled code sees them: a
        # boolean output takes the two booleans, an integer output takes an
        # exact number of its own range, and the inexact ones take anything
        fill_is_bool = out_dtype.kind == "b"
        fill_is_exact = out_dtype.kind in "iu"
        if fill_is_exact:
            info = np.iinfo(out_dtype)
            fill_min = int(info.min) if info.min < 0 else 0
            fill_max = int(info.max)
            # the exclusive bound, which a float holds exactly - it is what an
            # inexact fill value is compared against
            fill_max_exclusive = float(info.max) + 1.0
        else:
            fill_min, fill_max, fill_max_exclusive = 0, 0, 0.0
        message = FILL_NOT_CONVERTIBLE.format(dtype=out_dtype.type.__name__)
        return (
            out_dtype,
            aux_dtype,
            default_fill,
            message,
            fill_is_bool,
            fill_is_exact,
            fill_min,
            fill_max,
            fill_max_exclusive,
        )

    return prepare


# the three functions of the one signature of an operation: the stub (the
# python entry point), the typing function of its overload and the
# implementation that function returns.  Every variant spells the signature
# out three times rather than have it assembled, because numba compares the
# typing function with the implementation it returns - argument names and
# annotations included - and because the argument an operation has of its own
# is the one thing a caller of the kernel reads the signature for.


def _kernel_none(name: str, driver: Callable[..., Any], resolve: _PythonPrologue, prepare: _TypingPrologue) -> _Kernel:
    """The kernel of an operation that takes nothing of its own."""

    def stub(group_idx: Any, a: Any, size: int = -1, fill_value: float | None = None) -> Any:
        group_idx, a, out_dtype, aux_dtype, fill = resolve(group_idx, a, size, fill_value)
        return driver(group_idx, a, size, out_dtype, aux_dtype, fill)

    def typing(group_idx: Any, a: Any, size: int = -1, fill_value: float | None = None) -> Any:
        out_dtype, aux_dtype, default_fill, message, is_bool, is_exact, low, high, high_exclusive = prepare(
            group_idx, a, fill_value
        )

        def impl(group_idx: Any, a: Any, size: int = -1, fill_value: float | None = None) -> Any:
            _require_inputs(group_idx, a)
            if fill_value is None:
                return driver(group_idx, a, size, out_dtype, aux_dtype, default_fill)
            else:
                # the fill value has to fit the output (numba prunes this
                # branch away when the fill value is a None, which is the
                # branch above)
                _require_fittable(fill_value, is_bool, is_exact, low, high, high_exclusive, message)
                return driver(group_idx, a, size, out_dtype, aux_dtype, fill_value)

        impl.__qualname__ = f"{name}_impl"
        return impl

    return stub, typing


def _kernel_ddof(name: str, driver: Callable[..., Any], resolve: _PythonPrologue, prepare: _TypingPrologue) -> _Kernel:
    """The kernel of an operation that takes the ``ddof`` of the variance."""

    def stub(group_idx: Any, a: Any, ddof: int = 0, size: int = -1, fill_value: float | None = None) -> Any:
        group_idx, a, out_dtype, aux_dtype, fill = resolve(group_idx, a, size, fill_value)
        return driver(group_idx, a, size, out_dtype, aux_dtype, fill, ddof)

    def typing(group_idx: Any, a: Any, ddof: int = 0, size: int = -1, fill_value: float | None = None) -> Any:
        out_dtype, aux_dtype, default_fill, message, is_bool, is_exact, low, high, high_exclusive = prepare(
            group_idx, a, fill_value
        )

        def impl(group_idx: Any, a: Any, ddof: int = 0, size: int = -1, fill_value: float | None = None) -> Any:
            _require_inputs(group_idx, a)
            if fill_value is None:
                return driver(group_idx, a, size, out_dtype, aux_dtype, default_fill, ddof)
            else:
                _require_fittable(fill_value, is_bool, is_exact, low, high, high_exclusive, message)
                return driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, ddof)

        impl.__qualname__ = f"{name}_impl"
        return impl

    return stub, typing


def _kernel_q(name: str, driver: Callable[..., Any], resolve: _PythonPrologue, prepare: _TypingPrologue) -> _Kernel:
    """The kernel of an operation that takes a quantile."""

    def stub(group_idx: Any, a: Any, q: float = 0.5, size: int = -1, fill_value: float | None = None) -> Any:
        group_idx, a, out_dtype, aux_dtype, fill = resolve(group_idx, a, size, fill_value)
        return driver(group_idx, a, size, out_dtype, aux_dtype, fill, q)

    def typing(group_idx: Any, a: Any, q: float = 0.5, size: int = -1, fill_value: float | None = None) -> Any:
        out_dtype, aux_dtype, default_fill, message, is_bool, is_exact, low, high, high_exclusive = prepare(
            group_idx, a, fill_value
        )

        def impl(group_idx: Any, a: Any, q: float = 0.5, size: int = -1, fill_value: float | None = None) -> Any:
            _require_inputs(group_idx, a)
            if fill_value is None:
                return driver(group_idx, a, size, out_dtype, aux_dtype, default_fill, q)
            else:
                _require_fittable(fill_value, is_bool, is_exact, low, high, high_exclusive, message)
                return driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, q)

        impl.__qualname__ = f"{name}_impl"
        return impl

    return stub, typing


def _kernel_dx(name: str, driver: Callable[..., Any], resolve: _PythonPrologue, prepare: _TypingPrologue) -> _Kernel:
    """The kernel of an operation that takes the sample spacing ``dx`` of the
    trapezoid.

    The default is the integer 1 and not the float 1.0: a float in the slot of
    the driver compiles the kernel of the trapezoid a few percent slower than
    the integer it used to carry - the nan variant measured ~3% slower with it.
    """

    def stub(group_idx: Any, a: Any, dx: float = 1, size: int = -1, fill_value: float | None = None) -> Any:
        group_idx, a, out_dtype, aux_dtype, fill = resolve(group_idx, a, size, fill_value)
        return driver(group_idx, a, size, out_dtype, aux_dtype, fill, dx)

    def typing(group_idx: Any, a: Any, dx: float = 1, size: int = -1, fill_value: float | None = None) -> Any:
        out_dtype, aux_dtype, default_fill, message, is_bool, is_exact, low, high, high_exclusive = prepare(
            group_idx, a, fill_value
        )

        def impl(group_idx: Any, a: Any, dx: float = 1, size: int = -1, fill_value: float | None = None) -> Any:
            _require_inputs(group_idx, a)
            if fill_value is None:
                return driver(group_idx, a, size, out_dtype, aux_dtype, default_fill, dx)
            else:
                _require_fittable(fill_value, is_bool, is_exact, low, high, high_exclusive, message)
                return driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, dx)

        impl.__qualname__ = f"{name}_impl"
        return impl

    return stub, typing


def _kernel_reverse(
    name: str, driver: Callable[..., Any], resolve: _PythonPrologue, prepare: _TypingPrologue
) -> _Kernel:
    """The kernel of an operation that takes the ``reverse`` flag of the sort."""

    def stub(group_idx: Any, a: Any, reverse: bool = False, size: int = -1, fill_value: float | None = None) -> Any:
        group_idx, a, out_dtype, aux_dtype, fill = resolve(group_idx, a, size, fill_value)
        return driver(group_idx, a, size, out_dtype, aux_dtype, fill, reverse)

    def typing(group_idx: Any, a: Any, reverse: bool = False, size: int = -1, fill_value: float | None = None) -> Any:
        out_dtype, aux_dtype, default_fill, message, is_bool, is_exact, low, high, high_exclusive = prepare(
            group_idx, a, fill_value
        )

        def impl(group_idx: Any, a: Any, reverse: bool = False, size: int = -1, fill_value: float | None = None) -> Any:
            _require_inputs(group_idx, a)
            if fill_value is None:
                return driver(group_idx, a, size, out_dtype, aux_dtype, default_fill, reverse)
            else:
                _require_fittable(fill_value, is_bool, is_exact, low, high, high_exclusive, message)
                return driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, reverse)

        impl.__qualname__ = f"{name}_impl"
        return impl

    return stub, typing


_KERNEL_BUILDERS: dict[Variant, Callable[[str, Callable[..., Any], _PythonPrologue, _TypingPrologue], _Kernel]] = {
    "none": _kernel_none,
    "ddof": _kernel_ddof,
    "q": _kernel_q,
    "dx": _kernel_dx,
    "reverse": _kernel_reverse,
}


def _make_stub(
    stub_name: str,
    driver_getter: Callable[[], Any],
    rule: Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: AScalar,
    doc: str | None,
    variant: Variant = "none",
    reject_fill: bool = False,
) -> Callable[..., Any]:
    """Create the kernel of one variant: the function of the name of its
    operation (``mean``, ``nanmean``) with the signature of that operation.

    The stub is one function with two ways in, and both of them end at the
    same compiled driver with the same arguments: from python it resolves the
    dtypes itself (``_python_prologue``) and calls the driver, from jitted code
    the ``@overload`` resolves the dtype rule at compile time
    (``_typing_prologue``) and returns the call.  Both of them first say no to
    the input the drivers were not written for, in the words of the same
    sentences - which is a fraction of what ``aggregate()`` asks, since a
    kernel has no axis to know, no indexed ``group_idx`` and no single value
    repeated over every label.

    ``fill_value=None`` asks for the function default, which - like the
    python-level DEFAULT_FILL_VALUE - depends on the output dtype (nan only
    fits inexact results); each path resolves it for itself, the jitted one in
    the typing function of the overload, the python one in its own prologue.

    The signature is the one of the operation: the argument it has of its own
    (the ``ddof`` of the variance, the ``q`` of the quantile, the ``dx`` of the
    trapezoid, the ``reverse`` of the sort) comes first, the ``size`` and the
    ``fill_value`` last - spelled out by the builder of the variant, which is
    also what gives the driver of that variant its signature.
    """

    resolve = _python_prologue(rule, fill_default, reject_fill, stub_name)
    prepare = _typing_prologue(rule, fill_default, reject_fill, stub_name)
    stub, typing = _KERNEL_BUILDERS[variant](stub_name, driver_getter(), resolve, prepare)

    stub.__name__ = stub.__qualname__ = stub_name
    stub.__module__ = __name__
    # the typing function and the implementation it returns carry the name of
    # the operation in their qualname - numba mangles the symbol of a
    # function from its module and qualname, and two of them that share it
    # would share the symbol
    typing.__name__ = f"_typing_{stub_name}"
    typing.__qualname__ = f"_typing_{stub_name}"
    typing.__module__ = __name__
    stub.__doc__ = (
        f"{doc or f'The {stub_name} of every group'} - the compiled kernel, "
        "callable from jitted code and from python with the same arguments."
    )

    # no jit_options cache here: the impl shim is compiled by numba's
    # overload machinery OUTSIDE the revisioned ``njit_stable`` namespace,
    # and its cache key does not reliably capture the identity of the
    # captured driver - stale impls of earlier refactors then keep being
    # served (observed as a nanfirst running plain-first machine code).
    # The shim itself compiles in microseconds, the expensive compilation
    # (the driver) stays cached through ``njit_stable``.
    # the overload speaks numba's protocol: it hands the typing function its
    # own types at compile time, and the function returns the implementation
    # it compiles - the two have to carry the same signature, which is why
    # the builders above write both of them down
    nb.extending.overload(stub)(typing)

    return stub


# ---------------------------------------------------------------------------
# the driver of a variant: the pass structure of ``build_kernel`` behind the
# signature of the operation.  The argument an operation has of its own keeps
# its name from the caller of the kernel to the update step of the operation,
# so nothing has to be looked up in a table of slots on the way.


def build_driver(name: str, passes: Any) -> Any:
    """The driver of an operation that takes nothing of its own."""

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
    ) -> Any:
        return passes(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, 0)

    return njit_stable(driver, name=f"{name}_driver")


def build_driver_ddof(name: str, passes: Any) -> Any:
    """The driver of an operation that takes the ``ddof`` of the variance."""

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        ddof: int,
    ) -> Any:
        return passes(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, ddof)

    return njit_stable(driver, name=f"{name}_driver")


def build_driver_dx(name: str, passes: Any) -> Any:
    """The driver of an operation that takes the sample spacing ``dx`` of the
    trapezoid."""

    def driver(
        group_idx: npt.NDArray[np.integer],
        a: npt.NDArray[np.number],
        size_arg: int,
        out_dtype: np.dtype,
        aux_dtype: np.dtype,
        fill_value: float,
        dx: float,
    ) -> Any:
        return passes(group_idx, a, size_arg, out_dtype, aux_dtype, fill_value, dx)

    return njit_stable(driver, name=f"{name}_driver")


# the pass structure knows the argument as ``extra``, so a driver builder
# exists for every name an operation uses for it - the two the composed
# kernels never take (``q`` and ``reverse``) are answered by the hand-written
# drivers of ``numpy_groupies.aggregate_numba.specials``
_DRIVER_BUILDERS: dict[Variant, Callable[[str, Any], Any]] = {
    "none": build_driver,
    "ddof": build_driver_ddof,
    "dx": build_driver_dx,
}


class KernelHandle(NamedTuple):
    """The compiled artifact of one operation variant.

    ``stub``    the kernel of the operation: the function it is exported
                under, by name, from the package - the same call from python
                and from jitted code
    ``driver``  the raw compiled driver - the python-level ``aggregate()``
                calls it with fully resolved arguments
    ``variant`` the argument the driver takes of its own, which is the one the
                caller of ``driver`` has to supply - ``aggregate_numba.ops``
                picks it out of the two that its own entry point takes
    ``forced_fill`` the seeding default of the output (or None when the
                output is seeded with the fill value itself)
    ``aux_widen``   whether the auxiliary dtype is the input dtype widened
                with float64 (the dispersion accumulators)
    ``nton``    whether the output has the input length (the cumulatives)
    ``fallback``    python-level callable for input dtypes the compiled
                driver cannot handle (complex medians go to the numpy
                implementation - numba cannot partition complex values)
    ``placeable``   which dtypes the compiled driver can take - None means
                the default: only complex input goes to the fallback
    ``post``    turns the result of the driver into the result of the caller
                (the collecting function assembles an object array of the
                groups out of the list the driver hands out)
    ``assembles``   the driver assembles the result instead of returning a
                plain array: the fill value is then the business of the
                post hook, not of the driver
    """

    name: str
    nans: bool
    stub: Callable[..., Any]
    # the driver returns the result array - or, for the collecting function, the
    # list of the arrays of the groups
    driver: Callable[..., AResult | list[AResult]]
    variant: Variant
    forced_fill: float | int | None
    aux_widen: bool
    nton: bool
    fallback: Callable[..., AResult] | None = None
    placeable: Callable[[np.dtype | None], bool] | None = None
    post: Callable[..., AResult] | None = None
    assembles: bool = False


class KernelPair(NamedTuple):
    """Plain and nan-variant handles of one operation.

    The four operations that do not skip nans have no nan variant - their
    ``nan_variant`` is None and the property raises there.
    """

    plain: KernelHandle
    nan_variant: KernelHandle | None

    @property
    def nan(self) -> KernelHandle:
        """The handle of the nan variant of the operation."""
        if self.nan_variant is None:
            raise ValueError(f"{self.plain.name} does not have a nan-version")
        return self.nan_variant


def build_kernel(
    name: str,
    *,
    # the building blocks are jitted - their state differs from operation to
    # operation, so their parameters stay open
    init: Callable[..., Any],
    update: Callable[..., None],
    rule: Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: AScalar = 0,
    forced_fill: float | None = None,
    seed: Callable[..., None] | None = None,
    record: Callable[..., None] | None = None,
    gate: Callable[..., None] | None = None,
    finish: Callable[..., Any] | None = None,
    untouched: Callable[..., npt.NDArray[np.bool_]] | None = None,
    update_nan: Callable[..., None] | None = None,
    nans: bool = True,
    variant: ComposedVariant = "none",
    aux_widen: bool = False,
    nton: bool = False,
    fallback: Callable[..., AResult] | None = None,
    placeable: Callable[[np.dtype | None], bool] | None = None,
    post: Callable[..., AResult] | None = None,
    assembles: bool = False,
    doc: str | None = None,
) -> KernelPair:
    """Compose the kernels of one operation from jitted building blocks.

    See the module docstring for the building-block signatures.  Both the
    plain and (unless ``nans=False``) the nan-guarded variant are built.  The
    nan variant uses ``update_nan`` when the operation provides one - a
    branchless select-form body that folds the nan handling into its stores
    so that LLVM keeps the loop vectorized - and otherwise falls back to the
    generic guard that skips nan values before they reach ``update``.

    ``variant`` names the argument the operation has of its own, if any: it
    decides the signature of the driver as well as the one of the kernel, so
    the two cannot drift apart.
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
    def _seed_none(st: Any, a: Any) -> None:
        pass

    def _record_none(st: Any, ri: int, i: int) -> None:
        pass

    def _gate_none(st: Any, n_values: int) -> None:
        pass

    def _finish_none(st: Any, g: int, fill_value: float, extra: Any) -> Any:
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

        def fill_untouched(st: Any, size: int, fill_value: float) -> None:
            # a custom fill value also belongs to the groups that never
            # received a value - their output was seeded with the forced
            # default of the operation
            if fill_value != forced_fill:
                un = untouched_c(st)
                for g in range(size):
                    if un[g]:
                        st.out[g] = fill_value

    else:

        def fill_untouched(st: Any, size: int, fill_value: float) -> None:
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
                    st: Any,
                    ri: int,
                    i: int,
                    val: float,
                    fill_value: float,
                    extra: Any,
                ) -> None:
                    if val == val:
                        update_c(st, ri, i, val, fill_value, extra)

                update_ = njit_stable(guarded_update, name=f"{op_name}_guarded_update")
        else:
            update_ = update_c

        def kernel(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            st: Any,
            fill_value: float,
            extra: Any,
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
                    raise ValueError(NEGATIVE_INDICES)
                if ri >= size:
                    raise ValueError(TOO_LARGE)
                update_(st, ri, i, a[i], fill_value, extra)
                record_c(st, ri, i)

        kernel = njit_stable(kernel, name=f"{op_name}_kernel")

        # the pass structure of the operation, behind the driver variant that
        # ``build_driver`` below builds around it: an operation with an
        # argument of its own costs the same pass structure as an operation
        # without one.  ``extra`` is that argument - the ddof of the variance,
        # the sample spacing of the trapezoid, the zero of an operation that
        # has nothing of its own.  The call of it is one call of a call, like
        # the building blocks it runs: a driver is reached once per aggregation.
        def passes(
            group_idx: npt.NDArray[np.integer],
            a: npt.NDArray[np.number],
            size_arg: int,
            out_dtype: np.dtype,
            aux_dtype: np.dtype,
            fill_value: float,
            extra: Any,
        ) -> Any:
            # numba's np.max reduction is several times faster than a scalar
            # loop here; negative indices slip through and are caught by the
            # bounds check inside the kernel below
            size = np.max(group_idx) + 1
            if size_arg >= 0:
                # an explicit output size may be larger than the largest
                # index, but never smaller
                if size > size_arg:
                    _raise_too_large_for_size(size_arg)
                size = size_arg
            st = init_c(size, len(group_idx), out_dtype, aux_dtype, a.dtype, fill_value, forced_fill)
            seed_c(st, a)
            kernel(group_idx, a, st, fill_value, extra, size)
            gate_c(st, len(group_idx))
            for g in range(size):
                st.out[g] = finish_c(st, g, fill_value, extra)
            fill_untouched(st, size, fill_value)
            return st.out

        passes = njit_stable(passes, name=f"{op_name}_passes")
        driver = _DRIVER_BUILDERS[variant](op_name, passes)

        stub = _make_stub(
            op_name,
            lambda: driver,
            rule,
            fill_default,
            doc,
            variant,
        )
        return KernelHandle(
            name=op_name,
            nans=nan_variant,
            stub=stub,
            driver=driver,
            variant=variant,
            forced_fill=forced_fill,
            aux_widen=aux_widen,
            nton=nton,
            fallback=fallback,
            placeable=placeable,
            post=post,
            assembles=assembles,
        )

    plain = build(False)
    nan_variant = build(True) if nans else None
    return KernelPair(plain, nan_variant)


def build_custom_kernel(
    name: str,
    *,
    driver_factory: Callable[[bool], Any],
    rule: Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: AScalar = 0,
    forced_fill: float | None = None,
    nans: bool = True,
    variant: Variant = "none",
    aux_widen: bool = False,
    nton: bool = False,
    fallback: Callable[..., AResult] | None = None,
    placeable: Callable[[np.dtype | None], bool] | None = None,
    post: Callable[..., AResult] | None = None,
    assembles: bool = False,
    reject_fill: bool = False,
    doc: str | None = None,
) -> KernelPair:
    """Compose the kernels of an operation with a hand-written driver.

    ``driver_factory(nans)`` returns the compiled driver, whose signature is
    the one of the ``variant`` - the same arguments the kernel of the operation
    is called with, behind the two arrays, the size, the dtype pair and the
    fill value.  Used by operations whose pass structure does not fit the
    accumulate/finish pattern (median groups the data with a counting sort
    before selecting, sort groups it before ordering).

    ``reject_fill`` says that the driver cannot place a fill value: the
    kernel then refuses one instead of ignoring it silently - a typing error
    to jitted code, a ValueError to python.
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
            variant,
            reject_fill,
        )
        return KernelHandle(
            name=stub_name_variant,
            nans=nan_variant,
            stub=stub,
            driver=driver,
            variant=variant,
            forced_fill=forced_fill,
            aux_widen=aux_widen,
            nton=nton,
            fallback=fallback,
            placeable=placeable,
            post=post,
            assembles=assembles,
        )

    plain = build(False)
    nan_variant = build(True) if nans else None
    return KernelPair(plain, nan_variant)
