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
the group loop, bounds checks and pass structure, and exposes it as a function
callable from jitted code and from python alike (the kernels of
``numpy_groupies.aggregate_numba``).  The python-level ``aggregate()`` calls
the same driver with fully resolved arguments, so there is exactly one
compiled artifact per operation.  Unused
building blocks are replaced by synthesized no-ops - the driver always runs
every stage, and LLVM folds the empty calls away.
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple, cast

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
# The words of the complaints live in ``numpy_groupies.common``: the
# python-level call and the kernels say the same things about the same
# faults.  What is left here is what only a kernel has to say - the flatness
# of its input and the numeric dtypes of its values.
_FLAT_INPUT = "the kernels take flat one-dimensional input"
_SCALAR_INPUT = "the kernels take one value per label - a scalar is not supported"
_NUMERIC_VALUES = "the kernels take numeric values"


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
    them, and the predicates of ``numpy_groupies.common`` ask the
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
            raise ValueError(TOO_LARGE_SIZE_PREFIX + str(size_arg))
        size = size_arg
    return size


# an argument an operation has of its own: the name it carries in the
# signature, the annotation, the default and the slot of the driver the value
# goes to ("ddof" or "q" - the two the driver knows)
Extra = tuple[tuple[str, str, Any, str], ...]

# the marker for an argument without a default - None is a default of its own
_NO_DEFAULT = object()


def _make_stub(
    stub_name: str,
    driver_getter: Callable[[], Any],
    rule: Callable[[nb.types.Type], tuple[np.dtype, np.dtype]],
    fill_default: AScalar,
    doc: str | None,
    extras: Extra = (),
    reject_fill: bool = False,
) -> Callable[..., Any]:
    """Create the kernel of one variant: the function of the name of its
    operation (``mean``, ``nanmean``) with the signature of that operation.

    The stub is one function with two ways in, and both of them end at the
    same compiled driver with the same arguments: from python it resolves the
    dtypes itself and calls the driver, from jitted code the ``@overload``
    resolves the dtype rule at compile time and returns the call.  Both of them
    first say no to the input the drivers were not written for, in the words of
    the same sentences - which is a fraction of what ``aggregate()`` asks, since
    a kernel has no axis to know, no indexed ``group_idx`` and no single value
    repeated over every label.

    ``fill_value=None`` asks for the function default, which - like the
    python-level DEFAULT_FILL_VALUE - depends on the output dtype (nan only
    fits inexact results); each path resolves it for itself, the jitted one
    in the overload builder, the python one in its own body.

    The signature is the one of the operation: the arguments it has of its own
    (the ``ddof`` of the variance, the ``q`` of the quantile, the ``dx`` of
    the trapezoid, the ``reverse`` of the sort) come first, the ``size`` and
    the ``fill_value`` last.  Numba compares the typing function of an
    overload with the implementation it returns - argument names and
    annotations included - so the three functions that carry the signature are
    spelled out from one description of it.
    """

    # the arguments of the signature, in order: the two that every operation
    # takes, the ones it has of its own, and the two that come last
    arguments: list[tuple[str, str, Any, str | None]] = [
        ("group_idx", "Any", _NO_DEFAULT, None),
        ("a", "Any", _NO_DEFAULT, None),
    ]
    arguments += [(name, annotation, default, slot) for name, annotation, default, slot in extras]
    arguments += [("size", "int", -1, None), ("fill_value", "float | None", None, None)]
    signature = ", ".join(
        name + ": " + annotation + ("" if default is _NO_DEFAULT else f" = {default!r}")
        for name, annotation, default, _ in arguments
    )
    # the two slots of the driver that carry an argument of its own: the
    # value that goes into each of them, or the default of the slot
    slot_values = {slot: next((name for name, _, _, s in extras if s == slot), None) or "" for slot in ("ddof", "q")}
    driver_args = ", ".join(
        slot_values[slot] if slot_values[slot] else repr(default) for slot, default in (("ddof", 0), ("q", 0.5))
    )
    # the driver of this operation cannot place a fill value - saying so
    # beats ignoring the argument silently.  Both calls of the kernel say it
    # in the words of the same sentence, each in the error of its own world.
    refusal = f"The {stub_name} kernel has no fill_value - an absent group holds an empty sequence"
    python_fill_check = "    if fill_value is not None:\n        raise ValueError(_refusal)\n" if reject_fill else ""
    typing_fill_check = (
        "        if fill_value is not None:\n            raise nb.TypingError(_refusal)\n" if reject_fill else ""
    )

    # the two value checks a call of a kernel cannot be trusted to skip: an
    # empty group_idx leaves the driver with nothing to reduce, and a short ``a``
    # has it read a value past the end of the array, which numba never catches.
    # Both entry points of every kernel run them - the body of the stub and the
    # compiled implementation of its overload - from this one spelling, each with
    # the indentation of the function it lands in.  The driver runs neither:
    # aggregate() has validated by hand on its way in, and validation of the same
    # call twice is what the fast path of the wrapper exists to avoid.
    def value_checks(indent: str) -> str:
        return (
            f"{indent}if len(group_idx) == 0:\n"
            f"{indent}    raise ValueError(_empty_index)\n"
            f"{indent}if len(a) != len(group_idx):\n"
            f"{indent}    raise ValueError(_length_mismatch)\n"
        )

    # The three functions with the one signature: the stub, the typing function
    # of the overload and the implementation it returns - built from one
    # source, so that they cannot drift apart.  The stub is the python entry
    # point and does by hand, once per call, what the overload does by rule
    # once per compiled signature: the two dtype resolutions are the same
    # rule, read from the numba type of the values on one side and from the
    # numpy dtype behind ``_array_type`` on the other.
    #
    # The fill value is checked in both worlds as well: the python call hands
    # it to numpy (the very function the python-level call uses), the jitted
    # one cannot call numpy and so runs the same rules as compiled code -
    # the output dtype decides them at compile time.
    source = f"""
def _stub({signature}):
    group_idx, a = _kernel_inputs(group_idx, a)
{python_fill_check}{value_checks("    ")}    out_dtype, aux_dtype = _rule(_array_type(a.dtype))
    if fill_value is None:
        fill_value = _resolve_default_fill(_fill_default, out_dtype)
    else:
        fill_value = _check_fill_value(fill_value, out_dtype)
    return _driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, {driver_args})


def _typing({signature}):
    if getattr(group_idx, "ndim", -1) != 1 or getattr(a, "ndim", -1) != 1:
        # the drivers read one label per value: a multidimensional group_idx
        # they walk by row and answer with a result that is quietly wrong
        raise nb.TypingError(_flat_input)
    if not is_integer_dtype(_numpy_dtype(getattr(group_idx, "dtype", None))):
        # an index numba does not know as an index: a float one is rounded by
        # the caller, a bool one is not an index at all
        raise nb.TypingError(_integer_index)
    if not is_numeric_dtype(_numpy_dtype(getattr(a, "dtype", None))):
        raise nb.TypingError(_numeric_values)
    a_type = a
    out_dtype, aux_dtype = _rule(a_type)
    default_fill = _resolve_default_fill(_fill_default, out_dtype)
    fill_not_convertible = _fill_not_convertible.format(dtype=out_dtype.type.__name__)
    if not _reject_fill and not _fill_type_ok(fill_value):
        # a fill value that is not a number cannot be put into any output
        raise nb.TypingError(fill_not_convertible)
    # the rules of the fill value, as the compiled code sees them: a boolean
    # output takes the two booleans, an integer output takes an exact number
    # of its own range, and the inexact ones take anything
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

    def _impl({signature}):
{typing_fill_check}{value_checks("        ")}        if fill_value is None:
            return _driver(group_idx, a, size, out_dtype, aux_dtype, default_fill, {driver_args})
        else:
            # the fill value has to fit the output: a boolean output takes the
            # two booleans, an integer output an exact number of its own range
            # (numba prunes this branch away when the fill value is a None,
            # which is the branch above)
            if fill_is_bool:
                if fill_value != 0 and fill_value != 1:
                    raise ValueError(_fill_not_boolean)
            elif fill_is_exact:
                if not _fill_fits(fill_value, fill_min, fill_max, fill_max_exclusive):
                    raise ValueError(fill_not_convertible)
            return _driver(group_idx, a, size, out_dtype, aux_dtype, fill_value, {driver_args})

    return _impl
"""
    namespace: dict[str, Any] = {
        "Any": Any,
        "nb": nb,
        "np": np,
        "_rule": rule,
        "_fill_default": fill_default,
        "_driver": driver_getter(),
        "_resolve_default_fill": _resolve_default_fill,
        "_check_fill_value": check_fill_value_for_dtype,
        "_kernel_inputs": _kernel_inputs,
        "_array_type": _array_type,
        "is_integer_dtype": is_integer_dtype,
        "is_numeric_dtype": is_numeric_dtype,
        "_numpy_dtype": _numpy_dtype,
        "_fill_type_ok": _fill_type_ok,
        "_fill_fits": _fill_fits,
        "_reject_fill": reject_fill,
        "_flat_input": _FLAT_INPUT,
        "_numeric_values": _NUMERIC_VALUES,
        "_integer_index": NON_INTEGER_GROUP_IDX,
        "_empty_index": EMPTY_GROUP_IDX,
        "_length_mismatch": LENGTH_MISMATCH,
        "_fill_not_boolean": FILL_NOT_BOOLEAN,
        "_fill_not_convertible": FILL_NOT_CONVERTIBLE,
        "_refusal": refusal,
    }
    # the signature is the one of the operation, spelled out in one place
    exec(source, namespace)  # noqa: S102
    stub: Callable[..., Any] = namespace["_stub"]
    typing: Callable[..., Any] = namespace["_typing"]

    stub.__name__ = stub_name
    stub.__qualname__ = stub_name
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
    # they come from one source above
    nb.extending.overload(stub)(typing)

    return stub


class KernelHandle(NamedTuple):
    """The compiled artifact of one operation variant.

    ``stub``    the kernel of the operation: the function it is exported
                under, by name, from the package - the same call from python
                and from jitted code
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
    # the driver returns the result array - or, for the collecting function,
    # the list of the arrays of the groups
    driver: Callable[..., AResult | list[AResult]]
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
    extras: Extra = (),
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

    def _finish_none(st: Any, g: int, fill_value: float, ddof: int) -> Any:
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
            st: Any,
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
                    raise ValueError(NEGATIVE_INDICES)
                if ri >= size:
                    raise ValueError(TOO_LARGE)
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
            q: float,
        ) -> Any:
            # numba's np.max reduction is several times faster than a scalar
            # loop here; negative indices slip through and are caught by the
            # bounds check inside the kernel below
            size = np.max(group_idx) + 1
            if size_arg >= 0:
                # an explicit output size may be larger than the largest
                # index, but never smaller
                if size > size_arg:
                    raise ValueError(TOO_LARGE_SIZE_PREFIX + str(size_arg))
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
            extras,
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
    extras: Extra = (),
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

    ``driver_factory(nans)`` returns the compiled driver with the same
    signature as the composed ones - used by operations whose pass structure
    does not fit the accumulate/finish pattern (median groups the data with a
    counting sort before selecting, sort groups it before ordering).

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
            extras,
            reject_fill,
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
            placeable=placeable,
            post=post,
            assembles=assembles,
        )

    plain = build(False)
    nan_variant = build(True) if nans else None
    return KernelPair(plain, nan_variant)
