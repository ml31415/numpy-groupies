"""The vocabulary of the package: the names, the aliases and the registry.

``registry`` holds what every implementation shares - and it must not import
numpy, because the pure python implementation is loaded before numpy is
known to be available.
"""

from dataclasses import dataclass
from typing import get_args

import numpy as np
import pytest

from numpy_groupies.registry import (
    FuncName,
    Registry,
    Resolved,
    aliasing_py,
    build_dispatch,
    custom_callable,
    funcs_common,
    resolve_name,
)
from numpy_groupies.utils_np import aliasing


def test_the_names_of_the_literal_type_are_the_names_of_the_alias_table():
    """The literal type of the func parameter names every function by name.

    A name the table does not hold would be one the type checker refuses
    without a reason, and one the table holds but the literal does not would
    be one it refuses wrongly.
    """
    names = set(get_args(FuncName))
    assert names == {name for name in aliasing if isinstance(name, str)}
    assert names == {name for name in aliasing_py if isinstance(name, str)}


def test_build_dispatch_resolves_every_alias():
    """Every alias of every known function is a key of the fused table.

    Naming a function is then one lookup, and a function the backend does not
    implement is answered by that table too: the key is present, holding the name
    and no implementation, rather than missing and sending the caller off to a
    second table to find out which of the two it was.
    """
    op_dict = {"sum": sum, "min": min}
    aliasing = {"sum": "sum", "add": "sum", "min": "min", "amin": "min", "nansum": "nansum"}

    dispatch = build_dispatch(op_dict, aliasing)

    assert set(dispatch) == set(aliasing)
    assert dispatch["add"] == ("sum", sum)
    assert dispatch["amin"] == ("min", min)
    assert dispatch["nansum"] == ("nansum", None)


def test_a_name_is_resolved_without_asking_its_func_for_a_hash():
    """A func that cannot be a key names no function - and is not looked up to find

    that out.  ``resolve_name`` asks the type first, so the lookup that could not
    be made is never the thing that appears to have failed.
    """
    assert resolve_name([np.sum]) is None
    assert resolve_name({"func": "sum"}) is None
    assert resolve_name("amin") == "min"
    assert resolve_name(np.amin) is None  # the names-only table holds no objects
    assert resolve_name(np.amin, aliasing) == "min"


def test_a_func_that_is_neither_a_name_nor_a_callable_is_said_so():
    for func in (42, [1, 2], {"a": 1}, np.array([1.0])):
        with pytest.raises(ValueError, match="neither a valid function string nor a callable object"):
            custom_callable(func)


def _registry():
    """A registry with two known functions and a factory for the caller's own."""
    op_dict = {"sum": sum, "min": min}
    aliasing = {"sum": "sum", "add": "sum", "min": "min", "nansum": "nansum"}

    def custom(func):
        # the factories of the backends tell a func that named nothing from
        # one that can be run
        callable_ = custom_callable(func)
        return Resolved(callable_, callable_)

    return Registry(op_dict, aliasing, custom)


def test_the_registry_answers_both_kinds_of_func_with_one_lookup():
    """A known function and a callable of the caller's own come out of one lookup.

    The registry holds the aliases of the known functions, and a func that names
    no known function is answered by the factory of the backend - so that the
    caller asks one question and gets the operation of the func it asked about.
    """
    registry = _registry()

    funcname, impl = registry["add"]
    assert (funcname, impl) == ("sum", sum)
    # a function the backend does not implement is a key of the table too
    assert registry["nansum"] == ("nansum", None)

    # a callable of the caller's own: the factory makes it the operation
    def my_func(values):
        return sum(values)

    funcname, impl = registry[my_func]
    assert funcname is my_func
    assert impl is my_func


def test_the_registry_answers_a_func_that_cannot_be_a_key():
    """A func the table cannot hold as a key names no known function either.

    A dict asks its keys for a hash, which an object that defines ``__eq__``
    without ``__hash__`` does not have - the registry answers it the same way
    it answers a callable.
    """

    @dataclass
    class Callable:
        def __call__(self, values):
            return sum(values)

    registry = _registry()
    func = Callable()
    with pytest.raises(TypeError):
        registry[func]
    funcname, impl = registry.custom(func)
    assert funcname is func
    assert impl is func


def test_a_func_that_named_nothing_is_said_so_by_the_factory():
    """The factory of the backend tells a func that named nothing from one that can be run."""
    registry = _registry()
    with pytest.raises(ValueError, match="neither a valid function string nor a callable object"):
        registry[42]


def test_the_functions_that_hold_a_none_match_the_registry():
    """The one function whose result holds a None is the object array.

    ``validation`` states which functions may be filled with a None - the
    object array of ``array`` and nothing else - while the registry of the
    fill values knows every function of the package.  The two have to agree,
    or a function would be filled with a None its result cannot hold.
    """
    from numpy_groupies import common, validation

    assert validation.NONE_RESULT_FUNCTIONS == frozenset({"array"})
    assert frozenset(common.default_fill_values) - validation.NONE_RESULT_FUNCTIONS


def test_every_function_of_the_package_has_an_entry_point_on_every_backend():
    """Every function is reachable by its name, on every backend.

    The entry points of a backend are the functions of ``funcs_common`` and
    their nan-variants - so the list has to name every function of the
    package, or a function is reachable only through the ``func`` argument.
    """
    from numpy_groupies import aggregate_nb, aggregate_np, aggregate_py
    from numpy_groupies.aggregate_numpy.ops import _op_dict

    for backend in (aggregate_np, aggregate_nb, aggregate_py):
        for name in _op_dict:
            assert hasattr(backend, name), f"{backend!r} has no entry point for {name}"


def test_the_list_of_the_functions_names_every_function_of_the_package():
    """The list the entry points are built from names every function.

    ``funcs_common`` is the list of the functions of the package - the
    nan-variants are added to it, so a function is named by the one it skips
    the nans of.
    """
    from numpy_groupies.aggregate_numpy.ops import _op_dict

    names = set(funcs_common)
    for name in _op_dict:
        assert name in names or name.removeprefix("nan") in names, name
    # the functions that are implemented by the numba backend alone are named
    # as well - they are functions of the package, too
    assert {"cumprod", "cummax", "cummin"} <= names


def test_the_package_carries_the_functions_of_the_best_implementation():
    """Every function of the package is a name of the package itself.

    ``npg.mean(group_idx, a)`` is the mean of every group, computed by the
    best implementation available - and every backend has a name of its own,
    so that the implementation can be chosen explicitly.
    """
    import numpy_groupies as npg
    from numpy_groupies import aggregate_np, aggregate_py

    for name in funcs_common:
        assert hasattr(npg, name), f"the package has no {name}"
        assert hasattr(aggregate_np, name), f"the numpy backend has no {name}"
        assert hasattr(aggregate_py, name), f"the pure python backend has no {name}"
