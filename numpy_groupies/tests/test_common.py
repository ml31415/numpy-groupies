"""The vocabulary of the package: the names, the aliases and the registry.

``common`` holds what every implementation shares - and it must not import
numpy, because the pure python implementation is loaded before numpy is
known to be available.
"""

from dataclasses import dataclass
from typing import get_args

import numpy as np
import pytest

from numpy_groupies.common import (
    FuncName,
    Registry,
    Resolved,
    aliasing_py,
    build_dispatch,
    custom_callable,
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
