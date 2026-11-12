"""The vocabulary of the package: the names, the aliases and the registry.

``registry`` holds the names of the functions and the dispatch of a backend -
and it must not import
numpy, because the pure python implementation is loaded before numpy is
known to be available.
"""

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np
import pytest

import numpy_groupies
import numpy_groupies as npg
from numpy_groupies import functions, validation
from numpy_groupies.driver import aliasing
from numpy_groupies.npg_numpy.ops import _op_dict
from numpy_groupies.registry import (
    FuncName,
    Registry,
    Resolved,
    aliasing_py,
    build_dispatch,
    custom_callable,
    funcs_common,
    function_names,
    resolve_name,
)
from numpy_groupies.tests.helpers import (
    aliasing_of,
    assert_same_result,
    impl_name,
    is_supported,
)


def test_names_of_literal_type_are_names_of_alias_table():
    """The literal type of the func parameter names every function by name.

    A name the table does not hold would be one the type checker refuses
    without a reason, and one the table holds but the literal does not would
    be one it refuses wrongly.
    """
    names = set(get_args(FuncName))
    assert names == {name for name in aliasing if isinstance(name, str)}
    # the table without the numpy functions names the same functions
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
    assert dispatch["add"] == Resolved("sum", sum, None)
    assert dispatch["amin"] == Resolved("min", min, None)
    assert dispatch["nansum"] == Resolved("nansum", None, None)


def test_name_is_resolved_without_asking_func_for_hash():
    """A func that cannot be a key names no function - and is not looked up to find

    that out.  ``resolve_name`` asks the type first, so the lookup that could not
    be made is never the thing that appears to have failed.
    """
    assert resolve_name([np.sum]) is None
    assert resolve_name({"func": "sum"}) is None
    assert resolve_name("amin") == "min"
    assert resolve_name(np.amin) is None  # the names-only table holds no objects
    assert resolve_name(np.amin, aliasing) == "min"


def test_func_neither_name_nor_callable_is_reported():
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


def test_registry_answers_both_kinds_of_func_with_one_lookup():
    """A known function and a callable of the caller's own come out of one lookup.

    The registry holds the aliases of the known functions, and a func that names
    no known function is answered by the factory of the backend - so that the
    caller asks one question and gets the operation of the func it asked about.
    """
    registry = _registry()

    funcname, impl, temporal_op = registry["add"]
    assert (funcname, impl) == ("sum", sum)
    assert temporal_op is None
    # a function the backend does not implement is a key of the table too
    assert registry["nansum"] == Resolved("nansum", None, None)

    # a callable of the caller's own: the factory makes it the operation
    def my_func(values):
        return sum(values)

    funcname, impl, temporal_op = registry[my_func]
    assert funcname is my_func
    assert impl is my_func
    assert temporal_op is None


def test_registry_answers_unhashable_func():
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
    funcname, impl, temporal_op = registry.custom(func)
    assert funcname is func
    assert impl is func
    assert temporal_op is None


def test_func_named_nothing_is_reported_by_factory():
    """The factory of the backend tells a func that named nothing from one that can be run."""
    registry = _registry()
    with pytest.raises(ValueError, match="neither a valid function string nor a callable object"):
        registry[42]


def test_functions_holding_none_match_registry():
    """The one function whose result holds a None is the object array.

    ``validation`` states which functions may be filled with a None - the
    object array of ``array`` and nothing else - while the registry of the
    fill values knows every function of the package.  The two have to agree,
    or a function would be filled with a None its result cannot hold.
    """

    assert validation.NONE_RESULT_FUNCTIONS == frozenset({"array"})
    assert frozenset(functions.default_fill_values) - validation.NONE_RESULT_FUNCTIONS


def test_every_function_of_package_has_entry_point_on_every_backend(impl):
    """Every function is reachable by its name, on every backend.

    The entry points of a backend are the functions of ``funcs_common`` and
    their nan-variants - so the list has to name every function of the
    package, or a function is reachable only through the ``func`` argument.
    """

    for name in _op_dict:
        assert hasattr(impl._module, name), f"the {impl_name(impl)} backend has no entry point for {name}"


def _the_support_matrix_of_the_docs():
    """the table of the functions of the docs, as the backends and the names.

    The table of ``docs/functions.md`` names every function of the package
    and the backend that answers it - and a function that a backend has
    learned since the table was written is a hole in the table.
    """
    docs = Path(__file__).parents[2] / "docs" / "functions.md"
    if not docs.exists():  # pragma: no cover - the docs are not in the wheel
        pytest.skip("the docs are not there")
    columns = []
    rows = []
    for line in docs.read_text().splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|\s*(x|-)\s*\|\s*(x|-)\s*\|\s*(x|-)\s*\|", line)
        if match is None:
            if not columns:
                # the head of the table names the backends in its columns
                head = re.match(r"\|\s*Function\s*\|(.+)\|", line)
                if head and "numba" in head.group(1):
                    columns = [name.strip() for name in head.group(1).split("|")]
            continue
        names, numpy, numba, purepy = match.groups()
        if "…" in names or "variants" in names:
            # the rows of the nan-variants and of the custom callables name
            # no function of their own
            continue
        rows.append((tuple(n.strip() for n in names.split(",")), numpy, numba, purepy))
    assert rows, "the table of the supported functions is not in the docs"
    return columns, rows


def test_support_matrix_of_docs_agrees_with_backends(impl):
    """The table of the docs says what every backend answers - and it is right.

    A function that a backend has learned since the table was written is a
    hole in the table: the docs would send the reader to a backend that
    refuses the function.
    """
    columns, rows = _the_support_matrix_of_the_docs()
    # the columns of the table name the backends
    columns_of_the_backends = {"numpy": "numpy", "numba": "numba", "purepy": "pure python"}
    name_of_the_backend = impl_name(impl)
    column = columns.index(columns_of_the_backends[name_of_the_backend])

    for names, *marks in rows:
        for name, mark in zip(names, marks):
            supported = is_supported(impl, name)
            assert supported == (marks[column] == "x"), f"{name} and the {name_of_the_backend} backend"


def test_list_of_functions_names_every_function_of_package():
    """The list the entry points are built from names every function.

    ``funcs_common`` is the list of the functions of the package - the
    nan-variants are added to it, so a function is named by the one it skips
    the nans of.
    """

    names = set(funcs_common)
    for name in _op_dict:
        assert name in names or name.removeprefix("nan") in names, name
    # the functions that are implemented by the numba backend alone are named
    # as well - they are functions of the package, too
    assert {"cumprod", "cummax", "cummin"} <= names


def test_package_carries_functions_of_best_implementation(impl):
    """Every function of the package is a name of the package itself.

    ``npg.mean(group_idx, a)`` is the mean of every group, computed by the
    best implementation available - and every backend has a name of its own,
    so that the implementation can be chosen explicitly.
    """

    for name in funcs_common:
        assert hasattr(npg, name), f"the package has no {name}"
        assert hasattr(impl._module, name), f"the {impl_name(impl)} backend has no {name}"


def test_every_function_of_package_is_declared_for_type_checker():
    """The names the package hands out are the names the type checker knows.

    The functions of the package are built at runtime, out of the table of
    the entry points - and the type checker cannot see that table.  So the
    names are declared for it in the package itself, and the two have to be
    the same: a function the type checker does not know is one the caller
    cannot use without an error.
    """
    tree = ast.parse(inspect.getsource(numpy_groupies))
    declared = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.annotation, ast.Name)
        and node.annotation.id == "EntryPoint"
    }

    assert declared == set(function_names())


def test_package_carries_every_function_of_registry(impl):
    """Every function of the registry is a name of the package itself.

    The nan variants are functions of the package, too - they are named by the
    function they skip the nans of, and the entry points are built out of the
    names of the registry.
    """

    for name in function_names():
        assert hasattr(npg, name), f"the package has no {name}"
        assert hasattr(impl._module, name), f"the {impl_name(impl)} backend has no {name}"


@pytest.mark.parametrize("name", sorted(function_names()))
def test_entry_point_of_package_answers_as_function(name):
    """``npg.mean(group_idx, a)`` is ``npg.aggregate(group_idx, a, func="mean")``.

    The entry points are built at runtime, out of the table of the entry
    points of the backend - so the name in the package and the function of the
    aggregation have to be one and the same.
    """

    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    got = getattr(npg, name)(group_idx, a, size=2)
    want = npg.aggregate(group_idx, a, func=name, size=2)
    assert_same_result(got, want, name)


# the arguments a function has of its own - they have to reach the function
# through the entry point of the package
_ENTRY_POINT_ARGUMENTS = [
    ("var", {"ddof": 1}),
    ("std", {"ddof": 1}),
    ("quantile", {"q": 0.25, "method": "lower"}),
    ("percentile", {"q": 25.0}),
    ("trapezoid", {"dx": 2.0}),
    ("sort", {"reverse": True}),
    ("topk", {"k": 2, "largest": False}),
]


@pytest.mark.parametrize(("name", "kwargs"), _ENTRY_POINT_ARGUMENTS)
def test_entry_point_passes_arguments_of_function(name, kwargs):
    """The arguments of the function reach it through the entry point.

    The entry point of the package takes the arguments of its function - the
    ``ddof`` of the variance, the ``q`` of the quantile - and hands them on.
    """

    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    got = getattr(npg, name)(group_idx, a, size=2, **kwargs)
    want = npg.aggregate(group_idx, a, func=name, size=2, **kwargs)
    assert_same_result(got, want, name)


@pytest.mark.parametrize("alias", sorted(alias for alias in aliasing if isinstance(alias, str)))
def test_every_alias_answers_like_named_function(impl, alias):
    """Every alias of the package is another name of the function it names.

    The aliases are the names numpy knows - ``add`` for the sum, ``amin``
    for the minimum, ``count`` for the length - and a caller who writes one
    of them expects the answer of the function behind it.
    """
    table = aliasing_of(impl)
    if alias not in table:
        pytest.skip("the backend does not know the alias")
    canonical = table[alias]
    group_idx = np.array([0, 0, 1, 1, 1])
    a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    got = impl.aggregate(group_idx, a, func=alias, size=2)
    want = impl.aggregate(group_idx, a, func=canonical, size=2)
    assert_same_result(got, want, canonical)
