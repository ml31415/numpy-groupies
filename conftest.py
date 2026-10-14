"""
pytest configuration to silently discard test items with invalid parameter combinations
See: https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496
"""

import inspect


def pytest_configure(config):
    config.addinivalue_line("markers", "deselect_if(func): function to deselect tests from parametrization")


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        for m in item.iter_markers("deselect_if"):
            func = m.kwargs["func"]
            # only pass the kwargs the deselect function actually accepts, so
            # one deselect can sit alongside another without either being
            # fed arguments it did not ask for
            params = item.callspec.params
            accepted = inspect.signature(func).parameters
            kwargs = {name: params[name] for name in accepted if name in params}
            if func(**kwargs):
                removed.append(item)
                break
        else:
            kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
