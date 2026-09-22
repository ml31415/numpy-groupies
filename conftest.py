"""
pytest configuration to silently discard test items with invalid parameter combinations
See: https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "deselect_if(func): function to deselect tests from parametrization")


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        m = item.get_closest_marker("deselect_if")
        if m:
            func = m.kwargs["func"]
            if func(**item.callspec.params):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
