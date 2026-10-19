"""pytest configuration: command line options for the test suite.

Only command line options live here - fixtures and collection hooks are in
``numpy_groupies/tests/conftest.py``. ``pytest_addoption`` is only picked
up from the rootdir conftest, not from subdirectory conftests.
"""

from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--seed",
        type=int,
        default=42,
        help="Seed for the session-wide random number generator (rng fixture).",
    )
