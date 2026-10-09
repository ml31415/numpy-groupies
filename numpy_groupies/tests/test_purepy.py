"""The pure python implementation must not need numpy.

Even importing the package has to work when numpy is missing - that is what
the import guard in ``__init__`` exists for, and it only holds if neither
``aggregate_purepy`` nor anything it pulls in touches numpy.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_BLOCKER = """\
import sys


class _BlockNumpy:
    def find_spec(self, name, path=None, target=None):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError(f"{name} is blocked for this test")
        return None


sys.meta_path.insert(0, _BlockNumpy())
"""

_SCRIPT = textwrap.dedent(
    """
    import math

    import numpy_groupies as npg

    try:
        import numpy
    except ImportError:
        pass
    else:
        raise AssertionError("numpy should have been blocked")

    assert npg.aggregate_py([0, 0, 1], [1, 2, 3], func="sum") == [3, 3]
    # aliases and the dispatch table work without numpy
    assert npg.aggregate_py([0, 0, 1], [1, 2, 3], func="add") == [3, 3]
    # a float group missing from group_idx takes the nan default, an integer one takes 0
    assert math.isnan(npg.aggregate_py([0], [1.5], func="min", size=2)[1])
    assert npg.aggregate_py([0], [1], func="min", size=2)[1] == 0
    # complex input refuses a non-complex output dtype
    try:
        npg.aggregate_py([0, 0], [1 + 2j, 3 + 4j], func="sum", dtype=float)
    except TypeError:
        pass
    else:
        raise AssertionError("non-complex dtype should have been rejected")
    print("ok")
    """
)


def test_import_and_aggregate_without_numpy(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(_BLOCKER)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=False,  # the returncode is asserted below, with stderr in the message
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
