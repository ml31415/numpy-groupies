"""The helpers the implementations of the package share.

The modules of this package hold what the backends have in common:

- ``nptools`` - the facts of numpy: the dtypes, the shapes, the fills and the
  checks the values and the labels have to pass.
- ``temporal`` - the policy of the ``datetime64`` and the ``timedelta64``
  values: what the reductions do with a ``NaT``, what they refuse, and how
  the results are turned back into the dtype they came in.
- ``typing`` - the types the dispatchers carry the values in.
- ``cache`` - where the compiled kernels of numba keep their artifacts, and
  how they are purged.

Nothing here is re-exported: a caller names the module it needs, so that the
helpers of one concern do not mix with the helpers of another.
"""

from __future__ import annotations
