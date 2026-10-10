"""Optional interoperability registrations.

Imports interop modules when corresponding optional dependencies are present.
"""

__all__: tuple[str, ...] = ()

from importlib.util import find_spec

if find_spec("unxt") is not None:  # pragma: no cover - optional path
    from . import interop_unxt  # noqa: F401
