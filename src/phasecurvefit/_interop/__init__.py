"""Optional interoperability registrations.

Imports interop modules when corresponding optional dependencies are present.
"""

__all__: tuple[str, ...] = ()

from phasecurvefit._src.optional_deps import OptDeps

if OptDeps.UNXT.installed:  # pragma: no cover - optional path
    from . import interop_unxt  # noqa: F401
