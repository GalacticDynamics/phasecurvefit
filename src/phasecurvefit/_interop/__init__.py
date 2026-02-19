"""Optional interoperability registrations.

Imports interop modules when corresponding optional dependencies are present.
"""

__all__: tuple[str, ...] = ()

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # pylint: disable=invalid-enum-extension
    """Optional dependency flags for localflowwalk interop."""

    UNXT = auto()


if OptDeps.UNXT.installed:  # pragma: no cover - optional path
    from . import interop_unxt  # noqa: F401
