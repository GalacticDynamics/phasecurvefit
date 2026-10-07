"""Optional-dependency flags (a leaf module: imports nothing from phasecurvefit)."""

__all__: tuple[str, ...] = ("OptDeps",)

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):  # pylint: disable=invalid-enum-extension
    """Optional dependency flags for phasecurvefit."""

    JAXKD = auto()
    UNXT = auto()
