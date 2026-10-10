"""Type aliases for phasecurvefit.

This module defines common type aliases used throughout the library.
"""

__all__: tuple[str, ...] = (
    "FAny",
    "FLikeSz0",
    "ScalarComponents",
    "VectorComponents",
)


from typing import TypeAlias

from jaxtyping import Array, ArrayLike, Bool, Float, Int, Real

# Any-shaped float array
FAny: TypeAlias = Float[Array, "..."]  # noqa: UP040

# Scalar float type
ISz0: TypeAlias = Int[Array, ""]  # noqa: UP040
FSz0: TypeAlias = Float[Array, ""]  # noqa: UP040
FLikeSz0: TypeAlias = Float[ArrayLike, " "]  # noqa: UP040

# 1D array of ints
ISzN: TypeAlias = Int[Array, " N"]  # noqa: UP040

# 1D array of floats
FSzN: TypeAlias = Float[Array, " N"]  # noqa: UP040

RSz0: TypeAlias = Real[Array, ""]  # noqa: UP040
RSzN: TypeAlias = Real[Array, " N"]  # noqa: UP040
RLikeSzN: TypeAlias = Real[ArrayLike, " N"]  # noqa: UP040
RLikeSz0: TypeAlias = Real[ArrayLike, " "]  # noqa: UP040

BSzN: TypeAlias = Bool[Array, " N"]  # noqa: UP040

# Type aliases for component dictionaries
ScalarComponents: TypeAlias = dict[str, RLikeSz0]  # noqa: UP040
"""dict of component names to scalar arrays (single phase-space point)."""

VectorComponents: TypeAlias = dict[str, RLikeSzN]  # noqa: UP040
"""dict of component names to 1D arrays (multiple phase-space points)."""
