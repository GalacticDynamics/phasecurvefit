"""Type aliases for localflowwalk.

This module defines common type aliases used throughout the library.
"""

__all__: tuple[str, ...] = (
    "FAny",
    "FLikeSz0",
    "FLikeSzN",
    "ILikeSzN",
    "FSzD",
    "FSzND",
    "ScalarComponents",
    "VectorComponents",
)


from typing import TypeAlias

from jaxtyping import Array, ArrayLike, Float, Int

# Any-shaped float array
FAny: TypeAlias = Float[Array, "..."]  # noqa: UP040

# Scalar float type
ISz0: TypeAlias = Int[Array, ""]  # noqa: UP040
FSz0: TypeAlias = Float[Array, ""]  # noqa: UP040
FLikeSz0: TypeAlias = Float[ArrayLike, " "]  # noqa: UP040

# 1D array of ints
ISzN: TypeAlias = Int[Array, " N"]  # noqa: UP040
ILikeSzN: TypeAlias = Int[ArrayLike, " N"]  # noqa: UP040

# 1D array of floats
FLikeSzN: TypeAlias = Float[ArrayLike, " N"]  # noqa: UP040
FSzN: TypeAlias = Float[Array, " N"]  # noqa: UP040
FSzD: TypeAlias = Float[Array, " D"]  # noqa: UP040
FSzND: TypeAlias = Float[Array, " N D"]  # noqa: UP040

# Type aliases for component dictionaries
ScalarComponents: TypeAlias = dict[str, FLikeSz0]  # noqa: UP040
"""dict of component names to scalar arrays (single phase-space point)."""

VectorComponents: TypeAlias = dict[str, FLikeSzN]  # noqa: UP040
"""dict of component names to 1D arrays (multiple phase-space points)."""
