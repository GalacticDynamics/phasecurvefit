"""Type aliases for phasecurvefit.

This module defines common type aliases used throughout the library.
"""

__all__: tuple[str, ...] = (
    "ArrayComponents",
    "FAny",
    "FLikeSz0",
    "ScalarComponents",
    "VectorComponents",
)


from collections.abc import Mapping
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
# A read-only `Mapping`, not a `dict`: `dict` is invariant in its value type,
# so a `dict[str, Array]` would not be a `dict[str, RLikeSzN]`.
ScalarComponents: TypeAlias = Mapping[str, RLikeSz0]  # noqa: UP040
"""Mapping of component names to scalar arrays (single phase-space point)."""

VectorComponents: TypeAlias = Mapping[str, RLikeSzN]  # noqa: UP040
"""Mapping of component names to 1D arrays (multiple phase-space points)."""

ArrayComponents: TypeAlias = Mapping[str, Array]  # noqa: UP040
"""Mapping of component names to JAX arrays: what a computation returns.

`VectorComponents` is what a function accepts, so it admits any array-like.
A function that builds its result with `jax.numpy` returns this instead.
"""
