"""Phase-space operations for localflowwalk.

This module provides functions for computing distances, directions, and
similarities in phase-space. These are the low-level building blocks used
by the phase-flow walk algorithm.

Phase-space data is represented as dictionaries mapping component names
to JAX arrays.

Examples
--------
>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf

Compute distance between two points:

>>> pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
>>> pos_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
>>> pcf.w.euclidean_distance(pos_a, pos_b)
Array(5., dtype=float32, weak_type=True)

"""

__all__: tuple[str, ...] = (
    # Distance functions
    "euclidean_distance",
    # Direction functions
    "unit_direction",
    # Velocity functions
    "velocity_norm",
    "unit_velocity",
    # Similarity functions
    "cosine_similarity",
    # Utility functions
    "get_w_at",
)

from ._src.phasespace import (
    cosine_similarity,
    euclidean_distance,
    get_w_at,
    unit_direction,
    unit_velocity,
    velocity_norm,
)
