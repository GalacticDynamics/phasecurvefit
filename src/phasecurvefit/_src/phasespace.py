"""Phase-space operations for localflowwalk.

This module provides functions for computing distances, directions, and
similarities in phase-space. All functions operate on **scalar components**
(single phase-space points). Use `jax.vmap` to vectorize operations over
multiple points.

**Cartesian Coordinate System**: All functions operate in Cartesian space.
Position and velocity dictionaries must use standard Cartesian coordinate
keys: ``"x"`` (required), ``"y"`` (optional), ``"z"`` (optional).

Phase-space data is represented as two dictionaries: one for positions and
one for velocities, where keys represent Cartesian coordinate components.

The fundamental data structure is simply two `dict[str, FLikeSz0]`:
- `position`: Maps Cartesian component names ("x", "y", "z") to scalar position values
- `velocity`: Maps Cartesian component names ("x", "y", "z") to scalar velocity values

Both dictionaries must have the same keys, with ``"x"`` at minimum.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp

Single point (scalar components):

>>> pos = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
>>> vel = {"x": jnp.array(0.1), "y": jnp.array(0.2), "z": jnp.array(0.3)}

For arrays of points, use vmap:

>>> pos_array = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([4.0, 5.0, 6.0])}
>>> vel_array = {"x": jnp.array([0.1, 0.2, 0.3]), "y": jnp.array([0.4, 0.5, 0.6])}
>>> # Use vmap to compute norms for all points:
>>> from phasecurvefit._src.phasespace import velocity_norm
>>> jax.vmap(velocity_norm)(vel_array)
Array([0.41231057, 0.53851646, 0.6708204 ], dtype=float32)

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


import equinox as eqx
import jax.numpy as jnp
import jax.tree as jtu
import plum

from .custom_types import FAny, FLikeSz0, ISz0, ISzN, ScalarComponents, VectorComponents

# -----------------------------------------------------------------------------
# Distance computations
# -----------------------------------------------------------------------------


@eqx.filter_jit
def _diff_sq(a: FAny, b: FAny, /) -> FAny:
    return jnp.square(jnp.subtract(a, b))


@plum.dispatch
def euclidean_distance(q_a: ScalarComponents, q_b: ScalarComponents, /) -> FLikeSz0:
    """Compute Euclidean distance between two position points in Cartesian space.

    This function operates on scalar components only (single points).
    Use `jax.vmap` to compute distances for arrays of points.

    Parameters
    ----------
    q_a, q_b : ScalarComponents
        Position dictionaries with scalar Cartesian components (keys: "x", "y", "z").

    Returns
    -------
    FLikeSz0
        The Euclidean distance.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> q_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    >>> q_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    >>> float(euclidean_distance(q_a, q_b))
    5.0

    """
    diffsq = jtu.map(_diff_sq, q_a, q_b)
    jnit = (0 * q_a["x"]) ** 2
    sumsq = jtu.reduce(jnp.add, diffsq, initializer=jnit)
    return jnp.sqrt(sumsq)


# -----------------------------------------------------------------------------
# Direction computations
# -----------------------------------------------------------------------------


@plum.dispatch
def unit_direction(q_a: ScalarComponents, q_b: ScalarComponents, /) -> ScalarComponents:
    """Compute unit direction vector from position a to b in Cartesian space.

    This function operates on scalar components only (single points).
    Use `jax.vmap` to compute directions for arrays of points.

    Parameters
    ----------
    q_a, q_b : ScalarComponents
        Position dictionaries with scalar Cartesian components (keys: "x", "y", "z").

    Returns
    -------
    ScalarComponents
        Dictionary of unit direction Cartesian components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> q_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    >>> q_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    >>> udir = unit_direction(q_a, q_b)
    >>> float(udir["x"]), float(udir["y"])
    (0.6..., 0.8...)

    """
    diff = jtu.map(jnp.subtract, q_b, q_a)  # b - a = direction from a to b
    norm = euclidean_distance(q_a, q_b)
    # When norm is 0, division by 0 will give nan; use jnp.where to handle this
    # by returning the zero vector when norm == 0
    safe_division = jnp.where(norm != 0, norm, 1)  # Avoid division by zero
    result = jtu.map(lambda d: d / safe_division, diff)
    # If norm was 0, zero-out the result
    return jtu.map(lambda r: jnp.where(norm != 0, r, r * 0), result)


# -----------------------------------------------------------------------------
# Velocity computations
# -----------------------------------------------------------------------------


@plum.dispatch
def velocity_norm(velocity: ScalarComponents, /) -> FLikeSz0:
    """Compute the norm of a velocity vector in Cartesian space.

    This function operates on scalar components only (single velocity vector).
    Use `jax.vmap` to compute norms for arrays of velocities.

    Parameters
    ----------
    velocity : ScalarComponents
        Velocity dictionary with scalar Cartesian components (keys: "x", "y", "z").

    Returns
    -------
    FLikeSz0
        The Euclidean norm of the velocity.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> vel = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    >>> float(velocity_norm(vel))
    5.0

    """
    vel_sq = jtu.map(jnp.square, velocity)
    # Use reduce with proper initializer to handle unit-valued components
    init = (0 * velocity["x"]) ** 2
    sumsq = jtu.reduce(jnp.add, vel_sq, initializer=init)
    return jnp.sqrt(sumsq)


@plum.dispatch
def unit_velocity(velocity: ScalarComponents, /) -> ScalarComponents:
    """Compute the unit velocity vector in Cartesian space.

    This function operates on scalar components only (single velocity vector).
    Use `jax.vmap` to compute unit velocities for arrays of velocities.

    Parameters
    ----------
    velocity : ScalarComponents
        Velocity dictionary with scalar Cartesian components (keys: "x", "y", "z").

    Returns
    -------
    ScalarComponents
        Dictionary of unit velocity Cartesian components.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> vel = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    >>> uvel = unit_velocity(vel)
    >>> float(uvel["x"]), float(uvel["y"])
    (0.6..., 0.8...)

    """
    norm = velocity_norm(velocity)
    # When norm is 0, division by 0 will give nan; use jnp.where to handle this
    safe_division = jnp.where(norm != 0, norm, 1)  # Avoid division by zero
    result = jtu.map(lambda v: v / safe_division, velocity)
    # If norm was 0, zero-out the result
    return jtu.map(lambda r: jnp.where(norm != 0, r, r * 0), result)


# -----------------------------------------------------------------------------
# Similarity computations
# -----------------------------------------------------------------------------


@plum.dispatch
def cosine_similarity(vec_a: ScalarComponents, vec_b: ScalarComponents, /) -> FLikeSz0:
    """Compute cosine similarity between two vectors in Cartesian space.

    The cosine similarity is defined as the dot product of the vectors.
    For unit vectors, this equals the cosine of the angle between them.

    This function operates on scalar components only (single vectors).
    Use `jax.vmap` to compute similarities for arrays of vectors.

    Parameters
    ----------
    vec_a, vec_b : ScalarComponents
        Vector dictionaries with scalar Cartesian components (keys: "x", "y", "z").

    Returns
    -------
    FLikeSz0
        The cosine similarity (dot product).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # Parallel vectors
    >>> a = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> b = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> float(cosine_similarity(a, b))
    1.0

    >>> # Orthogonal vectors
    >>> a = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    >>> b = {"x": jnp.array(0.0), "y": jnp.array(1.0)}
    >>> float(cosine_similarity(a, b))
    0.0

    """
    vec_a = unit_velocity(vec_a)
    vec_b = unit_velocity(vec_b)
    products = jtu.map(jnp.multiply, vec_a, vec_b)
    # Use reduce with proper initializer to handle unit-valued components
    init = 0 * products[next(iter(products))]
    return jtu.reduce(jnp.add, products, initializer=init)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def get_w_at(
    q: VectorComponents, p: VectorComponents, idx: int | ISz0 | ISzN, /
) -> tuple[ScalarComponents | VectorComponents, ScalarComponents | VectorComponents]:
    """Extract a phase-space point at the given index.

    This function uses standard phase-space notation where:
    - q = position (generalized Cartesian coordinates: "x", "y", "z")
    - p = momentum/velocity (generalized Cartesian momenta: "x", "y", "z")
    - w = (q, p) = full phase-space point

    This extracts a single point (scalar components) from arrays.
    For extracting multiple points, use `jax.vmap` or array indexing directly.

    Parameters
    ----------
    q : VectorComponents
        Position dictionary with 1D array Cartesian values of shape (N,).
    p : VectorComponents
        Velocity/momentum dictionary with 1D array Cartesian values of shape (N,).
    idx : int | Array
        Index or indices to extract. Can be:
        - int: Extract a single point (returns scalar arrays)
        - 0-d Array: Extract a single point (returns scalar arrays)
        - 1-d Array: Extract multiple points (returns 1D arrays)

    Returns
    -------
    tuple[dict[str, Array], dict[str, Array]]
        The (position, velocity) tuple at the given index/indices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> pos = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([4.0, 5.0, 6.0])}
    >>> vel = {"x": jnp.array([0.1, 0.2, 0.3]), "y": jnp.array([0.4, 0.5, 0.6])}

    Extract a single point:

    >>> q, p = get_w_at(pos, vel, 1)
    >>> float(q["x"]), float(p["y"])
    (2.0, 0.5)

    Extract multiple points:

    >>> q, p = get_w_at(pos, vel, jnp.array([0, 2]))
    >>> list(q["x"])
    [Array(1., dtype=float32), Array(3., dtype=float32)]

    """
    f = lambda v: v[idx]  # noqa: E731
    return jtu.map(f, q), jtu.map(f, p)
