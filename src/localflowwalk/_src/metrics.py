"""Distance metrics for phase-space walks.

This module provides pluggable distance metrics that control how the
`walk_local_flow` algorithm selects the next point in a trajectory.
"""

__all__: tuple[str, ...] = (
    "AbstractDistanceMetric",
    "AlignedMomentumDistanceMetric",
    "SpatialDistanceMetric",
    "FullPhaseSpaceDistanceMetric",
)

from abc import abstractmethod
from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import FLikeSz0, FSzN, VectorComponents
from .phasespace import (
    cosine_similarity,
    euclidean_distance,
    unit_direction,
    unit_velocity,
)

vec_euclidean_distance = jax.vmap(euclidean_distance, in_axes=(None, 0))
vec_unit_direction = jax.vmap(unit_direction, in_axes=(None, 0))
vec_cosine_similarity = jax.vmap(cosine_similarity, in_axes=(None, 0))


class AbstractDistanceMetric(eqx.Module):
    """Abstract base class for distance metrics in phase-space walks.

    A distance metric computes modified distances between a current point and
    all candidate next points, incorporating both spatial and velocity information.
    Different metrics can implement different weighting schemes or use different
    phase-space representations.

    Examples
    --------
    >>> import localflowwalk as lfw
    >>> metric = lfw.metrics.AlignedMomentumDistanceMetric()
    >>> # Use with walk_local_flow via metric parameter

    """

    __citation__: ClassVar[str | None]

    @abstractmethod
    def __call__(
        self,
        current_pos: VectorComponents,
        current_vel: VectorComponents,
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_scale: FLikeSz0,
    ) -> FSzN:
        """Compute distances from current point to all candidate points.

        Parameters
        ----------
        current_pos : dict[str, Array]
            Position of the current point (scalar components).
        current_vel : dict[str, Array]
            Velocity of the current point (scalar components).
        positions : dict[str, Array]
            Positions of all points (array components).
        velocities : dict[str, Array]
            Velocities of all points (array components).
        metric_scale : float
            Scale parameter for the metric (interpretation depends on metric type).

        Returns
        -------
        distances : Array
            Modified distances from current point to all candidate points.

        """
        raise NotImplementedError  # pragma: no cover


@final
class SpatialDistanceMetric(AbstractDistanceMetric):
    r"""Position-only distance metric.

    Computes pure Euclidean distance in position space, ignoring velocity
    information entirely. This reduces to standard nearest-neighbor search.

    $$
    d = d_0
    $$

    where $d_0$ is the Euclidean distance between positions. The `metric_scale`
    parameter is ignored.

    This metric is useful when:
    - Velocity information is unreliable or unavailable
    - Pure spatial proximity is the desired ordering criterion
    - Comparing against baseline nearest-neighbor approaches

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw
    >>> metric = lfw.metrics.SpatialDistanceMetric()
    >>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    >>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}
    >>> current_pos = {k: v[0] for k, v in pos.items()}
    >>> current_vel = {k: v[0] for k, v in vel.items()}
    >>> distances = metric(current_pos, current_vel, pos, vel, metric_scale=0.0)
    >>> distances.shape
    (3,)

    """

    __citation__: ClassVar = None

    def __call__(
        self,
        current_pos: VectorComponents,
        current_vel: VectorComponents,
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_scale: FLikeSz0,
    ) -> FSzN:
        """Compute spatial distances only.

        Parameters
        ----------
        current_pos : dict[str, Array]
            Position of the current point (scalar components).
        current_vel : dict[str, Array]
            Velocity of the current point (scalar components). Ignored.
        positions : dict[str, Array]
            Positions of all points (array components).
        velocities : dict[str, Array]
            Velocities of all points (array components). Ignored.
        metric_scale : float
            Scale parameter. Ignored for spatial-only metric.

        Returns
        -------
        distances : Array
            Euclidean distances in position space.

        """
        del current_vel, velocities, metric_scale
        # Compute Euclidean distances in position space (vmap over array)
        return vec_euclidean_distance(current_pos, positions)


@final
class AlignedMomentumDistanceMetric(AbstractDistanceMetric):
    r"""Default momentum-based distance metric.

    Computes modified distance as:

    $$ d = d_0 + \lambda (1 - \cos\theta) $$

    where $d_0$ is the Euclidean distance in position space, $\theta$ is the
    angle between the current velocity and the direction to the candidate point,
    and $\lambda$ controls the relative importance of momentum alignment.

    When $\lambda = 0$, reduces to pure nearest-neighbor search in position
    space.  As $\lambda$ increases, points aligned with the current velocity
    direction are increasingly favored.

    This is the original phase-flow walk metric from Nibauer et al. (2022).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw
    >>> metric = lfw.metrics.AlignedMomentumDistanceMetric()
    >>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    >>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}
    >>> current_pos = {k: v[0] for k, v in pos.items()}
    >>> current_vel = {k: v[0] for k, v in vel.items()}
    >>> distances = metric(current_pos, current_vel, pos, vel, metric_scale=1.0)
    >>> distances.shape
    (3,)

    """

    __citation__: ClassVar[str] = (
        "https://ui.adsabs.harvard.edu/abs/2022ApJ...940...22N/abstract"
    )

    def __call__(
        self,
        current_pos: VectorComponents,
        current_vel: VectorComponents,
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_scale: FLikeSz0,
    ) -> FSzN:
        """Compute momentum-weighted distances.

        Parameters
        ----------
        current_pos : dict[str, Array]
            Position of the current point (scalar components).
        current_vel : dict[str, Array]
            Velocity of the current point (scalar components).
        positions : dict[str, Array]
            Positions of all points (array components).
        velocities : dict[str, Array]
            Velocities of all points (array components).
        metric_scale : float
            Momentum weight parameter (units of distance). When 0, reduces to
            pure nearest-neighbor search.

        Returns
        -------
        distances : Array
            Modified distances incorporating momentum alignment.

        """
        del velocities

        # Compute Euclidean distances in position space (vmap over array)
        d0 = vec_euclidean_distance(current_pos, positions)

        # Compute unit directions from current point to all points (vmap over array)
        unit_dirs = vec_unit_direction(current_pos, positions)

        # Compute unit velocity of current point (scalar operation)
        unit_vel = unit_velocity(current_vel)

        # Compute cosine similarity between unit velocity and all unit
        # directions (vmap)
        cos_sim = vec_cosine_similarity(unit_vel, unit_dirs)

        # Momentum distance: d = d0 + λ * (1 - cos_sim)
        return d0 + metric_scale * (1.0 - cos_sim)


@final
class FullPhaseSpaceDistanceMetric(AbstractDistanceMetric):
    r"""Full 6D phase-space distance metric.

    Computes the Euclidean distance in the full 6-dimensional phase space by
    combining position and velocity differences. The parameter `metric_scale`
    (with time units) converts velocity differences to position units.

    $$
    d = \sqrt{d_0^2 + (\tau \cdot d_v)^2}
    $$

    where:
    - $d_0$ is the Euclidean distance in position space
    - $d_v$ is the Euclidean distance in velocity space
    - $\tau$ is the time parameter (metric_scale) that converts velocity to
      position units

    This metric treats position and velocity symmetrically in phase space,
    without directional bias from momentum alignment. The `metric_scale`
    parameter determines the relative weighting of velocity differences.

    Physically, if we think of phase space as having position coordinates
    measured in kpc and velocity coordinates measured in kpc/Myr, then
    `metric_scale` with units of Myr converts velocity differences to kpc,
    allowing us to compute a true Euclidean distance in a uniformly scaled phase
    space.

    This metric is useful when:

    - Position and velocity information are equally important
    - You want true 6D proximity without momentum direction bias
    - The natural time scale of the system is known

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw
    >>> metric = lfw.metrics.FullPhaseSpaceDistanceMetric()
    >>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    >>> vel = {"x": jnp.array([1.0, 1.5, 2.0]), "y": jnp.array([0.5, 1.0, 1.5])}
    >>> current_pos = {k: v[0] for k, v in pos.items()}
    >>> current_vel = {k: v[0] for k, v in vel.items()}
    >>> # metric_scale=1.0 means 1 unit of velocity diff = 1 unit of position diff
    >>> distances = metric(current_pos, current_vel, pos, vel, metric_scale=1.0)
    >>> distances.shape
    (3,)

    """

    __citation__: ClassVar = None

    def __call__(
        self,
        current_pos: VectorComponents,
        current_vel: VectorComponents,
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_scale: FLikeSz0,
    ) -> FSzN:
        """Compute full 6D phase-space distances.

        Parameters
        ----------
        current_pos : dict[str, Array]
            Position of the current point (scalar components).
        current_vel : dict[str, Array]
            Velocity of the current point (scalar components).
        positions : dict[str, Array]
            Positions of all points (array components).
        velocities : dict[str, Array]
            Velocities of all points (array components).
        metric_scale : FLikeSz0
            Time parameter (tau) to convert velocity differences to position units.
            Higher values weight velocity differences more heavily.

        Returns
        -------
        distances : Array
            6D Euclidean distances in phase space.

        """
        # Compute position distances (vmap over array)
        d_pos = vec_euclidean_distance(current_pos, positions)

        # Compute velocity distances (vmap over array)
        d_vel = vec_euclidean_distance(current_vel, velocities)

        # Combine: sqrt(d_pos^2 + (metric_scale * d_vel)^2)
        return jnp.sqrt(d_pos**2 + (metric_scale * d_vel) ** 2)
