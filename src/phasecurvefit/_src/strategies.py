"""Neighbor query strategies and configuration for the walk.

Provides instance-based strategies used by ``walk_local_flow`` to find nearby
neighbors:
- ``BruteForce()``: Compute distances to all remaining points (default)
- ``KDTree(k=...)``: Use a KD-tree for spatial prefiltering
    (requires optional ``jaxkd`` dependency)

The ``WalkConfig`` class composes a strategy with a distance metric:

    config = WalkConfig(
        metric=FullPhaseSpaceDistanceMetric(),
        strategy=KDTree(k=50),
    )
    result = walk_local_flow(pos, vel, config=config, ...)
"""

__all__: tuple[str, ...] = (
    "AbstractQueryStrategy",
    "BruteForce",
    "KDTree",
)

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from .custom_types import FLikeSz0, VectorComponents
from .metrics import AbstractDistanceMetric


class QueryResult(NamedTuple):
    """Result of a neighbor query.

    Attributes
    ----------
    distances : Array
        Array of distances to all points (shape (n,))
    indices : Array
        Candidate indices (for kdtree, indices of k nearest neighbors) For
        brute-force, this is None or full index range.

    """

    distances: Array
    indices: Array | None = None


class AbstractQueryStrategy(ABC):
    """Abstract base class for neighbor query strategies.

    Strategies are minimally stateful. Configure via `__init__` (e.g., KD-tree
    `k`). Call `init(positions)` once to build and return a strategy state
    object. Then call `query(state, ...)` for each step.
    """

    @abstractmethod
    def init(self, positions: VectorComponents, /, *, metadata: object) -> object:
        """Build and return strategy state from positions.

        For brute-force, return an empty state (e.g., `None`).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def query(
        self,
        state: object,
        /,
        current_pos: dict[str, Array],
        current_vel: dict[str, Array],
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_fn: AbstractDistanceMetric,
        metric_scale: FLikeSz0,
    ) -> QueryResult:
        """Query for neighbors given current state.

        Parameters
        ----------
        state : object
            Strategy state returned by `init()`
        current_pos : dict
            Position of current point
        current_vel : dict
            Velocity of current point
        positions : dict
            All positions in dataset
        velocities : dict
            All velocities in dataset
        metric_fn : callable
            Distance metric function
        metric_scale : float
            Metric-dependent scale parameter

        Returns
        -------
        QueryResult
            Result containing distances and optionally candidate indices

        """
        raise NotImplementedError  # pragma: no cover


class BruteForce(AbstractQueryStrategy):
    """Brute-force strategy: compute distance to all points.

    This is the default strategy. It computes distances to all unvisited points
    and selects the nearest one. Most efficient for small datasets or when most
    points are still unvisited.
    """

    def init(self, positions: VectorComponents, /, *, metadata: object) -> object:  # noqa: ARG002
        """No persistent state for brute-force; return None."""
        return None

    def query(
        self,
        state: object,  # noqa: ARG002
        /,
        current_pos: dict[str, Array],
        current_vel: dict[str, Array],
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_fn: AbstractDistanceMetric,
        metric_scale: FLikeSz0,
    ) -> QueryResult:
        """Compute distances to all points using the metric."""
        distances = metric_fn(
            current_pos, current_vel, positions, velocities, metric_scale
        )
        return QueryResult(distances=distances, indices=None)


class KDTree(AbstractQueryStrategy):
    """KD-tree strategy: spatial query followed by metric-based selection.

    This strategy uses a KD-tree to find the k nearest neighbors spatially,
    then applies the metric to select the best one. Much more efficient for
    large datasets, especially when most points have been visited.

    The KD-tree is built once at the start of the walk.

    Requires jaxkd optional dependency.
    """

    def __init__(self, k: int = 50) -> None:
        """Initialize KD-tree strategy.

        Parameters
        ----------
        k : int, optional
            Number of nearest spatial neighbors to query. Default: 50.
            Increase for more thorough searches; decrease for speed.

        """
        self.k = k

        try:
            import jaxkd  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        except ImportError:
            msg = (
                "KDTree requires jaxkd optional dependency. "
                "Install with: uv add localflowwalk[kdtree]"
            )
            raise ImportError(msg) from None
        self._jaxkd = jaxkd

    def init(
        self,
        positions: VectorComponents,
        /,
        *,
        metadata: object,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Build KD-tree from positions and return strategy state."""
        pos_arrays = [positions[k] for k in sorted(positions.keys())]
        pos_flat = jnp.stack(pos_arrays, axis=-1)
        tree = self._jaxkd.build_tree(pos_flat)
        return {"tree": tree}

    def query(
        self,
        kd_state: dict[str, Any],
        /,
        current_pos: dict[str, Array],
        current_vel: dict[str, Array],
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_fn: AbstractDistanceMetric,
        metric_scale: FLikeSz0,
    ) -> QueryResult:
        """Query k nearest neighbors and apply metric.

        Returns distances to all points, but algorithm will use kdtree
        indices to filter candidates before metric selection.
        """
        # Flatten current position
        current_pos_arr = jnp.array(  # (n_dims,)
            [current_pos[k] for k in sorted(current_pos.keys())]
        )

        # Query k nearest neighbors using jaxkd top-level API
        indices, _ = self._jaxkd.query_neighbors(
            kd_state["tree"], current_pos_arr[None, :], k=self.k
        )
        indices = indices[0]  # (k,)

        # Compute metric distances to all points
        distances_metric = metric_fn(
            current_pos, current_vel, positions, velocities, metric_scale
        )

        return QueryResult(distances=distances_metric, indices=indices)
