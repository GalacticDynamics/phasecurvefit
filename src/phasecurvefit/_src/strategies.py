"""Neighbor query strategies and configuration for the walk.

Provides instance-based strategies used by the local-flow walk to find nearby
neighbors:
- ``BruteForce()``: Compute distances to all remaining points (default)
- ``KDTree(k=...)``: Use a KD-tree for spatial prefiltering
    (requires optional ``jaxkd`` dependency)

The ``WalkConfig`` class composes a strategy with a distance metric:

    config = WalkConfig(
        metric=FullPhaseSpaceDistanceMetric(),
        strategy=KDTree(k=50),
    )
    result = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(config=config))
"""

__all__: tuple[str, ...] = (
    "AbstractQueryStrategy",
    "BruteForce",
    "KDTree",
    "QueryResult",
)

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from .custom_types import FLikeSz0, ISz0, VectorComponents
from .metrics import AbstractDistanceMetric
from .optional_deps import OptDeps
from .phasespace import get_w_at

if OptDeps.JAXKD.installed:
    import jaxkd


class QueryResult(NamedTuple):
    """Result of a neighbor query.

    Attributes
    ----------
    distances : Array
        Metric distances. If ``indices`` is None, to every point (shape
        ``(n,)``); otherwise to each candidate, aligned with ``indices``
        (``distances[i]`` is the distance to point ``indices[i]``).
    indices : Array or None
        Candidate indices, or None if every point is a candidate. For `KDTree`
        these are the ``k + 1`` nearest spatial points, which include the
        current point itself; the walk excludes it as already visited.

    """

    distances: Array
    indices: Array | None = None


class AbstractQueryStrategy(ABC):
    """Abstract base class for neighbor query strategies.

    Strategies are minimally stateful. Configure via `__init__` (e.g., KD-tree
    `k`). The walk calls `init(positions)` once to build a strategy state, then
    `query_at(state, idx, ...)` at each step. Subclasses must implement `query`,
    which answers for an arbitrary position; the default `query_at` looks up
    the data point at ``idx`` and delegates to it. Override `query_at` only to
    use precomputed per-point state, as `KDTree` does.
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

    def query_at(
        self,
        state: object,
        current_idx: ISz0,
        /,
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_fn: AbstractDistanceMetric,
        metric_scale: FLikeSz0,
    ) -> QueryResult:
        """Query for neighbors of the data point at ``current_idx``.

        The walk only ever queries at data points, so it calls this rather
        than `query`. The default looks the point up and delegates to `query`;
        a strategy can override it to use precomputed per-point state (as
        `KDTree` does with its neighbor table).
        """
        current_pos, current_vel = get_w_at(positions, velocities, current_idx)
        return self.query(
            state,
            current_pos,
            current_vel,
            positions,
            velocities,
            metric_fn,
            metric_scale,
        )


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
    """KD-tree strategy: restrict each step to the k nearest spatial neighbors.

    `init` builds a KD-tree and queries the ``k`` nearest spatial neighbors of
    every point once, giving an ``(n, k + 1)`` neighbor table. Each walk step
    then looks up the current point's row and evaluates the metric on those
    candidates only, so a step's own work is O(k) rather than O(n). Building
    the table is one batched KD-tree query over all points; on CPU that
    one-off cost dominates, so this is faster than `BruteForce` only for large
    datasets (tens of thousands of points and up).

    Because only spatial neighbors are candidates, the walk can pick a
    different point than `BruteForce` would: one that is nearer under the
    metric but not among the ``k`` nearest in space is never considered.

    Requires jaxkd optional dependency.
    """

    def __init__(self, k: int = 50) -> None:
        """Initialize KD-tree strategy.

        Parameters
        ----------
        k : int, optional
            Number of candidate spatial neighbors to consider at each step,
            *excluding* the current point itself. Default: 50. Larger values
            allow longer jumps; smaller values keep the walk spatially local.
            Values larger than the number of points are clamped to consider
            every other point.

        """
        self.k = k

        if not OptDeps.JAXKD.installed:
            msg = (
                "KDTree requires jaxkd optional dependency. "
                "Install with: uv add phasecurvefit[kdtree]"
            )
            raise ImportError(msg)

    def _n_query(self, n_points: int) -> int:
        # Query one extra neighbor to make room for the current point itself.
        # Otherwise ``k`` yields only ``k - 1`` usable candidates and small
        # ``k`` can deadlock the walk: the sole non-self neighbor may already
        # be visited, leaving no candidate and terminating the walk
        # prematurely. Self is NOT dropped positionally: with coincident points
        # the tree may return a duplicate before self, so slicing off slot 0
        # would drop an unvisited duplicate and keep self. Instead self stays
        # in the candidate set and is excluded by index via the walk's visited
        # mask (the current point is always visited). Clamp to the point count
        # since jaxkd errors when ``k`` exceeds the tree size.
        return min(self.k + 1, n_points)

    def init(
        self,
        positions: VectorComponents,
        /,
        *,
        metadata: object,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Build the KD-tree and every point's neighbor table."""
        pos_flat = jnp.stack([positions[k] for k in sorted(positions)], axis=-1)
        tree = jaxkd.build_tree(pos_flat)
        n_query = self._n_query(pos_flat.shape[0])
        neighbors, _ = jaxkd.query_neighbors(tree, pos_flat, k=n_query)
        return {"tree": tree, "n_query": n_query, "neighbors": neighbors}

    def _candidates(
        self,
        indices: Array,
        current_pos: dict[str, Array],
        current_vel: dict[str, Array],
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_fn: AbstractDistanceMetric,
        metric_scale: FLikeSz0,
    ) -> QueryResult:
        cand_pos, cand_vel = get_w_at(positions, velocities, indices)
        distances = metric_fn(
            current_pos, current_vel, cand_pos, cand_vel, metric_scale
        )
        return QueryResult(distances=distances, indices=indices)

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
        """Query the KD-tree at an arbitrary position; metric on candidates only."""
        current_pos_arr = jnp.array([current_pos[k] for k in sorted(current_pos)])
        indices, _ = jaxkd.query_neighbors(
            kd_state["tree"], current_pos_arr[None, :], k=kd_state["n_query"]
        )
        return self._candidates(
            indices[0],
            current_pos,
            current_vel,
            positions,
            velocities,
            metric_fn,
            metric_scale,
        )

    def query_at(
        self,
        kd_state: dict[str, Any],
        current_idx: ISz0,
        /,
        positions: VectorComponents,
        velocities: VectorComponents,
        metric_fn: AbstractDistanceMetric,
        metric_scale: FLikeSz0,
    ) -> QueryResult:
        """Look up the data point's precomputed neighbors; O(k) per step."""
        current_pos, current_vel = get_w_at(positions, velocities, current_idx)
        return self._candidates(
            kd_state["neighbors"][current_idx],
            current_pos,
            current_vel,
            positions,
            velocities,
            metric_fn,
            metric_scale,
        )
