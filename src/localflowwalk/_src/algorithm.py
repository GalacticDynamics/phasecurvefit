"""Nearest Neighbors with Momentum algorithm implementation.

This module implements the Nearest Neighbors with Momentum (NN+p) algorithm
from Nibauer et al. (2022) for ordering phase-space observations in stellar
streams using both spatial proximity and velocity momentum.

The algorithm finds a path through phase-space observations that balances:
1. Spatial proximity (distance between neighboring points)
2. Velocity momentum (alignment of the velocity with the direction to the next point)

This is particularly useful for tracing stellar streams where stars follow
coherent trajectories through phase-space.

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning."

Examples
--------
>>> import jax.numpy as jnp
>>> import localflowwalk as lfw

Create some example phase-space data:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0]), "y": jnp.array([0.0, 0.5, 0.8, 1.2])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0]), "y": jnp.array([0.2, 0.2, 0.2, 0.2])}

Run the algorithm:

>>> result = walk_local_flow(pos, vel, start_idx=0, lam=0.5)
>>> len(result.ordered_indices)
4

"""

__all__: tuple[str, ...] = ("LocalFlowWalkResult", "StateMetadata", "walk_local_flow")

from collections.abc import Set
from typing import NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import plum
import quax
from jaxtyping import Array, Bool, Int

from zeroth import zeroth

from . import metrics, utils
from .custom_types import FLikeSz0, ISz0, ISzN, VectorComponents
from .phasespace import get_w_at
from .strategies import AbstractQueryStrategy, BruteForceStrategy


class StateMetadata(quax.Value):
    """Metadata container for walk_local_flow state.

    This holds optional context like unit systems that need to be passed
    through the algorithm state without participating in computation.

    StateMetadata is registered as a JAX PyTree LEAF (not a node), which means
    JAX treats it as an atomic, non-flattened object. This prevents JAX from:
    - Attempting to flatten the metadata dict
    - Trying to convert the unit system object to an array
    - Losing the metadata during tree operations like scan

    Examples
    --------
    >>> metadata = StateMetadata(usys="SI")
    >>> metadata["usys"]
    'SI'
    >>> "usys" in metadata
    True

    """

    _data: dict[str, object] = eqx.field(static=True)

    def __init__(self, **kwargs: object) -> None:
        """Initialize StateMetadata with keyword arguments as data dict."""
        object.__setattr__(self, "_data", kwargs)

    def __getitem__(self, key: str) -> object:
        """Allow dict-like access to metadata."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'in' checks for keys."""
        return key in self._data

    def __getattribute__(self, name: str) -> object:
        """Bypass Equinox method wrapping for metadata access."""
        return object.__getattribute__(self, name)

    def get(self, key: str, default: object = None) -> object:
        """Get with default value."""
        return self._data.get(key, default)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"StateMetadata({self._data!r})"

    @staticmethod
    def aval() -> jax.core.ShapedArray:
        """Return a placeholder abstract value so JAX tracing is satisfied."""
        return jax.core.ShapedArray((), jnp.dtype(bool))

    def materialise(self) -> Array:
        """Return placeholder value."""
        return jnp.array(True)


State: TypeAlias = tuple[  # noqa: UP040
    Array,  # ordered_indices: int array of shape (n_obs,)
    Array,  # visited_mask: float array of shape (n_obs,)
    int | ISz0,  # current_idx: scalar index
    int | ISz0,  # step: scalar iteration counter
    bool | Bool[Array, " "],  # should_stop: scalar flag
    # Array,  # Terminate array,
    # FLikeSz0,  # max_dist: scalar float or Quantity
    StateMetadata,  # metadata: contains dummy array so JAX preserves it
    # VectorComponents,  # positions
    # VectorComponents,  # velocities
    # FLikeSz0,  # lam: scalar float or Quantity
]


class LocalFlowWalkResult(NamedTuple):
    """Result of the Nearest Neighbors with Momentum algorithm.

    Attributes
    ----------
    ordered_indices : Int[Array, "n_obs"]
        Array of indices in the order found by the algorithm. Indices that were
        skipped (not visited) are marked as -1.
    positions : dict[str, Array]
        The original positions data.
    velocities : dict[str, Array]
        The original velocities data.

    """

    ordered_indices: ISzN
    positions: VectorComponents
    velocities: VectorComponents

    @property
    def n_visited(self) -> ISz0:
        """Number of observations that were visited (not skipped)."""
        return jnp.sum(self.ordered_indices >= 0)

    @property
    def n_skipped(self) -> ISz0:
        """Number of observations that were not visited (skipped)."""
        return jnp.sum(self.ordered_indices < 0)

    @property
    def all_visited(self) -> Bool[Array, " "]:
        """Whether all observations were visited (no skips)."""
        return jnp.all(self.ordered_indices >= 0)

    @property
    def skipped_indices(self) -> Int[Array, " n_skipped"]:
        """Indices of skipped observations (marked as -1 in ordered_indices)."""
        n_obs = len(self.ordered_indices)
        all_indices = jnp.arange(n_obs)
        # Get set of visited indices (filtering out -1)
        visited = self.ordered_indices[self.ordered_indices >= 0]
        # Find indices not in visited set using isin
        is_visited = jnp.isin(all_indices, visited)
        return all_indices[~is_visited]


@plum.dispatch
def walk_local_flow(
    positions: VectorComponents,
    velocities: VectorComponents,
    /,
    *,
    start_idx: int = 0,
    lam: FLikeSz0 = 1.0,
    max_dist: float = jnp.inf,
    terminate_indices: Set[int] | None = None,
    n_max: int | None = None,
    metric: metrics.AbstractDistanceMetric = metrics.FullPhaseSpaceDistanceMetric(),  # noqa: B008
    strategy: AbstractQueryStrategy = BruteForceStrategy(),  # noqa: B008
    metadata: StateMetadata = StateMetadata(),  # noqa: B008
) -> LocalFlowWalkResult:
    r"""Find an ordered path through phase-space using nearest neighbors with momentum.

    This implements Algorithm 1 from Nibauer et al. (2022). The algorithm
    greedily selects the next point in the sequence by minimizing a distance
    metric over full phase space (positions and velocities). The default
    metric, ``FullPhaseSpaceDistanceMetric``, uses true 6D Euclidean distance.

    Due to the momentum condition, the algorithm may terminate before visiting
    all points. Points that would require "going backwards" (against the
    velocity direction) receive high momentum penalties and may be skipped.
    As stated in the paper: "Due to the momentum condition, the algorithm
    inevitably passes over some stream particles without incorporating them
    into the nearest neighbors graph."

    Parameters
    ----------
    positions
        Position dictionary with 1D array values of shape (N,).
    velocities
        Velocity dictionary with 1D array values of shape (N,).
    start_idx : int, optional
        The index of the starting observation (default: 0).
    lam
        The momentum weight ($\lambda$). Higher values favor points whose direction
        from the current point aligns with the current velocity. Default: 1.0.
    max_dist
        Maximum allowable distance between neighbors. If the minimum distance
        exceeds this value, the algorithm terminates, leaving remaining points
        unvisited. This is key to the algorithm's ability to skip outliers.
        Default: jnp.inf (no limit).
    terminate_indices
        Set of indices at which to terminate the algorithm if reached.
        Default: None.
    n_max
        Maximum number of iterations. Default: None (process all points).
    metric
        Distance metric to use for computing modified distances. Custom metrics
        can be provided to implement alternative distance weighting schemes. The
        metric is static and does not change during the walk. Defaults to
        ``FullPhaseSpaceDistanceMetric`` (true phase-space distance).
    strategy : AbstractQueryStrategy, optional
        Neighbor query strategy instance to use. Defaults to
        `BruteForceStrategy()`. To enable spatial KD-tree prefiltering, pass
        an instance of `KDTreeStrategy(k=...)` (requires `jaxkd`).
    metadata
        Optional metadata to pass through the algorithm state without
        participating in computation. Useful for unit systems or other context.

    Returns
    -------
    LocalFlowWalkResult
        NamedTuple with fields:
        - "ordered_indices": tuple of indices in order (may be fewer than
          total points if algorithm terminated early due to max_dist)
        - "positions": original positions dict
        - "velocities": original velocities dict

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw

    Create phase-space data for a simple stream:

    >>> pos = {
    ...     "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    ...     "y": jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    ... }
    >>> vel = {
    ...     "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    ...     "y": jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    ... }

    Run the algorithm starting from index 0:

    >>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.5)
    >>> result.ordered_indices
    Array([0, 1, 2, 3, 4], dtype=int32)

    """
    # Get number of observations from first positions array
    key0 = zeroth(positions)
    n_obs = jnp.shape(positions[key0])[0]

    if start_idx < 0 or start_idx >= n_obs:
        msg = f"start_idx {start_idx} out of bounds for data with {n_obs} observations."
        raise ValueError(msg)

    if n_max is None:
        n_max = n_obs

    if terminate_indices is None:
        terminate_indices = set()

    # Initialize query strategy
    query_strategy = strategy
    query_state = query_strategy.init(positions, metadata=metadata)

    # Create ordered_indices array (-1 for unused slots)
    ordered_arr = -jnp.ones(n_obs, dtype=int)
    ordered_arr = ordered_arr.at[0].set(start_idx)

    # Create visited mask (0.0 for visited, 1.0 for available)
    visited = jnp.ones(n_obs)
    visited = visited.at[start_idx].set(0.0)

    # Convert terminate_indices to array outside the loop
    terminate_arr = (
        jnp.array(list(terminate_indices))
        if terminate_indices
        else jnp.array([], dtype=int)
    )

    state: State = (
        ordered_arr,
        visited,
        start_idx,
        1,
        False,
        # terminate_arr,
        # max_dist,
        metadata,
        # positions,
        # velocities,
        # lam,
    )

    def cond_fn(state: State, /) -> bool:
        """Continue looping if there are remaining points and iterations left."""
        _, _, current_idx, step, should_stop, *_ = state

        # Check if current index should terminate
        should_not_terminate = jnp.where(
            terminate_arr.size > 0,
            jnp.all(current_idx != terminate_arr),
            jnp.array(True),
        )

        return jnp.logical_and(
            jnp.logical_and(step < n_max, jnp.logical_not(should_stop)),
            should_not_terminate,
        )

    # NOTE: body_fn stays un-jitted; when walk_local_flow is wrapped in jax.jit,
    # equinox.internal.while_loop traces this body once per iteration, so a local
    # jit would be redundant. The bounded while_loop is scan-based and more efficient.
    def body_fn(state: State, /) -> State:
        """Process one iteration using dict-based phase-space operations."""
        (
            ordered_indices,
            visited_mask,
            current_idx,
            step,
            should_stop,
            # terminate_arr,
            # max_dist_state,
            metadata,
            # positions,
            # velocities,
            # lam,
        ) = state

        # Get current positions and velocity (scalar dicts)
        current_pos, current_vel = get_w_at(positions, velocities, current_idx)

        # Query strategy for candidate neighbors and compute distances
        query_result = query_strategy.query(
            query_state, current_pos, current_vel, positions, velocities, metric, lam
        )
        distances = query_result.distances
        candidate_indices = query_result.indices

        # If kdtree returns candidates, restrict to them; otherwise use all
        if candidate_indices is not None:
            distances_candidates = jnp.full_like(distances, jnp.inf)
            distances_candidates = distances_candidates.at[candidate_indices].set(
                distances[candidate_indices]
            )
        else:
            distances_candidates = distances

        # Mask visited points (where mask is 0) by setting inf
        inf_mask = jnp.full_like(distances_candidates, jnp.inf) * max_dist
        distances_masked = jnp.where(visited_mask > 0.5, distances_candidates, inf_mask)

        # Find nearest neighbor
        min_dist = jnp.min(distances_masked)
        best_idx = jnp.argmin(distances_masked)

        # Check termination BEFORE adding the point
        new_should_stop = min_dist > max_dist

        # Conditional update: only add if not terminating
        new_ordered = jnp.where(
            new_should_stop,
            ordered_indices,
            ordered_indices.at[step].set(best_idx),
        )
        new_mask = jnp.where(
            new_should_stop,
            visited_mask,
            visited_mask.at[best_idx].set(0.0),
        )
        new_step = jnp.where(new_should_stop, step, step + 1)

        return (
            new_ordered,
            new_mask,
            best_idx,
            new_step,
            new_should_stop,
            # terminate_arr,
            # max_dist,
            metadata,
            # positions,
            # velocities,
            # lam,
        )

    # Use custom bounded_while_loop for efficient scan-based implementation that
    # skips iterations once condition is met.  The parent walk_local_flow
    # wrapper handles quaxification, so we don't quaxify here.
    final_state = utils.bounded_while_loop(cond_fn, body_fn, state, max_steps=n_max)

    # Extract results - return arrays directly (no tuple conversion in traced context)
    final_ordered, final_visited, *_ = final_state

    return LocalFlowWalkResult(
        ordered_indices=final_ordered,  # Array with -1 for unvisited
        positions=dict(positions),
        velocities=dict(velocities),
    )


def get_ordered_w(
    res: LocalFlowWalkResult, /
) -> tuple[VectorComponents, VectorComponents]:
    """Get positions and velocities in the ordered sequence from a LocalFlowWalkResult.

    Filters out unvisited indices (marked as -1) and returns only the visited
    observations in the order they were traversed.

    Parameters
    ----------
    res
        The result from walk_local_flow.

    Returns
    -------
    positions : dict[str, Array]
        Position arrays reordered according to the algorithm's output,
        with unvisited observations removed.
    velocities : dict[str, Array]
        Velocity arrays reordered according to the algorithm's output,
        with unvisited observations removed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw
    >>> pos = {"x": jnp.array([3.0, 1.0, 2.0])}
    >>> vel = {"x": jnp.array([1.0, 1.0, 1.0])}
    >>> result = lfw.walk_local_flow(pos, vel, start_idx=1, lam=0.0)
    >>> ordered_pos, ordered_vel = lfw.get_ordered_w(result)

    """
    # Filter out -1 (unvisited) indices
    valid_mask = res.ordered_indices >= 0
    indices = res.ordered_indices[valid_mask]
    f = lambda v: v[indices]  # noqa: E731
    return jtu.map(f, res.positions), jtu.map(f, res.velocities)
