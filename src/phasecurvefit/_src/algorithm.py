"""Local Flow Walk algorithm implementation.

This is particularly useful for tracing stellar streams where stars follow
coherent trajectories through phase-space.

Examples
--------
>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf

Create some example phase-space data:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0]), "y": jnp.array([0.0, 0.5, 0.8, 1.2])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0]), "y": jnp.array([0.2, 0.2, 0.2, 0.2])}

Run the algorithm:

>>> result = walk_local_flow(pos, vel, start_idx=0, metric_scale=0.5)
>>> result.ordering
Array([0, 1, 2, 3], dtype=int32)

"""

__all__: tuple[str, ...] = (
    "WalkLocalFlowResult",
    "StateMetadata",
    "walk_local_flow",
    "combine_results",
)

from collections.abc import Iterator, Set
from dataclasses import KW_ONLY
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jtu
import plum
import quax
from jax_bounded_while import bounded_while_loop
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from zeroth import zeroth

from .abstract_result import AbstractResult
from .custom_types import BSzN, ISz0, ISzN, RLikeSz0, VectorComponents
from .phasespace import euclidean_distance, get_w_at
from .query_config import WalkConfig

vec_euclidean_distance = jax.jit(jax.vmap(euclidean_distance, in_axes=(None, 0)))


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
        return self._data[key]  # pylint: disable=unsubscriptable-object

    def __contains__(self, key: str) -> bool:
        """Allow 'in' checks for keys."""
        return key in self._data  # pylint: disable=unsupported-membership-test

    def __getattribute__(self, name: str) -> object:
        """Bypass Equinox method wrapping for metadata access."""
        return object.__getattribute__(self, name)

    def get(self, key: str, default: object = None) -> object:
        """Get with default value."""
        return self._data.get(key, default)  # pylint: disable=no-member

    def __repr__(self) -> str:
        """Return string representation."""
        return f"StateMetadata({self._data!r})"

    def __iter__(self) -> Iterator:
        return iter(self._data)

    @staticmethod
    def aval() -> jax.core.ShapedArray:
        """Return a placeholder abstract value so JAX tracing is satisfied."""
        return jax.core.ShapedArray((), jnp.dtype(bool))

    def materialise(self) -> Array:
        """Return placeholder value."""
        return jnp.array(True)


class WalkLocalFlowResult(AbstractResult):
    r"""Result of the local flow walk algorithm.

    This class represents the complete output of the phase-flow walk algorithm.
    It contains the walk ordering, original phase-space data, and provides methods
    for examining and interpolating along the discovered stream.

    Attributes
    ----------
    positions : dict[str, Array]
        Position dictionary with keys (e.g., "x", "y", "z") and values as
        1D arrays of shape (n_obs,). These are the original positions from
        the input, not reordered.
    velocities : dict[str, Array]
        Velocity dictionary with same keys and shape as ``positions``.
        These are the original velocities from the input, not reordered.
    indices : Int[Array, " n_obs"]
        Ordered indices of visited observations. Shape (n_obs,).
        Unvisited observations are marked with -1.

        The walk order can be extracted by filtering: `indices[indices >= 0]`.
        See :attr:`ordering` property for a convenience accessor.
    gamma_range : tuple[float, float]
        Valid range of the ordering parameter in `__call__`. Default is (0.0, 1.0).
        This is a static field and cannot be changed after construction.

    Notes
    -----
    The walk algorithm discovers a path through phase-space by following the
    local flow defined by the velocity field. The ordering encodes which
    observations form a coherent sequence along this path.

    **Key distinction**: ``indices`` is an array of length ``n_obs`` where the
    *position* in the array indicates the *order* in the walk, and the *value*
    at that position is the original observation index. For example::

        indices = [3, 7, 1, -1, 5, ...]
        #          ^ 1st visited observation is index 3
        #             ^ 2nd visited observation is index 7
        #                ^ 3rd visited observation is index 1
        #                   ^ 4th observation was not visited
        #                      ^ 5th visited observation is index 5

    Properties provide convenient access to:
    - :attr:`visited`: Boolean mask of visited observations
    - :attr:`ordering`: Indices in walk order (filtered non-negative)
    - :attr:`ordered`: Positions/velocities reordered by walk
    - :attr:`skipped_indices`: Indices of unvisited observations

    The interpolation method (:meth:`__call__`) enables smooth spatial
    interpolation along the discovered path using a continuous ordering
    parameter $\gamma \in [0, 1]$.

    Examples
    --------
    **Basic Usage: Extract Ordering and Properties**

    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> pos = {
    ...     "x": jnp.linspace(0, 10, 20),
    ...     "y": jnp.sin(jnp.linspace(0, 2 * 3.14159, 20)),
    ... }
    >>> vel = {"x": jnp.ones(20), "y": jnp.cos(jnp.linspace(0, 2 * 3.14159, 20))}
    >>> result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)
    >>> result.indices
    Array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19], dtype=int32)
    >>> result.n_visited
    Array(20, dtype=int32)
    >>> result.n_skipped
    Array(0, dtype=int32)

    **Accessing Ordered Data**

    >>> qs_ordered, vs_ordered = result.ordered
    >>> qs_ordered["x"].shape
    (20,)

    **Spatial Interpolation with Gamma Parameter**

    The walk result can be called as a function to interpolate spatial positions
    from an ordering parameter $\gamma \in [0, 1]$:

    >>> gamma = jnp.array([0.0, 0.5, 1.0])
    >>> positions_interp = result(gamma)
    >>> positions_interp["x"]
    Array([ 0.,  5., 10.], dtype=float32)

    **Scalar Interpolation**

    >>> pos_at_midpoint = result(0.5)
    >>> pos_at_midpoint["x"]
    Array(5., dtype=float32)

    **JAX Transformations: JIT Compilation**

    The interpolator is JIT-compatible for efficient compilation:

    >>> import jax
    >>> @jax.jit
    ... def get_position(gamma):
    ...     return result(gamma)
    >>> get_position(0.25)
    {'x': Array(2.5, dtype=float32), 'y': Array(0.9897884, dtype=float32)}

    **JAX Transformations: Vectorization with vmap**

    Interpolate multiple gamma values efficiently:

    >>> gamma_batch = jnp.linspace(0, 1, 100)
    >>> @jax.jit
    ... def interpolate_many(gammas):
    ...     return jax.vmap(result)(gammas)
    >>> positions_batch = interpolate_many(gamma_batch)
    >>> positions_batch["x"].shape
    (100,)

    **JAX Transformations: Automatic Differentiation**

    Compute gradients of positions with respect to the ordering parameter:

    >>> def loss(gamma):
    ...     pos = result(gamma)
    ...     return jnp.sum(pos["x"] ** 2 + pos["y"] ** 2)
    >>> grad_fn = jax.grad(loss)
    >>> grad_at_half = grad_fn(0.5)

    **Composition: JIT + vmap + grad**

    Combine transformations for maximum efficiency:

    >>> @jax.jit
    ... def compute_gradients(gammas):
    ...     return jax.vmap(jax.grad(loss))(gammas)
    >>> compute_gradients(jnp.linspace(0, 1, 50))
    Array([  0. , 5.6351056, 11.270211 , ...],  dtype=float32)

    >>> result.visited.shape
    (20,)

    """

    positions: VectorComponents
    velocities: VectorComponents
    indices: ISzN
    _: KW_ONLY
    gamma_range: tuple[float, float] = (0.0, 1.0)

    @property
    def visited(self) -> BSzN:
        """Boolean array indicating which observations were visited."""
        return self.indices >= 0

    @property
    def n_visited(self) -> ISz0:
        """Number of observations that were visited (not skipped)."""
        return jnp.sum(self.visited)

    @property
    def n_skipped(self) -> ISz0:
        """Number of observations that were not visited (skipped)."""
        return jnp.sum(~self.visited)

    @property
    def all_visited(self) -> Bool[Array, " "]:
        """Whether all observations were visited (no skips)."""
        return jnp.all(self.visited)

    @property
    def skipped_indices(self) -> Int[Array, " n_skipped"]:
        """Indices of skipped observations (marked as -1 in indices)."""
        all_indices = jnp.arange(len(self.indices))
        # Get set of visited indices (filtering out -1)
        visited = self.indices[self.visited]
        # Find indices not in visited set using isin
        is_visited = jnp.isin(all_indices, visited)
        return all_indices[~is_visited]

    @property
    def ordering(self) -> Int[Array, " n_visited"]:
        """Indices of visited observations in the order they were visited."""
        return self.indices[self.visited]

    @property
    def ordered(self) -> tuple[VectorComponents, VectorComponents]:
        """Positions and velocities ordered by the walk."""
        return order_w(self)

    def __call__(
        self, gamma: Array, /, *, key: PRNGKeyArray | None = None
    ) -> VectorComponents:
        r"""Interpolate spatial positions from ordering parameter $\gamma$.

        Uses linear interpolation between consecutive visited observations, with
        unvisited observations (indices of -1) automatically masked out using
        JIT-safe operations.

        Parameters
        ----------
        gamma : Array
            Ordering parameter in [min_gamma, max_gamma], shape (...).
        key : PRNGKeyArray, optional
            JAX random key. Not used by linear interpolation, provided for
            interface compatibility. Default is None.

        Returns
        -------
        positions : dict[str, Array]
            Interpolated spatial positions with same shape (...) as gamma.

        Notes
        -----
        This method uses JIT-safe masked operations to interpolate only
        visited observations. The indices of -1 (unvisited) are automatically
        handled without changing array shapes.

        The input gamma is automatically clipped to the valid range
        [min_gamma, max_gamma] and normalized to [0, 1] before interpolation.

        """
        del key

        # Convert gamma to JAX array to ensure error_if works with scalars
        gamma = jnp.asarray(gamma)

        # Number of visited observations
        n_visited = self.n_visited

        # Validate and normalize gamma to range
        min_gamma, max_gamma = self.gamma_range
        gamma_range_width = max_gamma - min_gamma
        # Error if gamma is out of valid range
        gamma = eqx.error_if(
            gamma, jnp.any(gamma < min_gamma), "gamma must be >= min_gamma"
        )
        gamma = eqx.error_if(
            gamma, jnp.any(gamma > max_gamma), "gamma must be <= max_gamma"
        )
        # Normalize to [0, 1]
        gamma_normalized = (gamma - min_gamma) / gamma_range_width

        # Create a mapping from walk order to original indices, clamping invalid
        # indices (those with -1) to 0 for safe indexing in JIT
        visited_indices = jnp.where(
            self.indices >= 0, self.indices, 0
        )  # Shape: (n_obs,) but only first n_visited are valid

        # Map normalized gamma from [0, 1] to [0, n_visited-1]
        indices_float = gamma_normalized * (n_visited - 1)

        # Get lower and upper indices for linear interpolation
        idx_lower = jnp.floor(indices_float).astype(jnp.int32)
        idx_upper = jnp.ceil(indices_float).astype(jnp.int32)

        # Clamp to valid range [0, n_visited-1]
        idx_lower = jnp.clip(idx_lower, 0, n_visited - 1)
        idx_upper = jnp.clip(idx_upper, 0, n_visited - 1)

        # Map to original indices using the visited_indices array
        # This works in JIT because visited_indices is an array of concrete values
        orig_idx_lower = visited_indices[idx_lower]
        orig_idx_upper = visited_indices[idx_upper]

        # Compute interpolation weights
        weights = indices_float - jnp.floor(indices_float)

        # Interpolate each component using tree.map for cleaner code
        def interpolate_component(q_vals: Array) -> Array:
            """Interpolate a single position component."""
            q_lower = q_vals[orig_idx_lower]
            q_upper = q_vals[orig_idx_upper]
            # Linear interpolation: (1-w)*q_lower + w*q_upper
            return (1 - weights) * q_lower + weights * q_upper

        return jax.tree.map(interpolate_component, self.positions)


Direction: TypeAlias = Literal["forward", "backward", "both"]  # noqa: UP040
State: TypeAlias = tuple[  # noqa: UP040
    Int[Array, " n_obs"],  # indices in walk order (-1 for skipped)
    Array,  # visited_mask: float array of shape (n_obs,)
    ISz0,  # cur_idx: scalar index
    ISz0,  # step: scalar iteration counter
    Bool[Array, ""],  # stop: scalar flag
    StateMetadata,  # metadata: contains dummy array so JAX preserves it
]


@plum.dispatch
def walk_local_flow(
    xs: VectorComponents,
    vs: VectorComponents,
    /,
    *,
    start_idx: int = 0,
    metric_scale: RLikeSz0 = 1.0,
    max_dist: float = jnp.inf,
    terminate_indices: Set[int] | None = None,
    n_max: int | None = None,
    config: WalkConfig = WalkConfig(),  # noqa: B008
    metadata: StateMetadata = StateMetadata(),  # noqa: B008
    direction: Direction = "forward",
) -> WalkLocalFlowResult:
    r"""Find an ordered path through phase-space using the local flow.

    Parameters
    ----------
    xs
        Position dictionary with 1D array values of shape (N,).
    vs
        Velocity dictionary with 1D array values of shape (N,).
    start_idx : int, optional
        The index of the starting observation (default: 0).
    metric_scale
        Metric-dependent scale parameter. Interpretation depends on the metric:
        - AlignedMomentumDistanceMetric: Momentum weight (distance units)
        - FullPhaseSpaceDistanceMetric: Time scale for velocity-to-position conversion
        - SpatialDistanceMetric: Ignored
        Default: 1.0.
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
    config : WalkConfig
        Configuration for neighbor queries, containing both the distance metric
        and the query strategy. Use ``WalkConfig(metric=..., strategy=...)`` to
        customize. Defaults to ``WalkConfig()`` which uses
        ``FullPhaseSpaceDistanceMetric`` with ``BruteForce``.

    metadata
        Optional metadata to pass through the algorithm state without
        participating in computation. Useful for unit systems or other context.
    direction
        Direction to walk the local flow. 'forward' walks along the velocity
        field, 'backward' walks against the velocity field, and 'both' walks in
        both directions.  Default is 'forward'.

    Returns
    -------
    WalkLocalFlowResult
        NamedTuple with fields:

        - "indices": ordered indices array with -1 for unvisited observations.
        - "visited": boolean array indicating visited observations.
        - "xs": original xs dict
        - "vs": original vs dict

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

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

    >>> result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=0.5)
    >>> result.ordering
    Array([0, 1, 2, 3, 4], dtype=int32)

    Walk in the backward direction:

    >>> result_backward = pcf.walk_local_flow(
    ...     pos, vel, start_idx=4, metric_scale=0.5, direction="backward"
    ... )
    >>> result_backward.indices
    Array([4, 3, 2, 1, 0], dtype=int32)

    """
    if direction == "both":
        kwargs = {
            "start_idx": start_idx,
            "metric_scale": metric_scale,
            "max_dist": max_dist,
            "terminate_indices": terminate_indices,
            "n_max": n_max,
            "config": config,
            "metadata": metadata,
        }
        result_forward = walk_local_flow(xs, vs, **kwargs, direction="forward")
        result_backward = walk_local_flow(xs, vs, **kwargs, direction="backward")
        return combine_results(result_forward, result_backward)

    # ---------------------------------------------------------------

    # Get number of observations from first xs array
    key0 = zeroth(xs)
    n_obs = jnp.shape(xs[key0])[0]

    # Validate start_idx - use plain Python check if not traced
    if start_idx < 0 or start_idx >= n_obs:
        msg = f"start_idx {start_idx} out of bounds for data with {n_obs} observations."
        raise ValueError(msg)

    # Set n_max to n_obs if not provided
    n_max = n_obs if n_max is None else n_max

    # Initialize terminate_indices as empty set if None
    terminate_indices = set() if terminate_indices is None else terminate_indices

    # Store original velocities for the result and optionally negate velocities
    # for backward walk (internal use only).
    vs_original = vs
    if direction == "backward":
        vs = jtu.map(jnp.negative, vs)

    # Extract metric and strategy from config
    query_state = config.strategy.init(xs, metadata=metadata)

    # Create ordered_idxs array (-1 for unused slots)
    ordered_arr = -jnp.ones(n_obs, dtype=int)
    ordered_arr = ordered_arr.at[0].set(start_idx)

    # Create visited mask (0.0 for visited, 1.0 for available)
    unvisited = jnp.ones(n_obs, dtype=bool)
    unvisited = unvisited.at[start_idx].set(False)

    # Convert terminate_indices to array outside the loop
    terminate_arr = (
        jnp.array(list(terminate_indices))
        if terminate_indices
        else jnp.array([], dtype=int)
    )

    # Pack the state tuple
    state: State = (ordered_arr, unvisited, start_idx, 1, False, metadata)

    def cond_fn(state: State, /) -> bool:
        """Continue looping if there are remaining points and iterations left."""
        # Partially unpack state for info related to stopping
        _, _, cur_idx, step, stop, *_ = state

        # Check if current index should terminate
        should_not_terminate = jnp.where(
            terminate_arr.size > 0, jnp.all(cur_idx != terminate_arr), jnp.array(True)
        )

        # Continue if we haven't reached max steps, not signaled to stop, and
        # haven't hit a terminate index
        return jnp.logical_and(
            jnp.logical_and(step < n_max, jnp.logical_not(stop)), should_not_terminate
        )

    # NOTE: body_fn stays un-jitted; when walk_local_flow is wrapped in jax.jit,
    # equinox.internal.while_loop traces this body once per iteration, so a
    # local jit would be redundant. The bounded while_loop is scan-based and
    # more efficient.
    def body_fn(state: State, /) -> State:
        """Process one iteration using dict-based phase-space operations."""
        # Unpack carry: path indices
        path, unvisited, cur_idx, step, _, metadata = state

        # Get current xs and velocity (scalar dicts)
        cur_x, cur_v = get_w_at(xs, vs, cur_idx)

        # Query strategy for candidate neighbors and compute distance
        query_result = config.strategy.query(
            query_state, cur_x, cur_v, xs, vs, config.metric, metric_scale
        )
        ds = query_result.distances
        candidate_idxs = query_result.indices

        if candidate_idxs is not None:
            ds_candidates = jnp.full_like(ds, jnp.inf)
            ds_candidates = ds_candidates.at[candidate_idxs].set(ds[candidate_idxs])
        else:
            ds_candidates = ds

        # Mask visited points (where mask is 0) by setting inf
        inf_mask = jnp.full_like(ds, jnp.inf) * max_dist
        ds_masked = jnp.where(unvisited, ds_candidates, inf_mask)

        # Find nearest neighbor (within candidates if provided)
        best_idx = jnp.argmin(ds_masked)
        best_dist = jnp.min(ds_masked)

        # Check termination BEFORE adding the point
        # Stop if:
        # 1. All unvisited points exceed max distance (min_dist > max_dist), OR
        # 2. No unvisited candidates remain (isinf best_dist), OR
        # 3. Selected point itself exceeds max spatial distance (forced to
        #    backtrack)
        spatial_ds = vec_euclidean_distance(cur_x, xs)
        spatial_ds_masked = jnp.where(unvisited, spatial_ds, inf_mask)
        min_dist = jnp.min(spatial_ds_masked)
        new_stop = jnp.logical_or(
            jnp.logical_or(min_dist > max_dist, jnp.isinf(best_dist)),
            spatial_ds[best_idx] > max_dist,
        )

        # Conditional update: only add if not terminating
        new_path = jnp.where(new_stop, path, path.at[step].set(best_idx))
        new_unvisited = jnp.where(
            new_stop, unvisited, unvisited.at[best_idx].set(False)
        )
        new_step = jnp.where(new_stop, step, step + 1)
        new_cur_idx = jnp.where(new_stop, cur_idx, best_idx)

        return (new_path, new_unvisited, new_cur_idx, new_step, new_stop, metadata)

    # Use custom bounded_while_loop for efficient scan-based implementation that
    # skips iterations once condition is met.  The parent walk_local_flow
    # wrapper handles quaxification, so we don't quaxify here.
    final_state = bounded_while_loop(cond_fn, body_fn, state, max_steps=n_max)

    # Extract results - return arrays directly
    final_ordered, *_ = final_state

    # Set gamma_range based on direction
    gamma_range = (-1.0, 0.0) if direction == "backward" else (0.0, 1.0)

    # Package results into WalkLocalFlowResult. This is a NamedTuple, so can be
    # unpacked easily.
    return WalkLocalFlowResult(
        positions=dict(xs),
        velocities=dict(vs_original),
        indices=final_ordered,
        gamma_range=gamma_range,
    )


def order_w(res: WalkLocalFlowResult, /) -> tuple[VectorComponents, VectorComponents]:
    """Get xs and vs in the ordered sequence from a WalkLocalFlowResult.

    Filters out unvisited indices (marked as -1) and returns only the visited
    observations in the order they were traversed.

    Parameters
    ----------
    res
        The result from walk_local_flow.

    Returns
    -------
    xs : dict[str, Array]
        Position arrays reordered according to the algorithm's output,
        with unvisited observations removed.
    vs : dict[str, Array]
        Velocity arrays reordered according to the algorithm's output,
        with unvisited observations removed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> pos = {"x": jnp.array([3.0, 1.0, 2.0])}
    >>> vel = {"x": jnp.array([1.0, 1.0, 1.0])}
    >>> result = pcf.walk_local_flow(pos, vel, start_idx=1, metric_scale=0.0)
    >>> ordered_pos, ordered_vel = pcf.order_w(result)

    """
    # Filter out -1 (unvisited) indices
    ordering = res.ordering  # pre-computed ordering of visited indices
    order = lambda x: x[ordering]  # noqa: E731
    return jtu.map(order, res.positions), jtu.map(order, res.velocities)


# ===================================================================


DedupCarry: TypeAlias = tuple[  # noqa: UP040
    Bool[Array, " n_obs"],  # seen_arr: tracking which indices have been seen
    Int[Array, " n_obs"],  # out_arr: accumulating deduplicated indices
    ISz0,  # count: current position in output array
]


def _dedup_step(carry: DedupCarry, idx: Array) -> tuple[DedupCarry, None]:
    """Remove duplicate indices while preserving order (scan step function).

    Used by {func}`combine_results` to deduplicate the concatenated
    forward/backward walk results. Skips invalid indices (-1) and previously
    seen indices, building an output array of unique valid indices in their
    original order.

    This is a scan step function designed to be called via {func}`jax.lax.scan`.
    It maintains a running state of which indices have been seen and accumulates
    unique indices into an output array.

    The function handles edge cases:

    - Invalid indices (-1) are skipped and never added to the output
    - Repeated valid indices are only added on first occurrence
    - Safe indexing prevents out-of-bounds access when idx=-1

    """
    seen_arr, out_arr, count = carry
    # Use safe indexing: map -1 (skipped) to 0 to avoid out-of-bounds access
    safe_idx = jnp.where(idx >= 0, idx, 0)
    # Check if this index was already seen
    prev_seen = seen_arr[safe_idx]
    # Only count as new if idx is valid (>= 0) and hasn't been seen
    is_new = jnp.logical_and(idx >= 0, jnp.logical_not(prev_seen))
    # Mark idx as seen (only if idx is valid, else preserve previous state)
    new_seen = seen_arr.at[safe_idx].set(jnp.where(idx >= 0, True, prev_seen))
    # Add idx to output if new, otherwise preserve current output position
    new_out = out_arr.at[count].set(jnp.where(is_new, idx, out_arr[count]))
    # Increment count only if we added a new element
    new_count = count + is_new.astype(count.dtype)
    return (new_seen, new_out, new_count), None


@plum.dispatch
def combine_results(
    result_fwd: WalkLocalFlowResult, result_bwd: WalkLocalFlowResult, /
) -> WalkLocalFlowResult:
    r"""Combine forward and reverse flow walk results into a single result.

    Takes the results from two separate walk_local_flow calls (one forward, one
    reverse) and combines them into a single coherent ordering. This is useful
    for tracing complete stellar streams that extend in both directions from a
    starting point.

    The combination strategy assumes both walks started from the same index:

    1. Forward walk: [start_idx, idx1_fwd, idx2_fwd, ...]
    2. Backward walk: [start_idx, idx1_bwd, idx2_bwd, ...]
    3. Combined: [idx2_bwd, idx1_bwd, start_idx, idx1_fwd, idx2_fwd, ...]

    Duplicate indices are removed while preserving order, ensuring each point
    appears only once in the final ordering.

    Parameters
    ----------
    result_fwd : WalkLocalFlowResult
        Result from a forward walk (direction='forward').
    result_bwd : WalkLocalFlowResult
        Result from a backward walk (direction='backward'), typically starting
        from the same index as the forward walk.

    Returns
    -------
    WalkLocalFlowResult
        {class}`~typing.NamedTuple` with combined ordering that includes both
        forward and backward traversals. Indices are ordered from the backward
        tail through the starting point to the forward tail. The gamma_range
        is stitched together: if result_bwd has gamma_range [min_bwd, 0] and
        result_fwd has gamma_range [0, max_fwd], the combined result has
        gamma_range [min_bwd, max_fwd].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    Create phase-space data for a stream extending in both directions:

    >>> pos = {
    ...     "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    ...     "y": jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    ... }
    >>> vel = {
    ...     "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    ...     "y": jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    ... }

    Run forward and backward walks from the middle:

    >>> result_fwd = pcf.walk_local_flow(pos, vel, start_idx=2, metric_scale=0.5)
    >>> result_bwd = pcf.walk_local_flow(
    ...     pos, vel, start_idx=2, metric_scale=0.5, direction="backward"
    ... )

    Combine the results:

    >>> result = pcf.combine_results(result_fwd, result_bwd)
    >>> result.indices
    Array([4, 3, 0, 1, 2], dtype=int32)

    This combines the forward walk (2→3→4) and backward walk (2→1→0) into a
    single ordering (0, 1, 2, 3, 4).

    Notes
    -----
    The combined ordering places the starting point near the center, with the
    backward walk indices on the left and forward walk indices on the right.
    This is useful for:

    - Tracing complete stellar streams from a central progenitor
    - Exploring both tidal tails of a disrupting stream
    - Verifying stream connectivity in both directions
    - Handling bifurcated or complex stream geometries

    """
    # Full equality check using efficient tree operations
    # Use tuples to combine positions and velocities for one map/reduce pass
    # (tuples compile more efficiently than dicts)
    matches = jtu.map(
        jnp.array_equal,
        (result_fwd.positions, result_fwd.velocities),
        (result_bwd.positions, result_bwd.velocities),
    )
    all_match = jtu.reduce(jnp.logical_and, matches)

    # Error if they don't match
    _ = eqx.error_if(
        all_match,
        jnp.logical_not(all_match),
        "result_fwd and result_bwd must have the same positions and velocities",
    )

    # Get start_idx (first element in both, should be the same)
    start_idx = result_fwd.indices[0]

    # Skip start_idx from each walk and backward the backward walk
    forward_tail = result_fwd.indices[1:]
    backward_tail = jnp.flip(result_bwd.indices[1:])

    # Concatenate: backward_tail + start_idx + forward_tail
    combined_raw = jnp.concat([backward_tail, jnp.array([start_idx]), forward_tail])

    # Deduplicate while preserving order (ignore -1 and repeats)
    n_obs = len(result_fwd.indices)
    seen = jnp.zeros(n_obs, dtype=bool)
    dedup_init = jnp.full_like(combined_raw, -1)

    (_, deduplicated, _), _ = jax.lax.scan(
        _dedup_step, (seen, dedup_init, jnp.array(0, dtype=int)), combined_raw
    )

    # Pad to original size with -1
    indices = jnp.full(n_obs, -1, dtype=int)
    indices = indices.at[:n_obs].set(deduplicated[:n_obs])

    # Combine gamma ranges: backward [-1, 0] + forward [0, 1] → [-1, 1]
    min_gamma_bwd, _ = result_bwd.gamma_range
    _, max_gamma_fwd = result_fwd.gamma_range
    combined_gamma_range = (min_gamma_bwd, max_gamma_fwd)

    return WalkLocalFlowResult(
        positions=result_fwd.positions,
        velocities=result_fwd.velocities,
        indices=indices,
        gamma_range=combined_gamma_range,
    )
