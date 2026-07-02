"""Unified result type for ordering algorithms.

``OrderingResult`` is the single concrete result returned by orderers. It holds
the four fields the autoencoder consumes (``positions``, ``velocities``,
``indices``, ``gamma_range``) plus an optional ``backbone`` polyline, and a
single backbone-aware ``__call__``:

- with a ``backbone`` (e.g. from ``MSTOrderer``): interpolate along the smooth
  tip-to-tip polyline (a clean centerline);
- without one (e.g. from the local-flow walk): interpolate along the ordered
  visited observations, reproducing the legacy walk behavior.

``WalkLocalFlowResult`` is a thin subclass of this class (see ``algorithm``),
preserving its public name while sharing all logic.
"""

__all__: tuple[str, ...] = ("OrderingResult",)

from dataclasses import KW_ONLY

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.custom_types import BSzN, ISz0, ISzN, VectorComponents


class OrderingResult(AbstractResult):
    r"""Unified result of an ordering algorithm.

    Attributes
    ----------
    positions : dict[str, Array]
        Original (not reordered) position components, 1D arrays of shape
        ``(n_obs,)``.
    velocities : dict[str, Array]
        Original velocity components, same shape as ``positions``.
    indices : Int[Array, " n_obs"]
        Ordered indices of visited observations; unvisited slots are ``-1``.
    gamma_range : tuple[float, float]
        Static valid range of the ordering parameter for ``__call__``.
    backbone : dict[str, Array] | None
        Optional ordered polyline (tip-to-tip) that ``__call__`` interpolates
        along. ``None`` for walk-style results, which interpolate along the
        ordered visited observations instead.

    """

    positions: VectorComponents
    velocities: VectorComponents
    indices: ISzN
    _: KW_ONLY
    gamma_range: tuple[float, float] = eqx.field(static=True, default=(0.0, 1.0))
    backbone: VectorComponents | None = None

    def __check_init__(self) -> None:
        """Reject a degenerate ``gamma_range`` (its width divides in ``__call__``)."""
        min_gamma, max_gamma = self.gamma_range
        if not max_gamma > min_gamma:
            msg = f"gamma_range must satisfy max > min; got {self.gamma_range}."
            raise ValueError(msg)

    # -- ordering introspection ------------------------------------------

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
        visited = self.indices[self.visited]
        is_visited = jnp.isin(all_indices, visited)
        return all_indices[~is_visited]

    @property
    def ordering(self) -> Int[Array, " n_visited"]:
        """Indices of visited observations in the order they were visited."""
        return self.indices[self.visited]

    @property
    def ordered(self) -> tuple[VectorComponents, VectorComponents]:
        """Positions and velocities ordered by the walk (visited only)."""
        order = lambda x: x[self.ordering]  # noqa: E731
        return (
            jax.tree.map(order, self.positions),
            jax.tree.map(order, self.velocities),
        )

    # -- interpolation ---------------------------------------------------

    def __call__(
        self, gamma: Array, /, *, key: PRNGKeyArray | None = None
    ) -> VectorComponents:
        r"""Interpolate spatial positions from ordering parameter $\gamma$.

        Uses linear interpolation between consecutive control points. The
        control points are the ``backbone`` polyline vertices when a backbone is
        present, otherwise the ordered visited observations. ``gamma`` is
        clipped to ``gamma_range`` and normalized to ``[0, 1]`` first.
        """
        del key
        gamma = jnp.asarray(gamma)

        min_gamma, max_gamma = self.gamma_range
        gamma_range_width = max_gamma - min_gamma
        gamma = eqx.error_if(
            gamma, jnp.any(gamma < min_gamma), "gamma must be >= min_gamma"
        )
        gamma = eqx.error_if(
            gamma, jnp.any(gamma > max_gamma), "gamma must be <= max_gamma"
        )
        gamma_normalized = (gamma - min_gamma) / gamma_range_width

        if self.backbone is not None:
            return self._interp_backbone(gamma_normalized)
        return self._interp_ordered(gamma_normalized)

    @staticmethod
    def _lerp_bracket(gamma_normalized: Array, n_control: Array) -> tuple:
        """Bracketing indices and weights for linear interpolation over n points."""
        indices_float = gamma_normalized * (n_control - 1)
        floor = jnp.floor(indices_float)
        ceil = jnp.ceil(indices_float)
        idx_lower = jnp.clip(floor.astype(jnp.int32), 0, n_control - 1)
        idx_upper = jnp.clip(ceil.astype(jnp.int32), 0, n_control - 1)
        return idx_lower, idx_upper, indices_float - floor

    def _interp_backbone(self, gamma_normalized: Array) -> VectorComponents:
        """Interpolate along the static backbone polyline vertices."""
        n_control = len(next(iter(self.backbone.values())))
        lo, hi, w = self._lerp_bracket(gamma_normalized, n_control)

        def interpolate_component(vals: Array) -> Array:
            return (1 - w) * vals[lo] + w * vals[hi]

        return jax.tree.map(interpolate_component, self.backbone)

    def _interp_ordered(self, gamma_normalized: Array) -> VectorComponents:
        """Interpolate along the ordered visited observations (legacy walk)."""
        visited_indices = jnp.where(self.indices >= 0, self.indices, 0)
        lo, hi, w = self._lerp_bracket(gamma_normalized, self.n_visited)
        orig_lo, orig_hi = visited_indices[lo], visited_indices[hi]

        def interpolate_component(q_vals: Array) -> Array:
            return (1 - w) * q_vals[orig_lo] + w * q_vals[orig_hi]

        return jax.tree.map(interpolate_component, self.positions)
