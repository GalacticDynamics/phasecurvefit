"""The local-flow orderer: a backward-compatible wrapper around the walk."""

__all__: tuple[str, ...] = ("LocalFlowOrderer",)

import equinox as eqx
import jax.numpy as jnp
import plum

from .base import AbstractOrderer
from phasecurvefit._src.algorithm import (
    Direction,
    StateMetadata,
    WalkLocalFlowResult,
    _local_flow_walk,
)
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.query_config import WalkConfig


class LocalFlowOrderer(AbstractOrderer):
    """Order tracers with the velocity-following local-flow walk.

    This is the primary way to run the walk, via the uniform orderer interface:
    ``pcf.order(positions, velocities)`` uses it by default. ``order()`` handles
    ``direction="both"`` internally via ``combine_results``. (The module-level
    ``walk_local_flow`` is a deprecated alias for the same computation.)

    Parameters
    ----------
    metric_scale
        Metric-dependent scale parameter.
    config
        Neighbor-query configuration (metric + strategy).
    start_idx
        Index of the starting observation.
    direction
        ``"forward"``, ``"backward"``, or ``"both"``.
    max_dist
        Maximum allowed neighbor distance.
    terminate_indices
        Indices that terminate the walk when reached.
    n_max
        Maximum number of iterations.

    """

    metric_scale: float = 1.0
    config: WalkConfig = eqx.field(default_factory=WalkConfig)
    start_idx: int = eqx.field(static=True, default=0)
    direction: Direction = eqx.field(static=True, default="forward")
    max_dist: float = jnp.inf
    terminate_indices: frozenset[int] | None = eqx.field(static=True, default=None)
    n_max: int | None = eqx.field(static=True, default=None)

    @plum.dispatch
    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,
    ) -> WalkLocalFlowResult:
        """Run the local-flow walk and return its result."""
        kwargs: dict[str, object] = {}
        if metadata is not None:
            kwargs["metadata"] = metadata
        return _local_flow_walk(
            positions,
            velocities,
            start_idx=self.start_idx,
            metric_scale=self.metric_scale,
            max_dist=self.max_dist,
            terminate_indices=self.terminate_indices,
            n_max=self.n_max,
            config=self.config,
            direction=self.direction,
            **kwargs,
        )
