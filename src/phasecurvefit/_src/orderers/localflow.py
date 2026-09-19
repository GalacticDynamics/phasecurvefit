"""The local-flow orderer: a backward-compatible wrapper around the walk."""

__all__: tuple[str, ...] = ("LocalFlowOrderer",)

import equinox as eqx
import jax.numpy as jnp
import plum

import dataclassish

from .base import AbstractOrderer, chord_along_ordering
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import (
    Direction,
    StateMetadata,
    WalkLocalFlowResult,
    _local_flow_walk,
)
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.query_config import WalkConfig


def _resolve_start_idx(start_idx: int | None, init: AbstractResult | None) -> int:
    """Pick where the walk starts, taking it from ``init`` when unset.

    An explicit index always wins. Otherwise the first observation of the prior
    stage's ordering is used: an ``MSTOrderer`` orders tip to tip along the
    graph diameter, so its first observation is an endpoint of the curve --
    which is what the walk needs and what the caller would otherwise have to
    work out by hand.

    Falls back to ``0`` with no prior stage, matching the old default.
    """
    if start_idx is not None:
        return start_idx
    if init is None:
        return 0
    indices = jnp.asarray(init.indices)
    visited = indices[indices >= 0]
    return 0 if visited.shape[0] == 0 else int(visited[0])


def _with_chord(result: WalkLocalFlowResult) -> WalkLocalFlowResult:
    """Attach the arc length along the walk path.

    The walk has no separate backbone -- its curve is the path through the
    visited observations -- so the chord is the cumulative distance along that
    path. ``chord`` is not a static field but the result is built inside
    ``_local_flow_walk``, so it is filled in afterwards.
    """
    return dataclassish.replace(
        result, chord=chord_along_ordering(result.positions, result.indices)
    )


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
        Index of the starting observation. ``None`` (the default) takes the
        start from ``init`` when the walk is chained after another orderer, and
        falls back to ``0`` when it is not.

        The walk has to be told where a curve *ends*, and picking that index by
        hand means knowing the answer in advance. An
        :class:`~phasecurvefit.orderers.MSTOrderer` finds the two tips itself --
        its ordering is the graph diameter, tip to tip -- so its first ordered
        observation is a genuine endpoint::

            (
                pcf.orderers.MSTOrderer(k=16, jump_cap=3.0)
                | pcf.orderers.LocalFlowOrderer()
            )

        An explicit ``start_idx`` always wins, chained or not.
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
    start_idx: int | None = eqx.field(static=True, default=None)
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
        init: AbstractResult | None = None,
    ) -> WalkLocalFlowResult:
        """Run the local-flow walk and return its result."""
        kwargs: dict[str, object] = {}
        if metadata is not None:
            kwargs["metadata"] = metadata
        result = _local_flow_walk(
            positions,
            velocities,
            start_idx=_resolve_start_idx(self.start_idx, init),
            metric_scale=self.metric_scale,
            max_dist=self.max_dist,
            terminate_indices=self.terminate_indices,
            n_max=self.n_max,
            config=self.config,
            direction=self.direction,
            **kwargs,
        )
        return _with_chord(result)
