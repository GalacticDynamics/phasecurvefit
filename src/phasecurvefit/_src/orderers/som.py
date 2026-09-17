"""The Self-Organizing Map orderer.

Trains a 1-D SOM on the tracers and orders them by their arc-length projection
onto the trained backbone. Most useful as a *refinement* stage after another
orderer -- the prototypes average over local noise, so the ordering is robust
to the small mistakes a greedy walk or an MST backbone makes in dense or
self-overlapping regions -- but it runs standalone too.

References
----------
Starkman, N., Bovy, J., Webb, J. J., Calvetti, D., & Somersalo, E. (2023).
*On the Fast Track: Rapid construction of stellar stream paths.*
MNRAS 522(4), 5022-5036. https://arxiv.org/abs/2212.00949

If you use this orderer in published work, please cite that paper. It
deviates from the paper's method in eight places, listed in full in
:doc:`/guides/som` -- they matter to anyone citing it for results
produced here.

"""

__all__: tuple[str, ...] = ("SOMOrderer",)

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import plum

from .base import AbstractOrderer, _check_component_keys
from .result import OrderingResult
from phasecurvefit._src import som as _som
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.metrics import (
    AbstractDistanceMetric,
    SpatialDistanceMetric,
)


@eqx.filter_jit
def _train_and_project(
    orderer: "SOMOrderer",
    proto_q: VectorComponents,
    proto_p: VectorComponents,
    sub_q: VectorComponents,
    sub_p: VectorComponents,
    sigma_start: float | None,
) -> tuple[VectorComponents, VectorComponents, "jnp.ndarray"]:
    """Run fit -> densify -> chord as one XLA program.

    Jitted so the metric's ``(N, M)`` intermediates fuse; run eagerly each is
    materialized separately, costing roughly 15x the memory at catalogue scale.
    ``init_prototypes`` stays outside so its ``eqx.error_if`` guard raises as an
    ordinary Python exception.
    """
    model = _som.SOM1D(
        proto_q,
        proto_p,
        orderer.metric,
        metric_scale=orderer.metric_scale,
        n_epochs=orderer.n_epochs,
        sigma_start=sigma_start,
        sigma_end=orderer.sigma_end,
        densify_factor=orderer.densify_factor,
    ).fit(sub_q, sub_p)
    backbone_q, backbone_p = model.backbone()
    lam = model.chord(sub_q, sub_p)
    return backbone_q, backbone_p, lam


class SOMOrderer(AbstractOrderer):
    """Order tracers by projection onto a trained 1-D Self-Organizing Map.

    Parameters
    ----------
    n_prototypes
        Number of SOM prototypes, ``K``. More prototypes track finer structure
        at the cost of following noise; the paper finds results insensitive to
        the exact value above roughly 10 per distinct segment of the curve.
    metric
        Phase-space distance metric used for best-matching-unit search and for
        backbone assignment.
    metric_scale
        Scale parameter handed to ``metric``. The default ``metric``,
        :class:`~phasecurvefit.metrics.SpatialDistanceMetric`, ignores it
        entirely, and passing a non-zero value with that metric is rejected at
        construction. For
        :class:`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric` it is a
        *time*, converting velocity differences into position units -- so its
        correct value depends on your unit system, and there is no
        unit-independent default. Defaults to ``0.0``, i.e. pure position
        distance. Set it to make velocity participate, choosing a time such
        that ``metric_scale * dv`` is
        comparable to the position separations you want it to compete with.
        Too large and "nearest prototype" becomes "nearest in velocity", which
        on a winding curve conflates points a whole turn apart.

        ``metric`` must be symmetric in the two points it compares.
        :class:`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` is not:
        it scores "forward along the direction of travel", which is what a
        greedy walk step needs and not what nearest-prototype means. Passing it
        collapses the lattice toward the curve's head.
    n_epochs
        Number of batch-Kohonen epochs.
    sigma_start, sigma_end
        Neighbourhood width in lattice units at the first and last epoch,
        annealed geometrically. When ``init`` is supplied, ``sigma_start=None``
        uses ``sigma_end`` instead: a wide start would smooth away the very
        ordering the prior stage just computed. Standalone, ``sigma_start=None``
        uses ``max(n_prototypes / 4, sigma_end)``, i.e. ``n_prototypes / 4``
        floored at ``sigma_end`` so the anneal is never inverted on a very small
        lattice.
    densify_factor
        Backbone samples per prototype segment.
    orient_by_velocity
        If ``True``, flip the result so the chord increases *along* the mean
        velocity, mirroring :class:`~phasecurvefit.orderers.MSTOrderer`'s
        option of the same name. The SOM has no progenitor anchor -- the paper
        fixes the lattice's first point to the origin (§2.2.2 step 1) and this
        implementation does not -- so a standalone ordering's direction is
        otherwise arbitrary: it comes from the sign of the initializer's
        principal-axis eigenvector, which is stable but meaningless.

        Note this **overrides** an inherited direction rather than deferring
        to it: chained after a stage that already fixed one, setting this can
        silently reverse that stage's choice. Default ``False``, matching
        ``MSTOrderer``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    >>> ang = jnp.linspace(0.0, jnp.pi, 60)
    >>> pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    >>> vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}

    >>> result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    >>> int(result.n_visited)
    60

    The result carries both a smooth backbone and the arc-length chord:

    >>> result.backbone["x"].shape
    (46,)
    >>> result.chord.shape
    (60,)

    As a refinement stage after another orderer:

    >>> chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.SOMOrderer(
    ...     n_prototypes=10
    ... )
    >>> int(pcf.order(pos, vel, chain).n_visited)
    60

    """

    n_prototypes: int = eqx.field(static=True, default=25)
    metric: AbstractDistanceMetric = eqx.field(default_factory=SpatialDistanceMetric)
    metric_scale: float = 0.0
    n_epochs: int = eqx.field(static=True, default=10)
    sigma_start: float | None = eqx.field(static=True, default=None)
    sigma_end: float = eqx.field(static=True, default=0.7)
    densify_factor: int = eqx.field(static=True, default=5)
    orient_by_velocity: bool = eqx.field(static=True, default=False)

    __citation__: ClassVar[str] = "https://arxiv.org/abs/2212.00949"

    def __check_init__(self) -> None:
        """Reject invalid configuration early, at construction."""
        if self.n_prototypes < 2:
            msg = f"n_prototypes must be >= 2, got {self.n_prototypes}."
            raise ValueError(msg)
        if self.densify_factor < 1:
            msg = f"densify_factor must be >= 1, got {self.densify_factor}."
            raise ValueError(msg)
        if self.n_epochs < 1:
            msg = f"n_epochs must be >= 1, got {self.n_epochs}."
            raise ValueError(msg)
        if self.sigma_end <= 0:
            msg = f"sigma_end must be positive, got {self.sigma_end}."
            raise ValueError(msg)
        if self.metric_scale != 0.0 and isinstance(self.metric, SpatialDistanceMetric):
            msg = (
                f"metric_scale={self.metric_scale} has no effect with "
                "SpatialDistanceMetric, which ignores it. For velocity to "
                "participate in the best-matching-unit search, pass "
                "metric=FullPhaseSpaceDistanceMetric() as well."
            )
            raise ValueError(msg)

    @plum.dispatch
    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,  # noqa: ARG002
        init: AbstractResult | None = None,
    ) -> OrderingResult:
        """Train a SOM and order the tracers by their chord parameter."""
        _check_component_keys(positions, velocities)

        full_q = {k: jnp.asarray(v) for k, v in positions.items()}
        full_p = {k: jnp.asarray(v) for k, v in velocities.items()}
        n_obs = int(next(iter(full_q.values())).shape[0])

        # Working set: everything, or exactly what a prior stage visited.
        # Points a prior stage rejected as outliers stay rejected.
        if init is None:
            work = jnp.arange(n_obs)
        else:
            # ``indices`` is the only ordering field ``AbstractResult`` guarantees.
            prior = jnp.asarray(init.indices)
            work = prior[prior >= 0]
        sub_q = {k: v[work] for k, v in full_q.items()}
        sub_p = {k: v[work] for k, v in full_p.items()}

        # With a prior stage, the subset is already in that stage's order, so
        # equi-frequency binning along it is binning along the identity.
        sub_ordering = None if init is None else jnp.arange(work.shape[0])

        proto_q, proto_p = _som.init_prototypes(
            sub_q, sub_p, n_prototypes=self.n_prototypes, ordering=sub_ordering
        )

        # The standalone default, ``K / 4``, smooths over a quarter of the
        # lattice on the first epoch and erases the prior stage's ordering. With
        # one supplied there is nothing to discover globally, so start at
        # ``sigma_end`` and do only the local smoothing the SOM is here for.
        sigma_start = (
            self.sigma_end
            if self.sigma_start is None and init is not None
            else self.sigma_start
        )
        backbone_q, backbone_p, lam = _train_and_project(
            self, proto_q, proto_p, sub_q, sub_p, sigma_start
        )

        if self.orient_by_velocity:
            # Flip so the chord runs along the mean velocity rather than
            # against it. Branchless to stay traceable; the arrays are
            # backbone-sized, so both branches are cheap.
            comps = sorted(backbone_q)
            bb = jnp.stack([backbone_q[k] for k in comps], axis=-1)
            bb_vel = jnp.stack([backbone_p[k] for k in comps], axis=-1)
            tangent = jnp.diff(bb, axis=0)
            vel_mid = 0.5 * (bb_vel[:-1] + bb_vel[1:])
            flip = jnp.sum(tangent * vel_mid) < 0.0
            total = jnp.sum(jnp.linalg.norm(tangent, axis=-1))
            # ``total - lam`` also reverses the end-cap extrapolations, which
            # sit outside [0, total], symmetrically.
            lam = jnp.where(flip, total - lam, lam)
            # Only ``backbone_q`` is carried on the result, so only it is
            # flipped; ``backbone_p`` is not read again.
            backbone_q = {k: jnp.where(flip, v[::-1], v) for k, v in backbone_q.items()}

        # ``stable=True`` is JAX's default, made explicit because the tie
        # behaviour is a contract: observations with equal chord values keep
        # the order they arrived in, i.e. the prior stage's ordering.
        ordered = work[jnp.argsort(lam, stable=True)]
        indices = (
            jnp.full(n_obs, -1, dtype=jnp.int32)
            .at[: ordered.shape[0]]
            .set(ordered.astype(jnp.int32))
        )
        chord_full = jnp.full(n_obs, jnp.nan, dtype=lam.dtype).at[work].set(lam)

        return OrderingResult(
            positions=full_q,
            velocities=full_p,
            indices=indices,
            gamma_range=(-1.0, 1.0),
            backbone=backbone_q,
            chord=chord_full,
        )
