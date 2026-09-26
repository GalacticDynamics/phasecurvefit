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

import warnings
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import plum

from .base import AbstractOrderer, _check_component_keys
from .result import OrderingResult
from phasecurvefit._src import som as _som
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import FSz0, FSzN, ISzN, VectorComponents
from phasecurvefit._src.metrics import (
    AbstractDistanceMetric,
    FullPhaseSpaceDistanceMetric,
    SpatialDistanceMetric,
)

# Rank correlation between the prior ordering and the refined one, below which
# the disagreement is worth a warning. Empirical; see ``_warn_if_disagrees``.
_DISAGREE_WARN = 0.5


@eqx.filter_jit
def _train_and_project(
    orderer: "SOMOrderer",
    proto_q: VectorComponents,
    proto_p: VectorComponents,
    sub_q: VectorComponents,
    sub_p: VectorComponents,
    sigma_start: float | None,
    metric: AbstractDistanceMetric,
    metric_scale: float | FSz0,
) -> tuple[VectorComponents, VectorComponents, FSzN]:
    """Run fit -> densify -> chord as one XLA program.

    Jitted so the metric's ``(N, M)`` intermediates fuse; run eagerly each is
    materialized separately, costing roughly 15x the memory at catalogue scale.
    ``init_prototypes`` stays outside so its ``eqx.error_if`` guard raises as an
    ordinary Python exception.
    """
    model = _som.SOM1D(
        proto_q,
        proto_p,
        metric,
        metric_scale=metric_scale,
        n_epochs=orderer.n_epochs,
        sigma_start=sigma_start,
        sigma_end=orderer.sigma_end,
        densify_factor=orderer.densify_factor,
    ).fit(sub_q, sub_p)
    backbone_q, backbone_p = model.backbone()
    # The functional `chord` against the backbone already in hand, rather than
    # `model.chord(...)`, which densifies a second time internally. XLA's CSE
    # collapses the duplicate, so this costs no measurable compile or run time
    # -- but it hands the tracer ~200 fewer StableHLO lines to build and then
    # optimize away (1149 -> 949 at n=5e3, K=40).
    lam = _som.chord(
        backbone_q, backbone_p, sub_q, sub_p, metric=metric, metric_scale=metric_scale
    )
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
        Phase-space distance metric for the best-matching-unit search and the
        backbone assignment. ``None`` (the default) follows ``init``: after a
        stage that used velocity it is
        :class:`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric`, and
        otherwise :class:`~phasecurvefit.metrics.SpatialDistanceMetric`.

        Ordering on position alone after a stage that used velocity undoes that
        stage's work: at a self-crossing the two branches are spatially
        coincident and only their velocities differ, so a position-only SOM
        re-conflates exactly what the earlier stage separated.

        ``metric`` must be symmetric in the two points it compares.
        :class:`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` is not:
        it scores "forward along the direction of travel", which is what a
        greedy walk step needs and not what nearest-prototype means. Passing it
        collapses the lattice toward the curve's head.
    metric_scale
        Scale handed to ``metric``. ``None`` (the default) derives it as
        ``sigma_phys / (2 |v|)`` when following a velocity-aware ``init``, and
        is ``0.0`` otherwise. The derived value makes the velocity term
        separate anti-parallel branches by about the distance the lattice can
        already resolve.

        Set it explicitly to override. It is a *time*, converting velocity
        differences into position units, so its right value depends on your
        unit system. Too large and "nearest prototype" becomes "nearest in
        velocity", which on a winding curve conflates points a whole turn
        apart. A non-zero value with
        :class:`~phasecurvefit.metrics.SpatialDistanceMetric`, which ignores
        it, is rejected at construction.
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
    metric: AbstractDistanceMetric | None = None
    metric_scale: float | None = None
    n_epochs: int = eqx.field(static=True, default=10)
    sigma_start: float | None = eqx.field(static=True, default=None)
    sigma_end: float = eqx.field(static=True, default=0.7)
    densify_factor: int = eqx.field(static=True, default=5)
    orient_by_velocity: bool = eqx.field(static=True, default=False)

    __citation__: ClassVar[str] = "https://arxiv.org/abs/2212.00949"

    def __check_init__(self) -> None:
        """Reject invalid configuration early, at construction."""
        if self.metric is not None and not self.metric.is_symmetric:
            # Documented on ``metric`` above; enforced here so the failure lands
            # at construction rather than deep inside a jitted fit.
            msg = (
                f"metric must be symmetric for nearest-prototype assignment; "
                f"{type(self.metric).__name__} is not. Passing it collapses the "
                f"lattice toward the curve's head."
            )
            raise ValueError(msg)
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
        if self.metric_scale is not None and jnp.ndim(self.metric_scale) != 0:
            # Shape is static even for a tracer, so this is safe to check when
            # the orderer is built inside a transform. Without it a non-scalar
            # reaches the truthiness test below and raises NumPy's generic
            # "truth value of an array ... is ambiguous", naming nothing.
            msg = (
                f"metric_scale must be a scalar; got shape "
                f"{jnp.shape(self.metric_scale)}. It is one time converting "
                f"velocity differences into position units, not a per-"
                f"observation array."
            )
            raise ValueError(msg)
        if self.metric_scale not in (None, 0.0) and isinstance(
            self.metric, SpatialDistanceMetric
        ):
            msg = (
                f"metric_scale={self.metric_scale} has no effect with "
                "SpatialDistanceMetric, which ignores it. For velocity to "
                "participate in the best-matching-unit search, pass "
                "metric=FullPhaseSpaceDistanceMetric() as well."
            )
            raise ValueError(msg)

    def _resolve_metric(
        self,
        sub_q: VectorComponents,
        sub_p: VectorComponents,
        init: AbstractResult | None,
    ) -> tuple[AbstractDistanceMetric, float | FSz0]:
        """Pick the metric and scale, following ``init`` when unset.

        A stage that orders on position alone, chained after one that used
        velocity, re-conflates whatever the earlier stage separated with it: at
        a self-crossing the two branches are spatially coincident and only
        their velocities differ. So when ``init`` reports ``velocity_aware`` and
        neither ``metric`` nor ``metric_scale`` was set explicitly, use the
        phase-space metric with a scale derived from the data.

        The derived scale is ``sigma_phys / (2 |v|)``: it makes the velocity
        term separate anti-parallel branches by about the same distance the
        lattice can already resolve. Larger and "nearest prototype" becomes
        "nearest in velocity", which on a winding curve conflates points a whole
        turn apart.
        """
        follows_velocity = init is not None and getattr(init, "velocity_aware", False)

        # The metric and the scale are *independent* choices. Resolving them
        # together is what made naming a velocity-aware ``metric`` force the
        # scale to zero: the caller asked for more velocity awareness and got a
        # numerically position-only SOM, still reporting ``velocity_aware``.
        if self.metric is not None:
            metric = self.metric
        elif self.metric_scale is not None:
            # A non-zero scale is a request for velocity to participate, so pick
            # a metric that uses it rather than one that discards it. An
            # explicit zero is a request for position only.
            metric = (
                FullPhaseSpaceDistanceMetric()
                if self.metric_scale
                else SpatialDistanceMetric()
            )
        else:
            metric = (
                FullPhaseSpaceDistanceMetric()
                if follows_velocity
                else SpatialDistanceMetric()
            )

        # ``None`` means "unset, derive"; every other value is an explicit
        # choice, ``0.0`` included -- testing truthiness here would make
        # ``metric_scale=0.0`` mean "unset" and quietly re-derive a non-zero
        # scale, which is the opposite of what the caller asked for.
        if self.metric_scale is not None:
            return metric, self.metric_scale
        if not (follows_velocity and metric.uses_velocity):
            # Nothing to derive: either there is no velocity-aware stage to
            # follow, or the metric discards the velocity term regardless.
            return metric, 0.0

        comps = sorted(sub_q)
        q = jnp.stack([sub_q[k] for k in comps], axis=-1)
        v = jnp.stack([sub_p[k] for k in comps], axis=-1)
        # ``sub_q`` is already in the prior stage's order, so consecutive
        # differences are steps along the track.
        length = jnp.sum(jnp.linalg.norm(jnp.diff(q, axis=0), axis=-1))
        sigma_phys = self.sigma_end * length / (self.n_prototypes - 1)
        speed = jnp.median(jnp.linalg.norm(v, axis=-1))
        # Left as a JAX scalar rather than ``float(scale)``: ``_train_and_project``
        # is ``eqx.filter_jit``-ed, which treats a Python float as static, so a
        # data-derived scale would trigger a recompile for every new dataset
        # (measured: 0.33 s vs 0.009 s on a cached shape).
        scale = jnp.where(speed > 0, sigma_phys / (2.0 * speed), 0.0)
        return metric, scale

    def _warn_if_disagrees(
        self, perm: ISzN, sub_q: VectorComponents, init: AbstractResult | None
    ) -> None:
        """Warn when the refined ordering disagrees wholesale with its input.

        A refinement stage normally keeps most of the order it is handed and
        adjusts locally. Wholesale disagreement has two causes and this cannot
        tell them apart: the prior ordering was poor and the SOM genuinely
        overhauled it, or the lattice is too coarse for the curve and the SOM
        has tangled a good ordering. Both are worth a look, so the warning
        names the smoothing length and asks the caller to compare.

        ``_DISAGREE_WARN`` is empirical. Measured on a self-intersecting
        epitrochoid, a lattice that destroys the ordering scores 0.28 while
        every lattice that improves it scores 0.89 or above.
        """
        if init is None or perm.shape[0] < 3:
            return
        # ``sub_q`` is in the prior stage's order, so the prior rank is just
        # position in the array. ``perm`` is the ordering ``order`` already
        # computed; inverting it gives each observation's new rank for the
        # price of a scatter, rather than a second sort.
        n = perm.shape[0]
        prior = jnp.arange(n)
        rank = jnp.zeros(n, dtype=perm.dtype).at[perm].set(prior)
        rho = float(jnp.abs(jnp.corrcoef(prior, rank)[0, 1]))
        if rho >= _DISAGREE_WARN:
            return
        comps = sorted(sub_q)
        q = jnp.stack([sub_q[k] for k in comps], axis=-1)
        length = float(jnp.sum(jnp.linalg.norm(jnp.diff(q, axis=0), axis=-1)))
        sigma_phys = self.sigma_end * length / (self.n_prototypes - 1)
        warnings.warn(
            f"SOMOrderer's ordering disagrees with the one it was given (rank "
            f"correlation {rho:.2f}). Either the prior ordering was poor and this "
            f"is a genuine overhaul, or the lattice is too coarse for the curve "
            f"and has tangled a good ordering: n_prototypes={self.n_prototypes} "
            f"gives a smoothing length of {sigma_phys:.3g}, "
            f"{100 * sigma_phys / length:.1f}% of the track, and any structure "
            f"finer than that is smoothed away. Compare the two orderings, and "
            f"raise n_prototypes if the curve turns or self-approaches more "
            f"tightly than the smoothing length.",
            UserWarning,
            stacklevel=3,
        )

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
            # The core's own range guard cannot see these: the subset below is
            # re-indexed to ``arange(len(work))`` before it reaches
            # ``init_prototypes``. Without this, an ``init`` carried over from a
            # larger dataset gathers out of bounds, which JAX silently clamps --
            # the result then reports every observation visited while several
            # carry a nan chord.
            prior = eqx.error_if(
                prior,
                jnp.any(prior >= n_obs),
                "init.indices contains an index past the end of the data; the "
                "prior result was computed on a different, larger dataset. "
                f"Indices must be < n_obs ({n_obs}), with -1 for unvisited.",
            )
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
        metric, metric_scale = self._resolve_metric(sub_q, sub_p, init)
        backbone_q, backbone_p, lam = _train_and_project(
            self, proto_q, proto_p, sub_q, sub_p, sigma_start, metric, metric_scale
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
        perm = jnp.argsort(lam, stable=True)
        self._warn_if_disagrees(perm, sub_q, init)
        ordered = work[perm]
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
            # The *resolved* metric, not ``self.metric``: standalone it is
            # position-only, but chained after a velocity-aware stage it is
            # derived from ``init``. A stage after this one reads the flag off
            # the result, so it has to report what was actually used.
            velocity_aware=metric.uses_velocity,
        )
