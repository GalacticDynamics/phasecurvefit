"""Tests for SOMOrderer."""

import re
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import phasecurvefit as pcf
from phasecurvefit import som
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.orderers import som as som_orderer


def test_som_orderer_visits_everything_standalone(arc):
    pos, vel, _ = arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=12))
    assert int(result.n_visited) == 200
    assert bool(result.all_visited)


def test_som_orderer_recovers_the_arc_order(arc, spearman):
    """Scored against the curve's own parameter, so the input order is irrelevant.

    Monotonicity is the claim, not exactness; the core's
    ``test_chord_is_exact_on_a_straight_backbone`` covers exactness.
    """
    curve = arc()
    result = pcf.order(
        curve.positions, curve.velocities, pcf.orderers.SOMOrderer(n_prototypes=15)
    )
    assert spearman(result, curve) > 0.99


def test_som_orderer_sets_backbone_and_chord(arc):
    pos, vel, _ = arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    assert result.backbone is not None
    assert result.backbone["x"].shape == (5 * (10 - 1) + 1,)
    assert result.chord is not None
    assert result.chord.shape == (200,)
    assert not bool(jnp.any(jnp.isnan(result.chord)))


def test_som_orderer_gamma_range(arc):
    pos, vel, _ = arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    assert result.gamma_range == (-1.0, 1.0)


def test_som_orderer_refines_a_prior_ordering(arc):
    pos, vel, _ = arc()
    chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.SOMOrderer(
        n_prototypes=12
    )
    result = pcf.order(pos, vel, chain)
    assert int(result.n_visited) == 200
    assert result.chord is not None


def test_som_orderer_chained_seeding_survives_more_than_one_turn(helix, spearman):
    """The chained path must seed ``init_prototypes`` with the prior ordering.

    Falling back to PCA binning (``ordering=None``) tangles the lattice on
    ``_helix``, while a genuinely threaded ordering still resolves cleanly.
    """
    curve = helix(turns=1.5, radius=5.0, height=10.0)
    chain = pcf.orderers.MSTOrderer(k=8, jump_cap=4.0) | pcf.orderers.SOMOrderer(
        n_prototypes=12
    )
    result = pcf.order(curve.positions, curve.velocities, chain)
    assert spearman(result, curve) > 0.99


def test_som_orderer_propagates_skips():
    """Points a prior stage rejected stay unvisited and carry nan chord."""
    xs = jnp.concatenate([jnp.linspace(0.0, 9.0, 40), jnp.array([30.0])])
    ys = jnp.concatenate([jnp.zeros(40), jnp.array([30.0])])
    pos = {"x": xs, "y": ys}
    vel = {"x": jnp.ones(41), "y": jnp.zeros(41)}
    chain = pcf.orderers.MSTOrderer(
        k=10, jump_cap=50.0, edge_clip_sigma=3.0
    ) | pcf.orderers.SOMOrderer(n_prototypes=8)
    result = pcf.order(pos, vel, chain)
    assert int(result.n_skipped) == 1
    assert int(jnp.sum(jnp.isnan(result.chord))) == 1


def test_som_orderer_rejects_mismatched_keys():
    pos = {"x": jnp.arange(10.0), "y": jnp.zeros(10)}
    vel = {"x": jnp.ones(10)}
    with pytest.raises(ValueError, match="same component keys"):
        pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=5))


def test_som_orderer_rejects_bad_hyperparameters():
    with pytest.raises(ValueError, match="n_prototypes"):
        pcf.orderers.SOMOrderer(n_prototypes=1)
    with pytest.raises(ValueError, match="densify_factor"):
        pcf.orderers.SOMOrderer(densify_factor=0)


def test_som_orderer_chained_below_default_n_prototypes_raises_clearly():
    """Chaining with fewer visited points than the default ``n_prototypes=25``.

    ``docs/guides/som.md`` and the tutorials chain ``MSTOrderer() |
    SOMOrderer()`` with no override, so this is the exact path a reader would
    hit on a dataset small enough (or clipped enough) to drop the visited
    count below 25. Every other chained test in this file lowers
    ``n_prototypes`` to fit its fixture, so this crash path itself had no
    coverage -- this asserts it fails with the core's own clear message
    rather than, say, silently misbehaving.
    """
    n = 20
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    chain = pcf.orderers.MSTOrderer(k=5, jump_cap=2.0) | pcf.orderers.SOMOrderer()
    with pytest.raises(ValueError, match="binning needs at least n_prototypes"):
        pcf.order(pos, vel, chain)


def test_som_orderer_citation_is_set():
    assert pcf.orderers.SOMOrderer.__citation__ == "https://arxiv.org/abs/2212.00949"


def test_som_orderer_accepts_any_abstract_result_as_init(arc):
    """``init`` only needs ``indices``, the field ``AbstractResult`` guarantees.

    ``ordering`` is an ``OrderingResult`` property, not part of the
    ``AbstractResult`` contract the ``order()`` signature advertises, so a
    custom result type must still be chainable into ``SOMOrderer``.
    """

    class BareResult(AbstractResult):
        """Minimal AbstractResult: no ``ordering`` property at all."""

        def __call__(self, gamma, /, *, key=None):
            raise NotImplementedError

    pos, vel, _ = arc(n=60)
    # Reverse order, with the last 10 points marked unvisited.
    indices = jnp.concatenate(
        [jnp.arange(49, -1, -1, dtype=jnp.int32), jnp.full(10, -1, dtype=jnp.int32)]
    )
    assert not hasattr(BareResult, "ordering")

    prior = BareResult(positions=pos, velocities=vel, indices=indices)
    # The prior ordering is deliberately arbitrary, so the SOM rightly reports
    # that it overhauled it. This test is about the `init` contract, not quality.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*disagrees with the one it was given.*"
        )
        result = pcf.orderers.SOMOrderer(n_prototypes=8).order(pos, vel, init=prior)

    assert int(result.n_visited) == 50
    assert int(result.n_skipped) == 10
    assert int(jnp.sum(jnp.isnan(result.chord))) == 10


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["flow+x", "flow-x"])
def test_orient_by_velocity_orders_along_the_flow(sign, straight):
    """With the flag set, the chord runs along the mean velocity either way.

    Without it the direction comes from the sign of the initializer's
    principal-axis eigenvector, which is stable but arbitrary -- the SOM has no
    progenitor anchor.
    """
    pos, vel, _ = straight(sign=sign)
    result = pcf.order(
        pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10, orient_by_velocity=True)
    )
    order = result.ordering
    first, last = float(pos["x"][order[0]]), float(pos["x"][order[-1]])
    assert (last - first) * sign > 0


def test_orient_by_velocity_is_a_noop_when_already_aligned(straight):
    pos, vel, _ = straight(sign=1.0)
    plain = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    oriented = pcf.order(
        pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10, orient_by_velocity=True)
    )
    assert jnp.array_equal(plain.indices, oriented.indices)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["flow+x", "flow-x"])
def test_orient_by_velocity_keeps_backbone_and_ordering_in_step(sign, straight):
    """The backbone must flip with the chord, or ``__call__`` desyncs from it.

    ``gamma_range`` runs (-1, 1) over the backbone, so interpolating at the low
    end must land near the first ordered observation, not the last.
    """
    pos, vel, _ = straight(sign=sign)
    result = pcf.order(
        pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10, orient_by_velocity=True)
    )
    order = result.ordering
    first_x = float(pos["x"][order[0]])
    last_x = float(pos["x"][order[-1]])

    lo = float(result(jnp.array(result.gamma_range[0]))["x"])
    hi = float(result(jnp.array(result.gamma_range[1]))["x"])

    assert abs(lo - first_x) < abs(lo - last_x)
    assert abs(hi - last_x) < abs(hi - first_x)


def test_som_orderer_rejects_a_scale_the_chosen_metric_ignores():
    """A non-zero scale with an explicit SpatialDistanceMetric is a silent no-op."""
    with pytest.raises(ValueError, match="no effect with SpatialDistanceMetric"):
        pcf.orderers.SOMOrderer(
            metric=pcf.metrics.SpatialDistanceMetric(), metric_scale=0.2
        )

    # The documented remedy constructs fine.
    pcf.orderers.SOMOrderer(
        metric=pcf.metrics.FullPhaseSpaceDistanceMetric(), metric_scale=0.2
    )


def test_a_bare_scale_selects_a_metric_that_uses_it(arc):
    """Ask for velocity with no metric; the scale must reach one that uses it."""
    pos, vel, _ = arc()
    metric, scale = pcf.orderers.SOMOrderer(metric_scale=0.2)._resolve_metric(
        pos, vel, None
    )
    assert isinstance(metric, pcf.metrics.FullPhaseSpaceDistanceMetric)
    assert scale == 0.2


def _max_coverage_gap(proto, pos, keys=("x", "y")):
    """Largest distance from any datum to its nearest prototype."""
    p = jnp.stack([proto[k] for k in keys], axis=-1)
    d = jnp.stack([pos[k] for k in keys], axis=-1)
    return float(
        jnp.max(jnp.min(jnp.linalg.norm(d[:, None] - p[None], axis=-1), axis=1))
    )


def test_ordering_result_carries_chord():
    pos = {"x": jnp.linspace(0.0, 3.0, 4), "y": jnp.zeros(4)}
    vel = {"x": jnp.ones(4), "y": jnp.zeros(4)}
    result = pcf.orderers.OrderingResult(
        positions=pos,
        velocities=vel,
        indices=jnp.arange(4),
        chord=jnp.array([0.0, 1.0, 2.0, 3.0]),
    )
    assert result.chord is not None
    assert result.chord.shape == (4,)


def test_ordering_result_chord_defaults_to_none():
    pos = {"x": jnp.linspace(0.0, 3.0, 4)}
    vel = {"x": jnp.ones(4)}
    result = pcf.orderers.OrderingResult(
        positions=pos, velocities=vel, indices=jnp.arange(4)
    )
    assert result.chord is None


def test_fit_with_the_default_metric_covers_the_data():
    """The shipped default must train a lattice that spans the stream.

    A metric that is asymmetric in the two points it compares -- one scoring
    "forward along the direction of travel", as a greedy walk step does --
    drags every prototype toward the curve's head instead, leaving the far end
    uncovered while still returning a plausible-looking lattice.
    """
    n = 400
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}

    orderer = pcf.orderers.SOMOrderer(n_prototypes=11)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=11)
    result = som.fit(
        pq,
        pp,
        pos,
        vel,
        metric=orderer._resolve_metric(pos, vel, None)[0],
        metric_scale=orderer.metric_scale,
        n_epochs=50,
    )

    # Ideal spacing for 11 prototypes over 10 units is 1.0, so a lattice that
    # tracks the data leaves a gap well under 1; a collapsed one leaves > 2.
    assert _max_coverage_gap(result.prototype_positions, pos) < 1.0


def test_som_orderer_default_metric_is_symmetric(arc):
    """The default metric must be symmetric in the two points it compares.

    Symmetry is the contract: "which prototype is this datum nearest" is
    meaningless under an asymmetric metric.

    The property must be checked at a *non-zero* scale. At the shipped
    ``metric_scale=0.0`` every metric in the package degenerates to plain
    position distance, so a symmetry assertion there passes for
    ``AlignedMomentumDistanceMetric`` too and guards nothing.
    """
    orderer = pcf.orderers.SOMOrderer()
    resolved, _ = orderer._resolve_metric(*arc()[:2], None)
    assert isinstance(resolved, pcf.metrics.SpatialDistanceMetric)

    a_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    a_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    b_pos = {"x": jnp.array(1.0), "y": jnp.array(0.5)}
    b_vel = {"x": jnp.array(-1.0), "y": jnp.array(0.3)}
    as_arr = lambda d: {k: v[None] for k, v in d.items()}

    def probe(metric, scale):
        fwd = float(metric(a_pos, a_vel, as_arr(b_pos), as_arr(b_vel), scale)[0])
        bwd = float(metric(b_pos, b_vel, as_arr(a_pos), as_arr(a_vel), scale)[0])
        return fwd, bwd

    for metric in (resolved, pcf.metrics.FullPhaseSpaceDistanceMetric()):
        fwd, bwd = probe(metric, 0.5)
        assert fwd == pytest.approx(bwd, rel=1e-6)

    # The asymmetric metric fails it, so the assertion can fail.
    fwd, bwd = probe(pcf.metrics.AlignedMomentumDistanceMetric(), 0.5)
    assert fwd != pytest.approx(bwd, rel=1e-6)


def _tight_helix(n=400, turns=1.5, pitch=0.35, scatter=0.02, seed=3):
    """Build a helix whose turns sit below the default smoothing length.

    ``sigma_phys = sigma_end * L / (K - 1)`` is what the SOM smooths over. On an
    open curve a wide initial neighbourhood costs nothing, so a loose fixture
    cannot tell a preserved prior ordering from a discarded one. Here the turns
    are close enough that a wide start merges them.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    ang = 2 * np.pi * turns * t
    jit = lambda a: jnp.asarray(a + rng.normal(0.0, scatter, n))
    pos = {"x": jit(np.cos(ang)), "y": jit(np.sin(ang)), "z": jit(pitch * turns * t)}
    vel = {
        "x": jnp.asarray(-np.sin(ang)),
        "y": jnp.asarray(np.cos(ang)),
        "z": jnp.asarray(np.full(n, pitch)),
    }
    return pos, vel


def test_chained_som_preserves_the_prior_ordering():
    """A prior ordering must survive the SOM, not be smoothed away by it.

    ``sigma_start`` defaults to ``K / 4``, which smooths over a quarter of the
    lattice on the first epoch. That is right when the SOM has to find the
    global ordering itself, and wrong when a previous stage already did: it
    erases that stage's work. ``SOMOrderer`` therefore holds sigma at
    ``sigma_end`` when ``init`` is supplied.
    """
    pos, vel = _tight_helix()
    truth = np.arange(pos["x"].shape[0])
    chain = (
        pcf.orderers.MSTOrderer(k=8, jump_cap=1.0, on_disconnected="largest")
        | pcf.orderers.SOMOrderer()
    )
    result = pcf.order(pos, vel, chain)

    ranks = np.empty(truth.shape[0], dtype=int)
    order = np.asarray(result.ordering)
    ranks[order] = np.arange(order.shape[0])
    rho = spearmanr(ranks, truth).statistic
    assert abs(rho) > 0.95


@pytest.fixture
def epitrochoid_mst(epitrochoid):
    """Return the epitrochoid with the velocity-aware MST it needs.

    The jump cap is scaled to the data's own nearest-neighbour spacing.
    """
    pos, vel, truth = epitrochoid()
    d = np.stack([np.asarray(pos["x"]), np.asarray(pos["y"])], axis=1)
    med = float(np.median(cKDTree(d).query(d, k=2)[0][:, 1]))
    base = pcf.orderers.MSTOrderer(
        k=16,
        jump_cap=8.0 * med,
        sever_cos_threshold=0.9,
        orient_by_velocity=True,
        on_disconnected="largest",
    )
    return pos, vel, truth, base


class TestVelocityAwarePropagation:
    """A velocity-aware prior stage must make the SOM velocity-aware too."""

    def test_mst_reports_velocity_awareness(self, arc):
        """Only the settings that steer the ordering count, not orient_by_velocity."""
        pos, vel, _ = arc()
        plain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0)
        assert plain.order(pos, vel).velocity_aware is False
        for kw in ({"sever_cos_threshold": 0.9}, {"velocity_weight": 0.5}):
            orderer = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, **kw)
            assert orderer.order(pos, vel).velocity_aware is True
        oriented = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, orient_by_velocity=True)
        assert oriented.order(pos, vel).velocity_aware is False

    def test_localflow_reports_its_metrics_answer(self, arc):
        """The flag follows the metric, not the scale.

        A zero scale makes a phase-space metric numerically position-only, but
        the walk is still *configured* to follow the flow. Keying on the metric
        also keeps the flag static: ``metric_scale`` is a differentiable leaf
        and is a tracer under ``jit``/``grad``.
        """
        pos, vel, _ = arc()
        default = pcf.orderers.LocalFlowOrderer()
        assert default.config.metric.uses_velocity is True
        assert default.order(pos, vel).velocity_aware is True

        # ...including at a zero scale: the metric still reads velocity.
        zero = pcf.orderers.LocalFlowOrderer(metric_scale=0.0)
        assert zero.order(pos, vel).velocity_aware is True

        # A position-only metric is the thing that makes it False.
        spatial = pcf.orderers.LocalFlowOrderer(
            config=pcf.WalkConfig(metric=pcf.metrics.SpatialDistanceMetric())
        )
        assert spatial.order(pos, vel).velocity_aware is False

    def test_som_follows_a_velocity_aware_init(self, arc):
        """The resolved metric tracks ``init``, and an explicit choice still wins."""
        pos, vel, _ = arc()
        som = pcf.orderers.SOMOrderer(n_prototypes=20)
        plain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0).order(pos, vel)
        velaware = pcf.orderers.MSTOrderer(
            k=8, jump_cap=3.0, sever_cos_threshold=0.9
        ).order(pos, vel)

        metric, scale = som._resolve_metric(pos, vel, plain)
        assert isinstance(metric, pcf.metrics.SpatialDistanceMetric)
        assert scale == 0.0

        metric, scale = som._resolve_metric(pos, vel, velaware)
        assert isinstance(metric, pcf.metrics.FullPhaseSpaceDistanceMetric)
        assert scale > 0.0

        explicit = pcf.orderers.SOMOrderer(
            n_prototypes=20, metric=pcf.metrics.SpatialDistanceMetric()
        )
        metric, scale = explicit._resolve_metric(pos, vel, velaware)
        assert isinstance(metric, pcf.metrics.SpatialDistanceMetric)

    def test_standalone_som_stays_position_only(self, arc):
        """With no ``init`` there is nothing to follow, so the default is unchanged."""
        pos, vel, _ = arc()
        som = pcf.orderers.SOMOrderer(n_prototypes=20)
        metric, scale = som._resolve_metric(pos, vel, None)
        assert isinstance(metric, pcf.metrics.SpatialDistanceMetric)
        assert scale == 0.0

    def test_velocity_awareness_survives_the_crossings(self, epitrochoid_mst, spearman):
        """The whole point: the chain must not re-conflate the crossed branches."""
        pos, vel, truth, base = epitrochoid_mst
        rho_base = spearman(base.order(pos, vel), truth)
        chained = (base | pcf.orderers.SOMOrderer(n_prototypes=40)).order(pos, vel)
        # Position-only is what the SOM did before it followed ``init``.
        position_only = (
            base
            | pcf.orderers.SOMOrderer(
                n_prototypes=40, metric=pcf.metrics.SpatialDistanceMetric()
            )
        ).order(pos, vel)
        rho_auto = spearman(chained, truth)
        rho_pos = spearman(position_only, truth)
        assert rho_pos < rho_base - 0.2, "fixture no longer exercises the failure"
        assert rho_auto > rho_pos + 0.2
        assert rho_auto >= rho_base


def test_som_warns_when_it_overhauls_the_prior_ordering():
    """A lattice too coarse for the curve tangles it; that must not be silent."""
    pos, vel = _tight_helix()
    base = pcf.orderers.MSTOrderer(
        k=8, jump_cap=1.0, on_disconnected="largest", sever_cos_threshold=0.9
    )
    prior = base.order(pos, vel)
    som = pcf.orderers.SOMOrderer(n_prototypes=3)
    rng = np.random.default_rng(0)
    perm = jnp.asarray(rng.permutation(int((prior.indices >= 0).sum())))
    sub = {k: v[prior.ordering] for k, v in pos.items()}
    kept = jnp.ones(perm.shape[0], dtype=bool)
    with pytest.warns(UserWarning, match="disagrees with the one it was given"):
        som._warn_if_disagrees(perm, kept, sub, prior)


def test_som_is_quiet_when_it_agrees_with_the_prior_ordering(arc):
    """The warning must not fire on the ordinary case of a small refinement."""
    pos, vel, _ = arc()
    base = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, sever_cos_threshold=0.9)
    prior = base.order(pos, vel)
    som = pcf.orderers.SOMOrderer(n_prototypes=40)
    sub = {k: v[prior.ordering] for k, v in pos.items()}
    perm = jnp.arange(int((prior.indices >= 0).sum()))
    kept = jnp.ones(perm.shape[0], dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        som._warn_if_disagrees(perm, kept, sub, prior)


def test_velocity_aware_stays_concrete_under_grad():
    """``metric_scale`` is differentiable, so the static flag must not trace it."""
    pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
    vel = {"x": jnp.array([1.0, 1.1, 1.2, 1.3])}

    def loss(metric_scale):
        result = pcf.order(
            pos,
            vel,
            pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=metric_scale),
        )
        return jnp.sum(result.indices.astype(jnp.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        jax.grad(loss)(1.0)


@pytest.mark.parametrize(
    ("kwargs", "velocity_aware", "expected_metric", "expected_scale"),
    [
        ({"metric_scale": 0.0}, True, pcf.metrics.SpatialDistanceMetric, 0.0),
        ({"metric_scale": 0.0}, False, pcf.metrics.SpatialDistanceMetric, 0.0),
        ({"metric_scale": 0.5}, False, pcf.metrics.FullPhaseSpaceDistanceMetric, 0.5),
        ({}, False, pcf.metrics.SpatialDistanceMetric, 0.0),
    ],
    ids=["explicit-zero-after-velocity", "explicit-zero", "explicit-scale", "unset"],
)
def test_explicit_metric_scale_is_honoured(
    kwargs, velocity_aware, expected_metric, expected_scale, arc
):
    """`metric_scale=0.0` is a choice, not "unset".

    Testing truthiness would make an explicit zero mean "follow ``init``", so a
    caller asking for position-only after a velocity-aware stage would silently
    get a derived non-zero scale instead.
    """
    pos, vel, _ = arc()
    init = pcf.orderers.OrderingResult(
        positions=pos,
        velocities=vel,
        indices=jnp.arange(next(iter(pos.values())).shape[0], dtype=jnp.int32),
        velocity_aware=velocity_aware,
    )
    metric, scale = pcf.orderers.SOMOrderer(n_prototypes=10, **kwargs)._resolve_metric(
        pos, vel, init
    )
    assert isinstance(metric, expected_metric)
    assert scale == expected_scale


def test_som_orderer_rejects_an_asymmetric_metric():
    """Documented on ``metric``; enforced at construction, not mid-fit.

    ``AlignedMomentumDistanceMetric`` scores "forward along the direction of
    travel", so every datum prefers what lies ahead of it and the lattice
    collapses toward the curve's head.
    """
    with pytest.raises(ValueError, match="metric must be symmetric"):
        pcf.orderers.SOMOrderer(metric=pcf.metrics.AlignedMomentumDistanceMetric())


class TestMetricAndScaleResolveIndependently:
    """`metric` and `metric_scale` are separate choices.

    Resolving them together meant naming a velocity-aware ``metric`` forced the
    scale to 0.0 -- asking for *more* velocity awareness and getting a
    numerically position-only SOM, which still reported ``velocity_aware``.
    """

    @staticmethod
    def _hairpin(n=50):
        """Two anti-parallel arms 0.04 apart: separable only with velocity."""
        s = np.linspace(0, 10, n)
        pos = {
            "x": jnp.asarray(np.r_[s, s[::-1]]),
            "y": jnp.asarray(np.r_[np.full(n, 0.02), np.full(n, -0.02)]),
        }
        vel = {"x": jnp.asarray(np.r_[np.ones(n), -np.ones(n)]), "y": jnp.zeros(2 * n)}
        return pos, vel

    @pytest.mark.parametrize(
        ("kwargs", "expect_metric", "expect_scale"),
        [
            ({}, "FullPhaseSpaceDistanceMetric", None),
            (
                {"metric": pcf.metrics.FullPhaseSpaceDistanceMetric()},
                "FullPhaseSpaceDistanceMetric",
                None,
            ),
            (
                {"metric": pcf.metrics.SpatialDistanceMetric()},
                "SpatialDistanceMetric",
                0.0,
            ),
            ({"metric_scale": 2.5}, "FullPhaseSpaceDistanceMetric", 2.5),
            ({"metric_scale": 0.0}, "SpatialDistanceMetric", 0.0),
        ],
        ids=["both-unset", "metric-only", "spatial-only", "scale-only", "scale-zero"],
    )
    def test_resolution_matrix(self, kwargs, expect_metric, expect_scale):
        """A velocity-aware ``init`` derives a scale unless one was given."""
        pos, vel = self._hairpin()
        head = pcf.orderers.MSTOrderer(k=6, jump_cap=5.0, velocity_weight=1.0)
        init = head.order(pos, vel)
        sub_q = {k: v[init.ordering] for k, v in pos.items()}
        sub_p = {k: v[init.ordering] for k, v in vel.items()}
        resolve = lambda **kw: pcf.orderers.SOMOrderer(
            n_prototypes=30, **kw
        )._resolve_metric(sub_q, sub_p, init)

        metric, scale = resolve(**kwargs)
        assert type(metric).__name__ == expect_metric
        if expect_scale is None:
            # Pinned against the default resolution, not a magic number: the
            # `metric-only` cell must derive exactly what `both-unset` does.
            assert float(scale) == pytest.approx(float(resolve()[1]))
            assert float(scale) > 0.0
        else:
            assert float(scale) == pytest.approx(expect_scale)

    def test_naming_the_metric_does_not_disable_velocity(self):
        """The end-to-end symptom: the arms interleaved instead of separating."""
        pos, vel = self._hairpin()
        head = pcf.orderers.MSTOrderer(k=6, jump_cap=5.0, velocity_weight=1.0)
        n = len(pos["x"]) // 2
        switches = {}
        for label, orderer in (
            ("default", pcf.orderers.SOMOrderer(n_prototypes=30)),
            (
                "explicit",
                pcf.orderers.SOMOrderer(
                    n_prototypes=30,
                    metric=pcf.metrics.FullPhaseSpaceDistanceMetric(),
                ),
            ),
        ):
            idx = np.asarray(pcf.order(pos, vel, head | orderer).indices)
            idx = idx[idx >= 0]
            switches[label] = int(np.abs(np.diff((idx >= n).astype(int))).sum())
        # One switch is a perfect traversal: down one arm and back the other.
        assert switches["default"] == 1
        assert switches["explicit"] == switches["default"]


@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("below -1", lambda n: jnp.concatenate([jnp.arange(n - 3), jnp.full(3, -5)])),
        ("past the end", lambda n: jnp.arange(5, 5 + n)),
    ],
)
def test_som_orderer_rejects_init_indices_outside_the_contract(label, make):
    """`indices` must lie in [-1, n_obs), with -1 and only -1 for unvisited.

    A sentinel below -1 is not caught by an upper-bound check: the `>= 0`
    filter treats it as unvisited, so those observations vanish from the
    working set with no error and `n_visited` silently shrinks.
    """
    n = 40
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    init = pcf.orderers.OrderingResult(
        positions=pos, velocities=vel, indices=make(n), velocity_aware=True
    )
    with pytest.raises(ValueError, match=r"outside \[-1, n_obs\)"):
        pcf.orderers.SOMOrderer(n_prototypes=6).order(pos, vel, init=init)


def test_som_orderer_rejects_a_repeated_visited_index():
    """A repeat in `init.indices` must be caught here, not left to slip past.

    `sub_ordering` is a synthetic `arange` once chained (the subset is already
    in the prior stage's order), so `init_prototypes`'s own repeat guard never
    sees the real indices and can never catch this -- the check has to live in
    `order()` itself, against `init.indices` directly.
    """
    n = 40
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    # Index 1 repeated; index 2 never appears.
    indices = jnp.concatenate(
        [jnp.array([0, 1, 1, 3]), jnp.arange(4, n - 4), jnp.full(4, -1)]
    )
    init = pcf.orderers.OrderingResult(
        positions=pos, velocities=vel, indices=indices, velocity_aware=True
    )
    with pytest.raises(ValueError, match="repeated visited index"):
        pcf.orderers.SOMOrderer(n_prototypes=6).order(pos, vel, init=init)


def test_som_orderer_accepts_the_documented_minus_one_padding():
    """The guard must not reject legitimate -1 padding."""
    n = 40
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    padded = jnp.concatenate([jnp.arange(n - 3), -jnp.ones(3, dtype=jnp.int32)])
    init = pcf.orderers.OrderingResult(
        positions=pos, velocities=vel, indices=padded, velocity_aware=True
    )
    result = pcf.orderers.SOMOrderer(n_prototypes=6).order(pos, vel, init=init)
    assert int(result.n_visited) == n - 3


@pytest.mark.parametrize(
    "scale",
    [None, 0.0, 0.2, jnp.array(0.0), jnp.array(0.2)],
    ids=["none", "zero", "float", "jax-0d-zero", "jax-0d"],
)
def test_som_orderer_accepts_any_scalar_metric_scale(scale):
    """Python floats, 0-d arrays and `None` are all legitimate scalars."""
    assert pcf.orderers.SOMOrderer(n_prototypes=12, metric_scale=scale) is not None


@pytest.mark.parametrize("shape", [(1,), (2,)], ids=["one-element", "many"])
def test_som_orderer_rejects_a_non_scalar_metric_scale(shape):
    """A non-scalar previously hit NumPy's generic ambiguous-truth-value error.

    That message names neither `metric_scale` nor the shape, so the caller has
    nothing to go on. Shape is static even under a trace, so this check is safe
    when the orderer is built inside a transform.
    """
    with pytest.raises(ValueError, match="metric_scale must be a scalar"):
        pcf.orderers.SOMOrderer(n_prototypes=12, metric_scale=jnp.ones(shape))


class TestTracedMetricScale:
    """Choosing a metric from ``metric_scale`` is a *static* decision.

    It selects which distance-metric object runs, not a number, so it cannot
    be deferred to ``lax.cond``. A traced scale cannot answer it, and JAX's
    own "Attempted boolean conversion of traced array with shape bool[]"
    names neither the field nor the way out.

    Only ``jit`` reaches this. ``grad`` is fine -- a linearization tracer
    carries a concrete primal, so the comparison resolves -- and a Python
    ``float`` handed to ``filter_jit`` stays static, so the ordinary path of
    passing a built orderer through a transform never trips it.
    """

    @staticmethod
    def _curve(n=40):
        t = jnp.linspace(0.0, 2.0, n)
        return (
            {"x": jnp.cos(t), "y": jnp.sin(t)},
            {"x": -jnp.sin(t), "y": jnp.cos(t)},
        )

    def test_explicit_metric_makes_a_traced_scale_work_under_jit(self):
        """Naming the metric removes the only static question about the scale.

        ``isinstance(self.metric, ...)`` is static, so both the ``__check_init__``
        guard and ``_resolve_metric`` short-circuit before comparing the scale
        to zero -- and the orderer becomes buildable inside ``jit``.
        """
        pos, vel = self._curve()

        @jax.jit
        def run(scale):
            orderer = pcf.orderers.SOMOrderer(
                n_prototypes=8,
                metric_scale=scale,
                metric=pcf.metrics.FullPhaseSpaceDistanceMetric(),
            )
            return pcf.order(pos, vel, orderer).chord

        assert run(jnp.asarray(0.3)).shape == (40,)

    @pytest.mark.parametrize(
        ("metric", "remedy"),
        [
            (None, "Pass metric="),
            (pcf.metrics.SpatialDistanceMetric(), "concrete metric_scale"),
        ],
        ids=["metric-inferred-from-scale", "spatial-metric-effect-check"],
    )
    def test_a_traced_scale_without_an_explicit_answer_names_the_way_out(
        self, metric, remedy
    ):
        """The error must name the field *and* the remedy, not just complain.

        ``match`` pins the remedy substring, not merely the exception type: a
        guard that raises a bare ``TypeError`` is no better than the JAX error
        it replaces.
        """
        pos, vel = self._curve()

        @jax.jit
        def run(scale):
            orderer = pcf.orderers.SOMOrderer(
                n_prototypes=8, metric_scale=scale, metric=metric
            )
            return pcf.order(pos, vel, orderer).chord

        with pytest.raises(TypeError, match="metric_scale has no value"):
            run(jnp.asarray(0.3))
        with pytest.raises(TypeError, match=re.escape(remedy)):
            run(jnp.asarray(0.3))

    def test_grad_is_unaffected(self):
        """The rule is "no value", not "inside a transform".

        A ``grad`` trace carries a concrete primal, so the scale *does* have a
        value and the guard lets it through. (That gradient is identically
        zero -- ``metric_scale`` enters only through the best-matching-unit
        argmin, which is piecewise constant -- but the guard is not the right
        place to editorialise about that.)
        """
        pos, vel = self._curve()
        base = pcf.orderers.SOMOrderer(n_prototypes=8, metric_scale=0.3)

        def loss(scale):
            orderer = eqx.tree_at(lambda m: m.metric_scale, base, scale)
            return jnp.nansum(pcf.order(pos, vel, orderer).chord)

        assert jnp.isfinite(jax.grad(loss)(jnp.asarray(0.3)))


class TestDisagreementWarningIsBestEffort:
    """The disagreement warning is a diagnostic; it must never be the failure.

    It reports the smoothing length as a fraction of the track. Written as
    ``sigma_phys / length`` that fraction divides by a quantity that is
    exactly 0.0 for a coincident working set -- so the warning raised
    ``ZeroDivisionError`` instead of describing the problem.
    """

    @staticmethod
    def _disagreeing_perm(n, threshold):
        """Build a permutation whose rank correlation is below the gate."""
        rng = np.random.default_rng(0)
        for _ in range(500):
            perm = rng.permutation(n)
            rank = np.empty(n, dtype=int)
            rank[perm] = np.arange(n)
            if abs(np.corrcoef(np.arange(n), rank)[0, 1]) < threshold:
                return jnp.asarray(perm)
        pytest.fail("no sufficiently disagreeing permutation found")
        return None

    class _Init:
        velocity_aware = False

    def test_a_coincident_working_set_warns_instead_of_raising(self):
        """Every position identical makes the track length exactly zero."""
        n = 12
        orderer = pcf.orderers.SOMOrderer(n_prototypes=8)
        perm = self._disagreeing_perm(n, som_orderer._DISAGREE_WARN)
        sub_q = {"x": jnp.zeros(n), "y": jnp.zeros(n)}
        kept = jnp.ones(n, dtype=bool)

        with pytest.warns(UserWarning, match="disagrees with the one it was given"):
            orderer._warn_if_disagrees(perm, kept, sub_q, self._Init())

    def test_the_reported_fraction_is_unchanged_on_a_normal_track(self):
        """Cancelling ``length`` must not alter the number it used to print.

        Without this the crash is fixed by reporting something else.
        ``sigma_end / (n_prototypes - 1)`` is the fraction analytically, so
        pin it against that rather than against a recorded string.
        """
        n = 12
        orderer = pcf.orderers.SOMOrderer(n_prototypes=8, sigma_end=0.7)
        perm = self._disagreeing_perm(n, som_orderer._DISAGREE_WARN)
        sub_q = {"x": jnp.linspace(0.0, 3.0, n), "y": jnp.zeros(n)}
        kept = jnp.ones(n, dtype=bool)

        with pytest.warns(UserWarning, match="% of the track") as record:
            orderer._warn_if_disagrees(perm, kept, sub_q, self._Init())

        expected = 100 * 0.7 / (8 - 1)
        assert f"{expected:.1f}% of the track" in str(record[0].message)

    def test_rejected_points_do_not_count_as_reordering(self):
        """A sort key of +inf sends every rejected point to the tail of ``perm``.

        Naively correlating that against its original position reads as
        wholesale reordering even when the *kept* points kept perfect rank
        order (correlation ~1.0 restricted to them) -- rejection showing up as
        reordering, not reordering itself. Here half the points are rejected
        and dumped in reverse at the tail: the full-set correlation is -0.75
        (would have warned under the old, unfiltered computation), the
        kept-only correlation is ~1.0 (must not warn).
        """
        n = 20
        orderer = pcf.orderers.SOMOrderer(n_prototypes=8, sigma_end=0.7)
        perm = jnp.asarray(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        )
        kept = jnp.arange(n) >= 10
        sub_q = {"x": jnp.linspace(0.0, 3.0, n), "y": jnp.zeros(n)}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            orderer._warn_if_disagrees(perm, kept, sub_q, self._Init())


class TestTracingAStandaloneVersusChainedStage:
    """Standalone the orderer traces; chained it cannot, and must say why.

    The working set is whatever a prior stage visited, so its size depends on
    that stage's values. Under a transform ``init.indices`` is a tracer and no
    shape fixed at trace time holds the result, so the chained case cannot
    compile at all -- the point is that it fails *legibly*.
    """

    @staticmethod
    def _curve(n=60):
        t = jnp.linspace(0.0, 2.0, n)
        return (
            {"x": jnp.cos(t), "y": jnp.sin(t)},
            {"x": -jnp.sin(t), "y": jnp.cos(t)},
        )

    def test_standalone_traces(self):
        """The property the chained case is measured against."""
        pos, vel = self._curve()
        run = jax.jit(
            lambda q, p: pcf.order(q, p, pcf.orderers.SOMOrderer(n_prototypes=8)).chord
        )
        assert run(pos, vel).shape == (60,)

    def test_chained_under_jit_names_the_limit(self):
        """A bare ``NonConcreteBooleanIndexError`` names neither cause nor cure.

        ``match`` pins the remedy, not just the type: the whole value of the
        guard is that it points somewhere.
        """
        pos, vel = self._curve()
        chain = pcf.orderers.LocalFlowOrderer(
            metric_scale=0.0
        ) | pcf.orderers.SOMOrderer(n_prototypes=8)

        with pytest.raises(TypeError, match="cannot be traced when chained"):
            jax.jit(lambda q, p: pcf.order(q, p, chain).chord)(pos, vel)
        with pytest.raises(TypeError, match=re.escape("outside `jit`")):
            jax.jit(lambda q, p: pcf.order(q, p, chain).chord)(pos, vel)

    def test_chaining_still_works_untraced(self):
        """The guard must not cost the ordinary, documented usage."""
        pos, vel = self._curve()
        chain = pcf.orderers.LocalFlowOrderer(
            metric_scale=0.0
        ) | pcf.orderers.SOMOrderer(n_prototypes=8)

        result = pcf.order(pos, vel, chain)
        assert int(result.n_visited) == 60
        assert np.isfinite(np.asarray(result.chord)).all()


def _contaminated_half_arc(n_arc=300, n_out=60, scatter=0.21, radius=5.0, seed=0):
    """Half-turn arc plus uniform background contamination, per `#61`'s own measurement.

    Cross-track scatter 0.21, uniform background sprinkled through the
    bounding box. Returns ``(pos, vel, is_outlier)`` with the contaminants
    appended after the arc points, each carrying a velocity uncorrelated with
    the arc.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_arc)
    ang = np.pi * t
    x = radius * np.cos(ang) + rng.normal(0.0, scatter, n_arc)
    y = radius * np.sin(ang) + rng.normal(0.0, scatter, n_arc)
    xi = rng.uniform(x.min(), x.max(), n_out)
    yi = rng.uniform(y.min(), y.max(), n_out)
    pos = {"x": jnp.asarray(np.r_[x, xi]), "y": jnp.asarray(np.r_[y, yi])}
    vel = {
        "x": jnp.asarray(np.r_[-np.sin(ang), rng.normal(0.0, 1.0, n_out)]),
        "y": jnp.asarray(np.r_[np.cos(ang), rng.normal(0.0, 1.0, n_out)]),
    }
    is_outlier = np.r_[np.zeros(n_arc, bool), np.ones(n_out, bool)]
    return pos, vel, is_outlier


def _mean_dist_to_true_arc(backbone_q, radius=5.0, n_ref=2000):
    """Mean distance from backbone points to the nearest point on the true arc.

    The regression measure `#61` itself asks for: backbone error against
    ground truth, not ordering quality -- ordering quality (rank correlation)
    stays high under contamination even as the reconstructed track degrades,
    so it cannot detect this failure.
    """
    ang = np.linspace(0.0, np.pi, n_ref)
    ref = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=-1)
    bb = np.stack([np.asarray(backbone_q["x"]), np.asarray(backbone_q["y"])], axis=-1)
    tree = cKDTree(ref)
    dist, _ = tree.query(bb)
    return float(dist.mean())


class TestSOMOutlierClip:
    """Optional sigma-clipping of SOM quantization error (``outlier_clip_sigma``)."""

    def test_none_is_a_noop(self):
        """``outlier_clip_sigma=None`` (default) leaves every point visited."""
        pos, vel, _ = _contaminated_half_arc(n_out=0)
        res = pcf.orderers.SOMOrderer(n_prototypes=25).order(pos, vel)
        assert int(res.n_skipped) == 0

    def test_rejects_contamination_keeps_arc(self):
        """Clipping rejects most of the scattered background, keeps the arc.

        Not all of it: some contaminants land close enough to the curve, by
        chance, to be indistinguishable from genuine scatter (measured: 34/60
        at this seed). That is the nature of the fixture, not a bug.
        """
        pos, vel, is_outlier = _contaminated_half_arc(n_arc=300, n_out=60)
        res = pcf.orderers.SOMOrderer(n_prototypes=25, outlier_clip_sigma=3.0).order(
            pos, vel
        )
        visited = {int(i) for i in np.asarray(res.indices) if i >= 0}
        rejected = set(range(is_outlier.size)) - visited
        n_out_rejected = sum(is_outlier[i] for i in rejected)
        n_in_rejected = sum(1 for i in rejected if not is_outlier[i])
        assert n_out_rejected >= 0.5 * int(is_outlier.sum())  # most background gone
        assert n_in_rejected <= 0.05 * int((~is_outlier).sum())  # few genuine cut

    def test_clean_arc_not_clipped(self):
        """A clean arc with no contamination loses no genuine points.

        Not a universal guarantee: a fixed-sigma threshold has a small,
        expected false-positive rate over many points (measured: 0 or 1 of 300
        depending on the noise draw). This seed's draw happens to clip none.
        """
        pos, vel, _ = _contaminated_half_arc(n_out=0, seed=1)
        res = pcf.orderers.SOMOrderer(n_prototypes=25, outlier_clip_sigma=3.0).order(
            pos, vel
        )
        assert int(res.n_skipped) == 0

    def test_backbone_error_improves_under_contamination(self):
        """The failure #61 measured: ordering quality hides it, backbone error does not.

        The default degrades sharply as contamination rises; clipping keeps
        the error a fraction of the unclipped default at the same
        contamination level.
        """
        default_err, clipped_err = [], []
        for n_out in (0, 60):
            pos, vel, _ = _contaminated_half_arc(n_arc=300, n_out=n_out)
            res_default = pcf.orderers.SOMOrderer(n_prototypes=25).order(pos, vel)
            res_clipped = pcf.orderers.SOMOrderer(
                n_prototypes=25, outlier_clip_sigma=3.0
            ).order(pos, vel)
            default_err.append(_mean_dist_to_true_arc(res_default.backbone))
            clipped_err.append(_mean_dist_to_true_arc(res_clipped.backbone))

        assert default_err[1] > 2.0 * default_err[0]  # unclipped: degrades sharply
        assert clipped_err[1] < 0.5 * default_err[1]  # clipped: at least half the error

    def test_invalid_sigma_raises(self):
        """A non-positive ``outlier_clip_sigma`` is rejected at construction."""
        with pytest.raises(ValueError, match="outlier_clip_sigma"):
            pcf.orderers.SOMOrderer(outlier_clip_sigma=0.0)

    def test_invalid_max_iters_raises(self):
        """``outlier_clip_max_iters < 1`` is rejected at construction."""
        with pytest.raises(ValueError, match="outlier_clip_max_iters"):
            pcf.orderers.SOMOrderer(outlier_clip_sigma=3.0, outlier_clip_max_iters=0)

    def test_composes_with_a_prior_stage(self):
        """Chained: the SOM's own rejections narrow, never override, init's.

        ``MSTOrderer(jump_cap=3.0, ...)`` here is loose enough to leave most of
        the contamination in its own working set (measured: 59/60), which is
        the point -- it isolates what the SOM's own clipping contributes on
        top of whatever ``init`` already decided, rather than depending on a
        prior stage to have done the rejecting.
        """
        pos, vel, is_outlier = _contaminated_half_arc(n_arc=300, n_out=60)
        mst = pcf.orderers.MSTOrderer(k=10, jump_cap=3.0, edge_clip_sigma=3.0)
        init = mst.order(pos, vel)
        prior_visited = {int(i) for i in np.asarray(init.indices) if i >= 0}

        baseline = pcf.orderers.SOMOrderer(n_prototypes=25).order(pos, vel, init=init)
        chained = pcf.orderers.SOMOrderer(
            n_prototypes=25, outlier_clip_sigma=3.0
        ).order(pos, vel, init=init)
        baseline_visited = {int(i) for i in np.asarray(baseline.indices) if i >= 0}
        visited = {int(i) for i in np.asarray(chained.indices) if i >= 0}

        assert visited <= prior_visited  # never readmits what init rejected
        assert visited <= baseline_visited  # the clip only ever removes
        newly_rejected = baseline_visited - visited
        assert len(newly_rejected) > 0  # the clip does something on this init
        assert sum(is_outlier[i] for i in newly_rejected) >= 0.9 * len(newly_rejected)

    def test_standalone_traces_under_jit_and_vmap(self):
        """Matches the module's own standalone traceability contract."""
        pos, vel, _ = _contaminated_half_arc(n_arc=60, n_out=0)
        orderer = pcf.orderers.SOMOrderer(n_prototypes=10, outlier_clip_sigma=3.0)

        chord = jax.jit(lambda q, p: orderer.order(q, p).chord)(pos, vel)
        assert chord.shape == (60,)

        batched_pos = {k: jnp.stack([v, v]) for k, v in pos.items()}
        batched_vel = {k: jnp.stack([v, v]) for k, v in vel.items()}
        out = jax.vmap(lambda q, p: orderer.order(q, p).chord)(batched_pos, batched_vel)
        assert out.shape == (2, 60)


class TestNonFiniteVelocities:
    """NaN velocities are missing data, policed by ``nan_policy``; inf is an error.

    Missing velocities are common in catalogues (no radial velocity). Summed
    into the batch update, a single NaN reached every prototype through the
    neighbourhood, so ``orient_by_velocity`` compared against NaN and never
    flipped -- silently ordering against the flow.
    """

    @staticmethod
    def _spoil(vel, value):
        """Set one tracer's x velocity."""
        return {"x": vel["x"].at[57].set(value), "y": vel["y"]}

    @staticmethod
    def _against_the_flow(straight, sign):
        """Build a line whose *unflipped* SOM ordering runs against ``sign``.

        ``straight(sign=sign)`` mirrors its positions with ``sign``, which
        mirrors the initializer's axis too, so there the unflipped SOM already
        agrees with the flow and a test would pass without ever flipping.
        """
        pos, _, _ = straight(sign=-sign)
        return pos, {"x": jnp.full(200, 10.0 * sign), "y": jnp.zeros(200)}

    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["flow+x", "flow-x"])
    def test_omit_orients_along_the_flow(self, straight, sign):
        """A missing velocity no longer stops the flip."""
        pos, vel = self._against_the_flow(straight, sign)
        vel = self._spoil(vel, jnp.nan)
        orderer = pcf.orderers.SOMOrderer(orient_by_velocity=True, nan_policy="omit")
        order = orderer.order(pos, vel).ordering
        first, last = float(pos["x"][order[0]]), float(pos["x"][order[-1]])
        assert (last - first) * sign > 0

    @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["flow+x", "flow-x"])
    def test_a_stationary_tracer_is_data_not_missing(self, straight, sign):
        """A zero velocity neither raises nor stops the flip."""
        pos, vel = self._against_the_flow(straight, sign)
        vel = self._spoil(vel, 0.0)
        orderer = pcf.orderers.SOMOrderer(orient_by_velocity=True)
        order = orderer.order(pos, vel).ordering
        first, last = float(pos["x"][order[0]]), float(pos["x"][order[-1]])
        assert (last - first) * sign > 0

    @pytest.mark.parametrize(
        "kw",
        [
            {"orient_by_velocity": True},
            {
                "metric": pcf.metrics.FullPhaseSpaceDistanceMetric(),
                "metric_scale": 0.1,
            },
        ],
        ids=["orient", "velocity-aware-metric"],
    )
    def test_nan_raises_by_default_when_velocities_are_read(self, straight, kw):
        """Silently skipping a missing velocity has to be asked for."""
        pos, vel, _ = straight()
        vel = self._spoil(vel, jnp.nan)
        with pytest.raises(ValueError, match="nan_policy='omit'"):
            pcf.orderers.SOMOrderer(**kw).order(pos, vel)

    def test_nan_is_fine_when_velocities_are_not_read(self, straight):
        """A position-only stage never reads velocities, so never raises."""
        pos, vel, _ = straight()
        vel = self._spoil(vel, jnp.nan)
        result = pcf.orderers.SOMOrderer().order(pos, vel)
        assert int(result.n_visited) == 200

    @pytest.mark.parametrize("policy", ["raise", "omit"])
    @pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["+inf", "-inf"])
    def test_inf_raises_under_either_policy(self, straight, policy, bad):
        """Inf is not a missing measurement, so ``omit`` does not cover it."""
        pos, vel, _ = straight()
        vel = self._spoil(vel, bad)
        orderer = pcf.orderers.SOMOrderer(orient_by_velocity=True, nan_policy=policy)
        with pytest.raises(ValueError, match="infinite velocity"):
            orderer.order(pos, vel)

    def test_raises_under_jit(self, straight):
        """Traced, the check still fires at run time."""
        pos, vel, _ = straight()
        vel = self._spoil(vel, jnp.nan)
        orderer = pcf.orderers.SOMOrderer(n_prototypes=10, orient_by_velocity=True)
        with pytest.raises(RuntimeError, match="nan_policy='omit'"):
            jax.block_until_ready(
                jax.jit(lambda q, p: orderer.order(q, p).chord)(pos, vel)
            )

    def test_rejects_an_unknown_nan_policy(self):
        """A typo fails at construction, not at the first NaN."""
        with pytest.raises(ValueError, match="nan_policy"):
            pcf.orderers.SOMOrderer(nan_policy="propagate")

    def test_prototype_velocities_stay_finite(self, straight):
        """One NaN tracer used to make every prototype velocity NaN."""
        pos, vel, _ = straight()
        vel = self._spoil(vel, jnp.nan)
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=30)
        assert bool(jnp.all(jnp.isfinite(pp["x"])))
        result = som.fit(pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
        assert bool(jnp.all(jnp.isfinite(result.prototype_velocities["x"])))

    def test_a_velocity_component_nobody_has_stays_finite(self, straight):
        """Every contributor non-finite: the prototype keeps a finite value."""
        pos, vel, _ = straight()
        vel = {"x": vel["x"], "y": jnp.full_like(vel["y"], jnp.nan)}
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=30)
        result = som.fit(pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
        assert bool(jnp.all(jnp.isfinite(result.prototype_velocities["y"])))

    def test_derived_scale_ignores_a_nan_speed(self, straight):
        """A NaN median speed used to zero the scale, disabling velocity."""
        pos, vel, _ = straight()
        init = pcf.orderers.MSTOrderer(k=6, jump_cap=2.0, velocity_weight=1.0).order(
            pos, vel
        )
        sub_q = {k: v[init.ordering] for k, v in pos.items()}
        sub_p = {k: v[init.ordering] for k, v in self._spoil(vel, jnp.nan).items()}
        _, scale = pcf.orderers.SOMOrderer()._resolve_metric(sub_q, sub_p, init)
        assert float(scale) > 0.0

    def test_omit_keeps_the_arms_apart_under_a_velocity_aware_metric(self):
        """A datum without a velocity is matched by position alone.

        Left NaN, its distance to every prototype is NaN, ``argmin`` sends it
        to prototype 0, and one such datum interleaves the anti-parallel arms.
        The prior stage sees clean velocities so only the SOM sees the NaN.
        """
        pos, vel = TestMetricAndScaleResolveIndependently._hairpin()
        n = len(pos["x"]) // 2
        init = pcf.orderers.MSTOrderer(k=6, jump_cap=5.0, velocity_weight=1.0).order(
            pos, vel
        )
        vel = {"x": vel["x"].at[20].set(jnp.nan), "y": vel["y"]}
        orderer = pcf.orderers.SOMOrderer(n_prototypes=30, nan_policy="omit")
        idx = np.asarray(orderer.order(pos, vel, init=init).ordering)
        # One switch is a perfect traversal: down one arm and back the other.
        assert int(np.abs(np.diff((idx >= n).astype(int))).sum()) == 1

    def test_omit_traces_under_jit_and_vmap(self, straight):
        """The masking adds no Python branching on traced values."""
        pos, vel, _ = straight()
        vel = self._spoil(vel, jnp.nan)
        orderer = pcf.orderers.SOMOrderer(
            n_prototypes=10, orient_by_velocity=True, nan_policy="omit"
        )

        chord = jax.jit(lambda q, p: orderer.order(q, p).chord)(pos, vel)
        assert bool(jnp.all(jnp.isfinite(chord)))

        batched_pos = {k: jnp.stack([v, v]) for k, v in pos.items()}
        batched_vel = {k: jnp.stack([v, v]) for k, v in vel.items()}
        out = jax.vmap(lambda q, p: orderer.order(q, p).chord)(batched_pos, batched_vel)
        assert out.shape == (2, 200)
