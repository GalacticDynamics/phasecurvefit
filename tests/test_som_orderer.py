"""Tests for SOMOrderer."""

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
from phasecurvefit._src.orderers.result import OrderingResult


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
    fq, _ = som.fit(
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
    assert _max_coverage_gap(fq, pos) < 1.0


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
    with pytest.warns(UserWarning, match="disagrees with the one it was given"):
        som._warn_if_disagrees(perm, sub, prior)


def test_som_is_quiet_when_it_agrees_with_the_prior_ordering(arc):
    """The warning must not fire on the ordinary case of a small refinement."""
    pos, vel, _ = arc()
    base = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, sever_cos_threshold=0.9)
    prior = base.order(pos, vel)
    som = pcf.orderers.SOMOrderer(n_prototypes=40)
    sub = {k: v[prior.ordering] for k, v in pos.items()}
    perm = jnp.arange(int((prior.indices >= 0).sum()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        som._warn_if_disagrees(perm, sub, prior)


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


def test_som_orderer_rejects_an_out_of_range_init():
    """`init` from a larger dataset gathered out of bounds; JAX clamps silently.

    The core's range guard cannot see this: the working set is re-indexed to
    ``arange(len(work))`` before it reaches ``init_prototypes``.
    """
    n = 40
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    init = OrderingResult(
        positions=pos, velocities=vel, indices=jnp.arange(5, 5 + n), velocity_aware=True
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="past the end of the data"):
        pcf.orderers.SOMOrderer(n_prototypes=6).order(pos, vel, init=init)


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
