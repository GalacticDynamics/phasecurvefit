"""Tests for SOMOrderer."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

import phasecurvefit as pcf
from phasecurvefit import som


def _arc(n: int = 200):
    ang = jnp.linspace(0.0, jnp.pi, n)
    pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    return pos, vel


def _helix(n: int = 200, turns: float = 1.5, r: float = 5.0, height: float = 10.0):
    """Build a 1.5-turn helix, outside init_prototypes's PCA-fallback precondition.

    Its first principal axis lies in the winding (x, y) plane (the pitch is
    kept modest relative to the radius so height variance does not dominate),
    so the ``ordering=None`` fallback doubles back on itself and tangles the
    lattice. ``_arc`` cannot discriminate that; this fixture can.
    """
    t = jnp.linspace(0.0, 1.0, n)
    ang = 2 * jnp.pi * turns * t
    dtheta_dt = 2 * jnp.pi * turns
    pos = {"x": r * jnp.cos(ang), "y": r * jnp.sin(ang), "z": height * t}
    vel = {
        "x": -r * dtheta_dt * jnp.sin(ang),
        "y": r * dtheta_dt * jnp.cos(ang),
        "z": jnp.full(n, height),
    }
    return pos, vel


def test_som_orderer_visits_everything_standalone():
    pos, vel = _arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=12))
    assert int(result.n_visited) == 200
    assert bool(result.all_visited)


def test_som_orderer_recovers_the_arc_order():
    pos, vel = _arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=15))
    ordering = np.asarray(result.ordering)
    # the input is already in arc order: rank-correlate the returned order
    # against the true index (Ruling P3 -- exactness is not a stable
    # contract; monotonicity is the claim, and it's covered exactly by the
    # core's test_chord_is_exact_on_a_straight_backbone).
    rho, _ = spearmanr(ordering, np.arange(ordering.shape[0]))
    assert abs(rho) > 0.99


def test_som_orderer_sets_backbone_and_chord():
    pos, vel = _arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    assert result.backbone is not None
    assert result.backbone["x"].shape == (5 * (10 - 1) + 1,)
    assert result.chord is not None
    assert result.chord.shape == (200,)
    assert not bool(jnp.any(jnp.isnan(result.chord)))


def test_som_orderer_gamma_range():
    pos, vel = _arc()
    result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    assert result.gamma_range == (-1.0, 1.0)


def test_som_orderer_refines_a_prior_ordering():
    pos, vel = _arc()
    chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.SOMOrderer(
        n_prototypes=12
    )
    result = pcf.order(pos, vel, chain)
    assert int(result.n_visited) == 200
    assert result.chord is not None


def test_som_orderer_chained_seeding_survives_more_than_one_turn():
    """The chained path must seed ``init_prototypes`` with the prior ordering.

    Falling back to PCA binning (``ordering=None``) tangles the lattice on
    ``_helix``, while a genuinely threaded ordering still resolves cleanly.
    """
    pos, vel = _helix()
    chain = pcf.orderers.MSTOrderer(k=8, jump_cap=4.0) | pcf.orderers.SOMOrderer(
        n_prototypes=12
    )
    result = pcf.order(pos, vel, chain)
    ordering = np.asarray(result.ordering)
    rho, _ = spearmanr(ordering, np.arange(ordering.shape[0]))
    assert abs(rho) > 0.99


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


def test_som_orderer_accepts_any_abstract_result_as_init():
    """``init`` only needs ``indices``, the field ``AbstractResult`` guarantees.

    ``ordering`` is an ``OrderingResult`` property, not part of the
    ``AbstractResult`` contract the ``order()`` signature advertises, so a
    custom result type must still be chainable into ``SOMOrderer``.
    """
    from phasecurvefit._src.abstract_result import AbstractResult

    class BareResult(AbstractResult):
        """Minimal AbstractResult: no ``ordering`` property at all."""

        def __call__(self, gamma, /, *, key=None):
            raise NotImplementedError

    pos, vel = _arc(n=60)
    # Reverse order, with the last 10 points marked unvisited.
    indices = jnp.concatenate(
        [jnp.arange(49, -1, -1, dtype=jnp.int32), jnp.full(10, -1, dtype=jnp.int32)]
    )
    assert not hasattr(BareResult, "ordering")

    prior = BareResult(positions=pos, velocities=vel, indices=indices)
    result = pcf.orderers.SOMOrderer(n_prototypes=8).order(pos, vel, init=prior)

    assert int(result.n_visited) == 50
    assert int(result.n_skipped) == 10
    assert int(jnp.sum(jnp.isnan(result.chord))) == 10


def _straight(n=200, sign=1.0):
    """Build a straight stream flowing in +x (sign=+1) or -x (sign=-1)."""
    x = jnp.linspace(0.0, 10.0, n) if sign > 0 else jnp.linspace(10.0, 0.0, n)
    return {"x": x, "y": jnp.zeros(n)}, {"x": jnp.full(n, sign), "y": jnp.zeros(n)}


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["flow+x", "flow-x"])
def test_orient_by_velocity_orders_along_the_flow(sign):
    """With the flag set, the chord runs along the mean velocity either way.

    Without it the direction comes from the sign of the initializer's
    principal-axis eigenvector, which is stable but arbitrary -- the SOM has no
    progenitor anchor.
    """
    pos, vel = _straight(sign=sign)
    result = pcf.order(
        pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10, orient_by_velocity=True)
    )
    order = result.ordering
    first, last = float(pos["x"][order[0]]), float(pos["x"][order[-1]])
    assert (last - first) * sign > 0


def test_orient_by_velocity_is_a_noop_when_already_aligned():
    pos, vel = _straight(sign=1.0)
    plain = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10))
    oriented = pcf.order(
        pos, vel, pcf.orderers.SOMOrderer(n_prototypes=10, orient_by_velocity=True)
    )
    assert jnp.array_equal(plain.indices, oriented.indices)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["flow+x", "flow-x"])
def test_orient_by_velocity_keeps_backbone_and_ordering_in_step(sign):
    """The backbone must flip with the chord, or ``__call__`` desyncs from it.

    ``gamma_range`` runs (-1, 1) over the backbone, so interpolating at the low
    end must land near the first ordered observation, not the last.
    """
    pos, vel = _straight(sign=sign)
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


def test_som_orderer_rejects_a_scale_the_default_metric_ignores():
    """A non-zero scale with SpatialDistanceMetric is a silent no-op; fail loudly.

    The default metric discards ``metric_scale`` outright, so a caller who sets
    one is asking for velocity to participate and would get nothing.
    """
    with pytest.raises(ValueError, match="no effect with SpatialDistanceMetric"):
        pcf.orderers.SOMOrderer(metric_scale=0.2)

    # The documented remedy constructs fine.
    pcf.orderers.SOMOrderer(
        metric=pcf.metrics.FullPhaseSpaceDistanceMetric(), metric_scale=0.2
    )


def _max_coverage_gap(proto, pos, keys=("x", "y")):
    """Largest distance from any datum to its nearest prototype."""
    p = jnp.stack([proto[k] for k in keys], axis=-1)
    d = jnp.stack([pos[k] for k in keys], axis=-1)
    return float(
        jnp.max(jnp.min(jnp.linalg.norm(d[:, None] - p[None], axis=-1), axis=1))
    )


def test_ordering_result_carries_chord():
    import phasecurvefit as pcf

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
    import phasecurvefit as pcf

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
        metric=orderer.metric,
        metric_scale=orderer.metric_scale,
        n_epochs=50,
    )

    # Ideal spacing for 11 prototypes over 10 units is 1.0, so a lattice that
    # tracks the data leaves a gap well under 1; a collapsed one leaves > 2.
    assert _max_coverage_gap(fq, pos) < 1.0


def test_som_orderer_default_metric_is_symmetric():
    """The default metric must be symmetric in the two points it compares.

    Symmetry is the contract: "which prototype is this datum nearest" is
    meaningless under an asymmetric metric.

    The property must be checked at a *non-zero* scale. At the shipped
    ``metric_scale=0.0`` every metric in the package degenerates to plain
    position distance, so a symmetry assertion there passes for
    ``AlignedMomentumDistanceMetric`` too and guards nothing.
    """
    orderer = pcf.orderers.SOMOrderer()
    assert isinstance(orderer.metric, pcf.metrics.SpatialDistanceMetric)

    a_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    a_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
    b_pos = {"x": jnp.array(1.0), "y": jnp.array(0.5)}
    b_vel = {"x": jnp.array(-1.0), "y": jnp.array(0.3)}
    as_arr = lambda d: {k: v[None] for k, v in d.items()}

    def probe(metric, scale):
        fwd = float(metric(a_pos, a_vel, as_arr(b_pos), as_arr(b_vel), scale)[0])
        bwd = float(metric(b_pos, b_vel, as_arr(a_pos), as_arr(a_vel), scale)[0])
        return fwd, bwd

    for metric in (orderer.metric, pcf.metrics.FullPhaseSpaceDistanceMetric()):
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
