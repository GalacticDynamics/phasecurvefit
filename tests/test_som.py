"""Tests for the JAX Self-Organizing Map core."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit import som


def _helix(n: int = 200, turns: float = 1.0):
    """Generate a 3-D helix with tangent velocities; t is the true curve parameter."""
    t = jnp.linspace(0.0, 1.0, n)
    ang = 2 * jnp.pi * turns * t
    pos = {"x": jnp.cos(ang), "y": jnp.sin(ang), "z": 2.0 * t}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang), "z": jnp.full(n, 2.0)}
    return pos, vel, t


def test_init_prototypes_shape():
    pos, vel, _ = _helix()
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=10)
    assert set(pq) == {"x", "y", "z"}
    assert pq["x"].shape == (10,)
    assert pp["z"].shape == (10,)


def test_init_prototypes_from_ordering_is_monotone():
    """Binning along a known ordering places prototypes in that order."""
    n = 100
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    ordering = jnp.arange(n)
    pq, _ = som.init_prototypes(pos, vel, n_prototypes=10, ordering=ordering)
    assert jnp.all(jnp.diff(pq["x"]) > 0)


def test_init_prototypes_reversed_ordering_reverses_prototypes():
    n = 100
    pos = {"x": jnp.linspace(0.0, 10.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    pq, _ = som.init_prototypes(pos, vel, n_prototypes=10, ordering=jnp.arange(n)[::-1])
    assert jnp.all(jnp.diff(pq["x"]) < 0)


def test_init_prototypes_pca_fallback_is_monotone():
    """With no ordering, binning follows the first principal axis.

    Rows are shuffled before being passed in, so the fixture is not already
    sorted along its own principal axis: the fallback must actively recover
    monotone prototypes from that shuffle.
    """
    n = 100
    noise_key, shuffle_key = jax.random.split(jax.random.key(0))
    t = jnp.linspace(0.0, 10.0, n)
    x = t + 0.01 * jax.random.normal(noise_key, (n,))
    y = 0.5 * t
    perm = jax.random.permutation(shuffle_key, n)
    pos = {"x": x[perm], "y": y[perm]}
    vel = {"x": jnp.ones(n), "y": jnp.full(n, 0.5)}
    pq, _ = som.init_prototypes(pos, vel, n_prototypes=8)
    dx = jnp.diff(pq["x"])
    assert jnp.all(dx > 0) or jnp.all(dx < 0)


def test_init_prototypes_rejects_too_few_points():
    pos = {"x": jnp.arange(3.0)}
    vel = {"x": jnp.ones(3)}
    with pytest.raises(ValueError, match="at least n_prototypes"):
        som.init_prototypes(pos, vel, n_prototypes=10)


def test_fit_moves_prototypes_onto_the_curve():
    """Trained prototypes sit close to the helix they were fit to."""
    import phasecurvefit as pcf

    pos, vel, _ = _helix(n=300)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=12)
    fq, fp = som.fit(
        pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric(), n_epochs=40
    )
    # every prototype is within a small distance of some datum
    proto = jnp.stack([fq["x"], fq["y"], fq["z"]], axis=-1)
    data = jnp.stack([pos["x"], pos["y"], pos["z"]], axis=-1)
    d = jnp.linalg.norm(proto[:, None, :] - data[None, :, :], axis=-1)
    assert float(jnp.max(jnp.min(d, axis=1))) < 0.2


def test_fit_preserves_lattice_order_along_the_curve():
    """The lattice stays monotone in the helix's z coordinate."""
    import phasecurvefit as pcf

    pos, vel, _ = _helix(n=300)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=12)
    fq, _ = som.fit(
        pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric(), n_epochs=40
    )
    dz = jnp.diff(fq["z"])
    assert jnp.all(dz > 0) or jnp.all(dz < 0)


def test_fit_is_jittable():
    import phasecurvefit as pcf

    pos, vel, _ = _helix(n=100)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=8)
    metric = pcf.metrics.SpatialDistanceMetric()
    fn = jax.jit(lambda a, b, c, d: som.fit(a, b, c, d, metric=metric, n_epochs=5))
    out_q, _ = fn(pq, pp, pos, vel)
    assert out_q["x"].shape == (8,)


def test_fit_is_deterministic():
    import phasecurvefit as pcf

    pos, vel, _ = _helix(n=100)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=8)
    metric = pcf.metrics.SpatialDistanceMetric()
    a, _ = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=10)
    b, _ = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=10)
    assert jnp.allclose(a["x"], b["x"])


def test_densify_vertex_count():
    pos = {"x": jnp.linspace(0.0, 4.0, 5), "y": jnp.zeros(5)}
    vel = {"x": jnp.ones(5), "y": jnp.zeros(5)}
    bq, bp = som.densify(pos, vel, factor=10)
    assert bq["x"].shape == (10 * (5 - 1) + 1,)
    assert bp["x"].shape == (10 * (5 - 1) + 1,)


def test_densify_of_a_straight_line_stays_straight():
    pos = {"x": jnp.linspace(0.0, 4.0, 5), "y": jnp.zeros(5)}
    vel = {"x": jnp.ones(5), "y": jnp.zeros(5)}
    bq, _ = som.densify(pos, vel, factor=8)
    assert float(jnp.max(jnp.abs(bq["y"]))) < 1e-5
    assert jnp.all(jnp.diff(bq["x"]) > 0)


def test_densify_is_uniform_in_arc_length():
    ang = jnp.linspace(0.0, jnp.pi, 12)
    pos = {"x": jnp.cos(ang), "y": jnp.sin(ang)}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    bq, _ = som.densify(pos, vel, factor=10)
    seg = jnp.hypot(jnp.diff(bq["x"]), jnp.diff(bq["y"]))
    assert float(jnp.std(seg) / jnp.mean(seg)) < 1e-3


def test_densify_passes_through_the_endpoints():
    pos = {"x": jnp.linspace(0.0, 4.0, 5), "y": jnp.array([0.0, 1.0, 0.0, 1.0, 0.0])}
    vel = {"x": jnp.ones(5), "y": jnp.zeros(5)}
    bq, _ = som.densify(pos, vel, factor=6)
    assert float(bq["x"][0]) == pytest.approx(0.0, abs=1e-5)
    assert float(bq["x"][-1]) == pytest.approx(4.0, abs=1e-5)


def test_densify_is_c1_not_a_polyline():
    """The backbone must turn smoothly, not corner at the old prototypes.

    A piecewise-linear polyline through the same prototypes corners by
    ``dtheta`` (the inter-prototype angle) at every interior vertex; the
    spline keeps the largest turn between backbone samples well under it.
    """
    ang = jnp.linspace(0.0, 0.6 * jnp.pi, 8)
    pos = {"x": jnp.cos(ang), "y": jnp.sin(ang)}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    bq, _ = som.densify(pos, vel, factor=20)

    heading = jnp.arctan2(jnp.diff(bq["y"]), jnp.diff(bq["x"]))
    turn = jnp.abs(jnp.diff(heading))
    turn = jnp.minimum(turn, 2 * jnp.pi - turn)

    dtheta = float(ang[1] - ang[0])
    assert float(jnp.max(turn)) < 0.25 * dtheta


def test_chord_is_exact_on_a_straight_backbone():
    """On a straight line the chord is the along-track distance."""
    import phasecurvefit as pcf

    bq = {"x": jnp.linspace(0.0, 10.0, 21), "y": jnp.zeros(21)}
    bp = {"x": jnp.ones(21), "y": jnp.zeros(21)}
    pos = {"x": jnp.array([0.0, 2.5, 7.5, 10.0]), "y": jnp.zeros(4)}
    vel = {"x": jnp.ones(4), "y": jnp.zeros(4)}
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert jnp.allclose(lam, jnp.array([0.0, 2.5, 7.5, 10.0]), atol=1e-4)


def test_chord_ignores_perpendicular_offset():
    import phasecurvefit as pcf

    bq = {"x": jnp.linspace(0.0, 10.0, 21), "y": jnp.zeros(21)}
    bp = {"x": jnp.ones(21), "y": jnp.zeros(21)}
    pos = {"x": jnp.array([5.0, 5.0]), "y": jnp.array([0.0, 1.5])}
    vel = {"x": jnp.ones(2), "y": jnp.zeros(2)}
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert float(lam[0]) == pytest.approx(float(lam[1]), abs=1e-4)


def test_chord_extrapolates_past_the_end_caps():
    import phasecurvefit as pcf

    bq = {"x": jnp.linspace(0.0, 10.0, 21), "y": jnp.zeros(21)}
    bp = {"x": jnp.ones(21), "y": jnp.zeros(21)}
    pos = {"x": jnp.array([-3.0, 13.0]), "y": jnp.zeros(2)}
    vel = {"x": jnp.ones(2), "y": jnp.zeros(2)}
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert float(lam[0]) < 0.0
    assert float(lam[1]) > 10.0


def test_chord_is_true_arc_length_on_a_non_uniform_backbone():
    """``chord`` must not assume the backbone is arc-length-uniform.

    Segment lengths here vary 4x along the backbone ([1,1,1,1,4,4,4,4]), so a
    formula multiplying vertex index by the *mean* segment length returns
    index-proportional values dressed up as arc length.
    """
    import phasecurvefit as pcf

    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    bq = {"x": x, "y": jnp.zeros_like(x)}
    bp = {"x": jnp.ones_like(x), "y": jnp.zeros_like(x)}
    # Query exactly at backbone vertices 0, 2, 4, 6, 8.
    qx = x[jnp.array([0, 2, 4, 6, 8])]
    pos = {"x": qx, "y": jnp.zeros(5)}
    vel = {"x": jnp.ones(5), "y": jnp.zeros(5)}
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert jnp.allclose(lam, jnp.array([0.0, 2.0, 4.0, 12.0, 20.0]), atol=1e-4)


def test_chord_is_rank_monotone_on_a_helix():
    """The full pipeline (init -> fit -> densify -> chord) recovers order.

    A 1.5-turn helix winds past the PCA fallback's ~1-turn precondition, so
    the ordering is seeded: this checks the pipeline, not the initializer.
    Projection precision is covered by
    ``test_chord_is_exact_on_a_straight_backbone`` and its neighbours.
    """
    from scipy.stats import spearmanr

    import phasecurvefit as pcf

    pos, vel, t = _helix(n=400, turns=1.5)
    metric = pcf.metrics.SpatialDistanceMetric()
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=20, ordering=jnp.argsort(t))
    fq, fp = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=60)
    bq, bp = som.densify(fq, fp, factor=10)
    lam = som.chord(bq, bp, pos, vel, metric=metric)
    rho = spearmanr(np.asarray(t), np.asarray(lam)).statistic
    assert abs(rho) > 0.99


def test_chord_is_rank_monotone_in_four_position_dimensions():
    """The full pipeline is genuinely N-D: nothing assumes 2 or 3 components.

    The ordering is seeded as above: this curve's ``cos(2*ang)`` component
    doubles back along the fallback's principal axis.
    """
    from scipy.stats import spearmanr

    import phasecurvefit as pcf

    n = 400
    t = jnp.linspace(0.0, 1.0, n)
    ang = 2 * jnp.pi * t
    pos = {
        "w": jnp.cos(ang),
        "x": jnp.sin(ang),
        "y": 2.0 * t,
        "z": jnp.cos(2 * ang),
    }
    vel = {k: jnp.gradient(v) for k, v in pos.items()}
    metric = pcf.metrics.SpatialDistanceMetric()
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=20, ordering=jnp.argsort(t))
    fq, fp = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=60)
    bq, bp = som.densify(fq, fp, factor=10)
    lam = som.chord(bq, bp, pos, vel, metric=metric)
    rho = spearmanr(np.asarray(t), np.asarray(lam)).statistic
    assert abs(rho) > 0.99


def test_chord_is_rank_monotone_with_the_pca_fallback_within_one_turn():
    """The PCA fallback (no ``ordering`` seeded) works inside its precondition.

    Pins the valid regime documented on ``init_prototypes``: a helix of at
    most one turn does not double back along its first principal axis, so
    the default equi-frequency binning produces a usable initial lattice. A
    smoke test -- ``test_init_prototypes_pca_fallback_is_monotone`` guards the
    fallback itself.
    """
    from scipy.stats import spearmanr

    import phasecurvefit as pcf

    pos, vel, t = _helix(n=400, turns=1.0)
    metric = pcf.metrics.SpatialDistanceMetric()
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=20)
    fq, fp = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=60)
    bq, bp = som.densify(fq, fp, factor=10)
    lam = som.chord(bq, bp, pos, vel, metric=metric)
    rho = spearmanr(np.asarray(t), np.asarray(lam)).statistic
    assert abs(rho) > 0.99


def test_som1d_round_trip():
    import phasecurvefit as pcf

    pos, vel, t = _helix(n=300)
    model = som.SOM1D.make(
        pos, vel, n_prototypes=15, metric=pcf.metrics.SpatialDistanceMetric()
    )
    trained = model.fit(pos, vel)
    lam = trained.chord(pos, vel)
    assert lam.shape == (300,)
    assert trained.n_prototypes == 15


def test_som1d_citation_is_set():
    assert som.SOM1D.__citation__ == "https://arxiv.org/abs/2212.00949"


def test_chord_is_jittable():
    import phasecurvefit as pcf

    bq = {"x": jnp.linspace(0.0, 10.0, 21), "y": jnp.zeros(21)}
    bp = {"x": jnp.ones(21), "y": jnp.zeros(21)}
    pos = {"x": jnp.array([1.0, 2.0]), "y": jnp.zeros(2)}
    vel = {"x": jnp.ones(2), "y": jnp.zeros(2)}
    metric = pcf.metrics.SpatialDistanceMetric()
    fn = jax.jit(lambda a, b, c, d: som.chord(a, b, c, d, metric=metric))
    assert fn(bq, bp, pos, vel).shape == (2,)


def test_init_prototypes_rejects_a_padded_ordering():
    """``indices`` is -1-padded; passing it raw must not silently mis-bin.

    Out-of-range indices do not raise on their own: the padding lands on a
    real observation and drags that bin's mean.
    """
    n = 12
    pos = {"x": jnp.linspace(0.0, 11.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    padded = jnp.concatenate([jnp.arange(n - 2, -1, -1), jnp.array([-1])])

    with pytest.raises(Exception, match="only valid indices"):
        som.init_prototypes(pos, vel, n_prototypes=5, ordering=padded)

    # The documented remedy works, and gives a different answer.
    clean = som.init_prototypes(pos, vel, n_prototypes=5, ordering=padded[padded >= 0])
    assert float(clean[0]["x"][-1]) < 1.0

    # The upper bound is guarded too. An index past the end is clamped to the
    # last element rather than raising, so it mis-bins exactly like the -1 case.
    too_big = jnp.concatenate([jnp.arange(n - 1), jnp.array([n + 7])])
    with pytest.raises(Exception, match="only valid indices"):
        som.init_prototypes(pos, vel, n_prototypes=5, ordering=too_big)


def test_fit_leaves_unreached_prototypes_in_place():
    """A lattice unit no datum reaches must not move to the coordinate origin.

    The neighbourhood is exactly 0 in float32 about 11 lattice units from the
    nearest best-matching unit, and the numerator underflows with it.
    """
    n = 300
    # A tight clump far from the origin: every BMU is one lattice unit, so the
    # far end of the lattice is unreached.
    pos = {"x": jnp.full(n, 10.0) + jnp.linspace(-0.01, 0.01, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    n_proto = 25
    pq = {"x": jnp.linspace(10.0, 10.5, n_proto), "y": jnp.zeros(n_proto)}
    pp = {"x": jnp.ones(n_proto), "y": jnp.zeros(n_proto)}

    fq, _ = som.fit(
        pq,
        pp,
        pos,
        vel,
        metric=pcf.metrics.SpatialDistanceMetric(),
        n_epochs=1,
        sigma_start=0.7,
        sigma_end=0.7,
    )

    # Nothing teleports to the origin, and every prototype stays near the data.
    assert int(jnp.sum(fq["x"] == 0.0)) == 0
    assert float(jnp.min(fq["x"])) > 9.0
