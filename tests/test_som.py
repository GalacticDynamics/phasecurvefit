"""Tests for the JAX Self-Organizing Map core."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

import phasecurvefit as pcf
from phasecurvefit import som
from phasecurvefit._src.som import _safe_norm


def test_init_prototypes_shape(helix):
    pos, vel, _ = helix()
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=10)
    assert set(pq) == {"x", "y", "z"}
    assert pq["x"].shape == (10,)
    assert pp["z"].shape == (10,)


def test_init_prototypes_from_ordering_is_monotone(line):
    """Binning along a known ordering places prototypes in that order."""
    n = 100
    pos, vel = line(n, 10.0)
    ordering = jnp.arange(n)
    pq, _ = som.init_prototypes(pos, vel, n_prototypes=10, ordering=ordering)
    assert jnp.all(jnp.diff(pq["x"]) > 0)


def test_init_prototypes_reversed_ordering_reverses_prototypes(line):
    n = 100
    pos, vel = line(n, 10.0)
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


def test_fit_moves_prototypes_onto_the_curve(helix):
    """Trained prototypes sit close to the helix they were fit to."""
    pos, vel, _ = helix(n=300)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=12)
    fq, fp = som.fit(
        pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric(), n_epochs=40
    )
    # every prototype is within a small distance of some datum
    proto = jnp.stack([fq["x"], fq["y"], fq["z"]], axis=-1)
    data = jnp.stack([pos["x"], pos["y"], pos["z"]], axis=-1)
    d = jnp.linalg.norm(proto[:, None, :] - data[None, :, :], axis=-1)
    assert float(jnp.max(jnp.min(d, axis=1))) < 0.2


def test_fit_preserves_lattice_order_along_the_curve(helix):
    """The lattice stays monotone in the helix's z coordinate."""
    pos, vel, _ = helix(n=300)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=12)
    fq, _ = som.fit(
        pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric(), n_epochs=40
    )
    dz = jnp.diff(fq["z"])
    assert jnp.all(dz > 0) or jnp.all(dz < 0)


def test_fit_is_jittable(helix):
    pos, vel, _ = helix(n=100)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=8)
    metric = pcf.metrics.SpatialDistanceMetric()
    fn = jax.jit(lambda a, b, c, d: som.fit(a, b, c, d, metric=metric, n_epochs=5))
    out_q, _ = fn(pq, pp, pos, vel)
    assert out_q["x"].shape == (8,)


def test_fit_is_deterministic(helix):
    pos, vel, _ = helix(n=100)
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=8)
    metric = pcf.metrics.SpatialDistanceMetric()
    a, _ = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=10)
    b, _ = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=10)
    assert jnp.allclose(a["x"], b["x"])


def test_densify_vertex_count(line):
    pos, vel = line(5, 4.0)
    bq, bp = som.densify(pos, vel, factor=10)
    assert bq["x"].shape == (10 * (5 - 1) + 1,)
    assert bp["x"].shape == (10 * (5 - 1) + 1,)


def test_densify_of_a_straight_line_stays_straight(line):
    pos, vel = line(5, 4.0)
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


def test_densify_passes_through_the_endpoints(flat_vel):
    pos = {"x": jnp.linspace(0.0, 4.0, 5), "y": jnp.array([0.0, 1.0, 0.0, 1.0, 0.0])}
    vel = flat_vel(5)
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


def test_chord_is_exact_on_a_straight_backbone(line, flat_vel):
    """On a straight line the chord is the along-track distance.

    Most queries sit *between* vertices. The backbone has 21 vertices over
    [0, 10], i.e. spacing 0.5, so 2.3 / 4.75 / 7.1 are all off-vertex: an
    implementation that snapped to the nearest vertex and skipped the
    sub-vertex projection entirely would return 2.5 / 5.0 / 7.0 instead.
    """
    bq, bp = line(21, 10.0)
    want = jnp.array([0.0, 2.3, 4.75, 7.1, 10.0])
    pos = {"x": want, "y": jnp.zeros(5)}
    vel = flat_vel(5)
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert jnp.allclose(lam, want, atol=1e-4)


def test_chord_ignores_perpendicular_offset(line, flat_vel):
    bq, bp = line(21, 10.0)
    pos = {"x": jnp.array([5.0, 5.0]), "y": jnp.array([0.0, 1.5])}
    vel = flat_vel(2)
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert float(lam[0]) == pytest.approx(float(lam[1]), abs=1e-4)


def test_chord_extrapolates_past_the_end_caps(line, flat_vel):
    bq, bp = line(21, 10.0)
    pos = {"x": jnp.array([-3.0, 13.0]), "y": jnp.zeros(2)}
    vel = flat_vel(2)
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert float(lam[0]) < 0.0
    assert float(lam[1]) > 10.0


def test_chord_is_true_arc_length_on_a_non_uniform_backbone(flat_vel):
    """``chord`` must not assume the backbone is arc-length-uniform.

    Segment lengths here vary 4x along the backbone ([1,1,1,1,4,4,4,4]), so a
    formula multiplying vertex index by the *mean* segment length returns
    index-proportional values dressed up as arc length.
    """
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    bq = {"x": x, "y": jnp.zeros_like(x)}
    bp = {"x": jnp.ones_like(x), "y": jnp.zeros_like(x)}
    # Query exactly at backbone vertices 0, 2, 4, 6, 8.
    qx = x[jnp.array([0, 2, 4, 6, 8])]
    pos = {"x": qx, "y": jnp.zeros(5)}
    vel = flat_vel(5)
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert jnp.allclose(lam, jnp.array([0.0, 2.0, 4.0, 12.0, 20.0]), atol=1e-4)


def _four_dimensional(n=400):
    """Build a 4-D curve whose ``cos(2*ang)`` component doubles back on the PCA axis."""
    t = jnp.linspace(0.0, 1.0, n)
    ang = 2 * jnp.pi * t
    pos = {"w": jnp.cos(ang), "x": jnp.sin(ang), "y": 2.0 * t, "z": jnp.cos(2 * ang)}
    vel = {k: jnp.gradient(v) for k, v in pos.items()}
    return pos, vel, t


@pytest.mark.parametrize(
    ("build", "seeded"),
    [
        (lambda h: h(n=400, turns=1.5), True),
        (lambda _h: _four_dimensional(), True),
        (lambda h: h(n=400, turns=1.0), False),
    ],
    ids=["helix-1.5-turns", "four-dimensional", "helix-1-turn-pca-fallback"],
)
def test_chord_recovers_the_curve_order(helix, build, seeded):
    """Run init -> fit -> densify -> chord and recover the true parameter.

    ``seeded`` says whether the ordering is handed to ``init_prototypes``. The
    two seeded curves wind past the PCA fallback's ~1-turn precondition, so
    leaving it to the fallback would test the initializer rather than the
    pipeline; the third stays inside it and exercises the fallback end to end.

    This is a pipeline-level check. Projection precision is covered by
    ``test_chord_is_exact_on_a_straight_backbone`` and its neighbours.
    """
    pos, vel, t = build(helix)
    metric = pcf.metrics.SpatialDistanceMetric()
    ordering = jnp.argsort(t) if seeded else None
    pq, pp = som.init_prototypes(pos, vel, n_prototypes=20, ordering=ordering)
    fq, fp = som.fit(pq, pp, pos, vel, metric=metric, n_epochs=60)
    bq, bp = som.densify(fq, fp, factor=10)
    lam = som.chord(bq, bp, pos, vel, metric=metric)
    assert abs(spearmanr(np.asarray(t), np.asarray(lam)).statistic) > 0.99


def test_som1d_round_trip(helix):
    curve = helix(n=300)
    pos, vel, t = curve
    model = som.SOM1D.make(
        pos, vel, n_prototypes=15, metric=pcf.metrics.SpatialDistanceMetric()
    )
    trained = model.fit(pos, vel)
    lam = trained.chord(pos, vel)
    assert lam.shape == (300,)
    assert trained.n_prototypes == 15


def test_som1d_citation_is_set():
    assert som.SOM1D.__citation__ == "https://arxiv.org/abs/2212.00949"


def test_chord_is_jittable(line, flat_vel):
    bq, bp = line(21, 10.0)
    pos = {"x": jnp.array([1.0, 2.0]), "y": jnp.zeros(2)}
    vel = flat_vel(2)
    metric = pcf.metrics.SpatialDistanceMetric()
    fn = jax.jit(lambda a, b, c, d: som.chord(a, b, c, d, metric=metric))
    assert fn(bq, bp, pos, vel).shape == (2,)


def test_init_prototypes_rejects_a_padded_ordering(line):
    """``indices`` is -1-padded; passing it raw must not silently mis-bin.

    Out-of-range indices do not raise on their own: the padding lands on a
    real observation and drags that bin's mean.
    """
    n = 12
    pos, vel = line(n, 11.0)
    padded = jnp.concatenate([jnp.arange(n - 2, -1, -1), jnp.array([-1])])

    with pytest.raises(eqx.EquinoxRuntimeError, match="only valid indices"):
        som.init_prototypes(pos, vel, n_prototypes=5, ordering=padded)

    # The documented remedy works, and gives a different answer.
    clean = som.init_prototypes(pos, vel, n_prototypes=5, ordering=padded[padded >= 0])
    assert float(clean[0]["x"][-1]) < 1.0

    # The upper bound is guarded too. An index past the end is clamped to the
    # last element rather than raising, so it mis-bins exactly like the -1 case.
    too_big = jnp.concatenate([jnp.arange(n - 1), jnp.array([n + 7])])
    with pytest.raises(eqx.EquinoxRuntimeError, match="only valid indices"):
        som.init_prototypes(pos, vel, n_prototypes=5, ordering=too_big)


def test_fit_leaves_unreached_prototypes_in_place(flat_vel):
    """A lattice unit no datum reaches must not move to the coordinate origin.

    The neighbourhood is exactly 0 in float32 about 11 lattice units from the
    nearest best-matching unit, and the numerator underflows with it.
    """
    n = 300
    # A tight clump far from the origin: every BMU is one lattice unit, so the
    # far end of the lattice is unreached.
    pos = {"x": jnp.full(n, 10.0) + jnp.linspace(-0.01, 0.01, n), "y": jnp.zeros(n)}
    vel = flat_vel(n)
    n_proto = 25
    pq = {"x": jnp.linspace(10.0, 10.5, n_proto), "y": jnp.zeros(n_proto)}
    pp = flat_vel(n_proto)

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


class TestComponentKeyValidation:
    """`fit` and `chord` stack every dict against one key tuple."""

    def test_fit_rejects_a_mismatched_component(self, line):
        """A prototype dict missing a data component must not reach the loop."""
        pos, vel = line(20, 9.0)
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=6)
        del pq["y"]
        with pytest.raises(ValueError, match="same component keys"):
            som.fit(pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())

    @pytest.mark.parametrize(
        "fn",
        ["init_prototypes", "fit", "densify", "chord"],
    )
    def test_empty_component_dicts_are_rejected(self, fn):
        """All four entry points raised a bare `StopIteration` with no message."""
        metric = pcf.metrics.SpatialDistanceMetric()
        calls = {
            "init_prototypes": lambda: som.init_prototypes({}, {}, n_prototypes=4),
            "fit": lambda: som.fit({}, {}, {}, {}, metric=metric),
            "densify": lambda: som.densify({}, {}, factor=3),
            "chord": lambda: som.chord({}, {}, {}, {}, metric=metric),
        }
        with pytest.raises(ValueError, match="has no components"):
            calls[fn]()

    def test_densify_rejects_mismatched_prototype_keys(self):
        """A velocity component the positions lack was silently dropped."""
        pq = {"x": jnp.arange(4.0), "y": jnp.zeros(4)}
        pp = {"x": jnp.ones(4)}
        with pytest.raises(ValueError, match="same component keys"):
            som.densify(pq, pp, factor=3)

    @pytest.mark.parametrize(
        ("fn", "match"),
        [
            (
                lambda: som.densify(
                    {"x": jnp.array([0.0])}, {"x": jnp.array([1.0])}, factor=5
                ),
                "at least 2 prototypes",
            ),
            (
                lambda: som.chord(
                    {"x": jnp.array([0.0])},
                    {"x": jnp.array([1.0])},
                    {"x": jnp.array([0.0, 1.0])},
                    {"x": jnp.ones(2)},
                    metric=pcf.metrics.SpatialDistanceMetric(),
                ),
                "at least 2 vertices",
            ),
        ],
        ids=["densify", "chord"],
    )
    def test_a_single_vertex_is_not_a_backbone(self, fn, match):
        """One vertex has no segment: `densify` returned it, `chord` IndexError'd."""
        with pytest.raises(ValueError, match=match):
            fn()

    def test_chord_rejects_a_mismatched_backbone(self, line):
        """An extra data component would otherwise be dropped silently."""
        pos, vel = line(20, 9.0)
        bq, bp = line(5, 9.0)
        pos["z"] = jnp.zeros(20)
        vel["z"] = jnp.zeros(20)
        with pytest.raises(ValueError, match="same component keys"):
            som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())


def test_densify_does_not_perturb_a_clean_backbone():
    """The tie-breaking nudge must not touch a strictly increasing arc length.

    It shifts every entry by ``i * eps``, which accumulates with vertex count,
    so applying it unconditionally rescaled the track by ~1e-4 at large M.
    """
    ang = jnp.linspace(0.0, jnp.pi, 12)
    pos = {"x": jnp.cos(ang), "y": jnp.sin(ang)}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    bq, _ = som.densify(pos, vel, factor=40)
    seg = jnp.hypot(jnp.diff(bq["x"]), jnp.diff(bq["y"]))
    # tighter than test_densify_is_uniform_in_arc_length: the drift showed up
    # as a systematic spacing gradient
    assert float(jnp.std(seg) / jnp.mean(seg)) < 1e-4


def test_densify_still_handles_coincident_prototypes(flat_vel):
    """Duplicate vertices give a zero-length segment; the nudge must still fire."""
    pos = {"x": jnp.array([0.0, 1.0, 1.0, 2.0, 3.0]), "y": jnp.zeros(5)}
    vel = flat_vel(5)
    bq, _ = som.densify(pos, vel, factor=6)
    assert bool(jnp.all(jnp.isfinite(bq["x"])))
    assert jnp.all(jnp.diff(bq["x"]) >= 0)


class TestSymmetryRequirement:
    """Nearest-prototype assignment needs ``d(a, b) == d(b, a)``."""

    def test_aligned_momentum_really_is_asymmetric(self):
        """The ``is_symmetric`` flag must describe behaviour, not just intent.

        The metric reads only the *query* point's velocity (``del velocities``),
        so swapping the pair changes the alignment term.
        """
        m = pcf.metrics.AlignedMomentumDistanceMetric()
        aq = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        av = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        bq = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        bv = {"x": jnp.array(0.0), "y": jnp.array(1.0)}
        arr = lambda d: {k: v[None] for k, v in d.items()}
        ab = float(m(aq, av, arr(bq), arr(bv), 0.5)[0])
        ba = float(m(bq, bv, arr(aq), arr(av), 0.5)[0])
        assert ab != pytest.approx(ba)
        assert m.is_symmetric is False

    @staticmethod
    def _call_fit(pos, vel, metric):
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=6)
        return som.fit(pq, pp, pos, vel, metric=metric)

    @staticmethod
    def _call_chord(pos, vel, metric):
        bq = {"x": jnp.linspace(0.0, 9.0, 5), "y": jnp.zeros(5)}
        bp = {"x": jnp.ones(5), "y": jnp.zeros(5)}
        return som.chord(bq, bp, pos, vel, metric=metric)

    @pytest.mark.parametrize("fn", ["_call_fit", "_call_chord"], ids=["fit", "chord"])
    def test_core_rejects_an_asymmetric_metric(self, fn, line):
        """Both public entry points guard, not just the orderer above them."""
        n = 20
        pos, vel = line(n, 9.0)
        call = getattr(self, fn)
        with pytest.raises(ValueError, match="not symmetric"):
            call(pos, vel, pcf.metrics.AlignedMomentumDistanceMetric())


def test_densify_and_chord_preserve_the_input_dtype(line):
    """Array creation must follow the data, not JAX's global default.

    Under ``jax_enable_x64`` a bare ``jnp.zeros(1)`` / ``jnp.linspace`` is
    float64, which silently widened a float32 backbone and everything
    downstream of it.
    """
    n = 12
    pos, vel = line(n, 11.0)
    bq, bp = som.densify(pos, vel, factor=5)
    lam = som.chord(bq, bp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())
    assert bq["x"].dtype == pos["x"].dtype
    assert lam.dtype == pos["x"].dtype


@pytest.mark.parametrize("k", [0, 1], ids=["empty", "single"])
def test_fit_rejects_a_degenerate_lattice(k):
    """`fit` was the one entry point without the >= 2 floor.

    ``K=0`` reached ``argmin`` on an empty ``(N, 0)`` matrix; ``K=1`` trained
    happily and only failed later in `densify`.
    """
    n = 20
    pos = {"x": jnp.linspace(0.0, 9.0, n)}
    vel = {"x": jnp.ones(n)}
    pq = {"x": jnp.zeros(k)}
    pp = {"x": jnp.zeros(k)}
    with pytest.raises(ValueError, match="at least 2 prototypes"):
        som.fit(pq, pp, pos, vel, metric=pcf.metrics.SpatialDistanceMetric())


class TestNumericalHazards:
    """Silent-wrong-answer regressions found by an independent review."""

    def test_binning_survives_an_int32_index_overflow(self):
        """``arange(m) * K`` wrapped once ``m * K > 2**31``.

        The wrapped negative segment ids were dropped by ``segment_sum`` and
        rescued by ``maximum(counts, 1)`` into prototypes stacked at the
        origin: 14% of the lattice at these sizes, with no error raised.
        """
        n, k = 100_000, 25_000
        assert n * k > 2**31, "test no longer exercises the overflow"
        pos = {"x": jnp.linspace(0.0, 1000.0, n), "y": jnp.zeros(n)}
        vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
        pq, _ = som.init_prototypes(pos, vel, n_prototypes=k, ordering=jnp.arange(n))
        # The wrong-answer signature was thousands of prototypes at exactly
        # 0.0; the first bin's mean is near but not equal to the origin.
        assert int(jnp.sum(pq["x"] == 0.0)) == 0
        assert jnp.all(jnp.diff(pq["x"]) > 0)
        assert float(pq["x"].max()) == pytest.approx(1000.0, rel=1e-3)

    def test_chord_is_invariant_to_the_position_scale(self):
        """``_TINY`` guarded ``|s|^2``, which carries (length)^2 units.

        A fixed 1e-12 floor took over the denominator for any segment shorter
        than 1e-6 in the caller's units, collapsing the projection toward the
        nearest vertex -- right in kpc, wrong in normalised coordinates.
        """
        metric = pcf.metrics.SpatialDistanceMetric()
        want = jnp.array([0.05, 2.55, 7.35])
        for scale in (1.0, 1e-6, 1e-8):
            bq = {"x": jnp.linspace(0.0, 10.0 * scale, 101), "y": jnp.zeros(101)}
            bp = {"x": jnp.ones(101), "y": jnp.zeros(101)}
            pos = {"x": want * scale, "y": jnp.zeros(3)}
            vel = {"x": jnp.ones(3), "y": jnp.zeros(3)}
            lam = som.chord(bq, bp, pos, vel, metric=metric) / scale
            assert jnp.allclose(lam, want, atol=1e-4), f"scale {scale}"

    def test_densify_is_differentiable(self):
        """The padded stencil duplicates the endpoints, so ``norm`` saw a 0/0 VJP.

        The forward value was guarded but the reverse pass was not, making
        every gradient through ``densify`` NaN at both ends.
        """

        def total_x(xs):
            pos = {"x": xs, "y": jnp.zeros_like(xs)}
            vel = {"x": jnp.ones_like(xs), "y": jnp.zeros_like(xs)}
            return jnp.sum(som.densify(pos, vel, factor=5)[0]["x"])

        grad = jax.grad(total_x)(jnp.linspace(0.0, 3.0, 4))
        assert jnp.all(jnp.isfinite(grad))

    def test_densify_factor_one_smooths_nothing_but_still_resamples(self):
        """The documented corner: `factor=1` is not a no-op.

        It evaluates the spline at one sample per segment, reproducing the
        prototypes to rounding, and then still redistributes them to equal
        arc-length spacing -- which is why it can cut corners.
        """
        ang = jnp.linspace(0.0, jnp.pi, 7)
        pq = {"x": jnp.cos(ang), "y": jnp.sin(ang)}
        pp = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
        bq, _ = som.densify(pq, pp, factor=1)

        assert bq["x"].shape == pq["x"].shape  # factor * (K - 1) + 1 == K
        # Close to the prototypes -- the spline adds no smoothing ...
        assert jnp.allclose(bq["x"], pq["x"], atol=1e-5)
        # ... but the arc-length resampling still moved them, so it is not a
        # no-op: an identity `densify` would be exactly equal.
        seg = jnp.hypot(jnp.diff(bq["x"]), jnp.diff(bq["y"]))
        assert float(jnp.std(seg) / jnp.mean(seg)) < 1e-3

    def test_densify_is_differentiable_through_coincident_prototypes(self):
        """Duplicated observations make `init_prototypes` emit coincident pairs.

        Flooring the knot distance in `_catmull_rom` fixed the padded *ends*
        only; `_resample_uniform` kept a bare `norm` whose 0/0 VJP NaN'd every
        gradient through an interior coincidence.
        """

        def total_x(xs):
            pos = {"x": xs, "y": jnp.zeros_like(xs)}
            vel = {"x": jnp.ones_like(xs), "y": jnp.zeros_like(xs)}
            pq, pp = som.init_prototypes(pos, vel, n_prototypes=8)
            return jnp.sum(som.densify(pq, pp, factor=5)[0]["x"])

        xs = jnp.asarray(np.repeat([0.0, 1.0, 2.0, 3.0], 8), dtype=jnp.float32)
        assert jnp.all(jnp.isfinite(jax.grad(total_x)(xs)))

    def test_chord_is_differentiable_through_a_coincident_backbone(self):
        """`chord`'s cumulative arc length had the same unguarded `norm`."""
        bx = jnp.asarray([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])  # vertices 2,3 coincide
        bp = {"x": jnp.ones(6), "y": jnp.zeros(6)}
        pos = {"x": jnp.asarray([0.5, 2.5, 3.5]), "y": jnp.zeros(3)}
        vel = {"x": jnp.ones(3), "y": jnp.zeros(3)}

        def total(b):
            return jnp.sum(
                som.chord(
                    {"x": b, "y": jnp.zeros(6)},
                    bp,
                    pos,
                    vel,
                    metric=pcf.metrics.SpatialDistanceMetric(),
                )
            )

        assert jnp.all(jnp.isfinite(jax.grad(total)(bx)))

    def test_safe_norm_matches_the_library_norm(self):
        """The gradient fix must not move any forward value."""
        rng = np.random.default_rng(0)
        for scale in (1.0, 1e-20, 1e18):
            d = jnp.asarray((rng.normal(0, 1, (200, 3)) * scale).astype(np.float32))
            assert jnp.allclose(_safe_norm(d), jnp.linalg.norm(d, axis=-1), rtol=1e-6)
        assert float(_safe_norm(jnp.zeros((1, 3)))[0]) == 0.0

    @pytest.mark.parametrize(
        "row",
        [[jnp.nan, 1.0], [jnp.inf, 1.0], [jnp.nan, jnp.inf]],
        ids=["nan", "inf", "nan+inf"],
    )
    def test_safe_norm_propagates_invalid_values(self, row):
        """It must not launder a NaN into 0.0.

        The guard originally tested ``sq > 0``, which a NaN fails just as an
        exact zero does -- so an invalid input came back as a plausible 0.0
        instead of the NaN ``jnp.linalg.norm`` would have given.
        """
        d = jnp.asarray([row])
        want = jnp.linalg.norm(d, axis=-1)
        got = _safe_norm(d)
        assert jnp.isnan(got) == jnp.isnan(want)
        assert jnp.array_equal(got, want, equal_nan=True)

    @pytest.mark.parametrize("dtype", [jnp.float32, bool], ids=["float", "bool-mask"])
    def test_init_prototypes_rejects_a_non_integer_ordering(self, dtype):
        """A float or mask `ordering` failed from inside the repeat guard.

        The old message named ``bincount`` -- a function the caller never
        invoked. Passing ``visited`` instead of ``ordering`` is the likely
        mistake, so the message names that.
        """
        n = 20
        pos = {"x": jnp.linspace(0.0, 9.0, n), "y": jnp.zeros(n)}
        vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
        bad = jnp.zeros(n, dtype=dtype) if dtype is bool else jnp.arange(n, dtype=dtype)
        with pytest.raises(TypeError, match="must be an integer index array"):
            som.init_prototypes(pos, vel, n_prototypes=5, ordering=bad)

    @pytest.mark.parametrize("dtype", [jnp.int8, jnp.int32, jnp.uint32])
    def test_init_prototypes_accepts_any_integer_width(self, dtype):
        """The guard must not reject legitimate integer index arrays."""
        n = 20
        pos = {"x": jnp.linspace(0.0, 9.0, n), "y": jnp.zeros(n)}
        vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
        pq, _ = som.init_prototypes(
            pos, vel, n_prototypes=5, ordering=jnp.arange(n, dtype=dtype)
        )
        assert jnp.all(jnp.diff(pq["x"]) > 0)

    def test_init_prototypes_rejects_a_repeated_index(self):
        """A non-permutation passed the range check and scrambled the lattice."""
        pos = {"x": jnp.linspace(0.0, 9.0, 20), "y": jnp.zeros(20)}
        vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
        dup = jnp.concatenate([jnp.arange(10), jnp.arange(10)])
        with pytest.raises(eqx.EquinoxRuntimeError, match="must not repeat"):
            som.init_prototypes(pos, vel, n_prototypes=5, ordering=dup)

    def test_a_visited_subset_is_still_accepted(self):
        """The repeat guard must not reject a prior stage's visited subset."""
        pos = {"x": jnp.linspace(0.0, 9.0, 20), "y": jnp.zeros(20)}
        vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
        pq, _ = som.init_prototypes(
            pos, vel, n_prototypes=5, ordering=jnp.arange(0, 20, 2)
        )
        assert jnp.all(jnp.diff(pq["x"]) > 0)
