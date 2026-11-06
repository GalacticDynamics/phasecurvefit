"""Tests for RidgeOrderer, against images and point clouds of a known arc.

The oracle is a half circle of radius 1: a backbone on the curve has radius 1
everywhere, and one that covers it runs from angle ~0 to ~pi.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.orderers.ridge import (
    _density,
    _estimate_bandwidth,
    _smooth_and_project,
)

pytest.importorskip("scipy")


def _arc_image(*, uniform: bool = False, gap: bool = False):
    """Pixels of a half circle of radius 1, as samples weighted by density.

    ``uniform`` gives a segmentation map (weight 1 within 0.15 of the curve),
    otherwise the density is a Gaussian of width 0.1 across it. ``gap`` halves
    the density over a stretch around the top of the arc.
    """
    g = jnp.linspace(-1.5, 1.5, 61)
    X, Y = jnp.meshgrid(g, g)
    R = jnp.hypot(X, Y)
    if uniform:
        image = (jnp.abs(R - 1.0) < 0.15) * 1.0
    else:
        image = jnp.exp(-((R - 1.0) ** 2) / (2 * 0.1**2))
    image = image * (Y > 0)
    if gap:
        image = image * (1.0 - 0.5 * (jnp.abs(X) < 0.3))
    keep = image.ravel() > 0.05
    pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
    vel = jax.tree.map(jnp.zeros_like, pos)
    return pos, vel, image.ravel()[keep]


def _radius_err(backbone):
    return np.abs(np.hypot(backbone["x"], backbone["y"]) - 1.0)


def _angle_span(backbone):
    ang = np.arctan2(backbone["y"], backbone["x"])
    return float(np.max(ang) - np.min(ang))


class TestRidgeOrderer:
    """RidgeOrderer finds the centre line of images and point clouds."""

    def test_blurred_image(self):
        """A blurred arc: the backbone follows the curve from end to end."""
        pos, vel, w = _arc_image()
        result = pcf.order(
            pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1), weights=w
        )
        assert float(np.median(_radius_err(result.backbone))) < 0.03
        assert _angle_span(result.backbone) > 0.9 * np.pi

    def test_uniform_image(self):
        """A segmentation map: the ridge of the smoothed mask is the centre line."""
        pos, vel, w = _arc_image(uniform=True)
        result = pcf.order(
            pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1), weights=w
        )
        assert float(np.median(_radius_err(result.backbone))) < 0.03
        assert _angle_span(result.backbone) > 0.9 * np.pi

    def test_gap_is_bridged(self):
        """A stretch at half density does not split the backbone."""
        pos, vel, w = _arc_image(gap=True)
        result = pcf.order(
            pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1), weights=w
        )
        assert _angle_span(result.backbone) > 0.9 * np.pi

    def test_point_cloud(self):
        """Unweighted noisy samples along the arc work too."""
        key_a, key_r = jax.random.split(jax.random.key(0))
        ang = jax.random.uniform(key_a, (400,), minval=0.0, maxval=jnp.pi)
        rad = 1.0 + 0.05 * jax.random.normal(key_r, (400,))
        pos = {"x": rad * jnp.cos(ang), "y": rad * jnp.sin(ang)}
        vel = jax.tree.map(jnp.zeros_like, pos)
        result = pcf.order(pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1))
        assert float(np.median(_radius_err(result.backbone))) < 0.03
        assert result.weights is None

    def test_duplicates_count_like_weight(self):
        """In the density, a sample given twice counts like one of double weight."""
        pts = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.15, 0.0]])
        w = np.array([1.0, 2.0, 1.0, 1.0])
        dup = np.vstack([pts, pts[[1]]])  # sample 1 twice, each of weight 1
        np.testing.assert_allclose(
            _density(dup, np.ones(5), 0.05)[:4], _density(pts, w, 0.05)
        )

    def test_duplicated_samples(self):
        """An image with every third pixel given twice still finds the arc."""
        pos, vel, w = _arc_image()
        dup = np.arange(0, w.shape[0], 3)

        def cat(a):
            return jnp.concatenate([a, a[dup]])

        result = pcf.order(
            jax.tree.map(cat, pos),
            jax.tree.map(cat, vel),
            pcf.orderers.RidgeOrderer(bandwidth=0.1),
            weights=cat(w),
        )
        assert float(np.median(_radius_err(result.backbone))) < 0.03
        assert _angle_span(result.backbone) > 0.9 * np.pi

    @pytest.mark.parametrize("n", [0, 1])
    def test_too_few_samples(self, n):
        """With fewer than 2 samples the ordering is the identity, as for the MST."""
        pos = {"x": jnp.arange(n, dtype=float), "y": jnp.zeros(n)}
        vel = jax.tree.map(jnp.zeros_like, pos)
        result = pcf.orderers.RidgeOrderer(bandwidth=0.1).order(pos, vel)
        np.testing.assert_array_equal(result.indices, np.arange(n))
        np.testing.assert_array_equal(result.chord, np.zeros(n))

    def test_single_vertex_path(self):
        """A path of one vertex is returned as is, with every chord 0."""
        pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        backbone, chord = _smooth_and_project(pts, np.array([1]), 0.1)
        np.testing.assert_array_equal(backbone, pts[[1]])
        np.testing.assert_array_equal(chord, np.zeros(3))

    def test_result_contract(self):
        """Every sample is ordered, by its chord, and the weights are carried."""
        pos, vel, w = _arc_image()
        result = pcf.order(
            pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1), weights=w
        )
        n = w.shape[0]
        assert int(result.n_visited) == n
        np.testing.assert_array_equal(np.sort(result.indices), np.arange(n))
        assert bool(jnp.all(jnp.diff(result.chord[result.indices]) >= 0))
        np.testing.assert_allclose(result.weights, w)
        assert result.gamma_range == (-1.0, 1.0)
        assert not result.velocity_aware

    def test_seeds_a_som(self):
        """``RidgeOrderer | SOMOrderer`` refines the weighted centre line."""
        pos, vel, w = _arc_image()
        chain = pcf.orderers.RidgeOrderer(bandwidth=0.1) | pcf.orderers.SOMOrderer(
            n_prototypes=20, metric=pcf.metrics.SpatialDistanceMetric()
        )
        result = pcf.order(pos, vel, chain, weights=w)
        assert float(np.median(_radius_err(result.backbone))) < 0.02
        assert int(result.n_visited) == w.shape[0]


def _soft_arc(seed: int = 0):
    """Make a soft mask on the arc: widening, brighter in one stretch, noisy.

    Like a segmentation probability map: the weight falls from the arc to 0 at
    the edge, the half-width grows from 0.1 to 0.2 along the arc, and every
    pixel carries 30% multiplicative noise.
    """
    g = jnp.linspace(-1.5, 1.5, 61)
    X, Y = jnp.meshgrid(g, g)
    theta = jnp.arctan2(Y, X)
    half_width = 0.1 + 0.1 * theta / jnp.pi
    across = jnp.clip(1.0 - ((jnp.hypot(X, Y) - 1.0) / half_width) ** 2, 0.0)
    along = 1.0 + 0.5 * jnp.exp(-0.5 * ((theta - 1.0) / 0.3) ** 2)
    noise = jax.random.lognormal(jax.random.key(seed), 0.3, X.shape)
    image = (along * across * noise * (Y > 0)).ravel()
    keep = image > 0
    pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
    return pos, jax.tree.map(jnp.zeros_like, pos), image[keep]


def _estimate(pos, w):
    pts = np.stack([np.asarray(pos[c]) for c in sorted(pos)], axis=1)
    return _estimate_bandwidth(pts, np.asarray(w), k=8, core=0.2, trim=0.5)


class TestBandwidthEstimate:
    """With ``bandwidth=None`` the width is estimated from the data."""

    def test_blurred_width(self):
        """A Gaussian blur of sigma 0.1 gives a bandwidth of about 0.1."""
        pos, _, w = _arc_image()
        assert 0.08 < _estimate(pos, w) < 0.12

    def test_uniform_width(self):
        """A uniform band of half-width r gives about its RMS, r / sqrt(3)."""
        pos, _, w = _arc_image(uniform=True)
        assert 0.07 < _estimate(pos, w) < 0.12  # r / sqrt(3) = 0.087

    def test_point_cloud_width(self):
        """Samples scattered by 0.05 about the arc give about 0.05."""
        key_a, key_r = jax.random.split(jax.random.key(0))
        ang = jax.random.uniform(key_a, (400,), minval=0.0, maxval=jnp.pi)
        rad = 1.0 + 0.05 * jax.random.normal(key_r, (400,))
        pos = {"x": rad * jnp.cos(ang), "y": rad * jnp.sin(ang)}
        assert 0.04 < _estimate(pos, jnp.ones(400)) < 0.065

    def test_subsample_matches(self):
        """An image too large to estimate on in full gives the same width."""
        g = jnp.linspace(-1.5, 1.5, 151)
        X, Y = jnp.meshgrid(g, g)
        R = jnp.hypot(X, Y)
        image = (jnp.exp(-((R - 1.0) ** 2) / (2 * 0.1**2)) * (Y > 0)).ravel()
        keep = image > 0.05
        assert int(keep.sum()) > 2000  # past the subsample limit
        pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
        coarse, _, w = _arc_image()
        assert _estimate(pos, image[keep]) == pytest.approx(
            _estimate(coarse, w), rel=0.1
        )

    def test_collinear_floor(self):
        """Samples exactly on a line still get a positive width."""
        pos = {"x": jnp.linspace(0.0, 1.0, 50), "y": jnp.zeros(50)}
        assert _estimate(pos, jnp.ones(50)) >= 0.99 / 49  # the spacing, to rounding

    def test_soft_mask(self):
        """A widening, noisy probability map still finds the arc by default."""
        pos, vel, w = _soft_arc()
        result = pcf.order(pos, vel, pcf.orderers.RidgeOrderer(), weights=w)
        assert float(np.median(_radius_err(result.backbone))) < 0.03
        assert _angle_span(result.backbone) > 0.85 * np.pi


class TestRidgeOrdererValidation:
    """Bad configuration and inputs raise."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bandwidth": 0.0},
            {"bandwidth": jnp.array(-0.1)},
            {"bandwidth": 0.1, "k": 0},
            {"bandwidth": 0.1, "core": 0.0},
            {"bandwidth": 0.1, "trim": 1.5},
        ],
    )
    def test_bad_config(self, kwargs):
        """Non-positive bandwidth, k < 1, and fractions outside (0, 1] raise."""
        with pytest.raises(ValueError, match="must be"):
            pcf.orderers.RidgeOrderer(**kwargs)

    def test_bad_weights(self):
        """Weights of the wrong length raise."""
        pos, vel, w = _arc_image()
        with pytest.raises(ValueError, match="shape"):
            pcf.order(
                pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1), weights=w[:-1]
            )


class TestRidgeOrdererUnxt:
    """Quantity inputs give the same ordering as the stripped values."""

    def test_bare_bandwidth_is_in_usys_length(self):
        """With Quantity positions, a bare bandwidth is in the usys length unit."""
        u = pytest.importorskip("unxt")
        pos, vel, w = _arc_image()
        q = {k: u.Q(v, "kpc") for k, v in pos.items()}
        p = {k: u.Q(v, "km/s") for k, v in vel.items()}
        meta = pcf.StateMetadata(usys=u.unitsystems.galactic)  # length: kpc
        bare, unitful = (
            pcf.order(
                q,
                p,
                pcf.orderers.RidgeOrderer(bandwidth=bw),
                metadata=meta,
                weights=w,
            )
            for bw in (0.1, u.Q(100.0, "pc"))
        )
        np.testing.assert_allclose(
            u.ustrip("kpc", bare.chord), u.ustrip("kpc", unitful.chord), atol=1e-5
        )

    def test_quantity_matches_plain(self):
        """Positions in kpc with a bandwidth in pc match the bare-number run."""
        u = pytest.importorskip("unxt")
        pos, vel, w = _arc_image()
        plain = pcf.order(pos, vel, pcf.orderers.RidgeOrderer(bandwidth=0.1), weights=w)
        q = {k: u.Q(v, "kpc") for k, v in pos.items()}
        p = {k: u.Q(v, "km/s") for k, v in vel.items()}
        meta = pcf.StateMetadata(usys=u.unitsystems.galactic)
        unitful = pcf.order(
            q,
            p,
            pcf.orderers.RidgeOrderer(bandwidth=u.Q(100.0, "pc")),
            metadata=meta,
            weights=w,
        )
        # Exact order can differ where two samples' chords tie to round-off.
        np.testing.assert_allclose(
            u.ustrip("kpc", unitful.chord), plain.chord, atol=1e-5
        )
        np.testing.assert_allclose(
            u.ustrip("kpc", unitful.backbone["x"]), plain.backbone["x"], atol=1e-5
        )
