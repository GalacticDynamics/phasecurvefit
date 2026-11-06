"""Tests for RidgeOrderer, against images and point clouds of a known arc.

The oracle is a half circle of radius 1: a backbone on the curve has radius 1
everywhere, and one that covers it runs from angle ~0 to ~pi.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf

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
