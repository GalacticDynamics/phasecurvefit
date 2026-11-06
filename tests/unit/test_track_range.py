"""Tests for AutoencoderResult.track_range, the trimmed range for drawing a track.

The oracle is a blurred straight segment from x=0 to x=4: its track should end
at x=0 and x=4, while decoding the whole of ``gamma_range`` runs past them into
the faint tails of the blur.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.nn.result import _mask_ends, _track_range, _trim_ends

pytest.importorskip("scipy")


class TestTrimEnds:
    """The local half-maximum rule on hand-made distances along a track."""

    def test_trims_faint_cap(self):
        """A short faint stretch past the end, like a cap, is cut off."""
        along = jnp.linspace(0.0, 10.0, 400)
        weights = jnp.where(along <= 9.5, 1.0, 0.05)
        lo, hi = _trim_ends(along, weights)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(9.5, abs=0.3)

    def test_long_faint_tail_is_stream(self):
        """A faint stretch longer than the local window counts as stream."""
        along = jnp.linspace(0.0, 10.0, 400)
        weights = jnp.where(along <= 8.0, 1.0, 0.05)
        assert _trim_ends(along, weights)[1] == pytest.approx(10.0)

    def test_even_sampling_keeps_span(self):
        """Evenly spread, unweighted samples keep their whole span."""
        lo, hi = _trim_ends(jnp.linspace(0.0, 10.0, 400), jnp.ones(400))
        assert (lo, hi) == pytest.approx((0.0, 10.0))

    def test_fading_stream_is_not_cut(self):
        """A stream fading slowly to 30% at one end keeps that end."""
        along = jnp.linspace(0.0, 10.0, 400)
        lo, hi = _trim_ends(along, 1.0 - 0.07 * along)
        assert (lo, hi) == pytest.approx((0.0, 10.0))

    def test_too_few_samples_for_two_bins(self):
        """Under 20 samples there is one bin, kept whole."""
        lo, hi = _trim_ends(jnp.linspace(0.0, 1.0, 15), jnp.ones(15))
        assert (lo, hi) == pytest.approx((0.0, 1.0))

    def test_narrow_stretch_at_an_end_is_kept(self):
        """A stretch at 40% of the median near an end isn't mistaken for a cap."""
        along = jnp.linspace(0.0, 10.0, 400)
        weights = jnp.where(along < 3.0, 0.4, 1.0)
        lo, _ = _trim_ends(along, weights)
        assert lo == pytest.approx(0.0)


class TestTrackRangeFallbacks:
    """``_track_range`` falls back to the members' span when it can't trim."""

    @pytest.fixture(scope="class")
    def model(self):
        """Build an untrained model: the fallbacks return before decoding."""
        pos = {"x": jnp.linspace(0.0, 1.0, 20), "y": jnp.zeros(20)}
        vel = jax.tree.map(jnp.ones_like, pos)
        normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
        model = pcf.nn.PathAutoencoder.make(
            normalizer=normalizer, gamma_range=(-1.0, 1.0), key=jax.random.key(0)
        )
        return model, pos

    def test_no_members(self, model):
        """With no members, the range is the span of all gamma."""
        m, pos = model
        gamma = jnp.linspace(-0.3, 0.6, 20)
        rng = _track_range(m, pos, gamma, jnp.zeros(20, dtype=bool), None)
        assert rng == pytest.approx((-0.3, 0.6))

    def test_zero_weight_members(self, model):
        """Members of total weight 0 fall back to their span, without crashing."""
        m, pos = model
        gamma = jnp.linspace(-0.3, 0.6, 20)
        rng = _track_range(m, pos, gamma, jnp.ones(20, dtype=bool), jnp.zeros(20))
        assert rng == pytest.approx((-0.3, 0.6))

    def test_undecodable_track(self, model, monkeypatch):
        """A track that decodes to NaN everywhere falls back to the members' span."""
        m, pos = model
        monkeypatch.setattr(
            type(m),
            "decode",
            lambda _self, g, **_: {k: jnp.full(jnp.shape(g), jnp.nan) for k in pos},
        )
        gamma = jnp.linspace(-0.3, 0.6, 20)
        rng = _track_range(m, pos, gamma, jnp.ones(20, dtype=bool), None)
        assert rng == pytest.approx((-0.3, 0.6))

    def test_equal_gamma(self, model):
        """Every member at one gamma gives that single point."""
        m, pos = model
        rng = _track_range(m, pos, jnp.full(20, 0.3), jnp.ones(20, dtype=bool), None)
        assert rng == pytest.approx((0.3, 0.3))


def _capsule(length: float, half_width, pix: float = 0.02):
    """Grid points of a curve from 0 to ``length`` thickened by ``half_width``.

    Returns each point's distance along the curve (past the ends, along its
    extension) and across it, as `_mask_ends` receives them. ``half_width`` is a
    function of the distance along the curve, so the band can widen.
    """
    w_max = max(half_width(0.0), half_width(length))
    x, y = np.meshgrid(
        np.arange(-w_max, length + w_max, pix), np.arange(-w_max, w_max, pix)
    )
    x, y = x.ravel(), y.ravel()
    xc = np.clip(x, 0.0, length)  # nearest point of the curve
    inside = np.hypot(x - xc, y) <= np.vectorize(half_width)(xc)
    return x[inside], np.abs(y[inside])


class TestMaskEnds:
    """The cap rule on hand-made thickened curves with known ends."""

    def test_uniform_band(self):
        """A band of half-width 0.5 around a curve from 0 to 4 ends at 0 and 4."""
        along, across = _capsule(4.0, lambda _: 0.5)
        lo, hi = _mask_ends(along, across, 4.0)
        assert lo == pytest.approx(0.0, abs=0.03)
        assert hi == pytest.approx(4.0, abs=0.03)

    def test_widening_band(self):
        """A band widening from 0.2 to 0.6 still ends at 0 and 4."""
        along, across = _capsule(4.0, lambda s: 0.2 + 0.1 * s)
        lo, hi = _mask_ends(along, across, 4.0)
        assert lo == pytest.approx(0.0, abs=0.05)
        assert hi == pytest.approx(4.0, abs=0.05)

    def test_short_band(self):
        """A band shorter than it is wide still ends at its true ends."""
        along, across = _capsule(0.2, lambda _: 0.5)
        lo, hi = _mask_ends(along, across, 0.2)
        assert lo == pytest.approx(0.0, abs=0.03)
        assert hi == pytest.approx(0.2, abs=0.03)

    def test_overlapping_caps_keep_span(self):
        """Caps wider than the whole span leave nothing to cut, so it is kept."""
        along = np.array([-0.5, 0.0, 0.05, 0.5])
        across = np.array([0.0, 0.6, 0.6, 0.0])
        assert _mask_ends(along, across, 0.05) == pytest.approx((-0.5, 0.5))


def _mask_segment():
    """Pixels of a segment from (0, 0) to (4, 0) thickened to half-width 0.3."""
    X, Y = jnp.meshgrid(jnp.arange(-1.0, 5.0, 0.05), jnp.arange(-0.8, 0.8, 0.05))
    d = jnp.hypot(X - jnp.clip(X, 0.0, 4.0), Y)
    keep = (d <= 0.3).ravel()
    pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
    return pos, jax.tree.map(jnp.zeros_like, pos)


class TestMaskTrackEnds:
    """``TrainingConfig(track_ends="mask")`` ends the track at a mask's ends."""

    def test_mask_ends_at_the_segment_ends(self):
        """The track over ``track_range`` ends within 0.1 of x=0 and x=4."""
        pos, vel = _mask_segment()
        cfg = pcf.nn.TrainingConfig(
            n_epochs_encoder=200,
            n_epochs_decoder=150,
            n_epochs_both=0,
            batch_size=500,
            track_ends="mask",
            show_pbar=False,
        )
        _, fitted, _ = pcf.fit_track(
            pos, vel, key=jax.random.key(0), training_config=cfg
        )
        ends = np.sort(np.asarray(fitted(jnp.asarray(fitted.track_range))["x"]))
        np.testing.assert_allclose(ends, [0.0, 4.0], atol=0.1)

    def test_unknown_track_ends(self):
        """An unknown ``track_ends`` raises."""
        pos, vel = _mask_segment()
        model = pcf.nn.PathAutoencoder.make(
            normalizer=pcf.nn.StandardScalerNormalizer(pos, vel),
            gamma_range=(-1.0, 1.0),
            key=jax.random.key(0),
        )
        gamma = jnp.linspace(-0.9, 0.9, pos["x"].shape[0])
        member = jnp.ones(gamma.shape, dtype=bool)
        with pytest.raises(ValueError, match="track_ends"):
            _track_range(model, pos, gamma, member, None, ends="edge")


def _blurred_segment():
    """Pixels of a segment from (0, 0) to (4, 0), blurred with sigma 0.15."""
    X, Y = jnp.meshgrid(jnp.arange(-1.0, 5.0, 0.05), jnp.arange(-0.8, 0.8, 0.05))
    xs = jnp.linspace(0.0, 4.0, 400)
    d2 = jnp.min((X[..., None] - xs) ** 2 + Y[..., None] ** 2, axis=-1)
    image = jnp.exp(-d2 / (2 * 0.15**2))
    keep = image.ravel() > 0.02
    pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
    return pos, jax.tree.map(jnp.zeros_like, pos), image.ravel()[keep]


class TestTrackRangeFit:
    """A trained track ends at the ends of a blurred segment."""

    @pytest.fixture(scope="class")
    def fitted(self):
        """Fit the blurred segment once for the class."""
        pos, vel, w = _blurred_segment()
        cfg = pcf.nn.TrainingConfig(
            n_epochs_encoder=200,
            n_epochs_decoder=150,
            n_epochs_both=0,
            batch_size=500,
            show_pbar=False,
        )
        _, fitted, _ = pcf.fit_track(
            pos, vel, key=jax.random.key(0), weights=w, training_config=cfg
        )
        return fitted

    def test_track_range_reaches_the_ends(self, fitted):
        """Decoded at ``track_range``, the track ends within 0.1 of x=0 and x=4."""
        ends = np.sort(np.asarray(fitted(jnp.asarray(fitted.track_range))["x"]))
        np.testing.assert_allclose(ends, [0.0, 4.0], atol=0.1)

    def test_gamma_range_overshoots(self, fitted):
        """Control: the whole ``gamma_range`` runs past both ends."""
        ends = np.sort(np.asarray(fitted(jnp.asarray(fitted.gamma_range))["x"]))
        assert ends[0] < -0.1
        assert ends[1] > 4.1

    def test_within_gamma_range(self, fitted):
        """``track_range`` lies inside ``gamma_range``."""
        lo, hi = fitted.track_range
        assert fitted.gamma_range[0] <= lo
        assert lo < hi
        assert hi <= fitted.gamma_range[1]
