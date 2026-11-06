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
from phasecurvefit._src.nn.result import _track_range

pytest.importorskip("scipy")


class TestTrackRangeRule:
    """The half-median rule on hand-made gamma values."""

    def test_trims_faint_tail(self):
        """A faint tail past the end of a full-weight stretch is cut off."""
        gamma = jnp.linspace(-0.8, 1.0, 400)
        weights = jnp.where(gamma <= 0.8, 1.0, 0.05)
        lo, hi = _track_range(gamma, jnp.ones(400, dtype=bool), weights)
        assert lo == pytest.approx(-0.8)
        assert hi == pytest.approx(0.8, abs=0.05)

    def test_even_sampling_keeps_span(self):
        """Evenly spread, unweighted samples keep their whole span."""
        gamma = jnp.linspace(-0.9, 0.9, 400)
        lo, hi = _track_range(gamma, jnp.ones(400, dtype=bool), None)
        assert lo == pytest.approx(-0.9)
        assert hi == pytest.approx(0.9)

    def test_only_members_count(self):
        """Non-members don't widen the range."""
        gamma = jnp.linspace(-1.0, 1.0, 400)
        member = jnp.abs(gamma) <= 0.5
        lo, hi = _track_range(gamma, member, None)
        assert lo >= -0.5
        assert hi <= 0.5

    def test_no_members(self):
        """With no members, the range is the span of all gamma."""
        gamma = jnp.linspace(-0.3, 0.6, 20)
        lo, hi = _track_range(gamma, jnp.zeros(20, dtype=bool), None)
        assert (lo, hi) == pytest.approx((-0.3, 0.6))


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
        assert fitted.gamma_range[0] <= lo < hi <= fitted.gamma_range[1]
