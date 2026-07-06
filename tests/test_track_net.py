"""Tests for the swappable trainable decoder (AbstractTrackNet / FourierTrackNet)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phasecurvefit as pcf


class TestFourierTrackNet:
    """The Fourier-feature decoder variant."""

    def test_subclasses_abstract_track_net(self):
        """Both concrete decoders share the AbstractTrackNet interface."""
        assert issubclass(pcf.nn.FourierTrackNet, pcf.nn.AbstractTrackNet)
        assert issubclass(pcf.nn.TrackNet, pcf.nn.AbstractTrackNet)

    def test_output_shape(self):
        """A scalar gamma decodes to an (out_size,) position."""
        d = pcf.nn.FourierTrackNet(
            out_size=2, n_frequencies=4, width_size=32, depth=2, key=jr.key(0)
        )
        out = d(jnp.array(0.3))
        assert out.shape == (2,)

    def test_feature_count_is_1_plus_2k(self):
        """The Fourier embedding has 1 + 2*n_frequencies features."""
        d = pcf.nn.FourierTrackNet(
            out_size=2, n_frequencies=4, width_size=8, depth=1, key=jr.key(0)
        )
        assert d.n_frequencies == 4
        feats = d.features(jnp.array(0.2))
        assert feats.shape == (1 + 2 * 4,)

    def test_differentiable_in_gamma(self):
        """The tangent (velocity) loss needs jvp of the decoder w.r.t. gamma."""
        d = pcf.nn.FourierTrackNet(
            out_size=2, n_frequencies=4, width_size=32, depth=2, key=jr.key(0)
        )
        primal, tangent = jax.jvp(d, (jnp.array(0.3),), (jnp.array(1.0),))
        assert primal.shape == (2,)
        assert tangent.shape == (2,)
        assert jnp.all(jnp.isfinite(tangent))


class TestSwappableDecoder:
    """PathAutoencoder.make accepts a decoder override."""

    @pytest.fixture
    def norm(self):
        """Return a 2D standard-scaler normalizer fixture."""
        q = {"x": jnp.linspace(0.0, 1.0, 10), "y": jnp.zeros(10)}
        p = {"x": jnp.ones(10), "y": jnp.zeros(10)}
        return pcf.nn.StandardScalerNormalizer(q, p)

    def test_default_decoder_is_tracknet(self, norm):
        """With no decoder override, make() builds the default TrackNet."""
        ae = pcf.nn.PathAutoencoder.make(norm, gamma_range=(0.0, 1.0), key=jr.key(2))
        assert isinstance(ae.decoder, pcf.nn.TrackNet)

    def test_make_uses_passed_decoder(self, norm):
        """A passed decoder is used verbatim."""
        dec = pcf.nn.FourierTrackNet(
            out_size=2, n_frequencies=3, width_size=16, depth=1, key=jr.key(1)
        )
        ae = pcf.nn.PathAutoencoder.make(
            norm, gamma_range=(-1.0, 1.0), decoder=dec, key=jr.key(2)
        )
        assert isinstance(ae.decoder, pcf.nn.FourierTrackNet)
        assert ae.decoder is dec

    def test_out_size_mismatch_raises(self, norm):
        """A decoder whose out_size != n_spatial_dims is rejected."""
        # norm is 2D; a 3-output decoder is inconsistent.
        dec = pcf.nn.FourierTrackNet(
            out_size=3, n_frequencies=3, width_size=16, depth=1, key=jr.key(1)
        )
        with pytest.raises(ValueError, match="out_size"):
            pcf.nn.PathAutoencoder.make(
                norm, gamma_range=(0.0, 1.0), decoder=dec, key=jr.key(2)
            )
