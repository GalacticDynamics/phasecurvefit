"""Acceptance test: an MSTOrderer result feeds the autoencoder unchanged."""

import jax
import jax.numpy as jnp
import numpy as np

import phasecurvefit as pcf


def _clean_arc(n=80):
    t = np.linspace(0.0, 1.0, n)
    x = 10.0 * t
    y = 2.0 * np.sin(3.0 * t)
    pos = {"x": jnp.asarray(x), "y": jnp.asarray(y)}
    vel = {"x": jnp.ones(n), "y": jnp.asarray(6.0 * np.cos(3.0 * t))}
    return pos, vel


def _train(mst, *, n_epochs, seed=0):
    normalizer = pcf.nn.StandardScalerNormalizer(mst.positions, mst.velocities)
    k1, k2 = jax.random.split(jax.random.key(seed))
    model = pcf.nn.PathAutoencoder.make(normalizer, gamma_range=mst.gamma_range, key=k1)
    cfg = pcf.nn.TrainingConfig(
        n_epochs_encoder=n_epochs, n_epochs_both=n_epochs, show_pbar=False
    )
    result, *_ = pcf.nn.train_autoencoder(model, mst, config=cfg, key=k2)
    return result


class TestMSTAutoencoderIntegration:
    """Tests for MST autoencoder integration."""

    def test_mst_result_feeds_autoencoder(self):
        """MST result feeds autoencoder."""
        pos, vel = _clean_arc()
        mst = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        result = _train(mst, n_epochs=5)
        assert result is not None
        assert result.gamma.shape == (len(pos["x"]),)
        assert jnp.all(jnp.isfinite(result.gamma))

    def test_encoder_gamma_monotone_in_arclength(self):
        """Encoder gamma monotone in arclength."""
        pos, vel = _clean_arc(n=80)
        mst = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        result = _train(mst, n_epochs=200)
        gamma_in_order = np.asarray(result.gamma)[np.asarray(mst.ordering)]
        rho = np.corrcoef(np.arange(gamma_in_order.size), gamma_in_order)[0, 1]
        assert abs(rho) > 0.9
