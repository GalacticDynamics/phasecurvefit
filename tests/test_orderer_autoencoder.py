"""Acceptance test: an MSTOrderer result feeds the autoencoder unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

import phasecurvefit as pcf


def _epitrochoid(n=2048, noise=6.0, seed=1, scale=120.0, R=5.0, r=1.0, d=4.5):
    """Return a self-intersecting epitrochoid (5 internal loops per rotation)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(np.deg2rad(5.0), np.deg2rad(355.0), n)
    ratio = (R + r) / r
    x = scale * ((R + r) * np.cos(t) - d * np.cos(ratio * t)) / 5.0
    y = scale * ((R + r) * np.sin(t) - d * np.sin(ratio * t)) / 5.0
    dx = scale * (-(R + r) * np.sin(t) + d * ratio * np.sin(ratio * t)) / 5.0
    dy = scale * ((R + r) * np.cos(t) - d * ratio * np.cos(ratio * t)) / 5.0
    pos = {
        "x": jnp.asarray(x + rng.normal(0, noise, n)),
        "y": jnp.asarray(y + rng.normal(0, noise, n)),
    }
    vel = {"x": jnp.asarray(dx), "y": jnp.asarray(dy)}
    return pos, vel


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

    def test_decoder_traces_self_intersecting_curve(self):
        """The decoded mean path hugs the data on a self-intersecting rose.

        Regression for the Phase-2 target being denoised over the (locally noisy)
        encoder gamma instead of the ordering: the decoder faithfully learns a
        blurred target and the mean path sits far off the curve.
        """
        pos, vel = _epitrochoid()
        P = np.stack([np.asarray(pos["x"]), np.asarray(pos["y"])], axis=1)
        tree = cKDTree(P)
        med = float(np.median(tree.query(P, k=2)[0][:, 1]))
        mst = pcf.orderers.MSTOrderer(
            k=16,
            jump_cap=8.0 * med,
            sever_cos_threshold=0.9,
            orient_by_velocity=True,
            on_disconnected="largest",
        ).order(pos, vel)

        normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
        k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
        decoder = pcf.nn.FourierTrackNet(
            out_size=2, n_frequencies=10, width_size=128, depth=3, key=k1
        )
        model = pcf.nn.PathAutoencoder.make(
            normalizer, gamma_range=mst.gamma_range, decoder=decoder, key=k2
        )
        cfg = pcf.nn.TrainingConfig(
            n_epochs_encoder=500, n_epochs_decoder=800, n_epochs_both=0, show_pbar=False
        )
        result, *_ = pcf.nn.train_autoencoder(model, mst, config=cfg, key=k3)

        path = result(jnp.linspace(*result.gamma_range, 1000))
        A = np.stack([np.asarray(path["x"]), np.asarray(path["y"])], axis=1)
        hug = float(np.median(tree.query(A)[0]))
        assert hug < 4.0 * med, (
            f"decoder mean path {hug:.1f} too far (spacing {med:.1f})"
        )
