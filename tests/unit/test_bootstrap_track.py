"""Tests for ``pcf.bootstrap_track`` and ``pcf.TrackEnsemble``."""

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src import pipeline


def _fast_config():
    return pcf.nn.TrainingConfig(
        n_epochs_encoder=3,
        n_epochs_decoder=2,
        n_epochs_both=2,
        batch_size=10,
        show_pbar=False,
    )


@pytest.fixture(scope="module")
def ensemble():
    rng = np.random.default_rng(0)
    ang = jnp.linspace(0.0, jnp.pi, 40)
    pos = {
        "x": 5.0 * jnp.cos(ang) + rng.normal(0, 0.1, 40),
        "y": 5.0 * jnp.sin(ang) + rng.normal(0, 0.1, 40),
    }
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    return pcf.bootstrap_track(
        pos,
        vel,
        key=jax.random.key(0),
        n_samples=3,
        n_prototypes=8,
        training_config=_fast_config(),
    )


def _gamma(ens, n=7):
    lo, hi = ens.reference.track_range
    return jnp.linspace(lo, hi, n)


def test_makes_reference_and_members(ensemble):
    assert isinstance(ensemble, pcf.TrackEnsemble)
    assert isinstance(ensemble.reference, pcf.nn.AutoencoderResult)
    assert len(ensemble.members) == 3


def test_members_differ(ensemble):
    """Each member is fitted to its own resample, so the tracks differ."""
    g = jnp.array([0.0])
    a = ensemble.members[0].model.decode(g)["x"]
    b = ensemble.members[1].model.decode(g)["x"]
    assert not jnp.allclose(a, b)


def test_offsets_shape(ensemble):
    assert ensemble.offsets(_gamma(ensemble)).shape == (3, 7, 2)


def test_offsets_are_perpendicular_to_the_reference(ensemble):
    g = _gamma(ensemble)
    d = ensemble.offsets(g)

    def stack(q):
        return jnp.stack([q["x"], q["y"]], -1)

    ref = ensemble.reference.model.decode
    t = jax.vmap(jax.jacfwd(lambda x: stack(ref(x))))(g)
    t = t / jnp.linalg.norm(t, axis=-1, keepdims=True)
    along = jnp.sum(d * t[None], axis=-1)
    assert jnp.allclose(jnp.nan_to_num(along), 0.0, atol=1e-5)


def test_reference_as_its_own_member_has_zero_offset(ensemble):
    """Up to the spacing of the dense evaluation, a track is 0 from itself."""
    ens = pcf.TrackEnsemble(ensemble.reference, (ensemble.reference,))
    g = _gamma(ens)[1:-1]  # inside the ends, so the nearest point isn't an end
    assert jnp.nanmax(jnp.abs(ens.offsets(g))) < 1e-3
    assert jnp.all(ens.radius(g) < 1e-3)


def test_member_that_stops_short_gives_nan(ensemble):
    ref = ensemble.reference
    lo, hi = ref.track_range
    short = eqx.tree_at(lambda r: r.track_range, ref, (lo, lo + 0.5 * (hi - lo)))
    ens = pcf.TrackEnsemble(ref, (short,))
    d = ens.offsets(jnp.array([lo + 0.25 * (hi - lo), hi]))
    assert jnp.all(jnp.isfinite(d[0, 0]))
    assert jnp.all(jnp.isnan(d[0, 1]))


def test_radius_is_quantile_of_distance(ensemble):
    g = _gamma(ensemble)[1:-1]
    dist = jnp.linalg.norm(ensemble.offsets(g), axis=-1)
    r68 = ensemble.radius(g, 0.68)
    r95 = ensemble.radius(g, 0.95)
    assert r68.shape == (5,)
    full = jnp.all(jnp.isfinite(dist), axis=0)  # every member reaches
    assert jnp.allclose(r68[full], jnp.quantile(dist, 0.68, axis=0)[full])
    assert jnp.all(jnp.nan_to_num(r95 - r68) >= 0)


def test_radius_counts_members_that_stop_short_as_outside(ensemble):
    """With 2 of 3 members reaching a point, no 68% tube holds there."""
    ref = ensemble.reference
    lo, hi = ref.track_range
    short = eqx.tree_at(lambda r: r.track_range, ref, (lo, lo + 0.5 * (hi - lo)))
    ens = pcf.TrackEnsemble(ref, (ref, ref, short))
    g = jnp.array([lo + 0.25 * (hi - lo), lo + 0.9 * (hi - lo)])
    r = ens.radius(g, 0.68)
    assert jnp.isfinite(r[0])
    assert jnp.isnan(r[1])
    assert jnp.isfinite(ens.radius(g, 0.3)[1])


def test_passes_fit_track_options_through(monkeypatch):
    seen = []

    def fake_fit_track(pos, vel, **kw):
        seen.append(kw)
        return None, ensemble_result, None

    ensemble_result = object()
    monkeypatch.setattr(pipeline, "fit_track", fake_fit_track)
    pos = {"x": jnp.arange(10.0), "y": jnp.zeros(10)}
    vel = {"x": jnp.ones(10), "y": jnp.zeros(10)}
    ens = pipeline.bootstrap_track(
        pos, vel, key=jax.random.key(0), n_samples=2, outlier_mixture=None
    )
    assert len(seen) == 3
    assert all(kw["outlier_mixture"] is None for kw in seen)
    assert ens.members == (ensemble_result, ensemble_result)


def test_accepts_every_fit_track_option():
    """Every keyword of fit_track can be given to bootstrap_track."""
    fit = inspect.signature(pcf.fit_track).parameters
    boot = inspect.signature(pcf.bootstrap_track).parameters
    takes_any = any(p.kind is p.VAR_KEYWORD for p in boot.values())
    missing = [n for n in fit if n not in boot and n not in ("positions", "velocities")]
    assert takes_any or not missing


def test_n_samples_must_be_positive():
    pos = {"x": jnp.arange(10.0), "y": jnp.zeros(10)}
    vel = {"x": jnp.ones(10), "y": jnp.zeros(10)}
    with pytest.raises(ValueError, match="n_samples"):
        pcf.bootstrap_track(pos, vel, key=jax.random.key(0), n_samples=0)
