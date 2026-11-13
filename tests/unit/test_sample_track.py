"""Tests for ``pcf.sample_track``, ``pcf.track_model`` and ``pcf.TrackPosterior``."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

import unxt as u

import phasecurvefit as pcf
from phasecurvefit._src import posterior

N = 160


def _arc(n=N, seed=1):
    ang = jnp.linspace(0.0, jnp.pi, n)
    noise = 0.05 * jax.random.normal(jax.random.key(seed), (2, n))
    pos = {"x": 5 * jnp.cos(ang) + noise[0], "y": 5 * jnp.sin(ang) + noise[1]}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    return pos, vel


@pytest.fixture(scope="module")
def post():
    pos, vel = _arc()
    return pcf.sample_track(pos, vel, key=jax.random.key(0), n_draws=50)


def _interior(p, n=11):
    lo, hi = p.reference.track_range
    pad = 0.1 * (hi - lo)
    return jnp.linspace(lo + pad, hi - pad, n)


def test_returns_draws(post):
    assert isinstance(post, pcf.TrackPosterior)
    assert isinstance(post, pcf.TrackEnsemble)
    assert isinstance(post.reference, pcf.TrackDraw)
    assert len(post.members) == 50
    assert post.samples["sigma"].shape == (50,)
    assert post.losses is not None
    assert post.losses.shape == (4000,)


def test_band_covers_the_arc(post):
    """The true arc lies within a few band radii of the reference."""
    g = _interior(post)
    r = post.reference(g)
    err = np.abs(np.hypot(np.asarray(r["x"]), np.asarray(r["y"])) - 5.0)
    rad = np.asarray(post.radius(g, 0.95))
    assert np.all(np.isfinite(rad))
    assert np.median(err) < np.median(rad)


def test_width_matches_the_noise(post):
    w = np.asarray(post.reference.width(_interior(post)))
    np.testing.assert_allclose(np.median(w), 0.05, rtol=0.3)


def test_members_differ(post):
    a = post.members[0](0.0)["x"]
    b = post.members[1](0.0)["x"]
    assert not jnp.allclose(a, b)


# ---- the model ------------------------------------------------------------------


def _density(z, params, **kw):
    model = handlers.substitute(posterior.track_model, data=params)
    return float(log_density(model, (z,), {"n_q": 2, **kw}, {})[0])


def _params(z, gamma, **kw):
    opts: dict[str, Any] = {"n_basis": 16, "line_density": True}
    opts["width_varies"] = True
    opts |= {"velocity_tie": 0.05} | kw
    return posterior._start(z, gamma, np.arange(len(z)), n_q=2, **opts)


def test_weights_act_as_counts():
    """A weight of 2 gives the likelihood of the star written twice."""
    pos, vel = _arc(40)
    z, _, _ = posterior._scale(pos, vel)
    gamma = np.linspace(-1, 1, 40)
    params = _params(z, gamma)
    w = jnp.ones(40).at[:5].set(2.0)
    doubled = jnp.concatenate([z, z[:5]])
    np.testing.assert_allclose(
        _density(z, params, weights=w), _density(doubled, params), rtol=1e-5
    )


def test_tie_prefers_the_velocity_direction():
    """A track that runs against the velocities misfits every one of them."""
    pos, vel = _arc(40)
    z, _, _ = posterior._scale(pos, vel)
    gamma = np.linspace(-1, 1, 40)
    along = _density(z, _params(z, gamma))
    against = _density(z, _params(z, gamma[::-1]))
    assert along > against


def test_untied_has_free_velocity_channels():
    pos, vel = _arc(40)
    z, _, _ = posterior._scale(pos, vel)
    params = _params(z, np.linspace(-1, 1, 40), velocity_tie=None)
    assert params["beta"].shape == (6, 16)  # 4 mean channels, density, width
    assert np.isfinite(_density(z, params, velocity_tie=None))


# ---- options and errors ---------------------------------------------------------


def test_nuts_warns_when_chains_disagree():
    pos, vel = _arc(60)
    with pytest.warns(RuntimeWarning, match="R-hat"):
        post = pcf.sample_track(
            pos, vel, key=jax.random.key(0), method="nuts", n_warmup=5, n_draws=10
        )
    assert post.losses is None
    assert len(post.members) == 10


def test_unknown_method_raises():
    pos, vel = _arc(40)
    with pytest.raises(ValueError, match="method"):
        pcf.sample_track(pos, vel, key=jax.random.key(0), method="mcmc")


def test_non_finite_raises():
    pos, vel = _arc(40)
    vel = {"x": vel["x"].at[3].set(jnp.nan), "y": vel["y"]}
    with pytest.raises(ValueError, match="finite"):
        pcf.sample_track(pos, vel, key=jax.random.key(0))


def test_missing_numpyro_raises(monkeypatch):
    class NoNumpyro:
        class NUMPYRO:
            installed = False

    monkeypatch.setattr(posterior, "OptDeps", NoNumpyro)
    pos, vel = _arc(40)
    with pytest.raises(ImportError, match=r"phasecurvefit\[numpyro\]"):
        pcf.sample_track(pos, vel, key=jax.random.key(0))


def test_quantities_round_trip():
    """Quantity inputs give Quantity tracks, the same in kpc and in pc."""
    pos, vel = _arc(60)
    meta = pcf.StateMetadata(usys=u.unitsystems.galactic)
    vq = {k: u.Q(v, "km/s") for k, v in vel.items()}
    out = {}
    for unit, factor in (("kpc", 1.0), ("pc", 1000.0)):
        pq = {k: u.Q(v * factor, unit) for k, v in pos.items()}
        p = pcf.sample_track(
            pq, vq, key=jax.random.key(0), metadata=meta, n_steps=300, n_draws=5
        )
        out[unit] = p.reference(jnp.asarray(0.0))["y"]
        assert out[unit].unit == u.unit(unit)
    in_kpc = {k: np.asarray(u.ustrip("kpc", v)) for k, v in out.items()}
    np.testing.assert_allclose(in_kpc["pc"], in_kpc["kpc"], rtol=1e-3)


def test_positions_only_model():
    """With no velocity columns the model has no velocity width."""
    pos, vel = _arc(40)
    z, _, _ = posterior._scale(pos, vel)
    zq = z[:, :2]
    params = _params(zq, np.linspace(-1, 1, 40))
    assert "sigma_p" not in params
    assert np.isfinite(_density(zq, params))


def test_plain_options_and_quiet_nuts(monkeypatch):
    """Hard tie, no line density or width channel, a given ordering, NUTS."""
    monkeypatch.setattr(posterior, "_RHAT_WARN", np.inf)
    pos, vel = _arc(60)
    ordering = pcf.orderers.default_pipeline(pos, vel)
    post = pcf.sample_track(
        pos,
        vel,
        key=jax.random.key(0),
        method="nuts",
        ordering=ordering,
        n_warmup=5,
        n_draws=8,
        background=False,
        line_density=False,
        width_varies=False,
        velocity_tie=0.0,
    )
    assert post.reference.track_range == (-1.0, 1.0)
    assert "f_bg" not in post.samples
