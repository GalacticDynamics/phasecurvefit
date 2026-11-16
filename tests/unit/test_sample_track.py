"""Tests for ``pcf.sample_track``, ``pcf.track_model`` and ``pcf.TrackPosterior``."""

from typing import Any

import equinox as eqx
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


def _arc(n=N, seed=1, /):
    ang = jnp.linspace(0.0, jnp.pi, n)
    noise = 0.05 * jax.random.normal(jax.random.key(seed), (2, n))
    pos = {"x": 5 * jnp.cos(ang) + noise[0], "y": 5 * jnp.sin(ang) + noise[1]}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    return pos, vel


@pytest.fixture(scope="module")
def post():
    pos, vel = _arc()
    return pcf.sample_track(pos, vel, key=jax.random.key(0), n_draws=50)


def _interior(p, n=11, /):
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


def _density(z, params, /, **kw):
    model = handlers.substitute(posterior.track_model, data=params)
    return float(log_density(model, (z,), {"n_q": 2, **kw}, {})[0])


def _params(z, gamma, /, **kw):
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


# ---- audit fixes ----------------------------------------------------------------


def _radius_err(post, n=9, /):
    r = post.reference(_interior(post, n))
    return np.abs(np.hypot(np.asarray(r["x"]), np.asarray(r["y"])) - 5.0)


def test_reversed_ordering_is_turned_along_the_velocities():
    """A start that runs against the velocities is flipped, not fitted backwards."""
    pos, vel = _arc(80)
    o = pcf.orderers.default_pipeline(pos, vel)
    rev = eqx.tree_at(lambda r: r.indices, o, o.indices[::-1])
    post = pcf.sample_track(
        pos, vel, key=jax.random.key(0), ordering=rev, n_steps=1500, n_draws=10
    )
    assert np.median(_radius_err(post)) < 0.05


def test_positions_only_with_an_ordering():
    pos, _ = _arc(80)
    zero = {k: jnp.zeros(80) for k in pos}
    o = pcf.order(pos, zero, pcf.orderers.RidgeOrderer())
    post = pcf.sample_track(
        pos, None, key=jax.random.key(0), ordering=o, n_steps=1500, n_draws=10
    )
    assert "sigma_p" not in post.samples
    assert np.median(_radius_err(post)) < 0.05


def test_positions_only_needs_an_ordering():
    pos, _ = _arc(40)
    with pytest.raises(ValueError, match="needs an ordering"):
        pcf.sample_track(pos, None, key=jax.random.key(0))


@pytest.mark.parametrize(("method", "n_draws"), [("svi", 0), ("nuts", 6)])
def test_too_few_draws_raises(method, n_draws):
    pos, vel = _arc(40)
    with pytest.raises(ValueError, match="n_draws"):
        pcf.sample_track(
            pos, vel, key=jax.random.key(0), method=method, n_draws=n_draws
        )


def test_bad_weights_raise_with_an_ordering():
    pos, vel = _arc(40)
    o = pcf.orderers.default_pipeline(pos, vel)
    with pytest.raises(ValueError, match="non-negative"):
        pcf.sample_track(
            pos, vel, key=jax.random.key(0), ordering=o, weights=-jnp.ones(40)
        )


def test_ordering_of_other_tracers_raises():
    pos, vel = _arc(40)
    o = pcf.orderers.default_pipeline(*_arc(50))
    with pytest.raises(ValueError, match="ordering is of 50"):
        pcf.sample_track(pos, vel, key=jax.random.key(0), ordering=o)
    shifted = {k: v + 100.0 for k, v in pos.items()}
    o = pcf.orderers.default_pipeline(shifted, vel)
    with pytest.raises(ValueError, match="other tracers"):
        pcf.sample_track(pos, vel, key=jax.random.key(0), ordering=o)


def test_quantity_band_and_mixed_units():
    """The band carries a unit, and each component keeps its own unit."""
    pos, vel = _arc(60)
    meta = pcf.StateMetadata(usys=u.unitsystems.galactic)
    pq = {"x": u.Q(pos["x"], "kpc"), "y": u.Q(1000 * pos["y"], "pc")}
    vq = {k: u.Q(v, "km/s") for k, v in vel.items()}
    p = pcf.sample_track(
        pq, vq, key=jax.random.key(0), metadata=meta, n_steps=300, n_draws=10
    )
    g = _interior(p, 3)
    assert u.unit_of(p.reference(g)["y"]) == u.unit("pc")
    assert u.unit_of(p.radius(g)) == u.unit("kpc")
    assert u.unit_of(p.offsets(g)) == u.unit("kpc")


# ---- second review ------------------------------------------------------------


def test_direction_vote_ignores_interlopers():
    """60 interlopers with random velocities don't turn the start the wrong way."""
    rng = np.random.default_rng(3)
    ang = np.linspace(0.0, np.pi, 300)
    q = np.stack([5 * np.cos(ang), 5 * np.sin(ang)], 1) + rng.normal(0, 0.05, (300, 2))
    p = np.stack([-np.sin(ang), np.cos(ang)], 1)
    q = np.vstack([q, rng.uniform(-5, 5, (60, 2))])
    p = np.vstack([p, rng.normal(0, 1, (60, 2))])
    pos = {k: jnp.asarray(q[:, i]) for i, k in enumerate("xy")}
    vel = {k: jnp.asarray(p[:, i]) for i, k in enumerate("xy")}
    z, _, _ = posterior._scale(pos, vel)
    o = pcf.orderers.default_pipeline(pos, vel)
    rev = eqx.tree_at(lambda r: r.indices, o, o.indices[::-1])
    for start in (o, rev):
        idx, _ = posterior._start_gamma(z, 2, start, None)
        members = idx[idx < 300]
        step = np.diff(np.asarray(z)[members, :2], axis=0)
        along = np.sum(step * np.asarray(z)[members[1:], 2:], axis=1)
        assert np.mean(np.sign(along)) > 0.5


def test_stuck_chains_warn():
    """A chain stuck at one value has no R-hat; that counts as disagreeing."""
    with pytest.warns(RuntimeWarning, match="not computable"):
        posterior._warn_if_chains_disagree({"a": jnp.ones((2, 4, 3))})


def test_mismatched_velocity_keys_raise():
    pos, vel = _arc(40)
    vel = {"vx": vel["x"], "vy": vel["y"]}
    with pytest.raises(ValueError, match="components"):
        pcf.sample_track(pos, vel, key=jax.random.key(0))


def test_untied_three_dimensional_run():
    """End to end in 3-D with free velocity channels."""
    pos, vel = _arc(60)
    pos = {**pos, "z": 0.3 * pos["x"]}
    vel = {**vel, "z": 0.3 * vel["x"]}
    post = pcf.sample_track(
        pos, vel, key=jax.random.key(0), n_steps=300, n_draws=5, velocity_tie=None
    )
    assert post.samples["beta"].shape == (5, 8, 16)  # 6 free channels, density, width
    assert set(post.reference(0.0)) == {"x", "y", "z"}


# ---- third review -------------------------------------------------------------


def test_zero_velocities_point_to_velocities_none():
    pos, _ = _arc(40)
    zero = {k: jnp.zeros(40) for k in pos}
    with pytest.raises(ValueError, match="velocities=None"):
        pcf.sample_track(pos, zero, key=jax.random.key(0))


@pytest.mark.parametrize("tie", [-0.1, float("nan"), "loose"])
def test_bad_velocity_tie_raises(tie):
    pos, vel = _arc(40)
    with pytest.raises(ValueError, match="velocity_tie"):
        pcf.sample_track(pos, vel, key=jax.random.key(0), velocity_tie=tie)


def test_vote_ignores_repeated_positions():
    """Steps of zero length carry no direction, even when they are most steps."""
    q = np.array([[0.0, 0.0]] * 5 + [[1.0, 0.0], [2.0, 0.0]])
    zo = np.concatenate([q, np.tile([[1.0, 0.0]], (7, 1))], axis=1)
    assert not posterior._runs_backwards(zo, 2)
    assert posterior._runs_backwards(zo[::-1], 2)


def test_degenerate_orderings_raise():
    pos, vel = _arc(40)
    z, _, _ = posterior._scale(pos, vel)
    o = pcf.orderers.default_pipeline(pos, vel)
    one = eqx.tree_at(lambda r: r.indices, o, jnp.full(40, -1).at[0].set(0))
    with pytest.raises(ValueError, match="at least 2"):
        posterior._start_gamma(z, 2, one, None)
    with pytest.raises(ValueError, match="weight 0"):
        posterior._start_gamma(z, 2, o, jnp.zeros(40))


@pytest.mark.parametrize(
    ("option", "value"),
    [("n_steps", 0), ("n_grid", 1), ("n_basis", 0), ("n_draws", 3.0)],
)
def test_bad_counts_name_the_option(option, value):
    pos, vel = _arc(40)
    with pytest.raises(ValueError, match=option):
        pcf.sample_track(pos, vel, key=jax.random.key(0), **{option: value})


# ---- fifth review -------------------------------------------------------------


def test_draws_survive_jax_tree_operations_with_unsorted_keys():
    """JAX re-sorts dict keys; each component must keep its own column."""
    pos, vel = _arc(60)
    pos = {"y": pos["y"], "x": pos["x"]}
    vel = {"y": vel["y"], "x": vel["x"]}
    post = pcf.sample_track(pos, vel, key=jax.random.key(0), n_steps=300, n_draws=5)
    g = jnp.asarray(0.3)
    direct = post.reference(g)
    jitted = eqx.filter_jit(lambda p, gg: p.reference(gg))(post, g)
    mapped = jax.tree.map(lambda a: a, post.reference)(g)
    for out in (jitted, mapped):
        for k in ("x", "y"):
            np.testing.assert_allclose(out[k], direct[k], rtol=1e-5)


def test_numpy_counts_and_unused_options_are_accepted():
    pos, vel = _arc(40)
    post = pcf.sample_track(
        pos,
        vel,
        key=jax.random.key(0),
        n_steps=np.int64(50),
        n_draws=np.int32(3),
        n_warmup=0,  # NUTS only, so SVI doesn't check it
    )
    assert len(post.members) == 3


def test_ordering_in_other_units_raises():
    pos, vel = _arc(40)
    meta = pcf.StateMetadata(usys=u.unitsystems.galactic)
    pq = {k: u.Q(v, "kpc") for k, v in pos.items()}
    vq = {k: u.Q(v, "km/s") for k, v in vel.items()}
    o = pcf.orderers.default_pipeline(pos, vel)  # plain arrays
    with pytest.raises(ValueError, match="other units"):
        pcf.sample_track(pq, vq, key=jax.random.key(0), metadata=meta, ordering=o)
