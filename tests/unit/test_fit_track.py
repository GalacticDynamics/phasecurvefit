"""Tests for ``pcf.fit_track``, the one-call order+refine+train convenience path."""

import jax
import jax.numpy as jnp

import phasecurvefit as pcf


def _fast_config(**overrides):
    """Build a ``TrainingConfig`` small enough to run in a unit test."""
    defaults = {
        "n_epochs_encoder": 3,
        "n_epochs_decoder": 2,
        "n_epochs_both": 2,
        "batch_size": 10,
        "show_pbar": False,
    }
    return pcf.nn.TrainingConfig(**{**defaults, **overrides})


def test_fit_track_exists():
    assert callable(pcf.fit_track)


def test_runs_all_three_steps(arc):
    """Ordering, model build, and training all actually ran."""
    pos, vel, _ = arc(n=30)
    cfg = _fast_config()
    ordering, fitted, losses = pcf.fit_track(
        pos, vel, key=jax.random.key(0), n_prototypes=8, training_config=cfg
    )
    assert isinstance(ordering, pcf.orderers.OrderingResult)
    assert isinstance(fitted, pcf.nn.AutoencoderResult)
    assert losses.shape == (cfg.n_epochs,)
    assert jnp.all(jnp.isfinite(losses))


def test_ordering_result_matches_default_pipeline_alone(arc):
    """The ordering step is not silently different from calling it directly."""
    pos, vel, _ = arc(n=30)
    direct = pcf.orderers.default_pipeline(pos, vel, n_prototypes=8)
    ordering, _fitted, _losses = pcf.fit_track(
        pos, vel, key=jax.random.key(0), n_prototypes=8, training_config=_fast_config()
    )
    assert jnp.array_equal(ordering.indices, direct.indices)
    assert ordering.gamma_range == direct.gamma_range


def test_falls_back_below_n_prototypes_and_still_trains(arc):
    """Too few tracers for the SOM stage must not stop the pipeline overall.

    The MST-only fallback still hands train_autoencoder a valid result -- one
    with no SOM backbone of its own -- and training must adapt rather than
    assume the SOM ran.
    """
    pos, vel, _ = arc(n=10)
    ordering, fitted, losses = pcf.fit_track(
        pos,
        vel,
        key=jax.random.key(0),
        n_prototypes=15,
        training_config=_fast_config(batch_size=5),
    )
    assert isinstance(ordering, pcf.orderers.OrderingResult)
    assert ordering.backbone_size is not None  # the MST stage alone ran
    assert jnp.all(jnp.isfinite(losses))
    assert fitted.model.normalizer.n_spatial_dims == len(pos)


def test_model_is_queryable(arc):
    """The returned result's whole point: interpolate along the fitted track."""
    pos, vel, _ = arc(n=30)
    _ordering, fitted, _losses = pcf.fit_track(
        pos, vel, key=jax.random.key(0), n_prototypes=8, training_config=_fast_config()
    )
    out = fitted(jnp.array(0.0))
    assert set(out) == set(pos)


def test_same_key_reproduces_the_fit(arc):
    """The one top-level ``key`` argument must fully determine the result."""
    pos, vel, _ = arc(n=30)
    cfg = _fast_config()
    key = jax.random.key(0)
    _o1, fitted1, _l1 = pcf.fit_track(
        pos, vel, key=key, n_prototypes=8, training_config=cfg
    )
    _o2, fitted2, _l2 = pcf.fit_track(
        pos, vel, key=key, n_prototypes=8, training_config=cfg
    )
    out1 = fitted1(jnp.array(0.0))["x"]
    out2 = fitted2(jnp.array(0.0))["x"]
    assert jnp.array_equal(out1, out2)
