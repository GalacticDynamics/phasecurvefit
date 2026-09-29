"""Tests for the autoencoder normalizers."""

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.nn.normalize import AbstractNormalizer


def _plain():
    qs = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.5, 1.5, 2.5])}
    ps = {"vx": jnp.array([1.0, 1.0, 2.0]), "vy": jnp.array([0.0, 1.0, 1.0])}
    return qs, ps


def _quantity():
    u = pytest.importorskip("unxt")
    qs, ps = _plain()
    return (
        {k: u.Q(v, "kpc") for k, v in qs.items()},
        {k: u.Q(v, "km/s") for k, v in ps.items()},
    )


@pytest.mark.parametrize("make_data", [_plain, _quantity], ids=["array", "quantity"])
def test_standard_scaler_calls_base_init_once(monkeypatch, make_data):
    """Each ``__init__`` overload reaches the abstract base ``__init__`` once.

    The base is a plain (non-plum) no-op, so ``super().__init__`` must stop
    there rather than re-entering the subclass's plum dispatch.
    """
    calls = []
    monkeypatch.setattr(
        AbstractNormalizer, "__init__", lambda self, *_, **__: calls.append(self)
    )
    qs, ps = make_data()
    normalizer = pcf.nn.StandardScalerNormalizer(qs, ps)
    assert calls == [normalizer]
