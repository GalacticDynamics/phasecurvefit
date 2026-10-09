"""Smoke tests for a core-only install (no extras, no interop).

CI runs this file in an isolated environment with only the package's core
dependencies plus pytest, so it catches imports that are satisfied only
transitively (e.g. via ``unxt`` in the ``interop`` extra). Keep it
self-contained: no fixtures from ``conftest.py`` and no optional dependencies.

Skipped unless ``PHASECURVEFIT_SMOKE_TESTS=1`` (set by the CI smoke job), so the
full test matrix doesn't repeat them.
"""

import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf

pytestmark = pytest.mark.skipif(
    os.environ.get("PHASECURVEFIT_SMOKE_TESTS") != "1",
    reason="core-only smoke tests; set PHASECURVEFIT_SMOKE_TESTS=1 to run",
)


@pytest.mark.parametrize(
    "name", ["metrics", "neighbors", "nn", "orderers", "strats", "w"]
)
def test_public_submodules_import(name):
    """Each public submodule imports with core dependencies only."""
    importlib.import_module(f"phasecurvefit.{name}")


def _shuffled_line(n=20):
    """Points along +x moving in +x, in shuffled order."""
    perm = np.random.default_rng(0).permutation(n)
    x = jnp.linspace(0.0, 1.0, n)[perm]
    pos = {"x": x, "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    return pos, vel


@pytest.mark.parametrize(
    "make_orderer",
    [
        # Start at an end of the line: with no ``max_dist`` the walk would
        # otherwise reach one end and then jump back to the other half.
        lambda start: pcf.orderers.LocalFlowOrderer(metric_scale=0.1, start_idx=start),
        lambda _: pcf.orderers.MSTOrderer(k=5),
    ],
    ids=["localflow", "mst"],
)
def test_order_recovers_a_line(make_orderer):
    """`order` puts shuffled points on a line back in monotonic order."""
    pos, vel = _shuffled_line()
    start = int(jnp.argmin(pos["x"]))
    result = pcf.order(pos, vel, make_orderer(start))

    indices = np.asarray(result.indices)
    assert np.all(indices >= 0), "an orderer skipped points (-1 sentinel)"
    xs = np.asarray(pos["x"])[indices]
    diffs = np.diff(xs)
    assert len(xs) == len(pos["x"])
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_autoencoder_trains():
    """The README quickstart pipeline runs end to end and gives finite losses."""
    pos, vel = _shuffled_line()
    key = jax.random.key(0)
    result = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(metric_scale=0.1))

    normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
    autoencoder = pcf.nn.PathAutoencoder.make(
        normalizer, gamma_range=result.gamma_range, key=key
    )
    config = pcf.nn.TrainingConfig(n_epochs_encoder=2, n_epochs_both=2, show_pbar=False)
    _, _, losses = pcf.nn.train_autoencoder(autoencoder, result, config=config, key=key)

    assert np.all(np.isfinite(np.asarray(losses)))
