"""Shared fixtures for benchmark tests."""

import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

import phasecurvefit as pcf


def pytest_configure(config):
    """Configure pytest for benchmark tests."""
    # Register custom marker for benchmark tests
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test (may take long)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify benchmark tests to have longer timeout."""
    # For benchmark tests, disable the default timeout since they can take a while
    for item in items:
        # All tests in the benchmarks directory get extended timeout
        if "benchmarks" in str(item.fspath):
            # Apply a much longer timeout (30 minutes)
            item.add_marker(pytest.mark.timeout(1800))


@pytest.fixture
def simple_2d_stream(rng_key: PRNGKeyArray) -> tuple[dict, dict]:
    """Create a simple 2D stream for benchmarking.

    Returns
    -------
    tuple[dict, dict]
        (positions, velocities) dicts with 50 points

    """
    n_points = 50
    t = jnp.linspace(0, 10, n_points)

    pos = {"x": t, "y": 0.5 * jnp.sin(t)}
    vel = {"x": jnp.ones(n_points), "y": 0.5 * jnp.cos(t)}

    return pos, vel


@pytest.fixture
def medium_2d_stream(rng_key: PRNGKeyArray) -> tuple[dict, dict]:
    """Create a medium 2D stream (100 points).

    Returns
    -------
    tuple[dict, dict]
        (positions, velocities) dicts with 100 points

    """
    n_points = 100
    t = jnp.linspace(0, 20, n_points)

    pos = {"x": t, "y": 0.5 * jnp.sin(t)}
    vel = {"x": jnp.ones(n_points), "y": 0.5 * jnp.cos(t)}

    return pos, vel


@pytest.fixture
def large_2d_stream(rng_key: PRNGKeyArray) -> tuple[dict, dict]:
    """Create a large 2D stream (500 points).

    Returns
    -------
    tuple[dict, dict]
        (positions, velocities) dicts with 500 points

    """
    n_points = 500
    t = jnp.linspace(0, 50, n_points)

    pos = {"x": t, "y": 0.5 * jnp.sin(t)}
    vel = {"x": jnp.ones(n_points), "y": 0.5 * jnp.cos(t)}

    return pos, vel


@pytest.fixture
def simple_3d_stream(rng_key: PRNGKeyArray) -> tuple[dict, dict]:
    """Create a simple 3D stream (50 points).

    Returns
    -------
    tuple[dict, dict]
        (positions, velocities) dicts with 50 points in 3D

    """
    n_points = 50
    t = jnp.linspace(0, 10, n_points)

    pos = {
        "x": t,
        "y": 0.5 * jnp.sin(t),
        "z": 0.3 * jnp.cos(2 * t),
    }
    vel = {
        "x": jnp.ones(n_points),
        "y": 0.5 * jnp.cos(t),
        "z": -0.6 * jnp.sin(2 * t),
    }

    return pos, vel


@pytest.fixture
def simple_wlf_result(simple_2d_stream):
    """Create a phase-flow walk result."""
    pos, vel = simple_2d_stream
    return pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)


@pytest.fixture
def medium_wlf_result(medium_2d_stream):
    """Create a phase-flow walk result for 100 points."""
    pos, vel = medium_2d_stream
    return pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)


@pytest.fixture
def simple_autoencoder(simple_wlf_result, rng_key):
    """Create a simple autoencoder."""
    normalizer = pcf.nn.StandardScalerNormalizer(
        simple_wlf_result.positions, simple_wlf_result.velocities
    )
    return pcf.nn.PathAutoencoder.make(normalizer, gamma_range=(0.0, 1.0), key=rng_key)


@pytest.fixture
def medium_autoencoder(medium_wlf_result, rng_key):
    """Create autoencoder for medium dataset."""
    normalizer = pcf.nn.StandardScalerNormalizer(
        medium_wlf_result.positions, medium_wlf_result.velocities
    )
    return pcf.nn.PathAutoencoder.make(normalizer, gamma_range=(0.0, 1.0), key=rng_key)


@pytest.fixture
def trained_autoencoder(simple_2d_stream, rng_key):
    """Create and train an autoencoder."""
    pos, vel = simple_2d_stream
    walkresult = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

    normalizer = pcf.nn.StandardScalerNormalizer(
        walkresult.positions, walkresult.velocities
    )
    ae = pcf.nn.PathAutoencoder.make(
        normalizer, gamma_range=walkresult.gamma_range, key=rng_key
    )

    config = pcf.nn.TrainingConfig(n_epochs_encoder=5, n_epochs_both=5, show_pbar=False)
    result, _, _ = pcf.nn.train_autoencoder(ae, walkresult, config=config, key=rng_key)

    return result.model, result
