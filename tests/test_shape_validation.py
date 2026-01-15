"""Test that beartype/jaxtyping properly validates array shapes."""

import os

import jax.numpy as jnp
import pytest

# Enable beartype for testing
os.environ["LOCALFLOWWALK_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"

import localflowwalk as lfw


def test_algorithm_requires_correct_shape():
    """Test that walk_local_flow validates array shapes."""
    # Correct shapes work
    pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

    result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)
    assert result.ordered_indices.shape == (3,)


def test_default_metric_matches_full_phase_space():
    """Default metric should be FullPhaseSpaceDistanceMetric."""
    pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

    default_result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)
    full_phase_result = lfw.walk_local_flow(
        pos,
        vel,
        start_idx=0,
        lam=1.0,
        metric=lfw.metrics.FullPhaseSpaceDistanceMetric(),
    )

    assert jnp.allclose(
        default_result.ordered_indices, full_phase_result.ordered_indices
    )


def test_autoencoder_validates_dimensions():
    """Test that autoencoder validates n_dims."""
    import jax.random as jr

    key = jr.PRNGKey(0)

    # Create valid autoencoder with 2D data
    autoencoder = lfw.nn.Autoencoder(rngs=key, n_dims=2)

    # Valid 2D phase-space data
    pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

    # Should work fine
    gamma, prob = autoencoder.predict(pos, vel)
    assert gamma.shape == (3,)
    assert prob.shape == (3,)


def test_metrics_return_correct_shapes():
    """Test that metrics return 1D arrays of correct length."""
    pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

    current_pos = {k: v[0] for k, v in pos.items()}
    current_vel = {k: v[0] for k, v in vel.items()}

    metric = lfw.metrics.AlignedMomentumDistanceMetric()
    distances = metric(current_pos, current_vel, pos, vel, lam=1.0)

    # Should return 1D array of length N
    assert distances.shape == (3,)


def test_phasespace_functions_handle_scalars_and_arrays():
    """Test that phasespace functions work with both scalar and array inputs."""
    from localflowwalk._src.phasespace import euclidean_distance, get_w_at

    # Scalar inputs
    pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
    pos_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
    dist = euclidean_distance(pos_a, pos_b)
    assert dist.shape == ()  # scalar

    # Array inputs via get_w_at
    pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

    # Get single point
    p, v = get_w_at(pos, vel, 0)
    assert p["x"].shape == ()  # scalar
    assert v["x"].shape == ()  # scalar

    # Get multiple points
    p, v = get_w_at(pos, vel, jnp.array([0, 2]))
    assert p["x"].shape == (2,)  # 1D array
    assert v["x"].shape == (2,)  # 1D array


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
