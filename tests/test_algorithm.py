"""Tests for the Nearest Neighbors with Momentum algorithm."""

import jax.numpy as jnp
import pytest

import localflowwalk as lfw


class TestLocalFlowWalkResult:
    """Tests for LocalFlowWalkResult and accessors."""

    def test_result_structure(self):
        """Test that result has correct structure."""
        pos = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0)

        # Check that result has the expected attributes (NamedTuple)
        assert hasattr(result, "ordered_indices")
        assert hasattr(result, "skipped_indices")
        assert hasattr(result, "positions")
        assert hasattr(result, "velocities")
        # ordered_indices is now an array, not a tuple
        assert isinstance(result.ordered_indices, jnp.ndarray)
        # skipped_indices is computed via property
        assert isinstance(result.skipped_indices, jnp.ndarray)

    def test_get_ordered_w(self):
        """Test accessing ordered positions and velocities."""
        pos = {"x": jnp.array([3.0, 1.0, 2.0]), "y": jnp.array([30.0, 10.0, 20.0])}
        vel = {"x": jnp.array([0.3, 0.1, 0.2]), "y": jnp.array([3.0, 1.0, 2.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=1, lam=0.0)
        ordered_pos, ordered_vel = lfw.get_ordered_w(result)

        # Order starts from index 1, then finds nearest neighbors
        assert ordered_pos["x"].shape == (3,)
        assert ordered_vel["x"].shape == (3,)


class TestNearestNeighborsWithMomentum:
    """Tests for the main algorithm."""

    def test_simple_line(self):
        """Test with points on a simple line with aligned velocities."""
        # Points along x-axis with positive x velocity
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        # Should follow the line in order
        assert jnp.array_equal(result.ordered_indices, jnp.array([0, 1, 2, 3, 4]))

    def test_simple_line_reverse(self):
        """Test with points on a line, velocity pointing the other way."""
        # Points along x-axis with negative x velocity (pointing backward)
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel = {"x": jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])}

        # Start from middle, should go backward due to velocity
        result = lfw.walk_local_flow(pos, vel, start_idx=2, lam=1.0)

        # With momentum, should prefer going in velocity direction (backward)
        # First step from 2: velocity points to -x, so prefer 1 over 3
        assert result.ordered_indices[1] == 1

    def test_2d_stream(self):
        """Test with a 2D stream of points."""
        # Points along a diagonal with matching velocities
        pos = {
            "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
            "y": jnp.array([0.0, 1.0, 2.0, 3.0]),
        }
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
            "y": jnp.array([1.0, 1.0, 1.0, 1.0]),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        assert jnp.array_equal(result.ordered_indices, jnp.array([0, 1, 2, 3]))

    def test_lambda_zero_is_pure_nearest_neighbor(self):
        """Test that λ=0 gives pure nearest neighbor (no momentum)."""
        # Two points equidistant, but one in velocity direction
        pos = {
            "x": jnp.array([0.0, 1.0, -1.0]),
            "y": jnp.array([0.0, 0.0, 0.0]),
        }
        vel = {
            "x": jnp.array([1.0, 0.0, 0.0]),  # velocity points to +x
            "y": jnp.array([0.0, 0.0, 0.0]),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0)

        # With λ=0, it's pure distance, both are at distance 1
        # The algorithm will pick the first one found
        assert jnp.sum(result.ordered_indices >= 0) == 3
        assert result.ordered_indices[0] == 0

    def test_high_lambda_prefers_momentum_direction(self):
        """Test that high λ strongly prefers the momentum direction."""
        # Point at origin with velocity in +x direction
        # One point at (1, 0) - aligned with velocity
        # One point at (-1, 0) - opposite to velocity
        pos = {
            "x": jnp.array([0.0, 1.0, -1.0]),
            "y": jnp.array([0.0, 0.0, 0.0]),
        }
        vel = {
            "x": jnp.array([1.0, 0.0, 0.0]),  # velocity points to +x
            "y": jnp.array([0.0, 0.0, 0.0]),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=10.0)

        # With high λ, should strongly prefer point 1 (in velocity direction)
        assert result.ordered_indices[1] == 1

    def test_max_dist_termination(self):
        """Test that max_dist terminates the algorithm."""
        # Points with a gap
        pos = {"x": jnp.array([0.0, 1.0, 10.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0, max_dist=5.0)

        # Should stop before reaching point 2 (at x=10)
        assert not jnp.any(result.ordered_indices == 2)
        assert jnp.sum(result.ordered_indices >= 0) == 2

    def test_max_dist_skipped_indices(self):
        """Test that max_dist populates skipped_indices correctly."""
        # Points with a gap - point 2 is far away
        pos = {"x": jnp.array([0.0, 1.0, 10.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0, max_dist=5.0)

        # Point 2 should be skipped
        assert 2 in result.skipped_indices
        assert len(result.skipped_indices) == 1
        # All indices should be accounted for
        all_indices = set(
            result.ordered_indices[result.ordered_indices >= 0].tolist()
        ) | set(result.skipped_indices.tolist())
        assert all_indices == {0, 1, 2}

    def test_no_skipped_when_all_visited(self):
        """Test that skipped_indices is empty when all points visited."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0)

        assert len(result.skipped_indices) == 0
        assert jnp.sum(result.ordered_indices >= 0) == 3

    def test_n_max_limits_iterations(self):
        """Test that n_max limits the number of points."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0, n_max=3)

        assert jnp.sum(result.ordered_indices >= 0) == 3

    def test_terminate_indices(self):
        """Test that terminate_indices stops at specified points."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(
            pos, vel, start_idx=0, lam=0.0, terminate_indices={2}
        )

        # Should visit indices 0, 1, 2 and then stop
        valid_indices = result.ordered_indices[result.ordered_indices >= 0]
        assert len(valid_indices) == 3
        assert jnp.array_equal(valid_indices, jnp.array([0, 1, 2]))
        # Indices 3 and 4 should not be visited
        assert 3 not in valid_indices
        assert 4 not in valid_indices

    def test_invalid_start_idx_raises(self):
        """Test that invalid start_idx raises ValueError."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}

        with pytest.raises(ValueError, match="out of bounds"):
            lfw.walk_local_flow(pos, vel, start_idx=10)

        with pytest.raises(ValueError, match="out of bounds"):
            lfw.walk_local_flow(pos, vel, start_idx=-1)

    def test_single_point(self):
        """Test with a single point."""
        pos = {"x": jnp.array([1.0])}
        vel = {"x": jnp.array([1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0)

        assert jnp.array_equal(result.ordered_indices, jnp.array([0]))

    def test_3d_helix(self):
        """Test with points along a 3D helix-like structure."""
        t = jnp.linspace(0, 4 * jnp.pi, 20)
        pos = {
            "x": jnp.cos(t),
            "y": jnp.sin(t),
            "z": t / (4 * jnp.pi),  # gradually increases
        }
        # Tangent velocity
        vel = {
            "x": -jnp.sin(t),
            "y": jnp.cos(t),
            "z": jnp.ones_like(t) / (4 * jnp.pi),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        # Should roughly follow the helix order
        # Check that we visit all points
        assert jnp.sum(result.ordered_indices >= 0) == 20
        assert set(result.ordered_indices[result.ordered_indices >= 0].tolist()) == set(
            range(20)
        )


class TestAlgorithmIntegration:
    """Integration tests for the algorithm with realistic scenarios."""

    def test_curved_stream(self):
        """Test algorithm on a curved stream with varying velocity."""
        # Create a curved stream that bends
        n_points = 10
        t = jnp.linspace(0, jnp.pi / 2, n_points)

        # Arc of a circle
        pos = {
            "x": jnp.cos(t),
            "y": jnp.sin(t),
        }
        # Tangent velocity (derivative of position)
        vel = {
            "x": -jnp.sin(t),
            "y": jnp.cos(t),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=2.0)

        # With momentum, should follow the arc correctly
        # Check that consecutive points are neighbors in the result
        for i in range(len(result.ordered_indices) - 1):
            curr = result.ordered_indices[i]
            next_ = result.ordered_indices[i + 1]
            # Adjacent points in original should be within 2 indices of each other
            assert abs(next_ - curr) <= 3

    def test_noisy_stream(self):
        """Test algorithm on a stream with some noise."""
        import jax

        key = jax.random.PRNGKey(42)

        # Base stream along x-axis
        n_points = 20
        base_x = jnp.linspace(0, 10, n_points)
        base_y = jnp.zeros(n_points)

        # Add small noise
        key1, key2 = jax.random.split(key)
        noise_x = jax.random.normal(key1, (n_points,)) * 0.1
        noise_y = jax.random.normal(key2, (n_points,)) * 0.1

        pos = {
            "x": base_x + noise_x,
            "y": base_y + noise_y,
        }
        vel = {
            "x": jnp.ones(n_points),  # Moving in +x direction
            "y": jnp.zeros(n_points),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        # Should visit all points
        assert len(result.ordered_indices) == n_points
        assert set(result.ordered_indices[result.ordered_indices >= 0].tolist()) == set(
            range(n_points)
        )

        # The ordering should roughly follow the x-coordinate
        ordered_pos, _ = lfw.get_ordered_w(result)
        ordered_x = ordered_pos["x"]
        # Check that x is generally increasing (allow some local variation)
        avg_increase = jnp.mean(jnp.diff(ordered_x))
        assert avg_increase > 0  # Overall trend should be increasing

    def test_bidirectional_stream(self):
        """Test that algorithm can trace stream in either direction."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel_forward = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}
        vel_backward = {"x": jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])}

        # Start from middle with forward velocity
        result_forward = lfw.walk_local_flow(pos, vel_forward, start_idx=2, lam=5.0)
        # Start from middle with backward velocity
        result_backward = lfw.walk_local_flow(pos, vel_backward, start_idx=2, lam=5.0)

        # Forward should go 2 -> 3 -> 4 -> ...
        assert result_forward.ordered_indices[1] == 3

        # Backward should go 2 -> 1 -> 0 -> ...
        assert result_backward.ordered_indices[1] == 1

    def test_multidimensional_keys(self):
        """Test with Cartesian coordinate keys (x, y, z)."""
        pos = {
            "x": jnp.array([0.0, 1.0, 2.0]),
            "y": jnp.array([0.0, 0.5, 1.0]),
            "z": jnp.array([100.0, 101.0, 102.0]),
        }
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0]),
            "y": jnp.array([0.5, 0.5, 0.5]),
            "z": jnp.array([1.0, 1.0, 1.0]),
        }

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        assert jnp.sum(result.ordered_indices >= 0) == 3
        assert jnp.array_equal(result.ordered_indices, jnp.array([0, 1, 2]))
