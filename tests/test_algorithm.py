"""Tests for the phase flow walking algorithm."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import localflowwalk as lfw


class TestLocalFlowWalkResult:
    """Tests for LocalFlowWalkResult and accessors."""

    def test_result_structure(self):
        """Test that result has correct structure."""
        q = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([1.0, 2.0, 3.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=0.0)

        # Check that result has the expected attributes (NamedTuple)
        assert hasattr(result, "indices")
        assert hasattr(result, "skipped_indices")
        assert hasattr(result, "positions")
        assert hasattr(result, "velocities")
        # indices is now an array, not a tuple
        assert isinstance(result.indices, jnp.ndarray)
        # skipped_indices is computed via property
        assert isinstance(result.skipped_indices, jnp.ndarray)

    def test_order_w(self):
        """Test accessing ordered positions and velocities."""
        q = {"x": jnp.array([3.0, 1.0, 2.0]), "y": jnp.array([30.0, 10.0, 20.0])}
        p = {"x": jnp.array([0.3, 0.1, 0.2]), "y": jnp.array([3.0, 1.0, 2.0])}

        result = lfw.walk_local_flow(q, p, start_idx=1, metric_scale=0.0)
        ordered_pos, ordered_vel = lfw.order_w(result)

        # Order starts from index 1, then finds nearest neighbors
        assert ordered_pos["x"].shape == (3,)
        assert ordered_vel["x"].shape == (3,)


class TestNearestNeighborsWithMomentum:
    """Tests for the main algorithm."""

    def test_simple_line(self):
        """Test with points on a simple line with aligned velocities."""
        # Points along x-axis with positive x velocity
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=1.0)

        # Should follow the line in order
        assert jnp.array_equal(result.indices, jnp.array([0, 1, 2, 3, 4]))

    def test_simple_line_backward(self):
        """Test with points on a line, velocity pointing the other way."""
        # Points along x-axis with negative x velocity (pointing backward)
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])}

        # Start from middle, should go backward due to velocity
        result = lfw.walk_local_flow(q, p, start_idx=2, metric_scale=1.0)
        # With momentum, should prefer going in velocity direction (backward)
        # First step from 2: velocity points to -x, so prefer 1 over 3
        assert result.indices[1] == 1

    def test_backward_parameter(self):
        """Test the backward parameter negates velocity direction."""
        # Points along x-axis with positive x velocity
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Forward walk from leftmost point should go toward higher indices
        result_forward = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=1.0)
        # First step should be to index 1
        assert result_forward.indices[1] == 1

        # Reverse walk from rightmost point should go toward lower indices
        result_backward = lfw.walk_local_flow(
            q, p, start_idx=4, metric_scale=1.0, direction="backward"
        )
        # First step should be to index 3
        assert result_backward.indices[1] == 3

    def test_backward_equivalence(self):
        """Test that backward=True is equivalent to negating velocities."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Using backward parameter
        result_backward_param = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="backward"
        )

        # Manually negating velocities
        p_neg = {"x": -p["x"]}
        result_vel_negated = lfw.walk_local_flow(
            q, p_neg, start_idx=2, metric_scale=1.0
        )

        # Both should produce the same result
        assert jnp.array_equal(
            result_backward_param.indices, result_vel_negated.indices
        )

    def test_direction_both_basic(self):
        """Test direction='both' walks in both directions from start."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Start from middle point
        result = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="both"
        )

        assert result.all_visited
        assert set(result.indices.tolist()) == {0, 1, 2, 3, 4}

    def test_direction_both_visits_all(self):
        """Test that direction='both' visits all reachable points."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}

        # Start from near one end
        result = lfw.walk_local_flow(
            q, p, start_idx=1, metric_scale=1.0, direction="both"
        )

        # Should visit all points
        assert result.all_visited

    def test_direction_both_with_max_dist(self):
        """Test direction='both' respects max_dist in both directions."""
        # Points with gaps on both sides of start
        q = {"x": jnp.array([0.0, 1.0, 2.0, 10.0, 11.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Start from middle of left cluster
        result = lfw.walk_local_flow(
            q, p, start_idx=1, metric_scale=1.0, max_dist=2.0, direction="both"
        )

        # Should visit points in left cluster but not right cluster
        visited_set = set(result.ordering.tolist())
        assert 0 in visited_set
        assert 1 in visited_set
        assert 2 in visited_set
        # Should not reach the far cluster
        assert 3 not in visited_set
        assert 4 not in visited_set
        # Should have skipped indices
        assert 3 in result.skipped_indices
        assert 4 in result.skipped_indices

    def test_direction_both_2d_stream(self):
        """Test direction='both' on a 2D stream."""
        # Points along a diagonal
        q = {
            "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "y": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        }
        p = {
            "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "y": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        }

        result = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="both"
        )

        assert result.all_visited  # Should visit all points

    def test_direction_both_equivalence_to_combine(self):
        """Test that direction='both' gives similar results to manual combine."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}
        # Using direction='both'
        result_both = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="both"
        )

        # Manual combine
        result_fwd = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="forward"
        )
        result_bwd = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="backward"
        )
        result_combined = lfw.combine_flow_walks(result_fwd, result_bwd)

        # Both should visit the same points
        visited_both = set(result_both.ordering.tolist())
        visited_combined = set(result_combined.ordering.tolist())
        assert visited_both == visited_combined

    def test_direction_both_ordering(self):
        """Test that direction='both' orders points sensibly."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="both"
        )

        # Should contain all indices exactly once
        assert result.all_visited
        assert len(set(result.indices.tolist())) == 5

    def test_direction_both_endpoint_start(self):
        """Test direction='both' when starting from endpoint."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Start from leftmost point
        result_left = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=1.0, direction="both"
        )
        assert result_left.all_visited  # Should still visit all points

        # Start from rightmost point
        result_right = lfw.walk_local_flow(
            q, p, start_idx=4, metric_scale=1.0, direction="both"
        )
        assert result_right.all_visited  # Should still visit all points

    def test_2d_stream(self):
        """Test with a 2D stream of points."""
        # Points along a diagonal with matching velocities
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0]), "y": jnp.array([0.0, 1.0, 2.0, 3.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0]), "y": jnp.array([1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=1.0)

        assert jnp.array_equal(result.indices, jnp.array([0, 1, 2, 3]))

    def test_lambda_zero_is_pure_nearest_neighbor(self):
        """Test that λ=0 gives pure nearest neighbor (no momentum)."""
        # Two points equidistant, but one in velocity direction
        q = {"x": jnp.array([0.0, 1.0, -1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        p = {"x": jnp.array([1.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=0.0)

        # With λ=0, it's pure distance, both are at distance 1
        # The algorithm will pick the first one found
        assert result.all_visited
        assert result.indices[0] == 0

    def test_high_lambda_prefers_momentum_direction(self):
        """Test that high λ strongly prefers the momentum direction."""
        # Point at origin with velocity in +x direction
        # One point at (1, 0) - aligned with velocity
        # One point at (-1, 0) - opposite to velocity
        q = {"x": jnp.array([0.0, 1.0, -1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        p = {"x": jnp.array([1.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=10.0)

        # With high λ, should strongly prefer point 1 (in velocity direction)
        assert result.indices[1] == 1

    def test_max_dist_termination(self):
        """Test that max_dist terminates the algorithm."""
        # Points with a gap
        q = {"x": jnp.array([0.0, 1.0, 10.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=0.0, max_dist=5.0)

        # Should stop before reaching point 2 (at x=10)
        assert not result.all_visited
        assert result.n_visited == 2
        assert not jnp.any(result.indices == 2)

    def test_max_dist_skipped_indices(self):
        """Test that max_dist populates skipped_indices correctly."""
        # Points with a gap - point 2 is far away
        q = {"x": jnp.array([0.0, 1.0, 10.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=0.0, max_dist=5.0)

        # Point 2 should be skipped
        assert 2 in result.skipped_indices
        assert len(result.skipped_indices) == 1
        # All indices should be accounted for
        all_indices = set(result.ordering.tolist()) | set(
            result.skipped_indices.tolist()
        )
        assert all_indices == {0, 1, 2}

    def test_no_skipped_when_all_visited(self):
        """Test that skipped_indices is empty when all points visited."""
        q = {"x": jnp.array([0.0, 1.0, 2.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=0.0)

        assert result.all_visited

    def test_n_max_limits_iterations(self):
        """Test that n_max limits the number of points."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=0.0, n_max=3)

        assert not result.all_visited
        assert result.n_visited == 3

    def test_terminate_indices(self):
        """Test that terminate_indices stops at specified points."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=0.0, terminate_indices={2}
        )

        # Should visit indices 0, 1, 2 and then stop
        valid_indices = result.ordering
        assert len(valid_indices) == 3
        assert jnp.array_equal(valid_indices, jnp.array([0, 1, 2]))
        # Indices 3 and 4 should not be visited
        assert 3 not in valid_indices
        assert 4 not in valid_indices

    def test_invalid_start_idx_raises(self):
        """Test that invalid start_idx raises ValueError."""
        q = {"x": jnp.array([0.0, 1.0, 2.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0])}

        with pytest.raises(ValueError, match="out of bounds"):
            lfw.walk_local_flow(q, p, start_idx=10)

        with pytest.raises(ValueError, match="out of bounds"):
            lfw.walk_local_flow(q, p, start_idx=-1)

    def test_single_point(self):
        """Test with a single point."""
        q = {"x": jnp.array([1.0])}
        p = {"x": jnp.array([1.0])}

        result = lfw.walk_local_flow(q, p, start_idx=0)

        assert jnp.array_equal(result.indices, jnp.array([0]))

    def test_3d_helix(self):
        """Test with points along a 3D helix-like structure."""
        t = jnp.linspace(0, 4 * jnp.pi, 20)
        q = {"x": jnp.cos(t), "y": jnp.sin(t), "z": t / (4 * jnp.pi)}
        # Tangent velocity
        p = {"x": -jnp.sin(t), "y": jnp.cos(t), "z": jnp.ones_like(t) / (4 * jnp.pi)}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=1.0)

        # Should roughly follow the helix order
        # Check that we visit all points
        assert result.all_visited
        assert set(result.ordering.tolist()) == set(range(20))


class TestAlgorithmIntegration:
    """Integration tests for the algorithm with realistic scenarios."""

    def test_curved_stream(self):
        """Test algorithm on a curved stream with varying velocity."""
        # Create a curved stream that bends
        n_points = 10
        t = jnp.linspace(0, jnp.pi / 2, n_points)

        # Arc of a circle
        q = {"x": jnp.cos(t), "y": jnp.sin(t)}
        # Tangent velocity (derivative of position)
        p = {"x": -jnp.sin(t), "y": jnp.cos(t)}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=2.0)

        # With momentum, should follow the arc correctly
        # Check that consecutive points are neighbors in the result
        for i in range(len(result.indices) - 1):
            curr = result.indices[i]
            next_ = result.indices[i + 1]
            # Adjacent points in original should be within 2 indices of each other
            assert abs(next_ - curr) <= 3

    def test_noisy_stream(self):
        """Test algorithm on a stream with some noise."""
        key = jax.random.key(42)

        # Base stream along x-axis
        n_points = 20
        base_x = jnp.linspace(0, 10, n_points)
        base_y = jnp.zeros(n_points)

        # Add small noise
        key1, key2 = jax.random.split(key)
        noise_x = jax.random.normal(key1, (n_points,)) * 0.1
        noise_y = jax.random.normal(key2, (n_points,)) * 0.1

        q = {"x": base_x + noise_x, "y": base_y + noise_y}
        p = {"x": jnp.ones(n_points), "y": jnp.zeros(n_points)}

        result = lfw.walk_local_flow(q, p, start_idx=0, metric_scale=1.0)

        # Should visit all points
        assert result.all_visited
        assert set(result.ordering.tolist()) == set(range(n_points))

        # The ordering should roughly follow the x-coordinate
        ordered_pos, _ = lfw.order_w(result)
        ordered_x = ordered_pos["x"]
        # Check that x is generally increasing (allow some local variation)
        avg_increase = jnp.mean(jnp.diff(ordered_x))
        assert avg_increase > 0  # Overall trend should be increasing

    def test_bidirectional_stream(self):
        """Test that algorithm can trace stream in either direction."""
        # Test that velocity direction affects which neighbors are preferred
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel_forward = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}
        vel_backward = {"x": jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])}

        # Start from beginning with forward velocity should find more points
        # forward
        result_forward = lfw.walk_local_flow(
            pos, vel_forward, start_idx=0, metric_scale=5.0
        )
        # Start from beginning with backward velocity should still work (but
        # prefer backward)
        result_backward = lfw.walk_local_flow(
            pos, vel_backward, start_idx=0, metric_scale=5.0
        )

        # Both should visit all points eventually
        assert result_forward.all_visited
        assert result_backward.all_visited

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

        result = lfw.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

        assert result.all_visited
        assert jnp.array_equal(result.indices, jnp.array([0, 1, 2]))


class TestCombineFlowWalks:
    """Tests for the combine_flow_walks function."""

    def test_combine_simple_line(self):
        """Test combining forward and backward walks on a simple line."""
        # Points along x-axis with positive x velocity
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Start from middle, run both walks
        result_fwd = lfw.walk_local_flow(
            pos, vel, start_idx=2, metric_scale=1.0, direction="forward"
        )
        result_bwd = lfw.walk_local_flow(
            pos, vel, start_idx=2, metric_scale=1.0, direction="backward"
        )
        result = lfw.combine_flow_walks(result_fwd, result_bwd)

        # Should visit all 5 points
        assert result.all_visited

        # Start index should be in the result
        assert jnp.any(result.indices == 2)

    def test_combine_starts_in_middle(self):
        """Test that combined walk starts from a middle point."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Start from index 2 (middle)
        result_fwd = lfw.walk_local_flow(
            pos, vel, start_idx=2, metric_scale=1.0, direction="forward"
        )
        result_bwd = lfw.walk_local_flow(
            pos, vel, start_idx=2, metric_scale=1.0, direction="backward"
        )
        result = lfw.combine_flow_walks(result_fwd, result_bwd)

        # The result should have the start_idx in it
        assert jnp.any(result.ordering == 2)

    def test_combine_result_structure(self):
        """Test that combine result has correct structure."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}

        result_fwd = lfw.walk_local_flow(
            pos, vel, start_idx=1, metric_scale=0.5, direction="forward"
        )
        result_bwd = lfw.walk_local_flow(
            pos, vel, start_idx=1, metric_scale=0.5, direction="backward"
        )
        result = lfw.combine_flow_walks(result_fwd, result_bwd)

        # Should be a LocalFlowWalkResult
        assert hasattr(result, "indices")
        assert hasattr(result, "positions")
        assert hasattr(result, "velocities")
        assert isinstance(result.indices, jnp.ndarray)

    def test_combine_preserves_data(self):
        """Test that combine preserves original positions and velocities."""
        q = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.1, 0.1, 0.1])}

        res_fwd = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=1.0, direction="forward"
        )
        res_bwd = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=1.0, direction="backward"
        )
        res = lfw.combine_flow_walks(res_fwd, res_bwd)

        # Original data should be preserved
        assert jnp.array_equal(res.positions["x"], q["x"])
        assert jnp.array_equal(res.positions["y"], q["y"])
        assert jnp.array_equal(res.velocities["x"], p["x"])
        assert jnp.array_equal(res.velocities["y"], p["y"])

    def test_combine_can_extract_ordered_data(self):
        """Test that ordered data can be extracted from combined result."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}

        res_fwd = lfw.walk_local_flow(
            q, p, start_idx=1, metric_scale=0.5, direction="forward"
        )
        res_bwd = lfw.walk_local_flow(
            q, p, start_idx=1, metric_scale=0.5, direction="backward"
        )
        res = lfw.combine_flow_walks(res_fwd, res_bwd)

        # Should be able to extract ordered data
        ordered_q, ordered_p = lfw.order_w(res)

        # Should have arrays
        assert isinstance(ordered_q["x"], jnp.ndarray)
        assert isinstance(ordered_p["x"], jnp.ndarray)
        assert len(ordered_q["x"]) == len(ordered_p["x"])

    def test_combine_2d_stream(self):
        """Test combining walks on a 2D stream."""
        # Points along a diagonal
        q = {
            "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "y": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        }
        p = {
            "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "y": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        }

        res_fwd = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="forward"
        )
        res_bwd = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="backward"
        )
        res = lfw.combine_flow_walks(res_fwd, res_bwd)

        # Should visit all points
        assert res.all_visited

    def test_combine_different_start_indices(self):
        """Test combining from different starting indices."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # Combine from index 0
        res_fwd_0 = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=1.0, direction="forward"
        )
        res_bwd_0 = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=1.0, direction="backward"
        )
        res_0 = lfw.combine_flow_walks(res_fwd_0, res_bwd_0)

        # Combine from index 4
        res_fwd_4 = lfw.walk_local_flow(
            q, p, start_idx=4, metric_scale=1.0, direction="forward"
        )
        res_bwd_4 = lfw.walk_local_flow(
            q, p, start_idx=4, metric_scale=1.0, direction="backward"
        )
        res_4 = lfw.combine_flow_walks(res_fwd_4, res_bwd_4)

        # Both should visit all points (or similar counts)
        assert res_0.n_visited >= 3  # At least start and forward
        assert res_4.n_visited >= 3  # At least start and backward

    def test_combine_with_lam_parameter(self):
        """Test that metric_scale parameter affects combining."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # With metric_scale=0 (pure nearest neighbor)
        res_fwd_lam0 = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=0.0, direction="forward"
        )
        res_bwd_lam0 = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=0.0, direction="backward"
        )
        res_lam0 = lfw.combine_flow_walks(res_fwd_lam0, res_bwd_lam0)

        # With metric_scale=1.0 (balanced)
        res_fwd_lam1 = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="forward"
        )
        res_bwd_lam1 = lfw.walk_local_flow(
            q, p, start_idx=2, metric_scale=1.0, direction="backward"
        )
        res_lam1 = lfw.combine_flow_walks(res_fwd_lam1, res_bwd_lam1)

        # Both should be valid results
        assert res_lam0.n_visited >= 3
        assert res_lam1.n_visited >= 3

    def test_combine_with_max_dist(self):
        """Test that max_dist parameter works with combining."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 10.0, 11.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}

        # With max_dist=2, should not cross the gap
        res_fwd = lfw.walk_local_flow(
            q, p, start_idx=1, metric_scale=1.0, max_dist=2.0, direction="forward"
        )
        res_bwd = lfw.walk_local_flow(
            q, p, start_idx=1, metric_scale=1.0, max_dist=2.0, direction="backward"
        )
        res = lfw.combine_flow_walks(res_fwd, res_bwd)

        # Should have some skipped indices (can't cross the gap)
        assert res.n_skipped > 0

    def test_combine_validates_matching_data(self):
        """Test that combine raises error when positions/velocities don't match."""
        q1 = {"x": jnp.array([0.0, 1.0, 2.0])}
        p1 = {"x": jnp.array([1.0, 1.0, 1.0])}

        q2 = {"x": jnp.array([0.0, 1.0, 3.0])}  # Different data
        p2 = {"x": jnp.array([1.0, 1.0, 1.0])}

        # Create results with different positions
        res1 = lfw.walk_local_flow(
            q1, p1, start_idx=0, metric_scale=1.0, direction="forward"
        )
        res2 = lfw.walk_local_flow(
            q2, p2, start_idx=0, metric_scale=1.0, direction="backward"
        )

        # Should raise an error when combining
        with pytest.raises((eqx.EquinoxRuntimeError, ValueError)):
            lfw.combine_flow_walks(res1, res2)
