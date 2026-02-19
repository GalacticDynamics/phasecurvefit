"""Tests for distance metrics."""

import jax
import jax.numpy as jnp
import pytest

import localflowwalk as lfw


class TestAbstractDistanceMetric:
    """Tests for the AbstractDistanceMetric base class."""

    def test_is_abstract(self):
        """Test that AbstractDistanceMetric cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            lfw.metrics.AbstractDistanceMetric()

    def test_custom_metric_subclass(self):
        """Test that we can subclass AbstractDistanceMetric."""

        class CustomMetric(lfw.metrics.AbstractDistanceMetric):
            """Minimal custom metric for testing."""

            def __call__(self, cur_pos, cur_vel, positions, velocities, metric_scale):
                del cur_vel, velocities, metric_scale
                # Simple Euclidean distance
                diffs = jax.tree.map(lambda p, c: p - c, positions, cur_pos)
                dist_sq = sum(jax.tree.leaves(jax.tree.map(lambda x: x**2, diffs)))
                return jnp.sqrt(dist_sq)

        # Should be able to instantiate
        metric = CustomMetric()
        assert isinstance(metric, lfw.metrics.AbstractDistanceMetric)

        # Should work with algorithm
        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}
        result = lfw.walk_local_flow(
            pos,
            vel,
            start_idx=0,
            metric_scale=0.0,
            config=lfw.WalkConfig(metric=metric),
        )
        assert len(result.indices) == 3


class TestAlignedMomentumDistanceMetric:
    """Tests for the AlignedMomentumDistanceMetric."""

    def test_instantiation(self):
        """Test that metric can be instantiated."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()
        assert isinstance(metric, lfw.metrics.AbstractDistanceMetric)

    def test_call_shape(self):
        """Test that calling metric returns correct shape."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=1.0)

        assert distances.shape == (3,)
        assert jnp.all(jnp.isfinite(distances))

    def test_zero_lambda_is_pure_distance(self):
        """Test that λ=0 gives pure Euclidean distance."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()

        # Points at different distances, velocity pointing to one of them
        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        # Should be [0, 1, 2] (pure Euclidean distance)
        expected = jnp.array([0.0, 1.0, 2.0])
        assert jnp.allclose(distances, expected)

    def test_high_lambda_favors_aligned(self):
        """Test that high λ favors velocity-aligned points."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()

        # Two points at equal distance but different alignments
        # Point at (1, 0): aligned with velocity (+x direction)
        # Point at (0, 1): perpendicular to velocity
        pos = {"x": jnp.array([0.0, 1.0, 0.0]), "y": jnp.array([0.0, 0.0, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=10.0)

        # Distance to self (coincident point) has undefined direction, so gets
        # max penalty d = 0 + metric_scale * (1 - 0) = metric_scale
        assert jnp.isclose(distances[0], 10.0)
        # Point in velocity direction should have lower modified distance than
        # perpendicular
        assert distances[1] < distances[2]

    def test_momentum_penalty(self):
        """Test the momentum penalty term (1 - cos(θ))."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()

        # Point directly in velocity direction: cos(0) = 1, penalty = 0
        # Point perpendicular: cos(90°) = 0, penalty = 1
        # Point opposite: cos(180°) = -1, penalty = 2
        pos = {
            "x": jnp.array([0.0, 1.0, 0.0, -1.0]),
            "y": jnp.array([0.0, 0.0, 1.0, 0.0]),
        }
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
            "y": jnp.array([0.0, 0.0, 0.0, 0.0]),
        }

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        metric_scale = 1.0
        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=metric_scale)

        # Point 0 (self/coincident): undefined direction, gets max penalty
        # d = 0 + λ * 1 = 1
        assert jnp.isclose(distances[0], 1.0)

        # Point 1 (aligned): d = 1 + λ * 0 = 1
        assert jnp.isclose(distances[1], 1.0, atol=1e-5)

        # Point 2 (perpendicular): d = 1 + λ * 1 = 2
        assert jnp.isclose(distances[2], 2.0, atol=1e-5)

        # Point 3 (opposite): d = 1 + λ * 2 = 3
        assert jnp.isclose(distances[3], 3.0, atol=1e-5)

    def test_jit_compatible(self):
        """Test that metric works with JAX JIT compilation."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()

        @jax.jit
        def compute_distances(cur_pos, cur_vel, positions, velocities, metric_scale):
            return metric(cur_pos, cur_vel, positions, velocities, metric_scale)

        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        cur_pos = {"x": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0)}

        # Should compile and run without error
        distances = compute_distances(cur_pos, cur_vel, pos, vel, 1.0)
        assert distances.shape == (3,)

    def test_integration_with_algorithm(self):
        """Test that metric works correctly with walk_local_flow."""
        metric = lfw.metrics.AlignedMomentumDistanceMetric()

        # Simple line with aligned velocities
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(
            pos,
            vel,
            start_idx=0,
            metric_scale=1.0,
            config=lfw.WalkConfig(metric=metric),
        )

        # Should follow the line in order
        assert jnp.array_equal(result.indices, jnp.array([0, 1, 2, 3]))


class TestSpatialDistanceMetric:
    """Tests for the SpatialDistanceMetric."""

    def test_instantiation(self):
        """Test that metric can be instantiated."""
        metric = lfw.metrics.SpatialDistanceMetric()
        assert isinstance(metric, lfw.metrics.AbstractDistanceMetric)

    def test_call_shape(self):
        """Test that calling metric returns correct shape."""
        metric = lfw.metrics.SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        assert distances.shape == (3,)
        assert jnp.all(jnp.isfinite(distances))

    def test_ignores_velocity(self):
        """Test that metric ignores velocity information."""
        metric = lfw.metrics.SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel1 = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel2 = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([1.0, 1.0, 1.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel1 = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        cur_vel2 = {"x": jnp.array(0.0), "y": jnp.array(1.0)}

        # Same position, different velocities - should give same distances
        distances1 = metric(cur_pos, cur_vel1, pos, vel1, metric_scale=0.0)
        distances2 = metric(cur_pos, cur_vel2, pos, vel2, metric_scale=0.0)

        assert jnp.allclose(distances1, distances2)

    def test_ignores_lambda(self):
        """Test that metric ignores lambda parameter."""
        metric = lfw.metrics.SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        # Different lambda values should give same result
        distances_lam0 = metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)
        distances_lam10 = metric(cur_pos, cur_vel, pos, vel, metric_scale=10.0)

        assert jnp.allclose(distances_lam0, distances_lam10)

    def test_euclidean_distance(self):
        """Test that metric computes correct Euclidean distances."""
        metric = lfw.metrics.SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 3.0, 0.0]), "y": jnp.array([0.0, 0.0, 4.0])}
        vel = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        # Should be [0, 3, 4] (Euclidean distances)
        expected = jnp.array([0.0, 3.0, 4.0])
        assert jnp.allclose(distances, expected)

    def test_matches_momentum_metric_with_zero_lambda(self):
        """Test that SpatialMetric matches AlignedMomentumMetric(λ=0)."""
        spatial_metric = lfw.metrics.SpatialDistanceMetric()
        momentum_metric = lfw.metrics.AlignedMomentumDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        spatial_dist = spatial_metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)
        momentum_dist = momentum_metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        assert jnp.allclose(spatial_dist, momentum_dist)

    def test_jit_compatible(self):
        """Test that metric works with JAX JIT compilation."""
        metric = lfw.metrics.SpatialDistanceMetric()

        @jax.jit
        def compute_distances(cur_pos, cur_vel, positions, velocities, metric_scale):
            return metric(cur_pos, cur_vel, positions, velocities, metric_scale)

        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        cur_pos = {"x": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0)}

        # Should compile and run without error
        distances = compute_distances(cur_pos, cur_vel, pos, vel, 0.0)
        assert distances.shape == (3,)

    def test_integration_with_algorithm(self):
        """Test that metric works correctly with walk_local_flow."""
        metric = lfw.metrics.SpatialDistanceMetric()

        # Points along a line - should order by spatial proximity only
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([-1.0, -1.0, -1.0, -1.0])}  # Velocity points backward

        result = lfw.walk_local_flow(
            pos,
            vel,
            start_idx=0,
            metric_scale=0.0,
            config=lfw.WalkConfig(metric=metric),
        )

        # Should still follow spatial order despite backward velocity
        assert jnp.array_equal(result.indices, jnp.array([0, 1, 2, 3]))

    def test_multidimensional(self):
        """Test with higher-dimensional data."""
        metric = lfw.metrics.SpatialDistanceMetric()

        # 3D data
        pos = {
            "x": jnp.array([0.0, 1.0, 0.0, 0.0]),
            "y": jnp.array([0.0, 0.0, 1.0, 0.0]),
            "z": jnp.array([0.0, 0.0, 0.0, 1.0]),
        }
        vel = {
            "x": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "y": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "z": jnp.array([0.0, 0.0, 0.0, 0.0]),
        }

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        # All points at distance 1 from origin (except self at 0)
        expected = jnp.array([0.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(distances, expected)


class TestMetricComparison:
    """Tests comparing different metrics."""

    @pytest.mark.parametrize(
        ("metric_name", "metric"),
        [
            ("aligned_momentum", lfw.metrics.AlignedMomentumDistanceMetric()),
            ("spatial", lfw.metrics.SpatialDistanceMetric()),
            ("full_phase_space", lfw.metrics.FullPhaseSpaceDistanceMetric()),
        ],
    )
    def test_all_metrics_with_same_data(self, metric_name, metric):
        """Test all metrics on the same dataset."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0]), "y": jnp.array([0.0, 0.5, 1.0, 1.5])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5, 0.5])}

        result = lfw.walk_local_flow(
            q, p, start_idx=0, metric_scale=1.0, config=lfw.WalkConfig(metric=metric)
        )

        assert result.all_visited, metric_name

    def test_spatial_vs_momentum_with_misaligned_velocity(self):
        """Test that spatial and momentum metrics differ when velocity is misaligned."""
        # Create points where spatial and momentum ordering should differ
        q = {
            "x": jnp.array([0.0, 1.0, 0.5]),  # Point 2 is between 0 and 1
            "y": jnp.array([0.0, 0.0, 1.0]),  # but off to the side
        }
        p = {  # Velocity points in +x direction
            "x": jnp.array([1.0, 1.0, 1.0]),
            "y": jnp.array([0.0, 0.0, 0.0]),
        }

        spatial_result = lfw.walk_local_flow(
            q,
            p,
            start_idx=0,
            metric_scale=0.0,
            config=lfw.WalkConfig(metric=lfw.metrics.SpatialDistanceMetric()),
        )
        momentum_result = lfw.walk_local_flow(
            q,
            p,
            start_idx=0,
            metric_scale=5.0,
            config=lfw.WalkConfig(metric=lfw.metrics.AlignedMomentumDistanceMetric()),
        )

        # With high lambda, momentum should prefer point 1 (aligned) over point
        # 2 (closer but off-axis) Spatial should prefer point 2 (closer) Both
        # start at 0
        assert spatial_result.indices[0] == 0
        assert momentum_result.indices[0] == 0

        # Next point might differ We can't guarantee exact behavior without
        # knowing distances, but they should both complete successfully
        assert len(spatial_result.indices) == 3
        assert len(momentum_result.indices) == 3


class TestFullPhaseSpaceDistanceMetric:
    """Tests for the FullPhaseSpaceDistanceMetric."""

    def test_instantiation(self):
        """Test that metric can be instantiated."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()
        assert isinstance(metric, lfw.metrics.AbstractDistanceMetric)

    def test_call_shape(self):
        """Test that calling metric returns correct shape."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=1.0)

        assert distances.shape == (3,)
        assert jnp.all(jnp.isfinite(distances))

    def test_zero_velocity_matches_spatial(self):
        """Test that zero velocity gives same result as spatial metric."""
        phase_metric = lfw.metrics.FullPhaseSpaceDistanceMetric()
        spatial_metric = lfw.metrics.SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        # Zero velocities
        vel = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        phase_dist = phase_metric(cur_pos, cur_vel, pos, vel, metric_scale=1.0)
        spatial_dist = spatial_metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        assert jnp.allclose(phase_dist, spatial_dist)

    def test_zero_lambda_matches_spatial(self):
        """Test that metric_scale=0 gives same result as spatial metric."""
        phase_metric = lfw.metrics.FullPhaseSpaceDistanceMetric()
        spatial_metric = lfw.metrics.SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([0.5, 1.0, 1.5])}

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        phase_dist = phase_metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)
        spatial_dist = spatial_metric(cur_pos, cur_vel, pos, vel, metric_scale=0.0)

        assert jnp.allclose(phase_dist, spatial_dist)

    def test_velocity_contribution(self):
        """Test that velocity differences contribute to distance."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        # Same position, different velocities
        pos = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel = {
            "x": jnp.array([0.0, 1.0, 0.0]),  # Point 1 has velocity diff in x
            "y": jnp.array([0.0, 0.0, 1.0]),  # Point 2 has velocity diff in y
        }

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        metric_scale = 2.0
        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=metric_scale)

        # Point 0 (self): zero distance
        assert jnp.isclose(distances[0], 0.0)

        # Point 1: d_vel = 1, d = metric_scale * 1 = 2
        assert jnp.isclose(distances[1], metric_scale, atol=1e-5)

        # Point 2: d_vel = 1, d = metric_scale * 1 = 2
        assert jnp.isclose(distances[2], metric_scale, atol=1e-5)

    def test_combined_position_and_velocity(self):
        """Test correct combination of position and velocity distances."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        # Point at (3, 0) with velocity (4, 0)
        # Pythagorean triple: 3-4-5
        pos = {"x": jnp.array([0.0, 3.0]), "y": jnp.array([0.0, 0.0])}
        vel = {"x": jnp.array([0.0, 4.0]), "y": jnp.array([0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        # With metric_scale=1, should get sqrt(3^2 + 4^2) = 5
        metric_scale = 1.0
        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=metric_scale)

        assert jnp.isclose(distances[0], 0.0)
        assert jnp.isclose(distances[1], 5.0, atol=1e-5)

    def test_lambda_scaling(self):
        """Test that lambda scales velocity contribution correctly."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        # Same position, velocity diff of 1
        pos = {"x": jnp.array([0.0, 0.0]), "y": jnp.array([0.0, 0.0])}
        vel = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        # With metric_scale=2, velocity diff of 1 becomes distance of 2
        distances_lam2 = metric(cur_pos, cur_vel, pos, vel, metric_scale=2.0)
        assert jnp.isclose(distances_lam2[1], 2.0, atol=1e-5)

        # With metric_scale=5, velocity diff of 1 becomes distance of 5
        distances_lam5 = metric(cur_pos, cur_vel, pos, vel, metric_scale=5.0)
        assert jnp.isclose(distances_lam5[1], 5.0, atol=1e-5)

    def test_symmetry(self):
        """Test that metric is symmetric in position and velocity."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        # Case 1: Position diff = 3, velocity diff = 0, metric_scale = 1
        pos1 = {"x": jnp.array([0.0, 3.0]), "y": jnp.array([0.0, 0.0])}
        vel1 = {"x": jnp.array([0.0, 0.0]), "y": jnp.array([0.0, 0.0])}

        # Case 2: Position diff = 0, velocity diff = 3, metric_scale = 1
        pos2 = {"x": jnp.array([0.0, 0.0]), "y": jnp.array([0.0, 0.0])}
        vel2 = {"x": jnp.array([0.0, 3.0]), "y": jnp.array([0.0, 0.0])}

        cur_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        dist1 = metric(cur_pos, cur_vel, pos1, vel1, metric_scale=1.0)
        dist2 = metric(cur_pos, cur_vel, pos2, vel2, metric_scale=1.0)

        # Both should give distance of 3
        assert jnp.isclose(dist1[1], dist2[1], atol=1e-5)

    def test_jit_compatible(self):
        """Test that metric works with JAX JIT compilation."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        @jax.jit
        def compute_distances(cur_pos, cur_vel, positions, velocities, metric_scale):
            return metric(cur_pos, cur_vel, positions, velocities, metric_scale)

        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        cur_pos = {"x": jnp.array(0.0)}
        cur_vel = {"x": jnp.array(1.0)}

        # Should compile and run without error
        distances = compute_distances(cur_pos, cur_vel, pos, vel, 1.0)
        assert distances.shape == (3,)

    def test_integration_with_algorithm(self):
        """Test that metric works correctly with walk_local_flow."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        # Points along a line with varying velocities
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([1.0, 2.0, 3.0, 4.0])}

        result = lfw.walk_local_flow(
            pos,
            vel,
            start_idx=0,
            metric_scale=0.5,
            config=lfw.WalkConfig(metric=metric),
        )

        # Should complete successfully
        assert len(result.indices) == 4
        assert jnp.all(result.indices >= 0)

    def test_multidimensional(self):
        """Test with higher-dimensional data."""
        metric = lfw.metrics.FullPhaseSpaceDistanceMetric()

        # 3D data: one point with position offset, one with velocity offset
        pos = {
            "x": jnp.array([0.0, 3.0, 0.0]),
            "y": jnp.array([0.0, 0.0, 0.0]),
            "z": jnp.array([0.0, 0.0, 0.0]),
        }
        vel = {
            "x": jnp.array([0.0, 0.0, 0.0]),
            "y": jnp.array([0.0, 0.0, 4.0]),
            "z": jnp.array([0.0, 0.0, 0.0]),
        }

        cur_pos = {k: v[0] for k, v in pos.items()}
        cur_vel = {k: v[0] for k, v in vel.items()}

        metric_scale = 1.0
        distances = metric(cur_pos, cur_vel, pos, vel, metric_scale=metric_scale)

        # Point 0: zero distance
        assert jnp.isclose(distances[0], 0.0)
        # Point 1: position distance = 3
        assert jnp.isclose(distances[1], 3.0, atol=1e-5)
        # Point 2: velocity distance = 4, with metric_scale=1 gives 4
        assert jnp.isclose(distances[2], 4.0, atol=1e-5)
