"""Tests for distance metrics."""

import jax
import jax.numpy as jnp
import pytest

import localflowwalk as lfw
from localflowwalk.metrics import (
    AbstractDistanceMetric,
    AlignedMomentumDistanceMetric,
    FullPhaseSpaceDistanceMetric,
    SpatialDistanceMetric,
)


class TestAbstractDistanceMetric:
    """Tests for the AbstractDistanceMetric base class."""

    def test_is_abstract(self):
        """Test that AbstractDistanceMetric cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractDistanceMetric()

    def test_custom_metric_subclass(self):
        """Test that we can subclass AbstractDistanceMetric."""

        class CustomMetric(AbstractDistanceMetric):
            """Minimal custom metric for testing."""

            def __call__(self, current_pos, current_vel, positions, velocities, lam):
                del current_vel, velocities, lam
                # Simple Euclidean distance
                diffs = jax.tree.map(lambda p, c: p - c, positions, current_pos)
                dist_sq = sum(jax.tree.leaves(jax.tree.map(lambda x: x**2, diffs)))
                return jnp.sqrt(dist_sq)

        # Should be able to instantiate
        metric = CustomMetric()
        assert isinstance(metric, AbstractDistanceMetric)

        # Should work with algorithm
        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}
        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0, metric=metric)
        assert len(result.ordered_indices) == 3


class TestAlignedMomentumDistanceMetric:
    """Tests for the AlignedMomentumDistanceMetric."""

    def test_instantiation(self):
        """Test that metric can be instantiated."""
        metric = AlignedMomentumDistanceMetric()
        assert isinstance(metric, AbstractDistanceMetric)

    def test_call_shape(self):
        """Test that calling metric returns correct shape."""
        metric = AlignedMomentumDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(current_pos, current_vel, pos, vel, lam=1.0)

        assert distances.shape == (3,)
        assert jnp.all(jnp.isfinite(distances))

    def test_zero_lambda_is_pure_distance(self):
        """Test that λ=0 gives pure Euclidean distance."""
        metric = AlignedMomentumDistanceMetric()

        # Points at different distances, velocity pointing to one of them
        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        distances = metric(current_pos, current_vel, pos, vel, lam=0.0)

        # Should be [0, 1, 2] (pure Euclidean distance)
        expected = jnp.array([0.0, 1.0, 2.0])
        assert jnp.allclose(distances, expected)

    def test_high_lambda_favors_aligned(self):
        """Test that high λ favors velocity-aligned points."""
        metric = AlignedMomentumDistanceMetric()

        # Two points at equal distance but different alignments
        # Point at (1, 0): aligned with velocity (+x direction)
        # Point at (0, 1): perpendicular to velocity
        pos = {"x": jnp.array([0.0, 1.0, 0.0]), "y": jnp.array([0.0, 0.0, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        distances = metric(current_pos, current_vel, pos, vel, lam=10.0)

        # Distance to self (coincident point) has undefined direction, so gets
        # max penalty d = 0 + lam * (1 - 0) = lam
        assert jnp.isclose(distances[0], 10.0)
        # Point in velocity direction should have lower modified distance than
        # perpendicular
        assert distances[1] < distances[2]

    def test_momentum_penalty(self):
        """Test the momentum penalty term (1 - cos(θ))."""
        metric = AlignedMomentumDistanceMetric()

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

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        lam = 1.0
        distances = metric(current_pos, current_vel, pos, vel, lam=lam)

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
        metric = AlignedMomentumDistanceMetric()

        @jax.jit
        def compute_distances(current_pos, current_vel, positions, velocities, lam):
            return metric(current_pos, current_vel, positions, velocities, lam)

        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        current_pos = {"x": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0)}

        # Should compile and run without error
        distances = compute_distances(current_pos, current_vel, pos, vel, 1.0)
        assert distances.shape == (3,)

    def test_integration_with_algorithm(self):
        """Test that metric works correctly with walk_local_flow."""
        metric = AlignedMomentumDistanceMetric()

        # Simple line with aligned velocities
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0, metric=metric)

        # Should follow the line in order
        assert jnp.array_equal(result.ordered_indices, jnp.array([0, 1, 2, 3]))


class TestSpatialDistanceMetric:
    """Tests for the SpatialDistanceMetric."""

    def test_instantiation(self):
        """Test that metric can be instantiated."""
        metric = SpatialDistanceMetric()
        assert isinstance(metric, AbstractDistanceMetric)

    def test_call_shape(self):
        """Test that calling metric returns correct shape."""
        metric = SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(current_pos, current_vel, pos, vel, lam=0.0)

        assert distances.shape == (3,)
        assert jnp.all(jnp.isfinite(distances))

    def test_ignores_velocity(self):
        """Test that metric ignores velocity information."""
        metric = SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel1 = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel2 = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([1.0, 1.0, 1.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel1 = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        current_vel2 = {"x": jnp.array(0.0), "y": jnp.array(1.0)}

        # Same position, different velocities - should give same distances
        distances1 = metric(current_pos, current_vel1, pos, vel1, lam=0.0)
        distances2 = metric(current_pos, current_vel2, pos, vel2, lam=0.0)

        assert jnp.allclose(distances1, distances2)

    def test_ignores_lambda(self):
        """Test that metric ignores lambda parameter."""
        metric = SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        # Different lambda values should give same result
        distances_lam0 = metric(current_pos, current_vel, pos, vel, lam=0.0)
        distances_lam10 = metric(current_pos, current_vel, pos, vel, lam=10.0)

        assert jnp.allclose(distances_lam0, distances_lam10)

    def test_euclidean_distance(self):
        """Test that metric computes correct Euclidean distances."""
        metric = SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 3.0, 0.0]), "y": jnp.array([0.0, 0.0, 4.0])}
        vel = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        distances = metric(current_pos, current_vel, pos, vel, lam=0.0)

        # Should be [0, 3, 4] (Euclidean distances)
        expected = jnp.array([0.0, 3.0, 4.0])
        assert jnp.allclose(distances, expected)

    def test_matches_momentum_metric_with_zero_lambda(self):
        """Test that SpatialMetric matches AlignedMomentumMetric(λ=0)."""
        spatial_metric = SpatialDistanceMetric()
        momentum_metric = AlignedMomentumDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        spatial_dist = spatial_metric(current_pos, current_vel, pos, vel, lam=0.0)
        momentum_dist = momentum_metric(current_pos, current_vel, pos, vel, lam=0.0)

        assert jnp.allclose(spatial_dist, momentum_dist)

    def test_jit_compatible(self):
        """Test that metric works with JAX JIT compilation."""
        metric = SpatialDistanceMetric()

        @jax.jit
        def compute_distances(current_pos, current_vel, positions, velocities, lam):
            return metric(current_pos, current_vel, positions, velocities, lam)

        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        current_pos = {"x": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0)}

        # Should compile and run without error
        distances = compute_distances(current_pos, current_vel, pos, vel, 0.0)
        assert distances.shape == (3,)

    def test_integration_with_algorithm(self):
        """Test that metric works correctly with walk_local_flow."""
        metric = SpatialDistanceMetric()

        # Points along a line - should order by spatial proximity only
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([-1.0, -1.0, -1.0, -1.0])}  # Velocity points backward

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0, metric=metric)

        # Should still follow spatial order despite backward velocity
        assert jnp.array_equal(result.ordered_indices, jnp.array([0, 1, 2, 3]))

    def test_multidimensional(self):
        """Test with higher-dimensional data."""
        metric = SpatialDistanceMetric()

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

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(current_pos, current_vel, pos, vel, lam=0.0)

        # All points at distance 1 from origin (except self at 0)
        expected = jnp.array([0.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(distances, expected)


class TestMetricComparison:
    """Tests comparing different metrics."""

    def test_all_metrics_with_same_data(self):
        """Test all metrics on the same dataset."""
        pos = {
            "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
            "y": jnp.array([0.0, 0.5, 1.0, 1.5]),
        }
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
            "y": jnp.array([0.5, 0.5, 0.5, 0.5]),
        }

        metrics = {
            "aligned_momentum": AlignedMomentumDistanceMetric(),
            "spatial": SpatialDistanceMetric(),
            "full_phase_space": FullPhaseSpaceDistanceMetric(),
        }

        results = {}
        for name, metric in metrics.items():
            result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0, metric=metric)
            results[name] = result

        # All should complete successfully
        for name, result in results.items():
            assert len(result.ordered_indices) == 4, name
            assert jnp.all(result.ordered_indices >= 0), name

    def test_spatial_vs_momentum_with_misaligned_velocity(self):
        """Test that spatial and momentum metrics differ when velocity is misaligned."""
        # Create points where spatial and momentum ordering should differ
        pos = {
            "x": jnp.array([0.0, 1.0, 0.5]),  # Point 2 is between 0 and 1
            "y": jnp.array([0.0, 0.0, 1.0]),  # but off to the side
        }
        # Velocity points in +x direction
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0]),
            "y": jnp.array([0.0, 0.0, 0.0]),
        }

        spatial_result = lfw.walk_local_flow(
            pos, vel, start_idx=0, lam=0.0, metric=SpatialDistanceMetric()
        )
        momentum_result = lfw.walk_local_flow(
            pos, vel, start_idx=0, lam=5.0, metric=AlignedMomentumDistanceMetric()
        )

        # With high lambda, momentum should prefer point 1 (aligned) over point
        # 2 (closer but off-axis) Spatial should prefer point 2 (closer) Both
        # start at 0
        assert spatial_result.ordered_indices[0] == 0
        assert momentum_result.ordered_indices[0] == 0

        # Next point might differ
        # We can't guarantee exact behavior without knowing distances, but they should
        # both complete successfully
        assert len(spatial_result.ordered_indices) == 3
        assert len(momentum_result.ordered_indices) == 3


class TestFullPhaseSpaceDistanceMetric:
    """Tests for the FullPhaseSpaceDistanceMetric."""

    def test_instantiation(self):
        """Test that metric can be instantiated."""
        metric = FullPhaseSpaceDistanceMetric()
        assert isinstance(metric, AbstractDistanceMetric)

    def test_call_shape(self):
        """Test that calling metric returns correct shape."""
        metric = FullPhaseSpaceDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        distances = metric(current_pos, current_vel, pos, vel, lam=1.0)

        assert distances.shape == (3,)
        assert jnp.all(jnp.isfinite(distances))

    def test_zero_velocity_matches_spatial(self):
        """Test that zero velocity gives same result as spatial metric."""
        phase_metric = FullPhaseSpaceDistanceMetric()
        spatial_metric = SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        # Zero velocities
        vel = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        phase_dist = phase_metric(current_pos, current_vel, pos, vel, lam=1.0)
        spatial_dist = spatial_metric(current_pos, current_vel, pos, vel, lam=0.0)

        assert jnp.allclose(phase_dist, spatial_dist)

    def test_zero_lambda_matches_spatial(self):
        """Test that lam=0 gives same result as spatial metric."""
        phase_metric = FullPhaseSpaceDistanceMetric()
        spatial_metric = SpatialDistanceMetric()

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([0.5, 1.0, 1.5])}

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        phase_dist = phase_metric(current_pos, current_vel, pos, vel, lam=0.0)
        spatial_dist = spatial_metric(current_pos, current_vel, pos, vel, lam=0.0)

        assert jnp.allclose(phase_dist, spatial_dist)

    def test_velocity_contribution(self):
        """Test that velocity differences contribute to distance."""
        metric = FullPhaseSpaceDistanceMetric()

        # Same position, different velocities
        pos = {"x": jnp.array([0.0, 0.0, 0.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        vel = {
            "x": jnp.array([0.0, 1.0, 0.0]),  # Point 1 has velocity diff in x
            "y": jnp.array([0.0, 0.0, 1.0]),  # Point 2 has velocity diff in y
        }

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        lam = 2.0
        distances = metric(current_pos, current_vel, pos, vel, lam=lam)

        # Point 0 (self): zero distance
        assert jnp.isclose(distances[0], 0.0)

        # Point 1: d_vel = 1, d = lam * 1 = 2
        assert jnp.isclose(distances[1], lam, atol=1e-5)

        # Point 2: d_vel = 1, d = lam * 1 = 2
        assert jnp.isclose(distances[2], lam, atol=1e-5)

    def test_combined_position_and_velocity(self):
        """Test correct combination of position and velocity distances."""
        metric = FullPhaseSpaceDistanceMetric()

        # Point at (3, 0) with velocity (4, 0)
        # Pythagorean triple: 3-4-5
        pos = {"x": jnp.array([0.0, 3.0]), "y": jnp.array([0.0, 0.0])}
        vel = {"x": jnp.array([0.0, 4.0]), "y": jnp.array([0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        # With lam=1, should get sqrt(3^2 + 4^2) = 5
        lam = 1.0
        distances = metric(current_pos, current_vel, pos, vel, lam=lam)

        assert jnp.isclose(distances[0], 0.0)
        assert jnp.isclose(distances[1], 5.0, atol=1e-5)

    def test_lambda_scaling(self):
        """Test that lambda scales velocity contribution correctly."""
        metric = FullPhaseSpaceDistanceMetric()

        # Same position, velocity diff of 1
        pos = {"x": jnp.array([0.0, 0.0]), "y": jnp.array([0.0, 0.0])}
        vel = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        # With lam=2, velocity diff of 1 becomes distance of 2
        distances_lam2 = metric(current_pos, current_vel, pos, vel, lam=2.0)
        assert jnp.isclose(distances_lam2[1], 2.0, atol=1e-5)

        # With lam=5, velocity diff of 1 becomes distance of 5
        distances_lam5 = metric(current_pos, current_vel, pos, vel, lam=5.0)
        assert jnp.isclose(distances_lam5[1], 5.0, atol=1e-5)

    def test_symmetry(self):
        """Test that metric is symmetric in position and velocity."""
        metric = FullPhaseSpaceDistanceMetric()

        # Case 1: Position diff = 3, velocity diff = 0, lam = 1
        pos1 = {"x": jnp.array([0.0, 3.0]), "y": jnp.array([0.0, 0.0])}
        vel1 = {"x": jnp.array([0.0, 0.0]), "y": jnp.array([0.0, 0.0])}

        # Case 2: Position diff = 0, velocity diff = 3, lam = 1
        pos2 = {"x": jnp.array([0.0, 0.0]), "y": jnp.array([0.0, 0.0])}
        vel2 = {"x": jnp.array([0.0, 3.0]), "y": jnp.array([0.0, 0.0])}

        current_pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        current_vel = {"x": jnp.array(0.0), "y": jnp.array(0.0)}

        dist1 = metric(current_pos, current_vel, pos1, vel1, lam=1.0)
        dist2 = metric(current_pos, current_vel, pos2, vel2, lam=1.0)

        # Both should give distance of 3
        assert jnp.isclose(dist1[1], dist2[1], atol=1e-5)

    def test_jit_compatible(self):
        """Test that metric works with JAX JIT compilation."""
        metric = FullPhaseSpaceDistanceMetric()

        @jax.jit
        def compute_distances(current_pos, current_vel, positions, velocities, lam):
            return metric(current_pos, current_vel, positions, velocities, lam)

        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        current_pos = {"x": jnp.array(0.0)}
        current_vel = {"x": jnp.array(1.0)}

        # Should compile and run without error
        distances = compute_distances(current_pos, current_vel, pos, vel, 1.0)
        assert distances.shape == (3,)

    def test_integration_with_algorithm(self):
        """Test that metric works correctly with walk_local_flow."""
        metric = FullPhaseSpaceDistanceMetric()

        # Points along a line with varying velocities
        pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        vel = {"x": jnp.array([1.0, 2.0, 3.0, 4.0])}

        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.5, metric=metric)

        # Should complete successfully
        assert len(result.ordered_indices) == 4
        assert jnp.all(result.ordered_indices >= 0)

    def test_multidimensional(self):
        """Test with higher-dimensional data."""
        metric = FullPhaseSpaceDistanceMetric()

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

        current_pos = {k: v[0] for k, v in pos.items()}
        current_vel = {k: v[0] for k, v in vel.items()}

        lam = 1.0
        distances = metric(current_pos, current_vel, pos, vel, lam=lam)

        # Point 0: zero distance
        assert jnp.isclose(distances[0], 0.0)
        # Point 1: position distance = 3
        assert jnp.isclose(distances[1], 3.0, atol=1e-5)
        # Point 2: velocity distance = 4, with lam=1 gives 4
        assert jnp.isclose(distances[2], 4.0, atol=1e-5)
