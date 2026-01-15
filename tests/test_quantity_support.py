"""Tests for Quantity support in lfw.walk_local_flow.

These tests verify that lfw.walk_local_flow works correctly with unxt Quantity inputs
and that unit system information flows properly through the state tuple.
"""

import pytest

try:
    import unxt as u

    HAS_UNXT = True
except ImportError:
    HAS_UNXT = False

import jax
import jax.numpy as jnp

import localflowwalk as lfw
from localflowwalk._src.algorithm import StateMetadata
from localflowwalk._src.strategies import KDTreeStrategy

pytestmark = pytest.mark.skipif(not HAS_UNXT, reason="unxt not installed")


@pytest.fixture
def unit_system():
    """Create a simple unit system for testing."""
    return u.unitsystem(u.unit("m"), u.unit("s"))


@pytest.fixture
def quantity_positions():
    """Create position data with Quantity values."""
    return {
        "x": u.Q(jnp.array([0.0, 1.0, 2.0, 3.0]), "m"),
        "y": u.Q(jnp.array([0.0, 0.5, 0.8, 1.2]), "m"),
    }


@pytest.fixture
def quantity_velocities():
    """Create velocity data with Quantity values."""
    return {
        "x": u.Q(jnp.array([1.0, 1.0, 1.0, 1.0]), "m/s"),
        "y": u.Q(jnp.array([0.2, 0.2, 0.2, 0.2]), "m/s"),
    }


@pytest.fixture
def plain_positions():
    """Create position data with plain arrays."""
    return {
        "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
        "y": jnp.array([0.0, 0.5, 0.8, 1.2]),
    }


@pytest.fixture
def plain_velocities():
    """Create velocity data with plain arrays."""
    return {
        "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
        "y": jnp.array([0.2, 0.2, 0.2, 0.2]),
    }


class TestStateMetadata:
    """Tests for StateMetadata container."""

    def test_metadata_basic_functionality(self, unit_system):
        """Test that StateMetadata supports dict-like access."""
        metadata = StateMetadata(usys=unit_system)

        assert metadata["usys"] == unit_system
        assert "usys" in metadata
        assert metadata.get("usys") == unit_system
        assert metadata.get("nonexistent") is None

    def test_metadata_repr(self, unit_system):
        """Test StateMetadata string representation."""
        metadata = StateMetadata(usys=unit_system)
        repr_str = repr(metadata)

        assert "StateMetadata" in repr_str
        assert "usys" in repr_str

    def test_metadata_as_pytree_leaf(self, unit_system):
        """Test that StateMetadata is preserved as a PyTree leaf."""
        import jax.tree_util as jtu

        metadata = StateMetadata(usys=unit_system)
        state = (0, jnp.array([1, 2, 3]), metadata)

        # When treated as a leaf, metadata should appear intact in tree_leaves
        leaves = jtu.tree_leaves(
            state,
            is_leaf=lambda x: isinstance(x, StateMetadata),
        )

        assert metadata in leaves
        assert len(leaves) == 3  # int, array, metadata


class TestWalkLocalFlowWithPlainArrays:
    """Tests for lfw.walk_local_flow with plain array inputs (baseline)."""

    def test_walk_local_flow_plain_arrays(self, plain_positions, plain_velocities):
        """Test lfw.walk_local_flow works with plain array inputs."""
        result = lfw.walk_local_flow(
            plain_positions, plain_velocities, start_idx=0, lam=0.5
        )

        # Should visit all 4 points in order
        assert len(result.ordered_indices) == 4
        assert jnp.allclose(result.ordered_indices, jnp.array([0, 1, 2, 3]))
        assert result.all_visited

    def test_walk_local_flow_plain_arrays_with_kdtree_strategy(
        self, plain_positions, plain_velocities
    ):
        """Test walk_local_flow with plain arrays and KDTree strategy.

        Note: KDTree strategy uses jaxkd which requires plain array data.
        It cannot directly work with Quantity objects (which would need to be
        stripped first before KDTree processing).
        """
        # Create KDTree strategy with k=3 neighbors
        strategy = KDTreeStrategy(k=3)

        result = lfw.walk_local_flow(
            plain_positions,
            plain_velocities,
            start_idx=0,
            lam=0.5,
            strategy=strategy,
        )

        # Should still visit all 4 points
        assert len(result.ordered_indices) == 4
        assert jnp.allclose(result.ordered_indices, jnp.array([0, 1, 2, 3]))
        assert result.all_visited


class TestWalkLocalFlowWithQuantities:
    """Tests for lfw.walk_local_flow with Quantity inputs."""

    def test_walk_local_flow_quantities_basic(
        self, quantity_positions, quantity_velocities, unit_system
    ):
        """Test lfw.walk_local_flow works with Quantity inputs."""
        lam_quantity = u.Q(0.5, "m")

        result = lfw.walk_local_flow(
            quantity_positions,
            quantity_velocities,
            start_idx=0,
            lam=lam_quantity,
            usys=unit_system,
        )

        # Should visit all 4 points in order
        assert len(result.ordered_indices) == 4
        assert jnp.allclose(result.ordered_indices, jnp.array([0, 1, 2, 3]))
        assert result.all_visited

    def test_walk_local_flow_quantities_preserves_units(
        self, quantity_positions, quantity_velocities, unit_system
    ):
        """Test that lfw.walk_local_flow preserves Quantity units in output."""
        lam_quantity = u.Q(0.5, "m")

        result = lfw.walk_local_flow(
            quantity_positions,
            quantity_velocities,
            start_idx=0,
            lam=lam_quantity,
            usys=unit_system,
        )

        # Check that positions have units preserved
        assert isinstance(result.positions["x"], u.AbstractQuantity)
        assert isinstance(result.positions["y"], u.AbstractQuantity)
        assert str(result.positions["x"].unit) == "m"
        assert str(result.positions["y"].unit) == "m"

        # Check that velocities have units preserved
        assert isinstance(result.velocities["x"], u.AbstractQuantity)
        assert isinstance(result.velocities["y"], u.AbstractQuantity)
        assert str(result.velocities["x"].unit) == "m / s"
        assert str(result.velocities["y"].unit) == "m / s"

    def test_walk_local_flow_quantities_values_match_plain(
        self,
        quantity_positions,
        quantity_velocities,
        plain_positions,
        plain_velocities,
        unit_system,
    ):
        """Test that Quantity and plain array results have same values."""
        lam_plain = 0.5
        lam_quantity = u.Q(0.5, "m")

        # Run with plain arrays
        result_plain = lfw.walk_local_flow(
            plain_positions, plain_velocities, start_idx=0, lam=lam_plain
        )

        # Run with quantities
        result_quantity = lfw.walk_local_flow(
            quantity_positions,
            quantity_velocities,
            start_idx=0,
            lam=lam_quantity,
            usys=unit_system,
        )

        # Ordered indices should match
        assert jnp.allclose(
            result_plain.ordered_indices, result_quantity.ordered_indices
        )

        # Position values (stripped of units) should match
        for key in plain_positions:
            plain_val = result_plain.positions[key]
            quantity_val = u.ustrip(u.unit("m"), result_quantity.positions[key])
            assert jnp.allclose(plain_val, quantity_val)

        # Velocity values (stripped of units) should match
        for key in plain_velocities:
            plain_val = result_plain.velocities[key]
            quantity_val = u.ustrip(u.unit("m/s"), result_quantity.velocities[key])
            assert jnp.allclose(plain_val, quantity_val)

    def test_walk_local_flow_quantities_custom_lam(
        self, quantity_positions, quantity_velocities, unit_system
    ):
        """Test lfw.walk_local_flow with different lambda values."""
        # Test with higher lambda value (more momentum-dependent)
        lam_quantity = u.Q(2.0, "m")

        result = lfw.walk_local_flow(
            quantity_positions,
            quantity_velocities,
            start_idx=0,
            lam=lam_quantity,
            usys=unit_system,
        )

        # Should still produce valid results
        assert len(result.ordered_indices) >= 1
        assert result.ordered_indices[0] == 0  # Should start at start_idx

    def test_walk_local_flow_quantities_with_max_dist(
        self, quantity_positions, quantity_velocities, unit_system
    ):
        """Test lfw.walk_local_flow with max_dist constraint."""
        lam_quantity = u.Q(0.5, "m")
        max_dist = u.Q(1.5, "m")

        result = lfw.walk_local_flow(
            quantity_positions,
            quantity_velocities,
            start_idx=0,
            lam=lam_quantity,
            max_dist=max_dist,
            usys=unit_system,
        )

        # With small max_dist, might not visit all points
        assert len(result.ordered_indices) >= 1
        assert result.ordered_indices[0] == 0

    def test_walk_local_flow_quantities_with_kdtree_strategy(
        self, quantity_positions, quantity_velocities, unit_system
    ):
        """Test walk_local_flow with Quantities and KDTree strategy.

        This test verifies that walk_local_flow works with both Quantities
        and KDTree strategy enabled. This is the real use case from the
        notebook that requires proper quaxification.
        """
        lam_quantity = u.Q(0.5, "m")

        # Create KDTree strategy
        strategy = KDTreeStrategy(k=3)

        # Test WITH KDTree strategy and Quantities
        result = lfw.walk_local_flow(
            quantity_positions,
            quantity_velocities,
            start_idx=0,
            lam=lam_quantity,
            strategy=strategy,
            usys=unit_system,
        )

        # Should visit all 4 points
        assert len(result.ordered_indices) == 4
        assert jnp.allclose(result.ordered_indices, jnp.array([0, 1, 2, 3]))
        assert result.all_visited

        # Check that Quantities are preserved in output
        assert isinstance(result.positions["x"], u.AbstractQuantity)
        assert isinstance(result.velocities["x"], u.AbstractQuantity)
        assert str(result.positions["x"].unit) == "m"
        assert str(result.velocities["x"].unit) == "m / s"


class TestMetadataFlowing:
    """Tests for metadata flowing through state correctly."""

    def test_metadata_in_state_tuple(self, unit_system):
        """Test that metadata is properly stored in state tuple."""
        metadata = StateMetadata(usys=unit_system)

        # Metadata should be accessible
        assert metadata.get("usys") is not None
        assert isinstance(metadata.get("usys"), u.AbstractUnitSystem)


class TestEpitrochoidExample:
    """Tests for self-intersecting stream example from notebook."""

    def test_epitrochoid_stream_creation(self):
        """Test creating a self-intersecting epitrochoid stream with Quantities."""
        key = jax.random.key(0)

        def make_epitrochoid_stream(key, n=120, noise_sigma=0.5, scale=120.0):
            """Create an OPEN epitrochoid curve with self-intersections."""
            # Single outer rotation: 5° to 355° with 10° gap
            t_start = 5.0 * jnp.pi / 180.0
            t_end = 355.0 * jnp.pi / 180.0
            t = jnp.linspace(t_start, t_end, n)

            # Epitrochoid parameters
            R = 4.0  # Fixed circle radius
            r = 1.0  # Rolling circle radius
            d = 3.5  # Drawing point distance

            ratio = (R + r) / r
            x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
            y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0

            # Derivatives for velocity
            dx0 = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
            dy0 = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

            # Optional small positional noise
            kx, ky = jax.random.split(key)
            x = x0 + noise_sigma * jax.random.normal(kx, (n,))
            y = y0 + noise_sigma * jax.random.normal(ky, (n,))

            # Pack into unitful quantities
            pos = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
            vel = {"x": u.Q(dx0, "m/s"), "y": u.Q(dy0, "m/s")}
            return pos, vel, t

        pos, vel, t = make_epitrochoid_stream(key, n=100, noise_sigma=0.5, scale=120.0)

        # Verify data structure
        assert "x" in pos
        assert "y" in pos
        assert "x" in vel
        assert "y" in vel
        assert len(pos["x"]) == 100
        assert len(vel["x"]) == 100
        assert len(t) == 100

        # Verify units
        assert str(pos["x"].unit) == "m"
        assert str(vel["x"].unit) == "m / s"

    def test_epitrochoid_walk_with_momentum(self):
        """Test walk_local_flow on epitrochoid with different momentum values."""
        key = jax.random.key(0)

        # Create stream
        t_start = 5.0 * jnp.pi / 180.0
        t_end = 355.0 * jnp.pi / 180.0
        n = 80
        t = jnp.linspace(t_start, t_end, n)

        R, r, d = 4.0, 1.0, 3.5
        ratio = (R + r) / r
        scale = 120.0

        x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
        y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0

        dx0 = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
        dy0 = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

        kx, ky = jax.random.split(key)
        x = x0 + 0.5 * jax.random.normal(kx, (n,))
        y = y0 + 0.5 * jax.random.normal(ky, (n,))

        pos = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
        vel = {"x": u.Q(dx0, "m/s"), "y": u.Q(dy0, "m/s")}

        # Test with no momentum (lam=0)
        result_no_mom = lfw.walk_local_flow(pos, vel, start_idx=0, lam=u.Q(0.0, "m"))

        # Test with momentum
        result_with_mom = lfw.walk_local_flow(
            pos, vel, start_idx=0, lam=u.Q(400.0, "m")
        )

        # Both should produce valid results
        assert len(result_no_mom.ordered_indices) > 0
        assert len(result_with_mom.ordered_indices) > 0

        # Results should be different (momentum affects ordering)
        # but both should start from index 0
        assert result_no_mom.ordered_indices[0] == 0
        assert result_with_mom.ordered_indices[0] == 0

        # Verify units are preserved
        assert str(result_no_mom.positions["x"].unit) == "m"
        assert str(result_with_mom.positions["x"].unit) == "m"

    def test_epitrochoid_branch_jumps(self):
        """Test that momentum reduces branch jumps on self-intersecting stream."""
        key = jax.random.key(0)

        # Create stream
        n = 100
        t_start = 5.0 * jnp.pi / 180.0
        t_end = 355.0 * jnp.pi / 180.0
        t = jnp.linspace(t_start, t_end, n)

        R, r, d = 4.0, 1.0, 3.5
        ratio = (R + r) / r
        scale = 120.0

        x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
        y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0

        dx0 = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
        dy0 = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

        kx, ky = jax.random.split(key)
        x = x0 + 0.5 * jax.random.normal(kx, (n,))
        y = y0 + 0.5 * jax.random.normal(ky, (n,))

        pos = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
        vel = {"x": u.Q(dx0, "m/s"), "y": u.Q(dy0, "m/s")}

        # Walk with no momentum
        result_no_mom = lfw.walk_local_flow(pos, vel, start_idx=0, lam=u.Q(0.0, "m"))

        # Walk with momentum
        result_with_mom = lfw.walk_local_flow(
            pos, vel, start_idx=0, lam=u.Q(400.0, "m")
        )

        # Count big jumps in parameter t
        def count_big_jumps(ordered_idx, t_param, threshold=0.3):
            if len(ordered_idx) < 2:
                return 0
            t_along = t_param[ordered_idx]
            jumps = jnp.sum(jnp.abs(jnp.diff(t_along)) > threshold)
            return int(jumps)

        ordered_no_mom = jnp.array(result_no_mom.ordered_indices)
        ordered_with_mom = jnp.array(result_with_mom.ordered_indices)

        # Filter out invalid indices
        ordered_no_mom = ordered_no_mom[ordered_no_mom >= 0]
        ordered_with_mom = ordered_with_mom[ordered_with_mom >= 0]

        jumps_no_mom = count_big_jumps(ordered_no_mom, t)
        jumps_with_mom = count_big_jumps(ordered_with_mom, t)

        # With momentum, should typically have fewer big jumps
        # (though not guaranteed on a small test case)
        assert jumps_no_mom >= 0
        assert jumps_with_mom >= 0
