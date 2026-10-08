"""Tests for unxt interoperability with phasespace functions."""

import re

import pytest

import quaxed.numpy as jnp
import unxt as u

import phasecurvefit as pcf


class TestEuclideanDistanceQuantity:
    """Tests for euclidean_distance with Quantity-valued components."""

    def test_simple_distance(self):
        """Test distance calculation with simple Quantity values."""
        q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
        q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
        result = pcf.w.euclidean_distance(q_a, q_b)

        assert jnp.allclose(result, u.Q(5.0, "m"), atol=u.Q(1e-6, "m"))

    def test_distance_preserves_units(self):
        """Test that distance preserves input units."""
        q_a = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km")}
        q_b = {"x": u.Q(3.0, "km"), "y": u.Q(4.0, "km")}
        result = pcf.w.euclidean_distance(q_a, q_b)

        assert jnp.allclose(result, u.Q(5.0, "km"), atol=u.Q(1e-6, "km"))

    def test_identical_points(self):
        """Test that distance between identical points is zero."""
        q = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        result = pcf.w.euclidean_distance(q, q)

        assert jnp.allclose(result, u.Q(0.0, "m"), atol=u.Q(1e-10, "m"))

    def test_3d_distance(self):
        """Test distance with 3D coordinates."""
        q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        q_b = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(2.0, "m")}
        result = pcf.w.euclidean_distance(q_a, q_b)

        assert jnp.allclose(result, u.Q(3.0, "m"), atol=u.Q(1e-6, "m"))


class TestUnitDirectionQuantity:
    """Tests for unit_direction with Quantity-valued components."""

    def test_simple_direction(self):
        """Test unit direction calculation."""
        q_a = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
        q_b = {"x": u.Q(3.0, "m"), "y": u.Q(4.0, "m")}
        result = pcf.w.unit_direction(q_a, q_b)

        assert jnp.allclose(result["x"], u.Q(0.6, ""), atol=u.Q(1e-6, ""))
        assert jnp.allclose(result["y"], u.Q(0.8, ""), atol=u.Q(1e-6, ""))

    def test_direction_is_dimensionless(self):
        """Test that unit direction is dimensionless."""
        q_a = {"x": u.Q(0.0, "km"), "y": u.Q(0.0, "km")}
        q_b = {"x": u.Q(3.0, "km"), "y": u.Q(4.0, "km")}
        result = pcf.w.unit_direction(q_a, q_b)

        # Check dimensionless (unit should be dimensionless)
        assert result["x"].unit == u.unit("")
        assert result["y"].unit == u.unit("")

    def test_direction_has_unit_length(self):
        """Test that direction vector has unit norm."""
        q_a = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        q_b = {"x": u.Q(4.0, "m"), "y": u.Q(6.0, "m")}
        result = pcf.w.unit_direction(q_a, q_b)

        # Compute norm
        norm_sq = result["x"] ** 2 + result["y"] ** 2
        assert jnp.allclose(
            norm_sq,
            u.Q(1.0, ""),
            atol=u.Q(1e-6, ""),
        )


class TestVelocityNormQuantity:
    """Tests for velocity_norm with Quantity-valued components."""

    def test_simple_norm(self):
        """Test velocity norm calculation."""
        vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        result = pcf.w.velocity_norm(vel)

        assert jnp.allclose(result, u.Q(5.0, "m/s"), atol=u.Q(1e-6, "m/s"))

    def test_zero_velocity(self):
        """Test norm of zero velocity."""
        vel = {"x": u.Q(0.0, "m/s"), "y": u.Q(0.0, "m/s")}
        result = pcf.w.velocity_norm(vel)

        assert jnp.allclose(result, u.Q(0.0, "m/s"), atol=u.Q(1e-10, "m/s"))

    def test_3d_norm(self):
        """Test velocity norm with 3D velocity."""
        vel = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(2.0, "m/s")}
        result = pcf.w.velocity_norm(vel)

        assert jnp.allclose(
            result, u.Q(3.0, "m/s"), atol=u.Q(1e-6, "m/s")
        )  # sqrt(1 + 4 + 4) = 3


class TestUnitVelocityQuantity:
    """Tests for unit_velocity with Quantity-valued components."""

    def test_simple_unit_velocity(self):
        """Test unit velocity calculation."""
        vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        result = pcf.w.unit_velocity(vel)

        assert jnp.allclose(result["x"], u.Q(0.6, ""), atol=u.Q(1e-6, ""))
        assert jnp.allclose(result["y"], u.Q(0.8, ""), atol=u.Q(1e-6, ""))

    def test_unit_velocity_is_dimensionless(self):
        """Test that unit velocity is dimensionless."""
        vel = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s")}
        result = pcf.w.unit_velocity(vel)

        assert result["x"].unit == u.unit("")
        assert result["y"].unit == u.unit("")

    def test_unit_velocity_has_unit_length(self):
        """Test that unit velocity vector has unit norm."""
        vel = {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(2.0, "m/s")}
        result = pcf.w.unit_velocity(vel)

        norm_sq = result["x"] ** 2 + result["y"] ** 2 + result["z"] ** 2
        assert jnp.allclose(
            norm_sq,
            u.Q(1.0, ""),
            atol=u.Q(1e-6, ""),
        )


class TestCosineSimilarityQuantity:
    """Tests for cosine_similarity with Quantity-valued components."""

    def test_parallel_vectors(self):
        """Test cosine similarity of parallel unit vectors."""
        a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        b = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        result = pcf.w.cosine_similarity(a, b)

        # Result is dimensionless (dot product of unit vectors)
        assert jnp.allclose(result, u.Q(1.0, ""), atol=u.Q(1e-6, ""))

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        b = {"x": u.Q(0.0, "m/s"), "y": u.Q(1.0, "m/s")}
        result = pcf.w.cosine_similarity(a, b)

        # Orthogonal vectors have zero dot product (dimensionless)
        assert jnp.allclose(result, u.Q(0.0, ""), atol=u.Q(1e-6, ""))

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        a = {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        b = {"x": u.Q(-1.0, "m/s"), "y": u.Q(0.0, "m/s")}
        result = pcf.w.cosine_similarity(a, b)

        # Opposite vectors have negative dot product (dimensionless)
        assert jnp.allclose(result, u.Q(-1.0, ""), atol=u.Q(1e-6, ""))

    def test_similarity_is_dimensionless(self):
        """Test that cosine similarity is dimensionless."""
        a = {"x": u.Q(3.0, "km/s"), "y": u.Q(4.0, "km/s")}
        b = {"x": u.Q(3.0, "km/s"), "y": u.Q(4.0, "km/s")}
        result = pcf.w.cosine_similarity(a, b)

        # Dot product of unit vectors is dimensionless
        assert result.unit == u.unit("")


class TestUsysGuidance:
    """The missing-``usys`` error must name a way in that the caller has.

    The orderer dispatches accept ``metadata`` only, so the example in the error
    must be the ``metadata`` route, not a ``usys`` kwarg that does not exist.
    """

    @staticmethod
    def _quantity_data():
        return (
            {"x": u.Q(jnp.array([0.0, 1.0, 2.0]), "m")},
            {"x": u.Q(jnp.array([1.0, 1.0, 1.0]), "m/s")},
        )

    def test_order_names_the_metadata_route_it_actually_accepts(self):
        """``order`` has no ``usys`` kwarg, so suggesting one would misdirect."""
        q, p = self._quantity_data()
        with pytest.raises(TypeError, match=re.escape("order(q, p, metadata=")):
            pcf.order(q, p, pcf.orderers.LocalFlowOrderer(metric_scale=u.Q(1.0, "m")))

    def test_metadata_and_usys_together_merge(self):
        """The path that ``dict(metadata)`` used to break outright."""
        q, p = self._quantity_data()
        result = pcf.order(
            q,
            p,
            pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=u.Q(1.0, "m")),
            metadata=pcf.StateMetadata(note="carried", usys=u.unitsystems.si),
        )
        assert result.indices.shape == (3,)
