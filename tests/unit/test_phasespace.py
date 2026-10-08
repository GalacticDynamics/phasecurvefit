"""Tests for phase-space operations."""

import jax
import jax.numpy as jnp
import pytest

import phasecurvefit as pcf


class TestEuclideanDistance:
    """Tests for euclidean distance computations."""

    def test_point_to_point_2d(self):
        """Test Euclidean distance between two 2D points."""
        pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        pos_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}

        dist = pcf.w.euclidean_distance(pos_a, pos_b)
        assert float(dist) == pytest.approx(5.0)

    def test_point_to_point_3d(self):
        """Test Euclidean distance between two 3D points."""
        pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
        pos_b = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(2.0)}

        dist = pcf.w.euclidean_distance(pos_a, pos_b)
        assert float(dist) == pytest.approx(3.0)

    def test_zero_distance(self):
        """Test distance from a point to itself."""
        pos = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        dist = pcf.w.euclidean_distance(pos, pos)
        assert float(dist) == pytest.approx(0.0)

    def test_point_to_array(self):
        """Test distance from a point to multiple points using vmap."""
        pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        pos_arr = {"x": jnp.array([3.0, 0.0]), "y": jnp.array([4.0, 5.0])}

        # Vmap over the batched dimension (axis 0) of pos_arr values
        def dist_fn(pos_single):
            return pcf.w.euclidean_distance(pos, pos_single)

        dists = jax.vmap(dist_fn, in_axes=({"x": 0, "y": 0},))(pos_arr)
        assert float(dists[0]) == pytest.approx(5.0)
        assert float(dists[1]) == pytest.approx(5.0)


class TestCosineSimilarity:
    """Tests for cosine similarity computations."""

    def test_parallel(self):
        """Test cosine similarity of parallel vectors."""
        vec_a = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        vec_b = {"x": jnp.array(2.0), "y": jnp.array(0.0)}

        sim = pcf.w.cosine_similarity(vec_a, vec_b)
        assert float(sim) == pytest.approx(1.0)  # dot product of unit vectors

    def test_antiparallel(self):
        """Test cosine similarity of antiparallel vectors."""
        vec_a = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        vec_b = {"x": jnp.array(-1.0), "y": jnp.array(0.0)}

        sim = pcf.w.cosine_similarity(vec_a, vec_b)
        assert float(sim) == pytest.approx(-1.0)

    def test_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec_a = {"x": jnp.array(1.0), "y": jnp.array(0.0)}
        vec_b = {"x": jnp.array(0.0), "y": jnp.array(1.0)}

        sim = pcf.w.cosine_similarity(vec_a, vec_b)
        assert float(sim) == pytest.approx(0.0)

    def test_array_valued(self):
        """Test cosine similarity with array-valued components using vmap."""
        vec_a = {"x": jnp.array([1.0, 0.0]), "y": jnp.array([0.0, 1.0])}
        vec_b = {"x": jnp.array([1.0, 1.0]), "y": jnp.array([0.0, 0.0])}

        # Vmap over the batched dimension (axis 0) of both inputs
        def sim_fn(a, b):
            return pcf.w.cosine_similarity(a, b)

        sim = jax.vmap(sim_fn, in_axes=({"x": 0, "y": 0}, {"x": 0, "y": 0}))(
            vec_a, vec_b
        )
        assert sim.shape == (2,)
        assert float(sim[0]) == pytest.approx(1.0)
        assert float(sim[1]) == pytest.approx(0.0)


class TestVelocityFunctions:
    """Tests for velocity computations."""

    def test_velocity_norm_2d(self):
        """Test velocity norm for a 2D velocity."""
        vel = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
        norm = pcf.w.velocity_norm(vel)
        assert float(norm) == pytest.approx(5.0)

    def test_velocity_norm_3d(self):
        """Test velocity norm for a 3D velocity."""
        vel = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(2.0)}
        norm = pcf.w.velocity_norm(vel)
        assert float(norm) == pytest.approx(3.0)

    def test_unit_velocity_2d(self):
        """Test unit velocity for a 2D velocity."""
        vel = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
        uvel = pcf.w.unit_velocity(vel)

        assert float(uvel["x"]) == pytest.approx(0.6)
        assert float(uvel["y"]) == pytest.approx(0.8)

    def test_unit_velocity_3d(self):
        """Test unit velocity for a 3D velocity."""
        vel = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(2.0)}
        uvel = pcf.w.unit_velocity(vel)

        assert float(uvel["x"]) == pytest.approx(1.0 / 3.0)
        assert float(uvel["y"]) == pytest.approx(2.0 / 3.0)
        assert float(uvel["z"]) == pytest.approx(2.0 / 3.0)

    def test_unit_velocity_norm_is_one(self):
        """Test that unit velocity has norm 1."""
        vel = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
        uvel = pcf.w.unit_velocity(vel)
        norm = pcf.w.velocity_norm(uvel)
        assert float(norm) == pytest.approx(1.0)


class TestDirectionFunctions:
    """Tests for direction computations."""

    def test_unit_direction_2d(self):
        """Test unit direction between two 2D points."""
        pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        pos_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}

        udir = pcf.w.unit_direction(pos_a, pos_b)

        assert float(udir["x"]) == pytest.approx(0.6)
        assert float(udir["y"]) == pytest.approx(0.8)

    def test_unit_direction_point_to_array(self):
        """Test unit directions from a point to multiple points using vmap."""
        pos = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        pos_arr = {"x": jnp.array([3.0, 0.0]), "y": jnp.array([4.0, 5.0])}

        # Vmap over the batched dimension (axis 0) of pos_arr
        def dir_fn(pos_single):
            return pcf.w.unit_direction(pos, pos_single)

        udirs = jax.vmap(dir_fn, in_axes=({"x": 0, "y": 0},))(pos_arr)

        assert float(udirs["x"][0]) == pytest.approx(0.6)
        assert float(udirs["y"][0]) == pytest.approx(0.8)
        assert float(udirs["x"][1]) == pytest.approx(0.0)
        assert float(udirs["y"][1]) == pytest.approx(1.0)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_w_at_single_point(self):
        """Test extracting a single point from arrays."""
        pos = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([4.0, 5.0, 6.0])}
        vel = {"x": jnp.array([0.1, 0.2, 0.3]), "y": jnp.array([0.4, 0.5, 0.6])}

        q, p = pcf.w.get_w_at(pos, vel, 1)

        assert float(q["x"]) == pytest.approx(2.0)
        assert float(q["y"]) == pytest.approx(5.0)
        assert float(p["x"]) == pytest.approx(0.2)
        assert float(p["y"]) == pytest.approx(0.5)

    def test_get_w_at_multiple_points(self):
        """Test extracting multiple points from arrays."""
        pos = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([4.0, 5.0, 6.0])}
        vel = {"x": jnp.array([0.1, 0.2, 0.3]), "y": jnp.array([0.4, 0.5, 0.6])}

        q, _ = pcf.w.get_w_at(pos, vel, jnp.array([0, 2]))

        assert q["x"].shape == (2,)
        assert float(q["x"][0]) == pytest.approx(1.0)
        assert float(q["x"][1]) == pytest.approx(3.0)


class TestJAXCompatibility:
    """Tests for JAX compatibility."""

    def test_jit_euclidean_distance(self):
        """Test that euclidean_distance works with jax.jit."""

        @jax.jit
        def compute_dist(pos_a, pos_b):
            return pcf.w.euclidean_distance(pos_a, pos_b)

        pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        pos_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}

        result = compute_dist(pos_a, pos_b)
        assert float(result) == pytest.approx(5.0)

    def test_vmap_over_indices(self):
        """Test that we can vmap over array indices."""
        pos = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([4.0, 5.0, 6.0])}

        # Sum of position components at each index
        def sum_pos_at_index(i):
            return pos["x"][i] + pos["y"][i]

        indices = jnp.arange(3)
        sums = jax.vmap(sum_pos_at_index)(indices)

        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(sums, expected)

    def test_grad_through_distance(self):
        """Test that gradients work through distance computation."""

        def loss_fn(pos_b):
            pos_a = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
            return pcf.w.euclidean_distance(pos_a, pos_b)

        pos_b = {"x": jnp.array(3.0), "y": jnp.array(4.0)}
        grads = jax.grad(loss_fn)(pos_b)

        # Gradient of sqrt(x^2 + y^2) is (x/r, y/r)
        assert float(grads["x"]) == pytest.approx(0.6)
        assert float(grads["y"]) == pytest.approx(0.8)
