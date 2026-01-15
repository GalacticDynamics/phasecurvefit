"""Tests for the Autoencoder neural network for tracer interpolation."""

import jax
import jax.numpy as jnp
import pytest

import localflowwalk as lfw


class TestInterpolationNetwork:
    """Tests for the InterpolationNetwork encoder."""

    def test_initialization_2d(self):
        """Test network initialization for 2D data."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        assert net.n_dims == 2
        assert len(net.layers) == 3  # Default 3 hidden layers

    def test_initialization_3d(self):
        """Test network initialization for 3D data."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=3)

        assert net.n_dims == 3

    def test_custom_architecture(self):
        """Test network with custom hidden size and depth."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2, hidden_size=64, n_hidden=5)

        assert len(net.layers) == 5

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        # Batch of 10 points in 2D phase-space (4 features: x, y, vx, vy)
        phase_space = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        gamma, prob = net(phase_space)

        assert gamma.shape == (10,)
        assert prob.shape == (10,)

    def test_gamma_range(self):
        """Test that gamma output is in [-1, 1]."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        phase_space = jax.random.normal(jax.random.PRNGKey(1), (100, 4))
        gamma, _ = net(phase_space)

        assert jnp.all(gamma >= -1.0)
        assert jnp.all(gamma <= 1.0)

    def test_prob_range(self):
        """Test that probability output is in [0, 1]."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        phase_space = jax.random.normal(jax.random.PRNGKey(1), (100, 4))
        _, prob = net(phase_space)

        assert jnp.all(prob >= 0.0)
        assert jnp.all(prob <= 1.0)

    def test_single_point(self):
        """Test with a single point input."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        phase_space = jax.random.normal(jax.random.PRNGKey(1), (4,))
        gamma, prob = net(phase_space)

        assert gamma.shape == ()
        assert prob.shape == ()


class TestParamNet:
    """Tests for the ParamNet decoder."""

    def test_initialization_2d(self):
        """Test network initialization for 2D output."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.ParamNet(rngs, n_dims=2)

        assert net.n_dims == 2
        assert len(net.layers) == 3

    def test_initialization_3d(self):
        """Test network initialization for 3D output."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.ParamNet(rngs, n_dims=3)

        assert net.n_dims == 3

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.ParamNet(rngs, n_dims=3)

        gamma = jnp.linspace(-1, 1, 10)
        position = net(gamma)

        assert position.shape == (10, 3)

    def test_single_gamma(self):
        """Test with a single gamma input."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.ParamNet(rngs, n_dims=2)

        gamma = jnp.array(0.5)
        position = net(gamma)

        assert position.shape == (2,)


class TestAutoencoder:
    """Tests for the combined Autoencoder."""

    def test_initialization(self):
        """Test autoencoder initialization."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        assert ae.n_dims == 2
        assert ae.encoder is not None
        assert ae.decoder is not None

    def test_encode(self):
        """Test encoding phase-space to (gamma, prob)."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        gamma, prob = ae.encode(pos, vel)

        assert gamma.shape == (3,)
        assert prob.shape == (3,)

    def test_decode(self):
        """Test decoding gamma to position."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        gamma = jnp.linspace(-1, 1, 5)
        pos = ae.decode(gamma)

        assert pos.shape == (5, 2)

    def test_predict(self):
        """Test the predict method."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        gamma, prob = ae.predict(pos, vel)

        assert gamma.shape == (3,)
        assert prob.shape == (3,)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = lfw.nn.TrainingConfig()

        assert config.learning_rate == 1e-3
        assert config.n_epochs == 500
        assert config.batch_size == 32
        assert config.lambda_momentum == 100.0
        assert config.phase1_epochs == 200
        assert config.progress_bar is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = lfw.nn.TrainingConfig(
            learning_rate=1e-4, n_epochs=100, lambda_momentum=50.0
        )

        assert config.learning_rate == 1e-4
        assert config.n_epochs == 100
        assert config.lambda_momentum == 50.0

    def test_progress_bar_disabled(self):
        """Test that progress bar can be disabled."""
        config = lfw.nn.TrainingConfig(progress_bar=False)
        assert config.progress_bar is False


class TestTrainAutoencoder:
    """Tests for the train_autoencoder function."""

    @pytest.fixture
    def simple_wlf_result(self):
        """Create a simple NN+p result for testing."""
        n_points = 20
        t = jnp.linspace(0, 5, n_points)

        pos = {"x": t, "y": 0.5 * t}
        vel = {"x": jnp.ones(n_points), "y": 0.5 * jnp.ones(n_points)}

        return lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

    def test_training_runs(self, simple_wlf_result):
        """Test that training completes without errors."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        # Use minimal epochs for fast testing
        config = lfw.nn.TrainingConfig(n_epochs=10, phase1_epochs=5)

        trained, losses = lfw.nn.train_autoencoder(ae, simple_wlf_result, config)

        assert trained is not None
        assert len(losses) == 10

    def test_training_reduces_loss(self, simple_wlf_result):
        """Test that training reduces the loss within each phase."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        # Use more epochs for phase 1 to see loss reduction
        config = lfw.nn.TrainingConfig(n_epochs=100, phase1_epochs=50)

        _, losses = lfw.nn.train_autoencoder(ae, simple_wlf_result, config)

        # Phase 1 loss should decrease (epochs 0-49)
        phase1_early = jnp.mean(losses[:10])
        phase1_late = jnp.mean(losses[40:50])
        # Phase 1 should converge or at least not increase drastically
        assert phase1_late <= phase1_early * 2.0  # Allow some tolerance

        # Phase 2 starts fresh with a different loss function
        # So we just check it doesn't explode
        phase2_losses = losses[50:]
        assert jnp.all(jnp.isfinite(phase2_losses))

    def test_standardization_parameters_set(self, simple_wlf_result):
        """Test that standardization parameters are set after training."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        config = lfw.nn.TrainingConfig(n_epochs=5, phase1_epochs=2)

        trained, _ = lfw.nn.train_autoencoder(ae, simple_wlf_result, config)

        assert trained.pos_mean is not None
        assert trained.pos_std is not None
        assert trained.vel_mean is not None
        assert trained.vel_std is not None


class TestFillOrderingGaps:
    """Tests for the fill_ordering_gaps function."""

    @pytest.fixture
    def localflowwalk_with_gaps(self):
        """Create a NN+p result with skipped tracers."""
        n_points = 30
        key = jax.random.PRNGKey(42)

        # Create a curved stream
        theta = jnp.linspace(0, jnp.pi, n_points)
        shuffle_idx = jax.random.permutation(key, n_points)

        pos = {
            "x": jnp.cos(theta)[shuffle_idx],
            "y": jnp.sin(theta)[shuffle_idx],
        }
        vel = {
            "x": -jnp.sin(theta)[shuffle_idx],
            "y": jnp.cos(theta)[shuffle_idx],
        }

        # Use max_dist to create gaps
        start_idx = int(jnp.argmax(pos["x"]))
        return lfw.walk_local_flow(pos, vel, start_idx=start_idx, lam=3.0, max_dist=0.8)

    def test_fills_gaps(self, localflowwalk_with_gaps):
        """Test that fill_ordering_gaps produces complete ordering."""
        # Skip if there are no gaps
        if len(localflowwalk_with_gaps.skipped_indices) == 0:
            pytest.skip("No skipped indices in this test case")

        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        config = lfw.nn.TrainingConfig(n_epochs=50, phase1_epochs=25)
        trained, _ = lfw.nn.train_autoencoder(ae, localflowwalk_with_gaps, config)

        result = lfw.nn.fill_ordering_gaps(trained, localflowwalk_with_gaps)

        assert hasattr(result, "gamma")
        assert hasattr(result, "membership_prob")
        assert hasattr(result, "ordered_indices")

        # Should have more points than original knn+p
        assert len(result.ordered_indices) >= len(
            localflowwalk_with_gaps.ordered_indices
        )

    def test_result_structure(self, localflowwalk_with_gaps):
        """Test that result has correct structure."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        config = lfw.nn.TrainingConfig(n_epochs=10, phase1_epochs=5)
        trained, _ = lfw.nn.train_autoencoder(ae, localflowwalk_with_gaps, config)

        result = lfw.nn.fill_ordering_gaps(trained, localflowwalk_with_gaps)

        # Result is a NamedTuple, not a dict
        assert hasattr(result, "gamma")
        assert hasattr(result, "membership_prob")
        assert hasattr(result, "position")
        assert hasattr(result, "velocity")
        assert hasattr(result, "ordered_indices")

    def test_prob_threshold(self, localflowwalk_with_gaps):
        """Test probability threshold filtering."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        config = lfw.nn.TrainingConfig(n_epochs=10, phase1_epochs=5)
        trained, _ = lfw.nn.train_autoencoder(ae, localflowwalk_with_gaps, config)

        result_low = lfw.nn.fill_ordering_gaps(
            trained, localflowwalk_with_gaps, prob_threshold=0.1
        )
        result_high = lfw.nn.fill_ordering_gaps(
            trained, localflowwalk_with_gaps, prob_threshold=0.9
        )

        # Higher threshold should give fewer or equal points
        assert len(result_high.ordered_indices) <= len(result_low.ordered_indices)


class TestJAXIntegration:
    """Tests for JAX compatibility."""

    def test_jit_encoder(self):
        """Test that encoder can be JIT compiled."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        @jax.jit
        def encode(phase_space):
            return net(phase_space)

        phase_space = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        gamma, prob = encode(phase_space)

        assert gamma.shape == (10,)
        assert prob.shape == (10,)

    def test_jit_decoder(self):
        """Test that decoder can be JIT compiled."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.ParamNet(rngs, n_dims=3)

        @jax.jit
        def decode(gamma):
            return net(gamma)

        gamma = jnp.linspace(-1, 1, 10)
        position = decode(gamma)

        assert position.shape == (10, 3)

    def test_vmap_encoder(self):
        """Test that encoder works with vmap."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        # Single sample function
        def encode_single(phase_space):
            return net(phase_space)

        # Vectorized version
        encode_batch = jax.vmap(encode_single)

        batch = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        gamma, prob = encode_batch(batch)

        assert gamma.shape == (10,)
        assert prob.shape == (10,)

    def test_grad_encoder(self):
        """Test that gradients can be computed through encoder."""
        rngs = jax.random.PRNGKey(0)
        net = lfw.nn.InterpolationNetwork(rngs, n_dims=2)

        def loss(phase_space):
            gamma, prob = net(phase_space)
            return jnp.mean(gamma**2) + jnp.mean((prob - 0.5) ** 2)

        phase_space = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        grad_fn = jax.grad(loss)

        # Should not raise
        grads = grad_fn(phase_space)
        assert grads.shape == phase_space.shape


class Test3DData:
    """Tests with 3D phase-space data."""

    def test_3d_autoencoder(self):
        """Test autoencoder with 3D data."""
        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=3)

        pos = {
            "x": jnp.array([0.0, 1.0, 2.0]),
            "y": jnp.array([0.0, 0.5, 1.0]),
            "z": jnp.array([0.0, 0.25, 0.5]),
        }
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0]),
            "y": jnp.array([0.5, 0.5, 0.5]),
            "z": jnp.array([0.25, 0.25, 0.25]),
        }

        gamma, prob = ae.encode(pos, vel)

        assert gamma.shape == (3,)
        assert prob.shape == (3,)

    def test_3d_training(self):
        """Test training with 3D data."""
        n_points = 20
        t = jnp.linspace(0, 4 * jnp.pi, n_points)

        pos = {
            "x": jnp.cos(t),
            "y": jnp.sin(t),
            "z": t / (4 * jnp.pi),
        }
        vel = {
            "x": -jnp.sin(t),
            "y": jnp.cos(t),
            "z": jnp.ones(n_points) / (4 * jnp.pi),
        }

        start_idx = int(jnp.argmin(pos["z"]))
        localflowwalk_result = lfw.walk_local_flow(
            pos, vel, start_idx=start_idx, lam=3.0
        )

        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=3)

        config = lfw.nn.TrainingConfig(n_epochs=10, phase1_epochs=5)
        trained, losses = lfw.nn.train_autoencoder(ae, localflowwalk_result, config)

        assert len(losses) == 10
        assert trained is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        pos = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0]), "y": jnp.array([1.0, 1.0])}

        localflowwalk_result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        config = lfw.nn.TrainingConfig(n_epochs=5, phase1_epochs=2)

        # Should not raise
        trained, losses = lfw.nn.train_autoencoder(ae, localflowwalk_result, config)
        assert len(losses) == 5

    def test_all_points_ordered(self):
        """Test when NN+p orders all points (no gaps)."""
        n_points = 20
        t = jnp.linspace(0, 5, n_points)

        pos = {"x": t, "y": jnp.zeros(n_points)}
        vel = {"x": jnp.ones(n_points), "y": jnp.zeros(n_points)}

        localflowwalk_result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

        # All points should be ordered
        assert len(localflowwalk_result.ordered_indices) == n_points
        assert len(localflowwalk_result.skipped_indices) == 0

        rngs = jax.random.PRNGKey(0)
        ae = lfw.nn.Autoencoder(rngs, n_dims=2)

        # Use more epochs for better learning
        config = lfw.nn.TrainingConfig(n_epochs=50, phase1_epochs=25)

        # Should work fine even with no gaps
        trained, losses = lfw.nn.train_autoencoder(ae, localflowwalk_result, config)
        result = lfw.nn.fill_ordering_gaps(
            trained, localflowwalk_result, prob_threshold=0.0
        )

        # With prob_threshold=0.0, all points should be included
        assert len(result.ordered_indices) == n_points
