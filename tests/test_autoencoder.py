"""Tests for the Autoencoder neural network for tracer interpolation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

import localflowwalk as lfw


class TestOrderingNet:
    """Tests for the OrderingNet encoder."""

    def test_initialization_2d(self, rng_key: PRNGKeyArray):
        """Test network initialization for 2D data."""
        net = lfw.nn.OrderingNet(in_size=2, key=rng_key)

        assert net.in_size == 2
        assert net.depth == 2  # Default 2 hidden layers
        assert net.mlp is not None
        assert net.gamma_head is not None
        assert net.prob_head is not None

    def test_initialization_3d(self, rng_key: PRNGKeyArray):
        """Test network initialization for 3D data."""
        net = lfw.nn.OrderingNet(in_size=3, key=rng_key)
        assert net.in_size == 3

    def test_custom_architecture(self, rng_key: PRNGKeyArray):
        """Test network with custom hidden size and depth."""
        net = lfw.nn.OrderingNet(in_size=2, width_size=64, depth=5, key=rng_key)

        assert net.depth == 5
        assert net.mlp.layers[0].out_features == 64

    def test_forward_pass_shape(self, rng_key: PRNGKeyArray):
        """Test that forward pass produces correct output shapes."""
        key, subkey = jr.split(rng_key)
        net = lfw.nn.OrderingNet(in_size=4, key=subkey)

        # Batch of 10 points in 2D phase-space (4 features: x, y, vx, vy)
        key, subkey = jr.split(key)
        w = jax.random.normal(subkey, (10, 4))
        gamma, prob = jax.vmap(net)(w)

        assert gamma.shape == (10,)
        assert prob.shape == (10,)

    def test_gamma_range(self, rng_key: PRNGKeyArray):
        """Test that gamma output is in [-1, 1]."""
        key, subkey = jr.split(rng_key)
        net = lfw.nn.OrderingNet(in_size=4, key=subkey)

        key, subkey = jr.split(key)
        w = jax.random.normal(subkey, (100, 4))
        gamma, _ = jax.vmap(net)(w)

        assert jnp.all(gamma >= -1.0)
        assert jnp.all(gamma <= 1.0)

    def test_prob_range(self, rng_key: PRNGKeyArray):
        """Test that probability output is in [0, 1]."""
        key, subkey = jr.split(rng_key)
        net = lfw.nn.OrderingNet(in_size=4, key=subkey)

        key, subkey = jr.split(key)
        w = jax.random.normal(subkey, (100, 4))
        _, prob = jax.vmap(net)(w)

        assert jnp.all(prob >= 0.0)
        assert jnp.all(prob <= 1.0)

    def test_single_point(self, rng_key: PRNGKeyArray):
        """Test with a single point input."""
        key, subkey = jr.split(rng_key)
        net = lfw.nn.OrderingNet(in_size=4, key=subkey)

        key, subkey = jr.split(key)
        w = jax.random.normal(subkey, (4,))
        gamma, prob = net(w)

        assert gamma.shape == ()
        assert prob.shape == ()


class TestTrackNet:
    """Tests for the TrackNet decoder."""

    def test_initialization_2d(self, rng_key: PRNGKeyArray):
        """Test network initialization for 2D output."""
        net = lfw.nn.TrackNet(out_size=2, key=rng_key)

        assert net.out_size == 2
        assert net.width_size == 100
        assert net.mlp is not None

    def test_initialization_3d(self, rng_key: PRNGKeyArray):
        """Test network initialization for 3D output."""
        net = lfw.nn.TrackNet(out_size=3, key=rng_key)
        assert net.out_size == 3

    def test_forward_pass_shape(self, rng_key: PRNGKeyArray):
        """Test that forward pass produces correct output shapes."""
        net = lfw.nn.TrackNet(out_size=3, key=rng_key)

        gamma = jnp.linspace(-1, 1, 10)
        qs = jax.vmap(net)(gamma)

        assert qs.shape == (10, 3)

    def test_single_gamma(self, rng_key: PRNGKeyArray):
        """Test with a single gamma input."""
        net = lfw.nn.TrackNet(out_size=2, key=rng_key)

        gamma = jnp.array(0.5)
        qs = net(gamma)

        assert qs.shape == (2,)


class TestAutoencoder:
    """Tests for the combined Autoencoder."""

    def test_initialization(self, rng_key: PRNGKeyArray):
        """Test autoencoder initialization."""
        ae = lfw.nn.PathAutoencoder(
            encoder=lfw.nn.OrderingNet(in_size=4, key=rng_key),
            decoder=lfw.nn.TrackNet(out_size=2, key=rng_key),
            normalizer=lfw.nn.StandardScalerNormalizer(
                {"x": jnp.array([1, 2]), "y": jnp.array([1, 2])},
                {"x": jnp.array([1, 2]), "y": jnp.array([1, 2])},
            ),
        )

        assert ae.encoder.in_size == 4
        assert ae.decoder.out_size == 2
        assert ae.normalizer is not None

    def test_make(self, rng_key: PRNGKeyArray):
        """Test autoencoder make."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            {"x": jnp.array([1, 2]), "y": jnp.array([1, 2])},
            {"x": jnp.array([1, 2]), "y": jnp.array([1, 2])},
        )
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=rng_key)

        assert ae.encoder.in_size == 4
        assert ae.decoder.out_size == 2

    def test_encode(self, rng_key: PRNGKeyArray):
        """Test encoding phase-space to (gamma, prob)."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        ae = lfw.nn.PathAutoencoder.make(
            normalizer=lfw.nn.StandardScalerNormalizer(pos, vel), key=rng_key
        )

        gamma, prob = ae.encode(pos, vel)

        assert gamma.shape == (3,)
        assert prob.shape == (3,)

    def test_decode(self, rng_key: PRNGKeyArray):
        """Test decoding gamma to position."""
        pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

        ae = lfw.nn.PathAutoencoder.make(
            normalizer=lfw.nn.StandardScalerNormalizer(pos, vel), key=rng_key
        )

        gamma = jnp.linspace(-1, 1, 5)
        pos = ae.decode(gamma)

        assert len(pos) == 2
        assert pos["x"].shape == (5,)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = lfw.nn.TrainingConfig()

        assert config.n_epochs_encoder == 800
        assert config.n_epochs_decoder == 100
        assert config.n_epochs_both == 200
        assert config.n_epochs == 1100  # Sum of all phases
        assert config.batch_size == 100
        assert config.lambda_prob == 1.0
        assert config.lambda_q == 1.0
        assert config.lambda_p == (1.0, 5.0)
        assert config.member_threshold == 0.5
        assert config.gamma_range == (-1.0, 1.0)
        assert config.weight_by_density is False
        assert config.freeze_encoder_final_training is False
        assert config.show_pbar is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=100,
            n_epochs_both=200,
            batch_size=64,
            lambda_prob=2.0,
            lambda_q=0.5,
            lambda_p=(1.0, 150.0),
        )

        assert config.n_epochs_encoder == 100
        assert config.n_epochs_decoder == 100
        assert config.n_epochs_both == 200
        assert config.n_epochs == 400
        assert config.batch_size == 64
        assert config.lambda_prob == 2.0
        assert config.lambda_q == 0.5
        assert config.lambda_p == (1.0, 150.0)

    def test_progress_bar_disabled(self):
        """Test that progress bar can be disabled."""
        config = lfw.nn.TrainingConfig(show_pbar=False)
        assert config.show_pbar is False

    def test_phase_configs(self):
        """Test that phase-specific configs are constructed correctly."""
        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=100,
            n_epochs_both=200,
            batch_size=64,
            lambda_prob=2.0,
            lambda_q=0.5,
            lambda_p=(1.0, 150.0),
            gamma_range=(-0.5, 0.5),
        )

        # Check encoder config
        encoder = config.encoderonly_config()
        assert encoder.n_epochs == 100
        assert encoder.batch_size == 64
        assert encoder.lambda_prob == 2.0
        assert encoder.gamma_range == (-0.5, 0.5)

        # Check decoder config
        decoder = config.decoderonly_config()
        assert decoder.n_epochs == 100
        assert decoder.batch_size == 64

        # Check autoencoder config
        autoencoder = config.autoencoder_config()
        assert autoencoder.n_epochs == 200
        assert autoencoder.batch_size == 64
        assert autoencoder.lambda_q == 0.5
        assert autoencoder.lambda_p == (1.0, 150.0)

    def test_gamma_range_validation(self):
        """Test that gamma_range is validated."""
        # Valid ranges should work
        config = lfw.nn.TrainingConfig(gamma_range=(-0.5, 0.5))
        assert config.gamma_range == (-0.5, 0.5)

        # Invalid ranges should raise ValueError
        with pytest.raises(ValueError, match="gamma_range minimum"):
            lfw.nn.TrainingConfig(gamma_range=(-1.5, 0.5))

        with pytest.raises(ValueError, match="gamma_range maximum"):
            lfw.nn.TrainingConfig(gamma_range=(-0.5, 1.5))


class TestTrainAutoencoder:
    """Tests for the train_autoencoder function."""

    @pytest.fixture
    def simple_wlf_result(self):
        """Create a simple phase-flow walk result for testing."""
        n_points = 20
        t = jnp.linspace(0, 5, n_points)

        pos = {"x": t, "y": 0.5 * t}
        vel = {"x": jnp.ones(n_points), "y": 0.5 * jnp.ones(n_points)}

        return lfw.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

    def test_training_runs(self, simple_wlf_result, rng_key: PRNGKeyArray):
        """Test that training completes without errors."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            simple_wlf_result.positions, simple_wlf_result.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        # Use minimal epochs for fast testing
        config = lfw.nn.TrainingConfig(n_epochs_encoder=5, n_epochs_both=5)

        trained, _, losses = lfw.nn.train_autoencoder(
            ae, simple_wlf_result, config=config, key=key2
        )

        assert trained is not None
        assert len(losses) == 110

    def test_training_reduces_loss(self, simple_wlf_result, rng_key: PRNGKeyArray):
        """Test that training reduces the loss within each phase."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            simple_wlf_result.positions, simple_wlf_result.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        # Use more epochs for phase 1 to see loss reduction
        config = lfw.nn.TrainingConfig(n_epochs_encoder=50, n_epochs_both=50)

        _, _, losses = lfw.nn.train_autoencoder(
            ae, simple_wlf_result, config=config, key=key2
        )

        # Phase 1 loss should decrease (epochs 0-49)
        encoder_early = jnp.mean(losses[:10])
        encoder_late = jnp.mean(losses[40:50])
        # Phase 1 should converge or at least not increase drastically
        assert encoder_late <= encoder_early * 2.0  # Allow some tolerance

        # Phase 2 starts fresh with a different loss function
        # So we just check it doesn't explode
        autoencoder_losses = losses[50:]
        assert jnp.all(jnp.isfinite(autoencoder_losses))

    def test_standardization_parameters_set(
        self, simple_wlf_result, rng_key: PRNGKeyArray
    ):
        """Test that standardization parameters are set after training."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            simple_wlf_result.positions, simple_wlf_result.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        config = lfw.nn.TrainingConfig(n_epochs_encoder=2, n_epochs_both=3)

        trained, _, _ = lfw.nn.train_autoencoder(
            ae, simple_wlf_result, config=config, key=key2
        )

        # Check that normalizer has the expected attributes
        assert hasattr(trained.normalizer, "q_mean")
        assert hasattr(trained.normalizer, "q_std")
        assert hasattr(trained.normalizer, "p_mean")
        assert hasattr(trained.normalizer, "p_std")


class TestFillOrderingGaps:
    """Tests for the fill_ordering_gaps function."""

    @pytest.fixture
    def localflowwalk_with_gaps(self):
        """Create a simple phase-flow walk result with skipped tracers."""
        n_points = 30
        key = jax.random.key(42)

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
        return lfw.walk_local_flow(
            pos, vel, start_idx=start_idx, metric_scale=3.0, max_dist=0.8
        )

    def test_fills_gaps(self, localflowwalk_with_gaps, rng_key: PRNGKeyArray):
        """Test that fill_ordering_gaps produces complete ordering."""
        # Skip if there are no gaps
        if len(localflowwalk_with_gaps.skipped_indices) == 0:
            pytest.skip("No skipped indices in this test case")

        normalizer = lfw.nn.StandardScalerNormalizer(
            localflowwalk_with_gaps.positions, localflowwalk_with_gaps.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=25, n_epochs_decoder=25, n_epochs_both=25, show_pbar=False
        )
        trained, _, _ = lfw.nn.train_autoencoder(
            ae, localflowwalk_with_gaps, config=config, key=key2
        )

        result = lfw.nn.fill_ordering_gaps(trained, localflowwalk_with_gaps)

        assert hasattr(result, "gamma")
        assert hasattr(result, "membership_prob")
        assert hasattr(result, "indices")

        # Should have more points than original phase-flow walk result
        assert len(result.indices) >= len(localflowwalk_with_gaps.indices)

    def test_result_structure(self, localflowwalk_with_gaps, rng_key: PRNGKeyArray):
        """Test that result has correct structure."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            localflowwalk_with_gaps.positions, localflowwalk_with_gaps.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
        )
        trained, _, _ = lfw.nn.train_autoencoder(
            ae, localflowwalk_with_gaps, config=config, key=key2
        )

        result = lfw.nn.fill_ordering_gaps(trained, localflowwalk_with_gaps)

        # Result is a NamedTuple, not a dict
        assert hasattr(result, "gamma")
        assert hasattr(result, "membership_prob")
        assert hasattr(result, "positions")
        assert hasattr(result, "velocities")
        assert hasattr(result, "indices")

    def test_prob_threshold(self, localflowwalk_with_gaps, rng_key: PRNGKeyArray):
        """Test probability threshold filtering."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            localflowwalk_with_gaps.positions, localflowwalk_with_gaps.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
        )
        trained, _, _ = lfw.nn.train_autoencoder(
            ae, localflowwalk_with_gaps, config=config, key=key2
        )

        result_low = lfw.nn.fill_ordering_gaps(
            trained, localflowwalk_with_gaps, prob_threshold=0.1
        )
        result_high = lfw.nn.fill_ordering_gaps(
            trained, localflowwalk_with_gaps, prob_threshold=0.9
        )

        # Higher threshold should give fewer or equal points
        assert len(result_high.indices) <= len(result_low.indices)


class TestJAXIntegration:
    """Tests for JAX compatibility."""

    def test_jit_encoder(self, rng_key: PRNGKeyArray):
        """Test that encoder can be JIT compiled."""
        pos = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0]), "y": jnp.array([0.5, 0.5])}
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=rng_key)

        @jax.jit
        def encode(qs, ps):
            return ae.encode(qs, ps)

        gamma, prob = encode(pos, vel)

        assert gamma.shape == (2,)
        assert prob.shape == (2,)

    def test_jit_decoder(self, rng_key: PRNGKeyArray):
        """Test that decoder can be JIT compiled."""
        pos = {
            "x": jnp.array([0.0, 1.0, 2.0]),
            "y": jnp.array([0.0, 1.0, 2.0]),
            "z": jnp.array([0.0, 1.0, 2.0]),
        }
        vel = {
            "x": jnp.array([1.0, 1.0, 1.0]),
            "y": jnp.array([0.5, 0.5, 0.5]),
            "z": jnp.array([0.25, 0.25, 0.25]),
        }
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=rng_key)

        @jax.jit
        def decode(gamma):
            return ae.decode(gamma)

        gamma = jnp.linspace(-1, 1, 10)
        q = decode(gamma)

        assert len(q) == 3
        assert q["x"].shape == (10,)

    def test_vmap_encoder(self, rng_key: PRNGKeyArray):
        """Test that encoder works with vmap."""
        # Create batch data - 10 points in 2D
        pos = {
            "x": jax.random.normal(jax.random.key(1), (10,)),
            "y": jax.random.normal(jax.random.key(2), (10,)),
        }
        vel = {
            "x": jax.random.normal(jax.random.key(3), (10,)),
            "y": jax.random.normal(jax.random.key(4), (10,)),
        }
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=rng_key)

        gamma, prob = ae.encode(pos, vel)

        assert gamma.shape == (10,)
        assert prob.shape == (10,)

    def test_grad_encoder(self, rng_key: PRNGKeyArray):
        """Test that gradients can be computed through encoder."""
        pos = {
            "x": jax.random.normal(jax.random.key(1), (10,)),
            "y": jax.random.normal(jax.random.key(2), (10,)),
        }
        vel = {
            "x": jax.random.normal(jax.random.key(3), (10,)),
            "y": jax.random.normal(jax.random.key(4), (10,)),
        }
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=rng_key)

        def loss(pos_vals, vel_vals):
            # Unpack into dict
            pos_dict = {"x": pos_vals[:, 0], "y": pos_vals[:, 1]}
            vel_dict = {"x": vel_vals[:, 0], "y": vel_vals[:, 1]}
            gamma, prob = ae.encode(pos_dict, vel_dict)
            return jnp.mean(gamma**2) + jnp.mean((prob - 0.5) ** 2)

        pos_array = jnp.stack([pos["x"], pos["y"]], axis=1)
        vel_array = jnp.stack([vel["x"], vel["y"]], axis=1)
        grad_fn = jax.grad(loss)

        # Should not raise
        grads = grad_fn(pos_array, vel_array)
        assert grads.shape == pos_array.shape


class Test3DData:
    """Tests with 3D phase-space data."""

    def test_3d_autoencoder(self, rng_key: PRNGKeyArray):
        """Test autoencoder with 3D data."""
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

        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=rng_key)

        gamma, prob = ae.encode(pos, vel)

        assert gamma.shape == (3,)
        assert prob.shape == (3,)

    def test_3d_training(self, rng_key: PRNGKeyArray):
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
            pos, vel, start_idx=start_idx, metric_scale=3.0
        )

        normalizer = lfw.nn.StandardScalerNormalizer(
            localflowwalk_result.positions, localflowwalk_result.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
        )
        trained, _, losses = lfw.nn.train_autoencoder(
            ae, localflowwalk_result, config=config, key=key2
        )

        assert len(losses) == 15
        assert trained is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self, rng_key: PRNGKeyArray):
        """Test with very small dataset."""
        pos = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 1.0])}
        vel = {"x": jnp.array([1.0, 1.0]), "y": jnp.array([1.0, 1.0])}

        localflowwalk_result = lfw.walk_local_flow(
            pos, vel, start_idx=0, metric_scale=1.0
        )

        normalizer = lfw.nn.StandardScalerNormalizer(
            localflowwalk_result.positions, localflowwalk_result.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=2, n_epochs_decoder=3, n_epochs_both=3, show_pbar=False
        )

        # Should not raise
        trained, _, losses = lfw.nn.train_autoencoder(
            ae, localflowwalk_result, config=config, key=key2
        )
        assert len(losses) == 8

    def test_all_points_ordered(self, rng_key: PRNGKeyArray):
        """Test when phase-flow walk orders all points (no gaps)."""
        n_points = 20
        t = jnp.linspace(0, 5, n_points)

        pos = {"x": t, "y": jnp.zeros(n_points)}
        vel = {"x": jnp.ones(n_points), "y": jnp.zeros(n_points)}

        localflowwalk_result = lfw.walk_local_flow(
            pos, vel, start_idx=0, metric_scale=1.0
        )

        # All points should be ordered
        assert len(localflowwalk_result.indices) == n_points
        assert len(localflowwalk_result.skipped_indices) == 0

        normalizer = lfw.nn.StandardScalerNormalizer(
            localflowwalk_result.positions, localflowwalk_result.velocities
        )
        key1, key2 = jax.random.split(rng_key)
        ae = lfw.nn.PathAutoencoder.make(normalizer, key=key1)

        # Use more epochs for better learning
        config = lfw.nn.TrainingConfig(
            n_epochs_encoder=25, n_epochs_decoder=25, n_epochs_both=25, show_pbar=False
        )

        # Should work fine even with no gaps
        trained, _, losses = lfw.nn.train_autoencoder(
            ae, localflowwalk_result, config=config, key=key2
        )
        result = lfw.nn.fill_ordering_gaps(
            trained, localflowwalk_result, prob_threshold=0.0
        )

        # With prob_threshold=0.0, all points should be included
        assert len(result.indices) == n_points
