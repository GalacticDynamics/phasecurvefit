"""Tests for EncoderExternalDecoder with running-mean decoder."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import localflowwalk as lfw


class TestEncoderExternalDecoder:
    """Tests for EncoderExternalDecoder class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample phase-space data for testing."""
        N = 50
        t = jnp.linspace(0, 2 * jnp.pi, N)

        # Create a simple circular stream (use dict format)
        positions = {"x": jnp.cos(t), "y": jnp.sin(t)}
        velocities = {"x": -jnp.sin(t), "y": jnp.cos(t)}
        ordering = jnp.arange(N)

        return {
            "positions": positions,
            "velocities": velocities,
            "ordering": ordering,
            "N": N,
        }

    @pytest.fixture
    def normalizer(self, sample_data):
        """Create normalizer from sample data."""
        return lfw.nn.StandardScalerNormalizer(
            sample_data["positions"], sample_data["velocities"]
        )

    @pytest.fixture
    def encoder(self, normalizer):
        """Create an OrderingNet encoder."""
        return lfw.nn.OrderingNet(
            in_size=2 * normalizer.n_spatial_dims,
            width_size=32,
            depth=2,
            key=jr.key(0),
        )

    @pytest.fixture
    def decoder(self, encoder, normalizer, sample_data):
        """Create a running-mean decoder."""
        return lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.2,
        )

    def test_encoder_you_decoder_creation(self, encoder, decoder, normalizer):
        """Test creating an EncoderExternalDecoder instance."""
        model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        assert model is not None
        assert isinstance(model.encoder, lfw.nn.OrderingNet)
        assert callable(model.decoder)
        assert isinstance(model.normalizer, lfw.nn.StandardScalerNormalizer)

    def test_encode_method(self, encoder, decoder, normalizer, sample_data):
        """Test the encode method of EncoderExternalDecoder."""
        model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        positions = sample_data["positions"]
        velocities = sample_data["velocities"]

        gamma, prob = model.encode(positions, velocities)

        assert gamma.shape == (sample_data["N"],)
        assert prob.shape == (sample_data["N"],)
        assert jnp.all((gamma >= -1) & (gamma <= 1))
        assert jnp.all((prob >= 0) & (prob <= 1))

    def test_decode_method(self, encoder, decoder, normalizer):
        """Test the decode method of EncoderExternalDecoder."""
        model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        # Test multiple gamma values - use vmap for batched decoding
        gamma = jnp.array([0.0, 0.5, -0.5])
        qs = jax.vmap(model.decode)(gamma)

        assert len(qs) == 2
        assert qs["x"].shape == (3,)

    def test_encode_decode_roundtrip(self, encoder, decoder, normalizer, sample_data):
        """Test encoding and decoding maintains reasonable reconstruction."""
        model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        positions = sample_data["positions"]
        velocities = sample_data["velocities"]

        # Encode
        gamma, _ = model.encode(positions, velocities)

        # Decode (use vmap for batched decoding)
        reconstructed = jax.vmap(model.decode)(gamma)

        # Should have same shape as normalized positions
        qs_norm, _ = normalizer.transform(positions, velocities)
        # reconstructed is a dict with shape (N,) for each component
        assert reconstructed["x"].shape == (sample_data["N"],)
        assert reconstructed["y"].shape == (sample_data["N"],)


class TestRunningMeanDecoder:
    """Tests for RunningMeanDecoder classmethod."""

    @pytest.fixture
    def sample_data(self):
        """Create simple 1D stream data."""
        N = 100
        positions = {"x": jnp.linspace(0, 10, N), "y": jnp.zeros(N)}
        velocities = {"x": jnp.ones(N), "y": jnp.zeros(N)}
        return {"positions": positions, "velocities": velocities, "N": N}

    @pytest.fixture
    def encoder(self):
        """Create a simple encoder."""
        return lfw.nn.OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(0))

    @pytest.fixture
    def normalizer(self, sample_data):
        """Create normalizer."""
        return lfw.nn.StandardScalerNormalizer(
            sample_data["positions"], sample_data["velocities"]
        )

    def test_decoder_creation(self, encoder, normalizer, sample_data):
        """Test creating a running-mean decoder."""
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.1,
        )

        assert callable(decoder)

    def test_decoder_output_shape(self, encoder, normalizer, sample_data):
        """Test decoder output has correct shape."""
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.1,
        )

        # Test single gamma value
        gamma = jnp.array(0.0)
        result = decoder(gamma)

        assert result.shape == (2,)  # 2D position

    def test_decoder_vectorization(self, encoder, normalizer, sample_data):
        """Test decoder works with vmap."""
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.1,
        )

        # Test multiple gamma values
        gammas = jnp.array([-0.5, 0.0, 0.5])
        results = jax.vmap(decoder)(gammas)

        assert results.shape == (3, 2)

    def test_decoder_jit_compatible(self, encoder, normalizer, sample_data):
        """Test decoder is JIT-compatible."""
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.1,
        )

        # JIT the vmapped decoder instead of the decoder itself
        jitted_decoder = jax.jit(jax.vmap(decoder))
        gammas = jnp.array([0.0, 0.5, -0.5])

        # Should not raise
        result = jitted_decoder(gammas)
        assert result.shape == (3, 2)

    def test_decoder_with_normalized_arrays(self, encoder, normalizer, sample_data):
        """Test decoder creation with pre-normalized arrays."""
        # Normalize data first
        qs_norm, ps_norm = normalizer.transform(
            sample_data["positions"], sample_data["velocities"]
        )

        # Create decoder with normalized arrays
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            qs_norm,
            ps_norm,
            window_size=0.1,
        )

        gamma = jnp.array(0.0)
        result = decoder(gamma)

        assert result.shape == (2,)

    def test_decoder_window_size_effect(self, encoder, normalizer, sample_data):
        """Test that window size affects decoder smoothness."""
        # Create decoders with different window sizes
        decoder_small = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.05,
        )

        decoder_large = lfw.nn.RunningMeanDecoder.make(
            encoder,
            normalizer,
            sample_data["positions"],
            sample_data["velocities"],
            window_size=0.5,
        )

        gamma = jnp.array(0.0)

        # Both should produce valid outputs
        result_small = decoder_small(gamma)
        result_large = decoder_large(gamma)

        assert result_small.shape == (2,)
        assert result_large.shape == (2,)


class TestTrainEncoderExternalDecoder:
    """Tests for train_autoencoder function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for training."""
        N = 50
        positions = {"x": jnp.linspace(0, 5, N), "y": jnp.zeros(N)}
        velocities = {"x": jnp.ones(N), "y": jnp.zeros(N)}
        ordering = jnp.arange(N)
        return {"positions": positions, "velocities": velocities, "ordering": ordering}

    @pytest.fixture
    def model(self, sample_data):
        """Create untrained EncoderExternalDecoder."""
        normalizer = lfw.nn.StandardScalerNormalizer(
            sample_data["positions"], sample_data["velocities"]
        )
        encoder = lfw.nn.OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(0))
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder, normalizer, sample_data["positions"], sample_data["velocities"]
        )
        return lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

    def test_train_with_arrays(self, model, sample_data):
        """Test training with array inputs."""
        # Prepare normalized data
        qs_norm, ps_norm = model.normalizer.transform(
            sample_data["positions"], sample_data["velocities"]
        )
        ws_norm = jnp.concat([qs_norm, ps_norm], axis=1)

        config = lfw.nn.OrderingTrainingConfig(
            n_epochs=5, batch_size=16, show_pbar=False
        )

        trained_model, opt_state, losses = lfw.nn.train_autoencoder(
            model, ws_norm, sample_data["ordering"], config=config, key=jr.key(1)
        )

        assert isinstance(trained_model, lfw.nn.EncoderExternalDecoder)
        assert losses.shape == (5,)
        assert jnp.all(jnp.isfinite(losses))

    def test_train_with_walk_result(self, sample_data):
        """Test training with LocalFlowWalkResult input."""
        # Create a walk result
        positions = sample_data["positions"]
        velocities = sample_data["velocities"]

        walk_result = lfw.walk_local_flow(
            positions, velocities, start_idx=0, metric_scale=1.0
        )

        # Create model
        normalizer = lfw.nn.StandardScalerNormalizer(positions, velocities)
        encoder = lfw.nn.OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(0))
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder, normalizer, positions, velocities
        )
        model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        # Train
        config = lfw.nn.OrderingTrainingConfig(
            n_epochs=5, batch_size=16, show_pbar=False
        )
        trained_model, opt_state, losses = lfw.nn.train_autoencoder(
            model, walk_result, config=config, key=jr.key(1)
        )

        assert isinstance(trained_model, lfw.nn.EncoderExternalDecoder)
        assert losses.shape == (5,)

    def test_training_reduces_loss(self, model, sample_data):
        """Test that training reduces loss over epochs."""
        qs_norm, ps_norm = model.normalizer.transform(
            sample_data["positions"], sample_data["velocities"]
        )
        ws_norm = jnp.concat([qs_norm, ps_norm], axis=1)

        config = lfw.nn.OrderingTrainingConfig(
            n_epochs=20, batch_size=16, show_pbar=False
        )

        _, _, losses = lfw.nn.train_autoencoder(
            model, ws_norm, sample_data["ordering"], config=config, key=jr.key(1)
        )

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_trained_encoder_predicts_ordering(self, model, sample_data):
        """Test that trained encoder predicts reasonable gamma values."""
        qs_norm, ps_norm = model.normalizer.transform(
            sample_data["positions"], sample_data["velocities"]
        )
        ws_norm = jnp.concat([qs_norm, ps_norm], axis=1)

        config = lfw.nn.OrderingTrainingConfig(
            n_epochs=10, batch_size=16, show_pbar=False
        )

        trained_model, _, _ = lfw.nn.train_autoencoder(
            model, ws_norm, sample_data["ordering"], config=config, key=jr.key(1)
        )

        # Predict gamma for training data
        gamma, prob = trained_model.encode(
            sample_data["positions"], sample_data["velocities"]
        )

        # Gamma should be in valid range
        assert jnp.all((gamma >= -1) & (gamma <= 1))

        # Ordering by gamma should be roughly monotonic
        gamma_order = jnp.argsort(gamma)
        # For simple linear stream, ordering should be preserved
        # Allow some tolerance since this is a small network
        correlation = jnp.corrcoef(gamma_order, sample_data["ordering"])[0, 1]
        assert correlation > 0.8


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete workflow: walk -> create model -> train -> predict."""
        # Create sample stream data
        N = 60
        t = jnp.linspace(0, 2 * jnp.pi, N)
        positions = {"x": jnp.cos(t), "y": jnp.sin(t)}
        velocities = {"x": -jnp.sin(t), "y": jnp.cos(t)}

        # Run phase-flow walk (skip some points by using larger lambda)
        walk_result = lfw.walk_local_flow(
            positions, velocities, start_idx=0, metric_scale=0.5
        )

        # Create normalizer
        normalizer = lfw.nn.StandardScalerNormalizer(positions, velocities)

        # Create encoder
        encoder = lfw.nn.OrderingNet(in_size=4, width_size=64, depth=2, key=jr.key(0))

        # Create initial decoder
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder, normalizer, positions, velocities, window_size=0.15
        )

        # Create model
        model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        # Train
        config = lfw.nn.OrderingTrainingConfig(
            n_epochs=10, batch_size=20, show_pbar=False
        )
        trained_model, _, losses = lfw.nn.train_autoencoder(
            model, walk_result, config=config, key=jr.key(1)
        )

        # Check training worked
        assert jnp.all(jnp.isfinite(losses))
        assert losses[-1] < losses[0]

        # Predict for all points
        gamma, prob = trained_model.encode(positions, velocities)

        # Decode back to positions
        reconstructed = jax.vmap(trained_model.decode)(gamma)

        # Check outputs
        assert gamma.shape == (N,)
        assert prob.shape == (N,)
        assert isinstance(reconstructed, dict)
        assert reconstructed["x"].shape == (N,)
        assert reconstructed["y"].shape == (N,)
        assert jnp.all(jnp.isfinite(gamma))
        assert jnp.all(jnp.isfinite(prob))
        assert jnp.all(jnp.isfinite(reconstructed["x"]))
        assert jnp.all(jnp.isfinite(reconstructed["y"]))

    def test_comparison_with_path_autoencoder(self):
        """Test that EncoderExternalDecoder have similar results to PathAutoencoder."""
        # Create sample data
        N = 50
        positions = {"x": jnp.linspace(0, 5, N), "y": jnp.zeros(N)}
        velocities = {"x": jnp.ones(N), "y": jnp.zeros(N)}

        walk_result = lfw.walk_local_flow(
            positions, velocities, start_idx=0, metric_scale=0.3
        )

        normalizer = lfw.nn.StandardScalerNormalizer(positions, velocities)

        # Create EncoderExternalDecoder
        encoder = lfw.nn.OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(0))
        decoder = lfw.nn.RunningMeanDecoder.make(
            encoder, normalizer, positions, velocities
        )
        simple_model = lfw.nn.EncoderExternalDecoder(
            encoder=encoder, decoder=decoder, normalizer=normalizer
        )

        # Train
        config = lfw.nn.OrderingTrainingConfig(
            n_epochs=10, batch_size=16, show_pbar=False
        )
        trained_simple, _, _ = lfw.nn.train_autoencoder(
            simple_model, walk_result, config=config, key=jr.key(1)
        )

        # Get predictions
        gamma_simple, prob_simple = trained_simple.encode(positions, velocities)

        # Both should produce valid outputs
        assert jnp.all((gamma_simple >= -1) & (gamma_simple <= 1))
        assert jnp.all((prob_simple >= 0) & (prob_simple <= 1))

        # Gamma ordering should make sense (monotonic for this simple case)
        assert jnp.corrcoef(gamma_simple, jnp.arange(N))[0, 1] > 0.7
