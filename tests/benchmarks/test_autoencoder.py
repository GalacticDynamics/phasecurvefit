"""Benchmarks for autoencoder training."""

import jax.numpy as jnp

import localflowwalk as lfw


class TestAutoencoderTrainingBenchmarks:
    """Benchmarks for autoencoder training."""

    def test_full_autoencoder_training_simple(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark full autoencoder training on 50-point stream.

        Trains both ordering net (encoder) and track net (decoder).
        """
        config = lfw.nn.TrainingConfig(n_epochs_encoder=10, n_epochs_both=10)
        trained, _, losses = benchmark(
            lfw.nn.train_autoencoder,
            simple_autoencoder,
            simple_wlf_result,
            config=config,
            key=rng_key,
        )

        assert trained is not None
        assert len(losses) == 20  # 10 + 10 epochs

    def test_full_autoencoder_training_medium(
        self, benchmark, medium_autoencoder, medium_wlf_result, rng_key
    ):
        """Benchmark full autoencoder training on 100-point stream."""
        config = lfw.nn.TrainingConfig(n_epochs_encoder=20, n_epochs_both=20)
        trained, _, losses = benchmark(
            lfw.nn.train_autoencoder,
            medium_autoencoder,
            medium_wlf_result,
            config=config,
            key=rng_key,
        )

        assert trained is not None
        assert len(losses) == 40  # 20 + 20 epochs

    def test_full_autoencoder_training_high_epochs(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark full autoencoder training with many epochs."""
        config = lfw.nn.TrainingConfig(n_epochs_encoder=50, n_epochs_both=50)
        trained, _, losses = benchmark(
            lfw.nn.train_autoencoder,
            simple_autoencoder,
            simple_wlf_result,
            config=config,
            key=rng_key,
        )

        assert trained is not None
        assert len(losses) == 100

    def test_encoder_only_training_simple(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark encoder (OrderingNet) training only.

        This is phase 1 of the full training pipeline.
        """
        config = lfw.nn.OrderingTrainingConfig(n_epochs=5)
        ordering_net = simple_autoencoder.encoder

        # Prepare training data
        indices = simple_wlf_result.indices
        pos = simple_wlf_result.positions
        vel = simple_wlf_result.velocities

        # Create mask for valid (non-padded) entries
        is_valid = indices >= 0
        valid_indices = indices[is_valid]

        # Get ordered positions and velocities
        ordered_pos = {k: pos[k][valid_indices] for k in pos}
        ordered_vel = {k: vel[k][valid_indices] for k in vel}

        # Normalize
        ordered_pos_norm, ordered_vel_norm = simple_autoencoder.normalizer.transform(
            ordered_pos, ordered_vel
        )

        # Concatenate into feature vectors
        x_feat = jnp.concatenate([ordered_pos_norm, ordered_vel_norm], axis=1)

        trained, _, losses = benchmark(
            lfw.nn.train_ordering_net,
            ordering_net,
            x_feat,
            jnp.arange(len(x_feat)),
            config=config,
            key=rng_key,
        )

        assert trained is not None
        assert losses is not None

    def test_decoder_training_simple(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark decoder (TrackNet) training.

        The decoder is trained as part of the full training pipeline.
        """
        config = lfw.nn.TrainingConfig(n_epochs_encoder=5, n_epochs_both=5)

        # Train full autoencoder to get decoder training
        trained, _, _ = benchmark(
            lfw.nn.train_autoencoder,
            simple_autoencoder,
            simple_wlf_result,
            config=config,
            key=rng_key,
        )

        # The decoder is part of the trained autoencoder
        assert trained is not None
        assert trained.decoder is not None
