"""Benchmarks for autoencoder training."""

import jax.numpy as jnp

import phasecurvefit as pcf


class TestAutoencoderTrainingBenchmarks:
    """Benchmarks for autoencoder training."""

    def test_full_autoencoder_training_simple(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark full autoencoder training on 50-point stream.

        Trains both ordering net (encoder) and track net (decoder).
        """
        config = pcf.nn.TrainingConfig(
            n_epochs_encoder=10, n_epochs_both=10, show_pbar=False
        )
        trained, _, losses = benchmark(
            pcf.nn.train_autoencoder,
            simple_autoencoder,
            simple_wlf_result,
            config=config,
            key=rng_key,
        )

        assert trained is not None
        # Losses from: encoder (10) + decoder (100 default) + both (10) = 120
        assert len(losses) == 120

    def test_full_autoencoder_training_medium(
        self, benchmark, medium_autoencoder, medium_wlf_result, rng_key
    ):
        """Benchmark full autoencoder training on 100-point stream."""
        config = pcf.nn.TrainingConfig(
            n_epochs_encoder=20, n_epochs_both=20, show_pbar=False
        )
        trained, _, losses = benchmark(
            pcf.nn.train_autoencoder,
            medium_autoencoder,
            medium_wlf_result,
            config=config,
            key=rng_key,
        )

        assert trained is not None
        # Losses from: encoder (20) + decoder (100 default) + both (20) = 140
        assert len(losses) == 140

    def test_full_autoencoder_training_high_epochs(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark full autoencoder training with many epochs."""
        config = pcf.nn.TrainingConfig(
            n_epochs_encoder=50, n_epochs_both=50, show_pbar=False
        )
        trained, _, losses = benchmark(
            pcf.nn.train_autoencoder,
            simple_autoencoder,
            simple_wlf_result,
            config=config,
            key=rng_key,
        )

        assert trained is not None
        # Losses from: encoder (50) + decoder (100 default) + both (50) = 200
        assert len(losses) == 200

    def test_encoder_only_training_simple(
        self, benchmark, simple_autoencoder, simple_wlf_result, rng_key
    ):
        """Benchmark encoder (OrderingNet) training only.

        This is phase 1 of the full training pipeline.
        """
        config = pcf.nn.OrderingTrainingConfig(n_epochs=5, show_pbar=False)
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
            pcf.nn.train_ordering_net,
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
        config = pcf.nn.TrainingConfig(
            n_epochs_encoder=5, n_epochs_both=5, show_pbar=False
        )

        # Train full autoencoder to get decoder training
        trained, _, _ = benchmark(
            pcf.nn.train_autoencoder,
            simple_autoencoder,
            simple_wlf_result,
            config=config,
            key=rng_key,
        )

        # The decoder is part of the trained autoencoder
        assert trained is not None
        assert trained.decoder is not None
