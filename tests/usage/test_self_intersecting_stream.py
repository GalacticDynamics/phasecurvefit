"""Usage test: self-intersecting stream with autoencoder gamma learning.

This test reproduces the self-intersecting figure-eight stream example
to verify that the autoencoder correctly learns an affine parameter gamma
that unwraps the stream.

Reference: Nibauer et al. (2022), Appendix A - "Unwrapping Stellar Streams"
https://iopscience.iop.org/article/10.3847/1538-4357/ac93ee
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import localflowwalk as lfw


def make_self_intersecting_stream(
    key, n: int = 160, noise_sigma: float = 0.3, scale: float = 120.0
):
    """Create an OPEN epitrochoid curve with self-intersections.

    The stream completes ONE outer rotation (5° to 355°) with internal
    looping that creates self-intersections. The 10° gap prevents the
    curve from connecting back to its starting point.

    Parameters
    ----------
    key : PRNGKey
        Random key for noise.
    n : int
        Number of points.
    noise_sigma : float
        Positional noise standard deviation.
    scale : float
        Spatial scale of the stream.

    Returns
    -------
    pos : dict
        Position dictionary {"x": ..., "y": ...}.
    vel : dict
        Velocity dictionary {"x": ..., "y": ...}.
    t : Array
        True parameter for each point.

    """
    # Single outer rotation: 5° to 355° with 10° gap
    t_start = 5.0 * jnp.pi / 180.0
    t_end = 355.0 * jnp.pi / 180.0
    t = jnp.linspace(t_start, t_end, n)

    # Epitrochoid parameters: ratio=5 means 5 internal loops per outer rotation
    R = 4.0  # Fixed circle radius
    r = 1.0  # Rolling circle radius
    d = 2.8  # Drawing point distance (larger d creates bigger internal loops)

    ratio = (R + r) / r  # = 5 internal rotations per outer rotation
    x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
    y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0

    # Derivatives for velocity
    dx0 = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
    dy0 = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

    # Add noise
    kx, ky = jax.random.split(key)
    x = x0 + noise_sigma * jax.random.normal(kx, (n,))
    y = y0 + noise_sigma * jax.random.normal(ky, (n,))
    pos = {"x": x, "y": y}
    vel = {"x": dx0, "y": dy0}
    return pos, vel, t


class TestSelfIntersectingStream:
    """Tests for autoencoder on self-intersecting streams."""

    def test_walk_with_momentum_preserves_ordering(self):
        """Test that momentum term prevents branch jumps at crossing."""
        key = jax.random.key(42)
        pos, vel, t = make_self_intersecting_stream(key, n=160, scale=100.0)

        # Walk with and without momentum
        res_no_mom = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0)
        res_with_mom = lfw.walk_local_flow(pos, vel, start_idx=0, lam=50.0)

        # Count backward steps (non-monotonic) in true parameter t
        def count_backward_steps(t, ordered_idx, threshold=1e-6):
            valid = jnp.array(ordered_idx) >= 0
            valid_idx = jnp.array(ordered_idx)[valid]
            dt = t[valid_idx][1:] - t[valid_idx][:-1]
            return int(jnp.sum(dt < -threshold))

        back_no = count_backward_steps(t, res_no_mom.indices)
        back_yes = count_backward_steps(t, res_with_mom.indices)

        # Momentum should reduce backward steps
        assert back_yes <= back_no, (
            f"Momentum should reduce backward steps: "
            f"lam=0 has {back_no}, lam=50 has {back_yes}"
        )

    @pytest.mark.skip(reason="TODO: bad setup, ")
    def test_autoencoder_learns_gamma_from_true_ordering(self):
        """Test autoencoder learns correct gamma when given TRUE ordering.

        This tests the autoencoder algorithm in isolation. If we provide
        the correct ordering (sorted by true parameter t), the autoencoder
        should learn gamma values that correlate highly with t.

        This is a critical diagnostic: if this test fails, the autoencoder
        implementation itself is broken, not just the walk ordering.
        """
        key = jax.random.key(42)
        pos, vel, t = make_self_intersecting_stream(key, n=160, scale=100.0)

        # Create result with TRUE ordering (sorted by t)
        true_ordering = jnp.argsort(t)
        result = lfw.LocalFlowWalkResult(
            positions=pos,
            velocities=vel,
            indices=true_ordering,
        )

        # Train autoencoder
        key, model_key = jax.random.split(key)
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        autoencoder = lfw.nn.PathAutoencoder.make(normalizer, key=model_key)
        config = lfw.nn.TrainingConfig(show_pbar=False)
        trained, _, losses = lfw.nn.train_autoencoder(
            autoencoder, result, config=config, key=key
        )

        # Predict gamma for all tracers
        gamma_pred, _ = trained.encode(pos, vel)

        # Normalize t to [-1, 1] for comparison
        t_normalized = 2.0 * (t - t.min()) / (t.max() - t.min()) - 1.0

        # Compute correlation
        gamma_np = np.array(gamma_pred)
        t_np = np.array(t_normalized)
        correlation = np.corrcoef(gamma_np, t_np)[0, 1]

        # The correlation should be moderate to high (positive or negative,
        # since gamma direction is arbitrary). For a self-intersecting stream,
        # perfect correlation is challenging due to spatial ambiguities.
        assert abs(correlation) > 0.9, (
            f"Autoencoder should learn gamma correlated with true t. "
            f"Got correlation={correlation:.3f}, expected |corr| > 0.5. "
            f"This suggests a bug in the autoencoder implementation."
        )

    @pytest.mark.skip(reason="TODO: bad setup")
    def test_autoencoder_learns_gamma_from_walk_ordering(self):
        """Test autoencoder learns correct gamma from walk ordering.

        This is the full end-to-end test: walk → autoencoder → gamma.
        Success here means the full pipeline works.
        """
        key = jax.random.key(42)
        pos, vel, t = make_self_intersecting_stream(key, n=160, scale=100.0)

        # Run walk with momentum
        result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=50.0)

        # Train autoencoder
        key, model_key = jax.random.split(key)
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        autoencoder = lfw.nn.PathAutoencoder.make(normalizer, key=model_key)
        config = lfw.nn.TrainingConfig(
            show_pbar=False,
        )
        trained, _, losses = lfw.nn.train_autoencoder(
            autoencoder, result, config=config, key=key
        )

        # Predict gamma
        gamma_pred, _ = trained.encode(pos, vel)

        # Normalize t
        t_normalized = 2.0 * (t - t.min()) / (t.max() - t.min()) - 1.0

        # Compute correlation
        gamma_np = np.array(gamma_pred)
        t_np = np.array(t_normalized)
        correlation = np.corrcoef(gamma_np, t_np)[0, 1]

        # For end-to-end test, allow lower threshold than the ideal case
        # (walk ordering may not be perfect due to self-intersections)
        assert abs(correlation) > 0.9, (
            f"End-to-end autoencoder should learn gamma correlated with true t. "
            f"Got correlation={correlation:.3f}, expected |corr| > 0.7. "
            f"Either the walk has branch jumps or the autoencoder is broken."
        )

    @pytest.mark.skip(reason="TODO: bad setup")
    def test_gamma_monotonic_along_walk(self):
        """Test that predicted gamma increases monotonically along the walk path."""
        key = jax.random.key(42)
        pos, vel, t = make_self_intersecting_stream(key, n=160, scale=100.0)

        # Use true ordering for best case
        true_ordering = jnp.argsort(t)
        result = lfw.LocalFlowWalkResult(
            positions=pos, velocities=vel, indices=true_ordering
        )

        # Train autoencoder
        key, model_key = jax.random.split(key)
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        autoencoder = lfw.nn.PathAutoencoder.make(normalizer, key=model_key)
        config = lfw.nn.TrainingConfig(
            show_pbar=False,
        )
        trained, _, _ = lfw.nn.train_autoencoder(
            autoencoder, result, config=config, key=key
        )

        # Predict gamma
        gamma_pred, _ = trained.encode(pos, vel)

        # Get gamma values along the ordered path
        gamma_along_path = gamma_pred[true_ordering]

        # Compute fraction of monotonic steps
        dg = jnp.diff(gamma_along_path)
        # Count steps in dominant direction
        n_positive = jnp.sum(dg > 0)
        n_negative = jnp.sum(dg < 0)
        monotonic_frac = max(n_positive, n_negative) / len(dg)

        # At least 90% of steps should be in the same direction
        assert monotonic_frac > 0.9, (
            f"gamma should be mostly monotonic along path. "
            f"Got {monotonic_frac:.1%} monotonic steps."
        )


class TestDecoderReconstruction:
    """Tests for decoder position reconstruction."""

    @pytest.mark.skip(reason="TODO: bad setup")
    def test_decoder_reconstructs_positions(self):
        """Test that decoder can reconstruct positions from gamma."""
        key = jax.random.key(42)
        pos, vel, t = make_self_intersecting_stream(key, n=100, scale=100.0)

        # Use true ordering
        true_ordering = jnp.argsort(t)
        result = lfw.LocalFlowWalkResult(
            positions=pos, velocities=vel, indices=true_ordering
        )

        # Train
        key, model_key = jax.random.split(key)
        normalizer = lfw.nn.StandardScalerNormalizer(pos, vel)
        autoencoder = lfw.nn.PathAutoencoder.make(normalizer, key=model_key)
        config = lfw.nn.TrainingConfig(show_pbar=False)
        trained, _, _ = lfw.nn.train_autoencoder(
            autoencoder, result, config=config, key=key
        )

        # Predict gamma and decode back to position
        gamma_pred, _ = trained.encode(pos, vel)
        pos_recon = trained.decode(gamma_pred)

        # Compute reconstruction error
        x_err = jnp.mean((pos["x"] - pos_recon["x"]) ** 2)
        y_err = jnp.mean((pos["y"] - pos_recon["y"]) ** 2)
        total_err = float(jnp.sqrt(x_err + y_err))

        # Error should be small relative to scale
        scale = 100.0
        relative_err = total_err / scale

        assert relative_err < 0.2, (
            f"Decoder should reconstruct positions accurately. "
            f"Got relative error {relative_err:.2f}, expected < 0.2."
        )
