r"""Autoencoder Neural Network for interpolating skipped tracers.

This module implements the autoencoder neural network from Appendix A.2 of
Nibauer et al. (2022) for assigning $\gamma$ values to stream tracers that were
skipped by the NN+p algorithm.

The autoencoder consists of two parts:
1. **Interpolation Network**: Maps phase-space coordinates $(x, v) \to (\gamma, p)$
   where $\gamma \in [-1, 1]$ is the ordering parameter and $p \in [0, 1]$ is the
   membership probability.
2. **Param-Net (Decoder)**: Maps $\gamma \to x$, reconstructing the position from
   the ordering parameter.

Training follows a two-step process:
1. Train the interpolation network on ordered tracers from NN+p
2. Jointly train both networks with a momentum condition to refine $\gamma$ values

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning." Appendix A.2.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import localflowwalk as lfw

Create phase-space data and run NN+p:

>>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.sin(jnp.linspace(0, jnp.pi, 20))}
>>> vel = {"x": jnp.ones(20), "y": jnp.cos(jnp.linspace(0, jnp.pi, 20))}
>>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

Train autoencoder to interpolate skipped tracers:

>>> rngs = jax.random.PRNGKey(0)
>>> autoencoder = lfw.nn.Autoencoder(rngs=rngs, n_dims=2)
>>> cfg = lfw.nn.TrainingConfig(n_epochs=100)
>>> trained, losses = lfw.nn.train_autoencoder(autoencoder, result, config=cfg)

Predict $\gamma$ for all tracers (including skipped ones):

>>> gamma, prob = trained.predict(pos, vel)
>>> gamma
Array([-0.84798545, -0.8414864 , -0.8173885 , -0.7622074 , -0.66193575,
       -0.52575207, -0.38774183, -0.26474118, -0.15166219, -0.04374509,
        0.05862008,  0.15575506,  0.25443456,  0.3656837 ,  0.48988712,
        0.61111546,  0.7106663 ,  0.78016347,  0.82270974,  0.84560573],
        dtype=float32)

"""

__all__: tuple[str, ...] = (
    # Network components
    "InterpolationNetwork",
    "ParamNet",
    "Autoencoder",
    # Training functions
    "train_autoencoder",
    "AutoencoderResult",
    "TrainingConfig",
    # Convenience functions
    "fill_ordering_gaps",
)

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm.auto import tqdm

from .algorithm import LocalFlowWalkResult
from .custom_types import FAny, FSz0, FSzD, FSzN, FSzND, ISzN, VectorComponents


class InterpolationNetwork(eqx.Module):
    r"""Interpolation network: maps phase-space $(x, v) \to (\gamma, p)$.

    This network takes 6D (or 2D * n_dims) phase-space coordinates and outputs:
    - $\gamma \in [-1, 1]$: The ordering parameter along the stream
    - $p \in [0, 1]$: The membership probability (1 = likely stream member)

    The architecture follows Appendix B.3 of Nibauer et al. (2022):
    - Fully connected MLP with 3 hidden layers of 100 nodes each
    - tanh activation functions
    - Output $\gamma$ uses tanh to constrain to [-1, 1]
    - Output p uses sigmoid to constrain to [0, 1]

    The $\gamma$ output is scaled by a temperature parameter to prevent
    tanh saturation during training. Without this, the network tends to
    output saturated values near ±1 for all points, losing fine-grained
    ordering information.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key for initialization.
    n_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D).
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    """

    layers: list[eqx.nn.Linear]
    gamma_head: eqx.nn.Linear
    prob_head: eqx.nn.Linear
    n_dims: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        n_dims: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 3,
    ) -> None:
        self.n_dims = n_dims
        input_size = 2 * n_dims  # Position + velocity

        # Build layers
        layers = []
        in_features = input_size
        keys = jr.split(key, n_hidden + 2)
        for i in range(n_hidden):
            layers.append(eqx.nn.Linear(in_features, hidden_size, key=keys[i]))
            in_features = hidden_size
        self.layers = layers

        # Output heads
        self.gamma_head = eqx.nn.Linear(hidden_size, 1, key=keys[n_hidden])
        self.prob_head = eqx.nn.Linear(hidden_size, 1, key=keys[n_hidden + 1])

    def __call__(self, phase_space: Float[Array, "... D"]) -> tuple[FAny, FAny]:
        """Forward pass through the interpolation network.

        Parameters
        ----------
        phase_space : Array
            Phase-space coordinates of shape (..., 2*n_dims).
            For each point: [x, y, z, vx, vy, vz] (or 2D equivalent).

        Returns
        -------
        gamma : Array
            Ordering parameter in [-1, 1], shape (...).
        prob : Array
            Membership probability in [0, 1], shape (...).

        """
        # Handle batched input by vmapping over leading dimensions
        if phase_space.ndim > 1:
            # vmap over all batch dimensions
            return jax.vmap(self._forward)(phase_space)
        return self._forward(phase_space)

    def _forward(self, phase_space: FSzD) -> tuple[FSz0, FSz0]:
        """Forward pass for a single input vector."""
        x = phase_space
        for layer in self.layers:
            x = jax.nn.tanh(layer(x))

        # Output heads with appropriate activations
        # Scale gamma head by temperature to prevent tanh saturation during
        # early training. Lower values = wider output range but more saturation.
        # Value of 1.0 means no scaling; network learns full [-1, 1] range.
        gamma_temperature = 1.0
        gamma = jax.nn.tanh(self.gamma_head(x) / gamma_temperature).squeeze(-1)
        prob = jax.nn.sigmoid(self.prob_head(x)).squeeze(-1)

        return gamma, prob


class ParamNet(eqx.Module):
    r"""Param-Net (decoder): maps $\gamma \to$ position (x, y, z).

    This network reconstructs the stream track position from the ordering
    parameter $\gamma$. It serves as the second half of the autoencoder.

    The architecture follows Appendix B.1 of Nibauer et al. (2022):
    - Fully connected MLP with 3 hidden layers of 100 nodes each
    - tanh activation functions
    - Linear output layer

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key for initialization.
    n_dims : int
        Number of spatial dimensions (2 for 2D, 3 for 3D).
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    """

    layers: list[eqx.nn.Linear]
    output_layer: eqx.nn.Linear
    n_dims: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        n_dims: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 3,
    ) -> None:
        self.n_dims = n_dims

        # Build layers
        layers = []
        in_features = 1  # γ is a scalar
        keys = jr.split(key, n_hidden + 1)
        for i in range(n_hidden):
            layers.append(eqx.nn.Linear(in_features, hidden_size, key=keys[i]))
            in_features = hidden_size
        self.layers = layers

        # Output layer for position
        self.output_layer = eqx.nn.Linear(hidden_size, n_dims, key=keys[n_hidden])

    def __call__(self, gamma: FAny) -> Float[Array, "... D"]:
        """Forward pass through Param-Net.

        Parameters
        ----------
        gamma : Array
            Ordering parameter in [-1, 1], shape (...).

        Returns
        -------
        position : Array
            Reconstructed position of shape (..., n_dims).

        """
        # Handle batched input by vmapping over leading dimensions
        if gamma.ndim > 0:
            # vmap over all batch dimensions
            return jax.vmap(self._forward)(gamma)
        return self._forward(gamma)

    def _forward(self, gamma: FSz0) -> FSzD:
        """Forward pass for a single scalar input."""
        # Add feature dimension for linear layers
        x = gamma[..., None]

        for layer in self.layers:
            x = jax.nn.tanh(layer(x))

        return self.output_layer(x)


class Autoencoder(eqx.Module):
    r"""Autoencoder for stream tracer interpolation.

    Combines the InterpolationNetwork (encoder) and ParamNet (decoder)
    into a single autoencoder architecture that can:
    1. Predict $\gamma$ values for arbitrary phase-space points
    2. Reconstruct positions from $\gamma$ values
    3. Provide membership probabilities

    The autoencoder is trained in two phases:
    1. Train interpolation network on ordered tracers from NN+p
    2. Joint training with momentum condition to refine $\gamma$ values

    Parameters
    ----------
    rngs : PRNGKeyArray
        JAX random key for initialization.
    n_dims : int, optional
        Number of spatial dimensions. Default: 3.
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    Attributes
    ----------
    encoder : InterpolationNetwork
        The interpolation network mapping phase-space to $(\gamma, p)$.
    decoder : ParamNet
        The Param-Net mapping $\gamma$ to position.
    pos_mean : Array or None
        Mean of position data for standardization.
    pos_std : Array or None
        Std of position data for standardization.

    """

    encoder: InterpolationNetwork
    decoder: ParamNet
    n_dims: int = eqx.field(static=True)
    pos_mean: FSzD | None
    pos_std: FSzD | None
    vel_mean: FSzD | None
    vel_std: FSzD | None

    def __init__(
        self,
        rngs: PRNGKeyArray,
        n_dims: int = 3,
        hidden_size: int = 100,
        n_hidden: int = 3,
    ) -> None:
        self.n_dims = n_dims
        keys = jr.split(rngs, 2)

        self.encoder = InterpolationNetwork(keys[0], n_dims, hidden_size, n_hidden)
        self.decoder = ParamNet(keys[1], n_dims, hidden_size, n_hidden)

        # Standardization parameters (set during training)
        self.pos_mean = None
        self.pos_std = None
        self.vel_mean = None
        self.vel_std = None

    def encode(
        self, position: VectorComponents, velocity: VectorComponents
    ) -> tuple[FSzN, FSzN]:
        r"""Encode phase-space coordinates to $(\gamma, p)$.

        Parameters
        ----------
        position : VectorComponents
            Position dictionary with 1D array values.
        velocity : VectorComponents
            Velocity dictionary with 1D array values.

        Returns
        -------
        gamma : Array
            Ordering parameters in [-1, 1].
        prob : Array
            Membership probabilities in [0, 1].

        """
        # Stack to array
        keys = tuple(sorted(position.keys()))
        pos_arr = jnp.stack([position[k] for k in keys], axis=-1)
        vel_arr = jnp.stack([velocity[k] for k in keys], axis=-1)

        # Standardize if parameters are set
        if self.pos_mean is not None and self.pos_std is not None:
            pos_arr = (pos_arr - self.pos_mean) / (self.pos_std + 1e-8)
        if self.vel_mean is not None and self.vel_std is not None:
            vel_arr = (vel_arr - self.vel_mean) / (self.vel_std + 1e-8)

        # Concatenate position and velocity
        phase_space = jnp.concat([pos_arr, vel_arr], axis=-1)

        return self.encoder(phase_space)

    def decode(self, gamma: FSzN) -> FSzND:
        r"""Decode $\gamma$ to reconstructed position.

        Parameters
        ----------
        gamma : Array
            Ordering parameters in [-1, 1].

        Returns
        -------
        position : Array
            Reconstructed positions of shape (N, n_dims).
            Note: Returns standardized positions. Use decode_position()
            for unstandardized output.

        """
        return self.decoder(gamma)

    def decode_position(self, gamma: FSzN) -> VectorComponents:
        r"""Decode $\gamma$ to reconstructed position dictionary.

        This method handles unstandardization automatically.

        Parameters
        ----------
        gamma : Array
            Ordering parameters in [-1, 1].

        Returns
        -------
        position : VectorComponents
            Reconstructed position dictionary.

        """
        pos_arr = self.decode(gamma)

        # Unstandardize if parameters are set
        if self.pos_mean is not None and self.pos_std is not None:
            pos_arr = pos_arr * (self.pos_std + 1e-8) + self.pos_mean

        # Convert back to dict (assumes sorted keys)
        keys = [f"d{i}" for i in range(self.n_dims)]
        return {k: pos_arr[..., i] for i, k in enumerate(keys)}

    def predict(
        self, position: VectorComponents, velocity: VectorComponents
    ) -> tuple[FSzN, FSzN]:
        r"""Predict $\gamma$ and membership probability for phase-space points.

        Parameters
        ----------
        position : VectorComponents
            Position dictionary with 1D array values.
        velocity : VectorComponents
            Velocity dictionary with 1D array values.

        Returns
        -------
        gamma : Array
            Predicted ordering parameters in [-1, 1].
        prob : Array
            Predicted membership probabilities in [0, 1].

        """
        return self.encode(position, velocity)


class AutoencoderResult(NamedTuple):
    """Result of autoencoder training and prediction.

    Attributes
    ----------
    gamma : Array
        Ordering parameters for all tracers.
    membership_prob : Array
        Membership probabilities for all tracers.
    position : VectorComponents
        Original position data (dict with 1D arrays).
    velocity : VectorComponents
        Original velocity data (dict with 1D arrays).
    ordered_indices : Array
        Indices sorted by gamma value.

    """

    gamma: FSzN
    membership_prob: FSzN
    position: VectorComponents
    velocity: VectorComponents
    ordered_indices: ISzN


def _stack_phase_space(
    position: VectorComponents, velocity: VectorComponents
) -> tuple[FSzND, FSzND, tuple[str, ...]]:
    """Stack position and velocity dicts into arrays."""
    keys = tuple(sorted(position.keys()))
    pos_arr = jnp.stack([position[k] for k in keys], axis=-1)
    vel_arr = jnp.stack([velocity[k] for k in keys], axis=-1)
    return pos_arr, vel_arr, keys


def _compute_standardization(arr: FSzND) -> tuple[FSzD, FSzD]:
    """Compute mean and std for standardization."""
    return jnp.mean(arr, axis=0), jnp.std(arr, axis=0)


def _shuffle_and_batch(
    key: PRNGKeyArray,
    phase_space: Float[Array, "N 2D"],
    gamma_target: FSzN,
    prob_target: FSzN,
    mask: FSzN,
    batch_size: int,
) -> tuple[
    Float[Array, "B Bs 2D"],
    Float[Array, "B Bs"],
    Float[Array, "B Bs"],
    Float[Array, "B Bs"],
    int,
]:
    """Shuffle data and create padded batches for lax.scan.

    Returns arrays of shape (n_batches, batch_size, ...) suitable for lax.scan.
    The last batch is padded with zeros if needed.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for shuffling.
    phase_space, gamma_target, prob_target, mask : Array
        Training data arrays.
    batch_size : int
        Size of each batch.

    Returns
    -------
    batched_phase_space : Array of shape (n_batches, batch_size, D)
    batched_gamma : Array of shape (n_batches, batch_size)
    batched_prob : Array of shape (n_batches, batch_size)
    batched_mask : Array of shape (n_batches, batch_size)
    n_valid : int
        Number of valid (non-padded) samples in total.

    """
    n = phase_space.shape[0]

    # Shuffle indices
    perm = jr.permutation(key, n)
    phase_space = phase_space[perm]
    gamma_target = gamma_target[perm]
    prob_target = prob_target[perm]
    mask = mask[perm]

    # Compute number of batches (pad to make divisible by batch_size)
    n_batches = (n + batch_size - 1) // batch_size
    padded_n = n_batches * batch_size

    # Pad arrays
    pad_n = padded_n - n
    if pad_n > 0:
        phase_space = jnp.concat(
            [phase_space, jnp.zeros((pad_n, phase_space.shape[1]))], axis=0
        )
        gamma_target = jnp.concat([gamma_target, jnp.zeros(pad_n)], axis=0)
        prob_target = jnp.concat([prob_target, jnp.zeros(pad_n)], axis=0)
        mask = jnp.concat([mask, jnp.zeros(pad_n)], axis=0)

    # Reshape to (n_batches, batch_size, ...)
    batched_phase_space = phase_space.reshape(n_batches, batch_size, -1)
    batched_gamma = gamma_target.reshape(n_batches, batch_size)
    batched_prob = prob_target.reshape(n_batches, batch_size)
    batched_mask = mask.reshape(n_batches, batch_size)

    return batched_phase_space, batched_gamma, batched_prob, batched_mask, n


@dataclass
class TrainingConfig:
    r"""Configuration for autoencoder training.

    Parameters
    ----------
    learning_rate : float
        Learning rate for optimizer. Default: 1e-3.
    n_epochs : int
        Number of training epochs. Default: 500.
    batch_size : int
        Batch size for training. Default: 32.
    lambda_momentum : float
        Weight for momentum loss term ($\lambda$ in paper). Default: 100.0.
    n_random_samples : int
        Number of random samples for membership training. Default: 100.
    phase1_epochs : int
        Epochs for phase 1 (interpolation only). Default: 200.
    progress_bar : bool
        Whether to show a progress bar during training. Default: True.

    """

    learning_rate: float = 1e-3
    """Learning rate for optimizer."""

    n_epochs: int = 500
    """Number of training epochs."""

    batch_size: int = 32
    """Batch size for training."""

    lambda_momentum: float = 100.0
    """Weight for momentum loss term."""

    n_random_samples: int = 100
    """Number of random samples for membership training."""

    phase1_epochs: int = 200
    """Epochs for phase 1 (interpolation only)."""

    progress_bar: bool = True
    """Whether to show a progress bar during training."""


def _assign_gamma_init(
    ordered_indices: Int[Array, " N"],
    n_total: int,
    position: FSzND,
) -> FSzN:
    r"""Assign initial $\gamma$ values to ordered tracers with uniform spacing.

    Following Equation (A1) from the paper: > $\gamma_i \in [-1, 1]$. In
    practice, we divide this interval by the number > of stream tracers, such
    that $\gamma_{i+1} - \gamma_i$ is a positive constant.

    This means $\gamma$ values are uniformly spaced based on the ORDER of the
    tracers, NOT based on arc-length along the path. The order captures the
    topological structure of the stream, while uniform spacing allows the neural
    network to learn the non-uniform physical spacing.

    Parameters
    ----------
    ordered_indices : Array
        Array of indices from NN+p in order, with -1 for unvisited.
    n_total : int
        Total number of tracers.
    position : Array
        Position array of shape (N, n_dims). Not used (kept for API compat).

    Returns
    -------
    gamma : Array
        Initial $\gamma$ values (NaN for skipped tracers).

    """
    # Filter out unvisited indices (-1)
    valid_mask = ordered_indices >= 0
    valid_indices = ordered_indices[valid_mask]
    n_ordered = jnp.sum(valid_mask)

    if n_ordered <= 1:
        # Edge case: single point
        gamma = jnp.full(n_total, jnp.nan)
        # Only update if there's exactly one point
        first_idx = jnp.where(valid_mask, ordered_indices, -1)[0]
        gamma = jnp.where(n_ordered == 1, gamma.at[first_idx].set(0.0), gamma)
        return gamma  # noqa: RET504

    # Uniform spacing from -1 to 1 based on ORDER (not arc-length!)
    # Per paper: "divide this interval by the number of stream tracers,
    # such that γ_{i+1} - γ_i is a positive constant"
    gamma_values = jnp.linspace(-1.0, 1.0, n_ordered)

    # Initialize with NaN for skipped tracers
    gamma = jnp.full(n_total, jnp.nan)

    # Assign γ values to ordered tracers using vectorized operation
    gamma = gamma.at[valid_indices].set(gamma_values)

    return gamma  # noqa: RET504


def _interpolation_loss(
    encoder: InterpolationNetwork,
    phase_space: FSzND,
    gamma_target: FSzN,
    prob_target: FSzN,
    mask: FSzN,
    is_random: FSzN | None = None,
) -> FSz0:
    r"""Compute loss for interpolation network training.

    Loss from Equation (A2):
    $L = \sum_i |\gamma_\theta(w_i) - \gamma_{\text{init},i}|^2 + |p_\theta(w_i) - 1|^2
      + \sum_{\text{rand}} |p_\theta(w_{\text{rand}}) - 0|^2$

    IMPORTANT: The probability loss is only applied to:
    1. Ordered tracers (prob_target=1)
    2. Random samples (prob_target=0)

    Skipped tracers (mask=0 but not random) do NOT have probability loss.
    Their probability is learned indirectly through the momentum condition
    and reconstruction loss.

    Parameters
    ----------
    encoder : InterpolationNetwork
        The interpolation network.
    phase_space : Array
        Phase-space coordinates of shape (N, 2*n_dims).
    gamma_target : Array
        Target $\gamma$ values for ordered tracers.
    prob_target : Array
        Target probability (1 for ordered, 0 for random).
    mask : Array
        Mask for valid (ordered) tracers.
    is_random : Array, optional
        Mask indicating which points are random samples (1) vs actual tracers (0).
        If None, all points with mask=0 are assumed to be random samples.

    Returns
    -------
    loss : Array
        Scalar loss value.

    """
    gamma_pred, prob_pred = encoder(phase_space)

    # γ loss only for ordered tracers
    gamma_loss = jnp.where(mask > 0.5, (gamma_pred - gamma_target) ** 2, 0.0)

    # Probability loss for ordered (prob=1) and random (prob=0) only
    # NOT for skipped tracers (mask=0 but not random)
    if is_random is None:
        # Backward compatibility: assume mask=0 means random sample
        prob_loss = (prob_pred - prob_target) ** 2
    else:
        # Only apply prob loss to ordered (mask=1) or random (is_random=1)
        prob_loss_mask = (mask > 0.5) | (is_random > 0.5)
        prob_loss = jnp.where(prob_loss_mask, (prob_pred - prob_target) ** 2, 0.0)

    return jnp.mean(gamma_loss) + jnp.mean(prob_loss)


def _momentum_loss(
    autoencoder: Autoencoder,
    phase_space: FSzND,
    velocity: FSzND,
    pos_std: FSzD,
    dgamma: float = 0.01,
) -> FSz0:
    r"""Compute momentum loss for joint training.

    The momentum condition from Equation (A3) encourages the tangent
    to the decoded track ($dx/d\gamma$) to align with the velocity direction.
    This is crucial for properly ordering ALL tracers, including skipped ones.

    The loss is computed on ALL tracers (not just ordered ones) so that the
    network learns to predict correct $\gamma$ values for skipped tracers.

    Parameters
    ----------
    autoencoder : Autoencoder
        The autoencoder model.
    phase_space : Array
        Phase-space coordinates (standardized).
    velocity : Array
        Velocity vectors (unstandardized, physical units).
    pos_std : Array
        Position standard deviations for unstandardizing the decoder output.
    dgamma : float
        Step size for numerical derivative.

    Returns
    -------
    loss : Array
        Scalar momentum loss.

    """
    gamma_pred, _ = autoencoder.encoder(phase_space)

    # Compute dx/dγ numerically using the decoder
    pos_at_gamma_plus = autoencoder.decoder(gamma_pred + dgamma)
    pos_at_gamma_minus = autoencoder.decoder(gamma_pred - dgamma)

    # Use central difference for better numerical accuracy
    # The decoder outputs standardized positions, so we need to unstandardize
    # to get directions in physical space
    dx_dgamma_std = (pos_at_gamma_plus - pos_at_gamma_minus) / (2 * dgamma)

    # Unstandardize the direction (multiply by std to get physical units)
    # Note: We only care about direction, but different std values in each
    # dimension change the direction, so we must unstandardize
    pos_std_safe = pos_std + 1e-8
    dx_dgamma = dx_dgamma_std * pos_std_safe

    # Normalize to get unit tangent direction
    dx_norm = jnp.sqrt(jnp.sum(dx_dgamma**2, axis=-1, keepdims=True))
    T_hat = dx_dgamma / (dx_norm + 1e-8)

    # Normalize velocity to get unit velocity direction
    vel_norm = jnp.sqrt(jnp.sum(velocity**2, axis=-1, keepdims=True))
    v_hat = velocity / (vel_norm + 1e-8)

    # Use 1 - |cos(θ)| as loss to encourage alignment (either direction)
    # This is more robust than ||T_hat - v_hat||² which penalizes antiparallel
    cos_sim = jnp.sum(T_hat * v_hat, axis=-1)
    alignment_loss = 1.0 - jnp.abs(cos_sim)

    return jnp.mean(alignment_loss)


def _reconstruction_loss(
    decoder: ParamNet,
    gamma_target: FSzN,
    mask: FSzN,
    position: FSzND,
    pos_mean: FSzD,
    pos_std: FSzD,
) -> FSz0:
    r"""Compute reconstruction loss for the decoder only.

    IMPORTANT: This uses $\gamma_{\text{target}}$ (from walk ordering), NOT
    encoder's $\gamma_{\text{pred}}$.  This ensures the encoder learns to
    predict $\gamma_{\text{init}}$ via interpolation loss, while the decoder
    independently learns $\gamma \to \text{position}$ mapping.

    If we used $\gamma_{\text{pred}}$ from the encoder, the decoder's inability
    to reconstruct complex curves would corrupt the encoder's predictions
    through backprop.

    Parameters
    ----------
    decoder : ParamNet
        The decoder network.
    gamma_target : Array
        Target $\gamma$ values from walk ordering ($\gamma_{\text{init}}$).
    mask : Array
        Mask for valid (ordered) tracers.
    position : Array
        Target position coordinates.
    pos_mean, pos_std : Array
        Standardization parameters.

    Returns
    -------
    loss : Array
        Scalar reconstruction loss.

    """
    # Only compute reconstruction loss for ordered tracers (where we have gamma_target)
    pos_recon = decoder(gamma_target)

    # Standardize target for comparison
    pos_std_safe = pos_std + 1e-8
    position_standardized = (position - pos_mean) / pos_std_safe

    # Only count loss for ordered tracers (mask > 0.5)
    recon_err = (pos_recon - position_standardized) ** 2
    masked_err = jnp.where(mask[:, None] > 0.5, recon_err, 0.0)

    # Average over ordered tracers only
    n_ordered = jnp.sum(mask > 0.5) + 1e-8
    return jnp.sum(masked_err) / (n_ordered * position.shape[-1])


def _phase1_batch_step_impl(
    model: Autoencoder,
    opt_state: optax.OptState,
    batch_phase_space: FSzND,
    batch_gamma_target: FSzN,
    batch_prob_target: FSzN,
    batch_mask: FSzN,
    key: PRNGKeyArray,
    config: "TrainingConfig",
    n_dims: int,
    pos_mean: FSzD,
    pos_std: FSzD,
    vel_mean: FSzD,
    vel_std: FSzD,
    pos_std_safe: FSzD,
    vel_std_safe: FSzD,
    pos_min: FSzD,
    pos_max: FSzD,
    vel_min: FSzD,
    vel_max: FSzD,
    optimizer: optax.GradientTransformation,
) -> tuple[Autoencoder, optax.OptState, FSz0]:
    """Single batched training step for phase 1 (extracted for reuse).

    This function is defined at module level (not inside train_autoencoder) to
    avoid redefining it every epoch, which provides ~3-4% speedup.
    """
    # Generate random samples for membership training
    rand_key, _ = jr.split(key)
    n_rand = config.n_random_samples

    rand_keys = jr.split(rand_key, 2)
    rand_pos = jr.uniform(
        rand_keys[0], (n_rand, n_dims), minval=pos_min, maxval=pos_max
    )
    rand_vel = jr.uniform(
        rand_keys[1], (n_rand, n_dims), minval=vel_min, maxval=vel_max
    )

    # Standardize random samples
    rand_pos_std = (rand_pos - pos_mean) / pos_std_safe
    rand_vel_std = (rand_vel - vel_mean) / vel_std_safe
    rand_phase_space = jnp.concat([rand_pos_std, rand_vel_std], axis=-1)

    # Combined batch + random data
    all_phase_space = jnp.concat([batch_phase_space, rand_phase_space], axis=0)
    all_gamma_target = jnp.concat([batch_gamma_target, jnp.zeros(n_rand)], axis=0)
    all_prob_target = jnp.concat([batch_prob_target, jnp.zeros(n_rand)], axis=0)
    all_mask = jnp.concat([batch_mask, jnp.zeros(n_rand)], axis=0)

    # Unstandardize position for reconstruction loss
    batch_position = batch_phase_space[..., :n_dims] * pos_std_safe + pos_mean

    def loss_fn(m: Autoencoder) -> FSz0:
        interp_loss = _interpolation_loss(
            m.encoder,
            all_phase_space,
            all_gamma_target,
            all_prob_target,
            all_mask,
        )
        # Also train decoder in Phase 1 (critical for Phase 2 stability)
        # Use gamma_target (from walk ordering), NOT encoder's gamma_pred
        # This prevents decoder limitations from corrupting encoder learning
        recon_loss = _reconstruction_loss(
            m.decoder,
            batch_gamma_target,
            batch_mask,
            batch_position,
            pos_mean,
            pos_std,
        )
        return interp_loss + recon_loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def _phase2_batch_step_impl(
    model: Autoencoder,
    opt_state: optax.OptState,
    batch_phase_space: FSzND,
    batch_gamma_target: FSzN,
    batch_prob_target: FSzN,
    batch_mask: FSzN,
    key: PRNGKeyArray,
    config: "TrainingConfig",
    n_dims: int,
    pos_mean: FSzD,
    pos_std: FSzD,
    vel_mean: FSzD,
    vel_std: FSzD,
    pos_std_safe: FSzD,
    vel_std_safe: FSzD,
    pos_min: FSzD,
    pos_max: FSzD,
    vel_min: FSzD,
    vel_max: FSzD,
    optimizer: optax.GradientTransformation,
) -> tuple[Autoencoder, optax.OptState, FSz0]:
    r"""Single batched training step for phase 2 (extracted for reuse).

    This function is defined at module level (not inside train_autoencoder) to
    avoid redefining it every epoch, which provides ~3-4% speedup.

    Phase 2 includes:
    - Interpolation loss (to maintain $\gamma$ predictions)
    - Reconstruction loss (decoder accuracy)
    - Momentum loss (tangent-velocity alignment)
    """
    # Generate random samples
    rand_key, _ = jr.split(key)
    n_rand = config.n_random_samples

    rand_keys = jr.split(rand_key, 2)
    rand_pos = jr.uniform(
        rand_keys[0], (n_rand, n_dims), minval=pos_min, maxval=pos_max
    )
    rand_vel = jr.uniform(
        rand_keys[1], (n_rand, n_dims), minval=vel_min, maxval=vel_max
    )

    # Standardize random samples
    rand_pos_std = (rand_pos - pos_mean) / pos_std_safe
    rand_vel_std = (rand_vel - vel_mean) / vel_std_safe
    rand_phase_space = jnp.concat([rand_pos_std, rand_vel_std], axis=-1)

    # Unstandardize position and velocity for phase 2 losses
    batch_position = batch_phase_space[..., :n_dims] * pos_std_safe + pos_mean
    batch_velocity = batch_phase_space[..., n_dims:] * vel_std_safe + vel_mean

    def loss_fn(m: Autoencoder, /) -> FSz0:
        # Interpolation loss - same as phase 1 (includes random samples)
        n_batch = batch_phase_space.shape[0]

        all_phase_space = jnp.concat([batch_phase_space, rand_phase_space], axis=0)
        all_gamma_target = jnp.concat([batch_gamma_target, jnp.zeros(n_rand)], axis=0)
        all_prob_target = jnp.concat([batch_prob_target, jnp.zeros(n_rand)], axis=0)
        all_mask = jnp.concat([batch_mask, jnp.zeros(n_rand)], axis=0)
        # Mark which points are random samples (prob loss applies)
        # Batch points are NOT random, even if mask=0 (those are skipped tracers)
        all_is_random = jnp.concat([jnp.zeros(n_batch), jnp.ones(n_rand)], axis=0)
        interp_loss = _interpolation_loss(
            m.encoder,
            all_phase_space,
            all_gamma_target,
            all_prob_target,
            all_mask,
            is_random=all_is_random,
        )

        # Reconstruction loss (on batch data only, not random)
        # Use gamma_target (from walk ordering), NOT encoder's gamma_pred
        recon_loss = _reconstruction_loss(
            m.decoder,
            batch_gamma_target,
            batch_mask,
            batch_position,
            pos_mean,
            pos_std,
        )

        # Momentum loss on ALL tracers (including skipped) - this is the key
        # to learning correct γ values for skipped tracers
        mom_loss = _momentum_loss(m, batch_phase_space, batch_velocity, pos_std)

        return interp_loss + recon_loss + config.lambda_momentum * mom_loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def train_autoencoder(
    autoencoder: Autoencoder,
    localflowwalk_result: LocalFlowWalkResult,
    config: TrainingConfig | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Autoencoder, Float[Array, " E"]]:
    """Train the autoencoder on NN+p results.

    Training follows the two-phase process from Appendix A.2:
    1. Train interpolation network on ordered tracers
    2. Joint training with momentum condition

    Parameters
    ----------
    autoencoder : Autoencoder
        The autoencoder model to train.
    localflowwalk_result : LocalFlowWalkResult
        Result from walk_local_flow.
    config : TrainingConfig, optional
        Training configuration. Default: TrainingConfig().
    key : PRNGKeyArray, optional
        Random key for training. Default: jax.random.PRNGKey(0).

    Returns
    -------
    trained_autoencoder : Autoencoder
        The trained autoencoder model.
    losses : Array
        Training losses per epoch.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)
    >>> autoencoder = lfw.nn.Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=2)
    >>> trained, losses = lfw.nn.train_autoencoder(autoencoder, result)

    """
    if config is None:
        config = TrainingConfig()
    if key is None:
        key = jr.PRNGKey(0)

    position = localflowwalk_result.positions
    velocity = localflowwalk_result.velocities
    ordered_indices = localflowwalk_result.ordered_indices

    # Stack data
    pos_arr, vel_arr, keys = _stack_phase_space(position, velocity)
    n_total = pos_arr.shape[0]
    n_dims = pos_arr.shape[1]

    # Compute standardization parameters
    pos_mean, pos_std = _compute_standardization(pos_arr)
    vel_mean, vel_std = _compute_standardization(vel_arr)

    # Store standardization in autoencoder using functional update (Equinox style)
    autoencoder = eqx.tree_at(
        lambda m: (m.pos_mean, m.pos_std, m.vel_mean, m.vel_std),
        autoencoder,
        (pos_mean, pos_std, vel_mean, vel_std),
        is_leaf=lambda x: x is None,
    )

    # Standardize data
    pos_std_safe = pos_std + 1e-8
    vel_std_safe = vel_std + 1e-8
    pos_standardized = (pos_arr - pos_mean) / pos_std_safe
    vel_standardized = (vel_arr - vel_mean) / vel_std_safe
    phase_space = jnp.concat([pos_standardized, vel_standardized], axis=-1)

    # Assign initial γ values based on arc-length
    gamma_init = _assign_gamma_init(ordered_indices, n_total, pos_arr)
    mask = ~jnp.isnan(gamma_init)
    gamma_init_safe = jnp.where(mask, gamma_init, 0.0)

    # Probability targets: 1 for ordered, 0 for random samples
    prob_target = mask.astype(jnp.float32)

    # Prepare Phase 1 data: ONLY ordered tracers
    # Per the paper (Appendix A.2), Phase 1 trains only on ordered tracers (p=1)
    # and random samples (p=0). Skipped tracers are NOT included in Phase 1.
    ordered_mask_bool = mask.astype(bool)
    phase1_phase_space = phase_space[ordered_mask_bool]
    phase1_gamma = gamma_init_safe[ordered_mask_bool]
    phase1_prob = prob_target[ordered_mask_bool]
    phase1_mask = mask[ordered_mask_bool].astype(jnp.float32)

    # Phase 2 data: ALL tracers
    phase2_phase_space = phase_space
    phase2_gamma = gamma_init_safe
    phase2_prob = prob_target
    phase2_mask = mask.astype(jnp.float32)

    # Pre-compute data bounds ONCE (not per batch)
    pos_min = jnp.min(pos_arr, axis=0)
    pos_max = jnp.max(pos_arr, axis=0)
    vel_min = jnp.min(vel_arr, axis=0)
    vel_max = jnp.max(vel_arr, axis=0)

    # Create optax optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(autoencoder, eqx.is_array))

    losses = []

    # JIT-compile the batch step functions (defined outside loop for speed)
    phase1_step_jit = eqx.filter_jit(
        lambda model,
        state,
        b_phase,
        b_gamma,
        b_prob,
        b_mask,
        key: _phase1_batch_step_impl(
            model,
            state,
            b_phase,
            b_gamma,
            b_prob,
            b_mask,
            key,
            config,
            n_dims,
            pos_mean,
            pos_std,
            vel_mean,
            vel_std,
            pos_std_safe,
            vel_std_safe,
            pos_min,
            pos_max,
            vel_min,
            vel_max,
            optimizer,
        )
    )
    phase2_step_jit = eqx.filter_jit(
        lambda model,
        state,
        b_phase,
        b_gamma,
        b_prob,
        b_mask,
        key: _phase2_batch_step_impl(
            model,
            state,
            b_phase,
            b_gamma,
            b_prob,
            b_mask,
            key,
            config,
            n_dims,
            pos_mean,
            pos_std,
            vel_mean,
            vel_std,
            pos_std_safe,
            vel_std_safe,
            pos_min,
            pos_max,
            vel_min,
            vel_max,
            optimizer,
        )
    )

    # Training loop
    epoch_iter = range(config.n_epochs)
    if config.progress_bar:
        epoch_iter = tqdm(
            epoch_iter,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )

    def _phase1_scan_fn(
        carry: tuple[Autoencoder, optax.OptState],
        batch: tuple[FSzND, FSzN, FSzN, FSzN, PRNGKeyArray],
    ) -> tuple[tuple[Autoencoder, optax.OptState], FSz0]:
        model, state = carry
        b_phase, b_gamma, b_prob, b_mask, b_key = batch
        new_model, new_state, loss = phase1_step_jit(
            model, state, b_phase, b_gamma, b_prob, b_mask, b_key
        )
        return (new_model, new_state), loss

    def _phase2_scan_fn(
        carry: tuple[Autoencoder, optax.OptState],
        batch: tuple[FSzND, FSzN, FSzN, FSzN, PRNGKeyArray],
    ) -> tuple[tuple[Autoencoder, optax.OptState], FSz0]:
        model, state = carry
        b_phase, b_gamma, b_prob, b_mask, b_key = batch
        new_model, new_state, loss = phase2_step_jit(
            model, state, b_phase, b_gamma, b_prob, b_mask, b_key
        )
        return (new_model, new_state), loss

    for epoch in epoch_iter:
        key, epoch_key = jr.split(key)

        # Choose appropriate data for current phase
        if epoch < config.phase1_epochs:
            # Phase 1: Only ordered tracers
            (
                batched_phase_space,
                batched_gamma,
                batched_prob,
                batched_mask,
                _,
            ) = _shuffle_and_batch(
                epoch_key,
                phase1_phase_space,
                phase1_gamma,
                phase1_prob,
                phase1_mask,
                config.batch_size,
            )
            n_batches = batched_phase_space.shape[0]

            # Generate keys for each batch
            key, keys_key = jr.split(key)
            batch_keys = jr.split(keys_key, n_batches)

            # Run batches with JIT-compiled step function using lax.scan
            batch_inputs = (
                batched_phase_space,
                batched_gamma,
                batched_prob,
                batched_mask,
                batch_keys,
            )

            (autoencoder, opt_state), batch_losses = jax.lax.scan(
                _phase1_scan_fn, (autoencoder, opt_state), batch_inputs
            )
        else:
            # Phase 2: All tracers
            (
                batched_phase_space,
                batched_gamma,
                batched_prob,
                batched_mask,
                _,
            ) = _shuffle_and_batch(
                epoch_key,
                phase2_phase_space,
                phase2_gamma,
                phase2_prob,
                phase2_mask,
                config.batch_size,
            )
            n_batches = batched_phase_space.shape[0]

            # Generate keys for each batch
            key, keys_key = jr.split(key)
            batch_keys = jr.split(keys_key, n_batches)

            # Run batches with JIT-compiled step function using lax.scan
            batch_inputs = (
                batched_phase_space,
                batched_gamma,
                batched_prob,
                batched_mask,
                batch_keys,
            )

            (autoencoder, opt_state), batch_losses = jax.lax.scan(
                _phase2_scan_fn, (autoencoder, opt_state), batch_inputs
            )

        # Average loss for this epoch
        avg_loss = float(jnp.mean(batch_losses))
        losses.append(avg_loss)

        # Update progress bar with loss info
        if config.progress_bar and hasattr(epoch_iter, "set_postfix"):
            phase = "Phase 1" if epoch < config.phase1_epochs else "Phase 2"
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", phase=phase)

    return autoencoder, jnp.array(losses)


def fill_ordering_gaps(
    autoencoder: Autoencoder,
    localflowwalk_result: LocalFlowWalkResult,
    prob_threshold: float = 0.5,
) -> AutoencoderResult:
    r"""Use trained autoencoder to fill gaps in NN+p ordering.

    This function predicts $\gamma$ values for all tracers (including those
    skipped by NN+p) and returns a complete ordering.

    Parameters
    ----------
    autoencoder : Autoencoder
        Trained autoencoder model.
    localflowwalk_result : LocalFlowWalkResult
        Result from walk_local_flow.
    prob_threshold : float, optional
        Minimum membership probability to include. Default: 0.5.

    Returns
    -------
    result : AutoencoderResult
        Complete ordering including previously skipped tracers.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import localflowwalk as lfw

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)
    >>> autoencoder = lfw.nn.Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=2)
    >>> trained, _ = lfw.nn.train_autoencoder(autoencoder, result)
    >>> full_ordering = lfw.nn.fill_ordering_gaps(trained, result)

    """
    q = localflowwalk_result.positions
    p = localflowwalk_result.velocities

    # Predict gamma and probability for all tracers
    gamma, prob = autoencoder.predict(q, p)
    # Sort by gamma to get ordering
    sorted_indices = jnp.argsort(gamma)

    # Filter by probability threshold
    high_prob_mask = prob[sorted_indices] >= prob_threshold
    filtered_indices = sorted_indices[high_prob_mask]

    return AutoencoderResult(
        gamma=gamma,
        membership_prob=prob,
        position=dict(q),
        velocity=dict(p),
        ordered_indices=filtered_indices,
    )
