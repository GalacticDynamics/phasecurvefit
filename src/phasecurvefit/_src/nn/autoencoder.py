"""Autoencoder."""

__all__: tuple[str, ...] = ("PathAutoencoder", "train_autoencoder", "TrainingConfig")

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jtu
import jax_tqdm
import optax
import plum
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from .abstractautoencoder import AbstractAutoencoder
from .externaldecoder import RunningMeanDecoder
from .normalize import AbstractNormalizer
from .order_net import (
    OrderingNet,
    OrderingTrainingConfig,
    default_optimizer,
    train_ordering_net,
)
from .result import AutoencoderResult
from .track_net import TrackNet, TrackTrainingConfig, decoder_loss, train_track_net
from .utils import shuffle_and_batch
from phasecurvefit._src.algorithm import WalkLocalFlowResult
from phasecurvefit._src.custom_types import FLikeSz0, FSz0, FSzN

Gamma: TypeAlias = FSzN  # noqa: UP040


class PathAutoencoder(AbstractAutoencoder):
    r"""Autoencoder combining OrderingNet and TrackNet.

    This autoencoder is trained to assign $\gamma$ values to stream tracers
    that were skipped by the phase-flow walk algorithm. It consists of two
    parts:

    1. **Interpolation Network**: Maps phase-space coordinates $(x, v) \to (\gamma,
       p)$ where $\gamma \in [0, 1]$ is the ordering parameter and $p \in [0, 1]$
       is the membership probability.
    2. **Param-Net (Decoder)**: Maps $\gamma \to x$, reconstructing the position
       from the ordering parameter.

    """

    encoder: OrderingNet
    decoder: TrackNet
    normalizer: AbstractNormalizer

    __citation__: ClassVar[str] = (
        "https://ui.adsabs.harvard.edu/abs/2022ApJ...940...22N/abstract"
    )

    @property
    def gamma_range(self) -> tuple[float, float]:
        """Return the gamma range from the encoder."""
        return self.encoder.gamma_range

    @classmethod
    def make(
        cls,
        normalizer: AbstractNormalizer,
        *,
        gamma_range: tuple[float, float],
        ordering_width_size: int = 100,
        ordering_depth: int = 2,
        track_width_size: int = 128,
        track_depth: int = 3,
        key: PRNGKeyArray,
    ) -> "PathAutoencoder":
        key_encode, key_decode = jr.split(key)
        encoder = OrderingNet(
            in_size=2 * normalizer.n_spatial_dims,
            width_size=ordering_width_size,
            depth=ordering_depth,
            gamma_range=gamma_range,
            key=key_encode,
        )
        decoder = TrackNet(
            out_size=normalizer.n_spatial_dims,
            width_size=track_width_size,
            depth=track_depth,
            key=key_decode,
        )
        return cls(encoder=encoder, decoder=decoder, normalizer=normalizer)


# ============================================================


@eqx.filter_jit
def compute_weights(
    model: OrderingNet,
    ws: Float[Array, "N TwoF"],
    *,
    bandwidth: float = 0.02,
    key: PRNGKeyArray | None = None,
) -> Float[Array, " N"]:
    r"""Compute inverse density weights for phase-space samples.

    Uses Gaussian kernel density estimation (KDE) on predicted $\gamma$ values
    to compute sample weights inversely proportional to density. This provides
    importance weighting that upweights rare regions of the stream.

    The algorithm:
    1. Predict $\gamma$ values for all samples using the OrderingNet
    2. Compute KDE density at each $\gamma$ value
    3. Return inverse density as weights: $w_i = 1 / \text{density}(\gamma_i)$

    This matches the PyTorch implementation where samples in sparse regions
    of $\gamma$-space receive higher weights.

    Parameters
    ----------
    model : OrderingNet
        Trained interpolation network for predicting gamma values.
    ws : Array, shape (N, 2*n_dims)
        Phase-space coordinates (position + velocity).
    bandwidth : float, optional
        Gaussian kernel bandwidth for KDE. Default: 0.02.
    key : PRNGKeyArray, optional
        Random key for any stochastic operations (not used here, but included
        for signature consistency).

    Returns
    -------
    weights : Array, shape (N,)
        Inverse density weights for each sample.

    Notes
    -----
    The KDE density at point $\gamma_i$ is:

    $$ \hat{f}(\gamma_i) = \frac{1}{Nh} \sum_{j=1}^N
        K\left(\frac{\gamma_i - \gamma_j}{h}\right) $$

    where $K$ is the Gaussian kernel and $h$ is the bandwidth.

    """
    # Predict gamma values (only need gamma, not probability)
    gamma_predict, _ = jax.vmap(model, (0, None))(ws, key)

    # Compute pairwise distances in gamma space
    # Shape: (N, N) where entry (i, j) is |gamma_i - gamma_j|
    diff = gamma_predict[:, None] - gamma_predict[None, :]

    # Gaussian kernel: K(u) = exp(-0.5 * u^2) / sqrt(2π)
    # Normalization constant cancels when computing inverse weights
    kernel_vals = jnp.exp(-0.5 * (diff / bandwidth) ** 2)

    # KDE density estimate: mean of kernel evaluations
    density = jnp.mean(kernel_vals, axis=1)  # Shape: (N,)

    # Inverse density weights
    weights = 1.0 / density

    return weights  # noqa: RET504


@eqx.filter_jit
def compute_uniform_weights(
    model: OrderingNet,
    ws: Float[Array, " N TwoF"],
    *,
    bandwidth: float = -1,
    key: PRNGKeyArray | None = None,
) -> Float[Array, " N"]:
    """Compute uniform weights (all ones) for phase-space samples.

    Returns an array of ones with the same length as the input. This function
    has the same signature as `compute_weights` so it can be used as an
    alternative branch in `jax.lax.cond`.

    Parameters
    ----------
    model : OrderingNet
        Interpolation network (unused, but required for signature matching).
    ws : Array, shape (N, 2*n_dims)
        Phase-space coordinates (position + velocity).
    bandwidth : float, optional
        Kernel bandwidth (unused, but required for signature matching).
    key : PRNGKeyArray, optional
        Random key (unused, but required for signature matching).

    Returns
    -------
    weights : Array, shape (N,)
        Array of ones with length N.

    """
    del model, bandwidth, key  # Unused parameters for signature matching
    return jnp.ones(ws.shape[0], dtype=ws.dtype)


@dataclass
class EncoderDecoderTrainingConfig:
    r"""Configuration for Encoder + Decoder training."""

    _: KW_ONLY

    optimizer: optax.GradientTransformation = default_optimizer
    """Optax optimizer for training."""

    n_epochs: int = 200
    """Number of epochs for training."""

    batch_size: int = 100
    """Batch size for training."""

    lambda_q: float = 1.0
    """Weight for spatial reconstruction loss."""

    lambda_p: tuple[float, float] = (1.0, 5.0)
    """Weight schedule (start, stop) for velocity alignment loss."""

    member_threshold: float = 0.5
    """Membership p > threshold for identifying stream members."""

    freeze_encoder: bool = False
    """Whether to freeze the encoder during phase 2 training."""

    weight_by_density: bool | Mapping[str, object] = False
    """Whether to inverse density weight the samples. USE WITH CARE."""

    show_pbar: bool = True
    """Show an epoch progress bar via `tqdm`."""


@dataclass
class TrainingConfig:
    r"""Configuration for three-phase autoencoder training."""

    _: KW_ONLY

    # -------------------------------
    # Common configurations

    optimizer: optax.GradientTransformation = default_optimizer
    """Optax optimizer for training."""

    batch_size: int = 100
    """Batch size for training."""

    show_pbar: bool = True
    """Show an epoch progress bar via `tqdm`."""

    member_threshold: float = EncoderDecoderTrainingConfig.member_threshold
    """Membership p > threshold for identifying stream members."""

    # -------------------------------
    # Encoder-only training

    n_epochs_encoder: int = OrderingTrainingConfig.n_epochs
    """Number of epochs for Phase 1 training (OrderingNet)"""

    lambda_prob: float = OrderingTrainingConfig.lambda_prob
    """Weight for probability loss terms."""

    # -------------------------------
    # Decoder-only training

    n_epochs_decoder: int = TrackTrainingConfig.n_epochs
    """Number of epochs for Phase 2 training (TrackNet)"""

    # -------------------------------
    # Encoder + Decoder training

    n_epochs_both: int = EncoderDecoderTrainingConfig.n_epochs
    """Number of epochs for Phase 2 training (TrackNet)"""

    lambda_q: float = EncoderDecoderTrainingConfig.lambda_q
    """Weight for phase-2 spatial training."""

    lambda_p: tuple[float, float] = EncoderDecoderTrainingConfig.lambda_p
    """Weight range for phase-2 velocity training."""

    weight_by_density: bool | Mapping[str, object] = (
        EncoderDecoderTrainingConfig.weight_by_density
    )
    """Whether to inverse density weight the samples. USE WITH CARE."""

    freeze_encoder_final_training: bool = EncoderDecoderTrainingConfig.freeze_encoder
    """Whether to freeze the encoder during phase 2 training."""

    # =================================

    def __post_init__(self) -> None:
        pass

    @property
    def n_epochs(self) -> int:
        """Return the total number of epochs."""
        return self.n_epochs_encoder + self.n_epochs_decoder + self.n_epochs_both

    def encoderonly_config(self) -> OrderingTrainingConfig:
        """Construct the OrderingNet config."""
        return OrderingTrainingConfig(
            optimizer=self.optimizer,
            n_epochs=self.n_epochs_encoder,
            batch_size=self.batch_size,
            lambda_prob=self.lambda_prob,
            show_pbar=self.show_pbar,
        )

    def decoderonly_config(self) -> TrackTrainingConfig:
        """Construct the TrackNet config."""
        return TrackTrainingConfig(
            optimizer=self.optimizer,
            n_epochs=self.n_epochs_decoder,
            batch_size=self.batch_size,
            show_pbar=self.show_pbar,
        )

    def autoencoder_config(self) -> EncoderDecoderTrainingConfig:
        """Construct the Autoencoder config."""
        return EncoderDecoderTrainingConfig(
            optimizer=self.optimizer,
            n_epochs=self.n_epochs_both,
            batch_size=self.batch_size,
            show_pbar=self.show_pbar,
            lambda_q=self.lambda_q,
            lambda_p=self.lambda_p,
            member_threshold=self.member_threshold,
            freeze_encoder=self.freeze_encoder_final_training,
            weight_by_density=self.weight_by_density,
        )


@eqx.filter_value_and_grad
def compute_decoder_loss(
    model_dynamic: PathAutoencoder,
    model_static: PathAutoencoder,
    ws: Float[Array, " B TwoF"],
    weights: Float[Array, " B"],
    mask: Bool[Array, " B"],
    *,
    lambda_q: FLikeSz0,
    lambda_p: FLikeSz0,
    member_threshold: float,
    key: PRNGKeyArray,
) -> FSz0:
    r"""Compute decoder loss with gradients for Phase 2 training.

    This function computes the combined loss for spatial reconstruction and
    velocity alignment in the decoder training phase. It is decorated with
    `@eqx.filter_value_and_grad` to return both the loss value and gradients in
    a single pass.

    The loss combines two terms:

    1. Spatial reconstruction: L2 distance between measured and predicted positions
    2. Velocity alignment: Alignment between velocity direction and decoder tangent

    The decoder tangent $\hat{t}$ is computed from $\frac{\partial q}{\partial
    \gamma}$, the Jacobian of the decoder output with respect to $\gamma$,
    normalized to unit length. This represents the predicted direction of the
    stream at each point.

    """
    # Reconstruct full model from dynamic and static parts
    model = eqx.combine(model_dynamic, model_static)

    # Get phase-space q,p separation index
    if ws.shape[1] % 2 != 0:
        msg = "ord_w has the wrong shape"
        raise ValueError(msg)
    D = ws.shape[1] // 2

    # Unit velocity
    ps = ws[:, D:]
    ps_norm = jnp.linalg.norm(ps, axis=1, keepdims=True)
    ps_hat = jnp.where(ps_norm > 0, ps / ps_norm, jnp.zeros_like(ps))

    # Compute q_predict by w -encoder-> gamma -decoder-> q
    key, skey1, skey2 = jr.split(key, 3)
    gamma_predict, member_prob = jax.vmap(model.encoder, (0, None))(ws, skey1)
    q_predict = jax.vmap(model.decoder, (0, None))(gamma_predict, skey2)

    # Penalize mislabeling encoder members.
    is_member = member_prob > member_threshold
    # This drives member_prob to be > member_threshold for encoder members.
    false_negatives = mask & ~is_member
    loss_falseneg = jnp.mean(jnp.where(false_negatives, 1 - member_prob, 0.0))
    # This drives member_prob to be < member_threshold for encoder non-members.
    false_positives = ~mask & is_member
    loss_falsepos = jnp.mean(jnp.where(false_positives, member_prob, 0.0))
    # Total mismembership loss
    loss_mismember = loss_falseneg + loss_falsepos

    # Modify the mask to avoid training on non-members. This actually can't add
    # members (that would be an 'or' operation) and we penalize false positives
    # and negatives above, so this has a smaller effect.
    mask = mask & is_member

    # Compute dq/dgamma (Jacobian of decoder output w.r.t. gamma) elementwise
    # jax.jacobian gives us the derivative of decoder output w.r.t. its input
    # We vmap to compute this for each sample independently.
    def decoder_tangent(gamma: FSz0) -> Float[Array, " 3"]:
        # Use JVP (forward-mode AD) with basis vector
        return jax.jvp(model.decoder, (gamma,), (jnp.ones_like(gamma),))[1]

    gamma_sg = jax.lax.stop_gradient(gamma_predict)
    dq_dgamma = jax.vmap(decoder_tangent)(gamma_sg)
    dq_dgamma = jax.lax.stop_gradient(dq_dgamma)

    # Compute $\hat{t}$ from dq/dgamma
    t_norm = jnp.linalg.norm(dq_dgamma, axis=1, keepdims=True)
    t_hat = jnp.where(t_norm > 0, dq_dgamma / t_norm, jnp.zeros_like(dq_dgamma))

    loss_path = decoder_loss(
        qs_meas=ws[:, :D],
        weights=weights,
        qs_pred=q_predict,
        t_hat=t_hat,
        p_hat=ps_hat,
        mask=mask,
        lambda_q=lambda_q,
        lambda_p=lambda_p,
    )

    return loss_mismember + loss_path


@eqx.filter_jit
def make_step(
    model_dynamic: PathAutoencoder,
    model_static: PathAutoencoder,
    /,
    ws: Float[Array, " B TwoF"],
    weights: Float[Array, " B"],
    mask: Bool[Array, " B"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    *,
    lambda_q: FLikeSz0,
    lambda_p: FLikeSz0,
    member_threshold: float,
    key: PRNGKeyArray,
) -> tuple[FSz0, PathAutoencoder, optax.OptState]:
    r"""Run a single optimization step for Phase 2 decoder training.

    Computes loss and gradients via `compute_decoder_loss`, then applies
    gradient updates to the dynamic model parameters.

    """
    # Compute the loss
    loss, grads = compute_decoder_loss(
        model_dynamic,
        model_static,
        ws,
        weights=weights,
        mask=mask,
        lambda_q=lambda_q,
        lambda_p=lambda_p,
        member_threshold=member_threshold,
        key=key,
    )

    # Update the dynamic components of the model
    updates, opt_state = optimizer.update(grads, opt_state, model_dynamic)
    model_dynamic = cast("PathAutoencoder", eqx.apply_updates(model_dynamic, updates))
    return loss, model_dynamic, opt_state


# ===================================================================


BatchScanCarry: TypeAlias = tuple[eqx.Module, optax.OptState, PRNGKeyArray]  # noqa: UP040
BatchScanInputs: TypeAlias = tuple[  # noqa: UP040
    Bool[Array, " B"],  # batch_mask: True where batch has data, False where padded
    Float[Array, " B TwoF"],  # batch_ws : ordered positions
    Float[Array, " B"],  # batch_weights:
    FSz0,  # lambda_p: scalar array
]


def train_ordering_and_track_net(
    model: PathAutoencoder,
    all_ws: Float[Array, "N TwoF"],
    /,
    mask: Bool[Array, " N"],
    config: EncoderDecoderTrainingConfig,
    *,
    key: PRNGKeyArray,
) -> tuple[PathAutoencoder, optax.OptState, Float[Array, " {config.n_epochs}"]]:
    r"""Train the decoder (TrackNet) in Phase 2 of autoencoder training.

    This phase trains the decoder to reconstruct spatial positions from $\gamma$
    values while aligning with velocity directions. The encoder can optionally be
    frozen during this phase.

    The training uses lax.scan for efficient batching and supports:
    - Linear ramping of lambda_p from min to max over epochs
    - Optional KDE-based importance weighting
    - Membership probability filtering via member_threshold
    - Optional encoder freezing

    Parameters
    ----------
    model : PathAutoencoder
        The autoencoder model to train (encoder should already be trained).
    all_ws : Array, shape (N, 2*n_dims)
        All phase-space coordinates (positions + velocities).
    mask : Array, shape (N,)
        Binary mask where True = use for training (stream members).
    config : EncoderDecoderTrainingConfig
        Training configuration including epochs, batch size, loss weights, etc.
    key : PRNGKeyArray
        Random key for shuffling and batching.

    Returns
    -------
    model : PathAutoencoder
        Trained autoencoder with updated decoder (and encoder if not frozen).
    opt_state : optax.OptState
        Final optimizer state.
    losses : Array, shape (n_epochs,)
        Training loss per epoch.

    """
    # Compute weights
    # TODO: compute masked weights?
    key, subkey = jr.split(key)
    if not config.weight_by_density:
        weights = compute_uniform_weights(model, all_ws, key=subkey)
    elif isinstance(config.weight_by_density, Mapping):
        weights = compute_weights(
            model.encoder, all_ws, key=subkey, **config.weight_by_density
        )
    else:
        weights = compute_weights(model.encoder, all_ws)

    # Model surgery: partition out static components of the model
    if config.freeze_encoder:
        # Create a filter that is True for arrays, but False for the encoder
        # First apply is_array to the whole model
        filter_spec = jtu.map(eqx.is_array, model)
        # Then replace encoder subtree with a tree of False values
        encoder_false = jtu.map(lambda _: False, model.encoder)
        filter_spec = eqx.tree_at(lambda m: m.encoder, filter_spec, encoder_false)
    else:
        filter_spec = eqx.is_array
    model_dynamic, model_static = eqx.partition(model, filter_spec)

    # Optimizer setup - initialize with the dynamic model only
    optimizer = config.optimizer
    opt_state = optimizer.init(model_dynamic)

    # ----------------------------------------
    # Epoch Scan Function (per-epoch scan)

    batch_size = config.batch_size
    lambda_p_min, lambda_p_max = config.lambda_p

    def epoch_scan_fn(
        carry: BatchScanCarry, epoch_idx: int
    ) -> tuple[BatchScanCarry, FSz0]:
        # Unpack the carry
        model_dyn, opt_state, key = carry

        # Linearly ramp lambda_p from min to max over n_epochs
        frac = epoch_idx / (config.n_epochs - 1) if config.n_epochs > 1 else 0.0
        lambda_p = lambda_p_min + (lambda_p_max - lambda_p_min) * frac

        # Shuffle and batch data
        key, subkey = jr.split(key)
        b_mask, (b_ws, b_weights) = shuffle_and_batch(
            mask,
            all_ws,
            weights,
            batch_size=batch_size,
            key=subkey,
            pad_value=1,
        )

        # Broadcast lambda_p to match number of batches
        n_batches = b_ws.shape[0]
        b_lambda_p = jnp.broadcast_to(lambda_p, (n_batches,))

        # Scan over batches
        carry = (model_dyn, opt_state, key)
        carry, batch_losses = jax.lax.scan(
            cond_batch_scan_fn, carry, (b_mask, b_ws, b_weights, b_lambda_p)
        )

        # Use mean loss across all batches for this epoch
        avg_loss = jnp.mean(batch_losses)
        return carry, avg_loss

    # ----------------------------------------
    # Conditionally Run Batch Scan Function

    def cond_batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        """Run scanned batch step if there's data."""
        mask = inputs[0]
        return jax.lax.cond(
            jnp.any(mask), batch_scan_fn, null_batch_scan_fn, carry, inputs
        )

    def null_batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        """Don't run scanned batch step."""
        loss = jnp.array(0, dtype=jnp.result_type(*inputs[1:]))
        return carry, loss

    # ----------------------------------------
    # Batch Scan Function (per-batch scan)

    lambda_q = config.lambda_q
    member_threshold = config.member_threshold

    def batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        model_dyn, opt_state, key = carry
        mask, ws, weights, lambda_p = inputs

        # Single training step for this batch
        key, subkey = jr.split(key)
        loss, model_dyn, opt_state = make_step(
            model_dyn,
            model_static,
            ws=ws,
            weights=weights,
            mask=mask,
            opt_state=opt_state,
            optimizer=optimizer,
            lambda_q=lambda_q,
            lambda_p=lambda_p,
            member_threshold=member_threshold,
            key=subkey,
        )

        return (model_dyn, opt_state, key), loss

    # ----------------------------------------

    # Optionally wrap the epoch scan with a progress bar
    if config.show_pbar:
        epoch_scan_wrapped = jax_tqdm.scan_tqdm(
            config.n_epochs, desc="Training", unit="epoch", dynamic_ncols=True
        )(epoch_scan_fn)
    else:
        epoch_scan_wrapped = epoch_scan_fn

    # Prepare epoch indices and run scan over epochs, which scans over batches
    carry = (model_dynamic, opt_state, key)
    epoch_indices = jnp.arange(config.n_epochs)
    carry, epoch_losses = jax.lax.scan(epoch_scan_wrapped, carry, epoch_indices)
    model_dynamic, opt_state, _ = carry

    # Reconstruct model
    model = cast("PathAutoencoder", eqx.combine(model_dynamic, model_static))

    return model, opt_state, epoch_losses


# ===================================================================


@plum.dispatch
def train_autoencoder(
    model: PathAutoencoder,
    all_ws: Float[Array, " N TwoF"],
    ordering_indices: Int[Array, " N"],
    /,
    *,
    config: TrainingConfig | None = None,
    key: PRNGKeyArray,
) -> tuple[AutoencoderResult, dict[str, PyTree], Float[Array, " {config.n_epochs}"]]:
    r"""Train the PathAutoencoder in two phases.

    This function orchestrates the complete two-phase training procedure:

    **Phase 1 (OrderingNet/Encoder)**: Trains the encoder to predict $\gamma$
    (ordering parameter) and $p$ (membership probability) from phase-space
    coordinates. Uses the ordering from the walk algorithm as supervision.

    **Phase 2 (TrackNet/Decoder)**: Trains the decoder to reconstruct spatial
    positions from $\gamma$ while aligning with velocity directions. Uses the
    trained encoder to filter stream members based on membership probability
    threshold.

    Parameters
    ----------
    model : PathAutoencoder
        Untrained or partially trained autoencoder model.
    all_ws : Array, shape (N, 2*n_dims)
        All phase-space coordinates (positions + velocities).
    ordering_indices : Int[Array, " N"]
        Ordering indices from walk algorithm. Valid indices (>= 0) indicate
        ordered tracers; -1 indicates skipped/unordered tracers.
    config : TrainingConfig | None, optional
        Complete training configuration for both phases.
        If `None` (default), uses default configuration.
    key : PRNGKeyArray
        Random key for training (split internally for each phase).

    Returns
    -------
    result : AutoencoderResult
        Result containing the fully trained autoencoder and ordering data.
    opt_states : dict[str, optax.OptState]
        Dictionary with 'encoder', 'decoder' and 'both' optimizer states.
    losses : Array, shape (n_epochs_encoder + n_epochs_both,)
        Concatenated training losses from both phases.

    """
    # Build default config if none provided
    if config is None:
        config = TrainingConfig()

    # Split the keys
    keys: tuple[PRNGKeyArray, ...]
    key, *keys = jr.split(key, 6)

    # ===========================================
    # Train Encoder

    # Extract the configuration from the total config
    config_encoder = config.encoderonly_config()

    # Train the encoder
    encoder, encoder_opt_state, encoder_losses = train_ordering_net(
        model.encoder, all_ws, ordering_indices, config=config_encoder, key=keys[0]
    )

    # Model surgery: put the updated encoder back into the model
    model = eqx.tree_at(lambda m: m.encoder, model, encoder)

    # ===========================================
    # Train Decoder on the running mean

    # Use the trained encoder to compute gamma values for all member samples
    # TODO: this happens outside of JIT. Speed up.
    # TODO: add an optional exclusion for the progenitor
    gamma, prob = jax.jit(jax.vmap(model.encoder, in_axes=(0, None)))(all_ws, keys[1])
    is_member = prob > config.member_threshold

    # Then compute the running mean decoder targets for the member samples. This
    # gives us a denoised target for the decoder to train on, which is
    # especially important in the early stages of training when the encoder is
    # still noisy.
    D = all_ws.shape[1] // 2
    all_qs = all_ws[:, :D]
    mean_fn = RunningMeanDecoder(
        window_size=0.05,
        gamma_train=gamma,
        positions_train=all_qs,
        member_train=is_member,
    )
    mean_qs = jax.vmap(mean_fn, (0, None))(gamma, keys[2])

    # Train the decoder to reconstruct the running mean positions from gamma.
    config_decoder = config.decoderonly_config()
    decoder, decoder_opt_state, decoder_losses = train_track_net(
        model.decoder,
        gamma=gamma,
        qs_mean=mean_qs,
        mask=is_member,
        config=config_decoder,
        key=keys[2],
    )

    # Model surgery: put the updated decoder back into the model
    model = eqx.tree_at(lambda m: m.decoder, model, decoder)

    # ===========================================
    # Train Encoder & Decoder together

    # Extract the configuration from the total config
    config_autoencoder = config.autoencoder_config()

    # Train the decoder.
    model, autoencoder_opt_state, autoencoder_losses = train_ordering_and_track_net(
        model, all_ws, mask=is_member, config=config_autoencoder, key=keys[3]
    )

    # ===========================================
    # Return

    opt_states = {
        "encoder": encoder_opt_state,
        "decoder": decoder_opt_state,
        "both": autoencoder_opt_state,
    }
    losses = jnp.concat([encoder_losses, decoder_losses, autoencoder_losses])

    # Convert all_ws back to VectorComponents for AutoencoderResult
    D = all_ws.shape[1] // 2
    qs_norm = all_ws[:, :D]
    ps_norm = all_ws[:, D:]
    positions, velocities = model.normalizer.inverse_transform(qs_norm, ps_norm)

    # Encode to get gamma and membership_prob
    gamma, prob = model.encode(positions, velocities)
    # Sort by gamma to get ordering
    sorted_indices = jnp.argsort(gamma)
    # Filter by probability threshold
    high_prob_mask = prob[sorted_indices] >= config.member_threshold
    filtered_indices = sorted_indices[high_prob_mask]

    result = AutoencoderResult(
        model=model,
        positions=positions,
        velocities=velocities,
        indices=filtered_indices,
        gamma=gamma,
        membership_prob=prob,
        gamma_range=model.gamma_range,
    )

    return result, opt_states, losses


@plum.dispatch
def train_autoencoder(
    model: AbstractAutoencoder,
    walk_results: WalkLocalFlowResult,
    /,
    *,
    config: TrainingConfig | None = None,
    key: PRNGKeyArray,
) -> tuple[AutoencoderResult, dict[str, PyTree], Float[Array, " {config.n_epochs}"]]:
    # Transform walk results using the model's normalizer
    qs, ps = model.normalizer.transform(walk_results.positions, walk_results.velocities)
    ws = jnp.concatenate([qs, ps], axis=1)

    # Train the model
    result, opt_states, losses = train_autoencoder(
        model, ws, walk_results.indices, config=config, key=key
    )

    return result, opt_states, losses
