r"""Interpolation Network for interpolating skipped tracers."""

__all__: tuple[str, ...] = ("TrackNet", "decoder_loss")

import functools as ft
from dataclasses import KW_ONLY, dataclass
from typing import TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_tqdm
import optax
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Real

from .order_net import default_optimizer
from .scanmlp import ScanOverMLP
from .utils import masked_mean, shuffle_and_batch
from localflowwalk._src.custom_types import FSz0, RSz0, RSzN


class TrackNet(eqx.Module):
    r"""Param-Net (decoder): maps $\gamma \to$ position (x, y, z).

    This network reconstructs the stream track position from the ordering
    parameter $\gamma$. It serves as the second half of the autoencoder.

    The architecture follows Appendix B.1 of Nibauer et al. (2022).

    Uses scan-over-layers for improved compilation speed. See:
    https://docs.kidger.site/equinox/tricks/#improve-compilation-speed-with-scan-over-layers

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key for initialization.
    out_size : int
        Number of spatial dimensions (2 for 2D, 3 for 3D) for the track
        speed.
    hidden_size : int, optional
        Size of hidden layers. Default: 100.
    n_hidden : int, optional
        Number of hidden layers. Default: 3.

    """

    mlp: ScanOverMLP

    out_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        out_size: int = 3,
        width_size: int = 100,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        # Store static information
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth

        self.mlp = ScanOverMLP(
            in_size="scalar",
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    @ft.partial(eqx.filter_jit)
    def __call__(
        self, gamma: RSz0, /, key: PRNGKeyArray | None = None
    ) -> tuple[Float[Array, " {self.out_size}"], FSz0]:
        """Forward pass through Param-Net.

        Parameters
        ----------
        gamma : RSz0
            Ordering parameter in [-1, 1], shape (...).
        key : PRNGKeyArray | None
            Optional key.

        Returns
        -------
        position : Array
            Reconstructed position of shape (..., out_size).

        """
        return self.mlp(gamma, key=key)


# ===================================================================


@eqx.filter_jit
def decoder_loss(
    qs_meas: Real[Array, "N D"],
    weights: RSzN,
    qs_pred: Real[Array, "N D"],
    t_hat: RSzN,
    p_hat: RSzN,
    mask: Bool[Array, " N"],
    *,
    lambda_q: float = 100,
    lambda_p: float = 100,
) -> RSz0:
    r"""Loss for Param-Net decoder: position + momentum direction matching.

    Computes the reconstruction loss as defined in Appendix A.3 of Nibauer
    et al. (2022). The decoder reconstructs the stream track position and
    momentum direction (tangent vectors) from the ordering parameter $gamma$.

    The loss has two components:

    1. **Position MSE**: Squared error between true and predicted phase-space
       coordinates (position + velocity).
    2. **Momentum direction**: Squared error between true and predicted
       momentum direction (tangent vectors T and T_θ).

    The loss is weighted by element-wise weights to handle variable batch
    sizes with padding (only real data contributes).

    Parameters
    ----------
    qs_meas : Array, shape (N, D)
        True coordinate samples for the batch.
    weights : Array, shape (N,)
        Binary or continuous weights for each sample. Real data receives
        weight 1, padded samples receive weight 0.
    qs_pred : Array, shape (N, D)
        Predicted coordinates from Param-Net decoder.
    t_hat : Array, shape (N,)
        True momentum direction unit vectors (tangent vectors) for each sample.
    p_hat : Array, shape (N,)
        Predicted momentum direction unit vectors from Param-Net decoder.
    mask : Array, shape (N,)
        Mask for which values should be used, and which should be ignored.
    lambda_q : float, optional
        Weight for position reconstruction loss. Default: 1.0.
    lambda_p : float, optional
        Weight for momentum direction loss. Default: 100.0.

    Returns
    -------
    loss : Array, shape ()
        Scalar loss value.

    Notes
    -----
    The mathematical form is:

    $$ \\ell_\\theta(\\theta) = \\sum_{n=1}^N w_n \\left[
        \\lambda_q \\|x_n - x_\\theta(\\gamma_\\theta(x_n, v_n))\\|^2
        + \\lambda_p \\|T_n - T_\\theta(\\gamma_\\theta(x_n, v_n))\\|^2
    \\right] $$

    where:
    - $w_n$ are the sample weights
    - $x_n$ are true coordinates, $x_\\theta$ are predicted
    - $T_n$ are true tangent vectors, $T_\\theta$ are predicted
    - $\\lambda_q$ and $\\lambda_p$ control relative importance

    References
    ----------
    Nibauer et al. (2022), Appendix A.3: Decoder loss formulation for
    reconstructing stream track position and momentum direction.

    """
    # Spatial component
    sq_spatial_dist = jnp.sum(jnp.square(qs_meas - qs_pred), axis=1)
    spatial_l2 = masked_mean(weights * sq_spatial_dist, mask)

    # Velocity Alignment
    # TODO: replace this with the metric information
    sq_tangent_dist = jnp.sum(jnp.square(t_hat - p_hat), axis=1)
    tangent_l2 = masked_mean(weights * sq_tangent_dist, mask)

    # Full loss
    prefactor = 1 / (lambda_q + lambda_p)
    return prefactor * (lambda_q * spatial_l2 + lambda_p * tangent_l2)


@eqx.filter_value_and_grad
def compute_loss(
    model: TrackNet,
    gamma: Float[Array, " B"],
    qs_mean: Real[Array, " B D"],
    mask: Bool[Array, " B"],
    *,
    key: PRNGKeyArray | None = None,
) -> FSz0:
    """Compute loss and gradients for a batch of data."""
    # Predict positions from model
    qs_pred = jax.vmap(model, (0, None))(gamma, key)

    return decoder_loss(
        qs_meas=qs_mean,
        weights=jnp.ones_like(gamma),
        qs_pred=qs_pred,
        t_hat=jnp.zeros_like(qs_pred),  # Not used in current loss
        p_hat=jnp.zeros_like(qs_pred),  # Not used in current loss
        mask=mask,
        lambda_q=1,
        lambda_p=0,  # not used in current loss
    )


@eqx.filter_jit
def make_step(
    model_dynamic: TrackNet,
    model_static: TrackNet,
    gamma: Float[Array, " B"],
    qs_mean: Real[Array, " B D"],
    mask: Bool[Array, " B"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    *,
    key: PRNGKeyArray,
) -> tuple[FSz0, TrackNet, optax.OptState]:
    """Make a single optimization step for the decoder."""
    # Reconstruct full model from dynamic and static parts
    model = eqx.combine(model_dynamic, model_static)

    # Compute loss and gradients
    loss, grads = compute_loss(model, gamma, qs_mean, mask, key=key)

    # Update the dynamic components of the model
    updates, opt_state = optimizer.update(grads, opt_state, model_dynamic)
    model_dynamic = cast("TrackNet", optax.apply_updates(model_dynamic, updates))
    return loss, model_dynamic, opt_state


@dataclass
class TrackTrainingConfig:
    """Configuration for training the TrackNet decoder."""

    _: KW_ONLY

    optimizer: optax.GradientTransformation = default_optimizer
    """Optax optimizer for training. Default: AdamW with lr=1e-3, weight_decay=1e-7."""

    n_epochs: int = 100
    """Number of epochs for training."""

    batch_size: int = 100
    """Batch size for training."""

    show_pbar: bool = True
    """Whether to show an epoch progress bar via tqdm."""


BatchScanCarry: TypeAlias = tuple[eqx.Module, optax.OptState, PRNGKeyArray]  # noqa: UP040
BatchScanInputs: TypeAlias = tuple[  # noqa: UP040
    Bool[Array, " B"],  # mask
    Float[Array, " B"],  # gamma
    Real[Array, " B D"],  # qs_mean
]


def train_track_net(
    model: TrackNet,
    gamma: Float[Array, " N"],
    qs_mean: Real[Array, " N D"],
    mask: Bool[Array, " N"],
    *,
    config: TrackTrainingConfig | None = None,
    key: PRNGKeyArray,
) -> tuple[TrackNet, optax.OptState, Float[Array, " n_epochs"]]:
    """Train the TrackNet decoder using the provided data."""
    if config is None:
        config = TrackTrainingConfig()

    # Model surgery: partition out static components of the model
    filter_spec = eqx.is_array
    model_dynamic, model_static = eqx.partition(model, filter_spec)

    # Optimizer setup
    optimizer = config.optimizer
    opt_state = optimizer.init(model_dynamic)

    # ----------------------------------------
    # Epoch Scan Function (per-epoch scan)

    batch_size = config.batch_size

    def epoch_scan_fn(carry: BatchScanCarry, _: int) -> tuple[BatchScanCarry, FSz0]:
        """Run one scanned epoch (shuffle, batch, and train)."""
        # Unpack the carry
        model_dyn, opt_state, key = carry

        # Split key for this epoch
        key, subkey = jr.split(key, 2)

        # Shuffle and batch data
        b_mask, (b_gamma, b_qs_mean) = shuffle_and_batch(
            mask, gamma, qs_mean, batch_size=batch_size, key=subkey
        )

        # Scan over batches
        carry = (model_dyn, opt_state, key)
        x = (b_mask, b_gamma, b_qs_mean)
        carry, batch_losses = jax.lax.scan(cond_batch_scan_fn, carry, x)

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

    def batch_scan_fn(
        carry: BatchScanCarry, inputs: BatchScanInputs
    ) -> tuple[BatchScanCarry, FSz0]:
        """Run one scanned batch step.

        Notes
        -----
        Uses a partitioned model to keep the scan carry as arrays-only where
        possible, then re-combines with static structure for each step.

        """
        model_dyn, opt_state, key = carry
        mask, gamma, qs_mean = inputs

        # Single training step for this batch
        key, subkey = jr.split(key)
        loss, model_dyn, opt_state = make_step(
            model_dyn,
            model_static,
            gamma=gamma,
            qs_mean=qs_mean,
            mask=mask,
            opt_state=opt_state,
            optimizer=optimizer,
            key=subkey,
        )

        return (model_dyn, opt_state, key), loss

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
    model = cast("OrderingNet", eqx.combine(model_dynamic, model_static))

    return model, opt_state, epoch_losses
