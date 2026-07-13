r"""Interpolation Network for interpolating skipped tracers."""

__all__: tuple[str, ...] = (
    "AbstractTrackNet",
    "TrackNet",
    "FourierTrackNet",
    "TrackNetTrainer",
    "decoder_loss",
)

import abc
import functools as ft
from dataclasses import KW_ONLY, dataclass
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Real

from jaxmore.nn import masked_mean

from .order_net import default_optimizer
from .trainer import AbstractEqxScanTrainer, EqxTrainCarry
from phasecurvefit._src.custom_types import FSz0, RSz0, RSzN


class AbstractTrackNet(eqx.Module):
    r"""Interface for a trainable decoder mapping $\gamma \to$ position.

    A track net is the second half of the autoencoder: it reconstructs the
    spatial track position from the scalar ordering parameter $\gamma$.

    Subclasses must:

    - expose ``out_size`` (the number of spatial dimensions), and
    - implement ``__call__(gamma, /, key=None) -> position`` of shape
      ``(out_size,)``, **differentiable in** ``gamma`` (Phase-2 training takes a
      ``jvp`` of the decoder w.r.t. $\gamma$ for the velocity/tangent loss).

    Concrete variants (all interchangeable via ``PathAutoencoder.make(decoder=...)``):

    - :class:`TrackNet` — a plain MLP on $\gamma$ (default).
    - :class:`FourierTrackNet` — Fourier features on $\gamma$, for sharp /
      self-intersecting tracks a plain MLP over-smooths.
    """

    out_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(
        self, gamma: RSz0, /, key: PRNGKeyArray | None = None
    ) -> Float[Array, " {self.out_size}"]:
        """Reconstruct the position at ordering parameter ``gamma``."""
        ...


class TrackNet(AbstractTrackNet):
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

    mlp: eqx.nn.MLP

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

        self.mlp = eqx.nn.MLP(
            in_size="scalar",
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            scan=True,
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
            Ordering parameter in [0, 1], shape (...).
        key : PRNGKeyArray | None
            Optional key.

        Returns
        -------
        position : Array
            Reconstructed position of shape (..., out_size).

        """
        return self.mlp(gamma, key=key)


class FourierTrackNet(AbstractTrackNet):
    r"""Decoder with Fourier features on $\gamma$ before the MLP.

    Maps $\gamma$ to ``[gamma, sin(pi k gamma), cos(pi k gamma)]`` for
    ``k = 1 .. n_frequencies`` and feeds those ``1 + 2*n_frequencies`` features
    to an MLP. The Fourier features give the network the high-frequency capacity
    to represent sharp or self-intersecting tracks (e.g. multi-petal curves)
    that a plain :class:`TrackNet` tends to over-smooth. ``sin/cos(pi k gamma)``
    complete one period over a unit-width $\gamma$-range, so ``n_frequencies``
    controls how sharp a curve the decoder can render.

    Parameters
    ----------
    out_size : int
        Number of spatial dimensions.
    n_frequencies : int
        Number of Fourier modes ``k``. Higher captures sharper structure.
    width_size, depth : int
        MLP hidden-layer size and number of hidden layers.
    key : PRNGKeyArray
        JAX random key for initialization.

    Examples
    --------
    Create a Fourier decoder for 2D tracks:

    >>> import jax.random as jr
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    >>> # Create a decoder mapping gamma to 2D positions
    >>> decoder = pcf.nn.FourierTrackNet(
    ...     out_size=2, n_frequencies=8, width_size=128, depth=3, key=jr.key(0)
    ... )
    >>> decoder.out_size
    2

    Decode a single position from the ordering parameter:

    >>> # Evaluate decoder at gamma=0.5 (middle of track)
    >>> gamma_val = jnp.array(0.5)
    >>> position = decoder(gamma_val)
    >>> position.shape
    (2,)

    The Fourier embedding converts ``gamma`` to higher-dimensional features
    before the MLP, allowing sharp / multi-petal structure:

    >>> # Inspect the Fourier features
    >>> features = decoder.features(jnp.array(0.3))
    >>> features.shape  # (1 + 2 * 8,)
    (17,)

    Compute the tangent vector (velocity) for the tangent/momentum loss:

    >>> import jax
    >>> gamma = jnp.array(0.5)
    >>> position, tangent = jax.jvp(decoder, (gamma,), (jnp.array(1.0),))
    >>> tangent.shape  # Tangent vector in position space
    (2,)

    Use with ``PathAutoencoder.make()`` to build a complete autoencoder
    with Fourier decoding instead of the default plain MLP:

    >>> positions = {"x": jnp.linspace(0, 1, 20), "y": jnp.linspace(0, 1, 20)}
    >>> velocities = {"x": jnp.ones(20), "y": jnp.ones(20)}
    >>> normalizer = pcf.nn.StandardScalerNormalizer(positions, velocities)

    >>> # Create decoder with specific parameters for sharp tracks
    >>> decoder_sharp = pcf.nn.FourierTrackNet(
    ...     out_size=2, n_frequencies=16, width_size=256, depth=4, key=jr.key(1)
    ... )
    >>> autoencoder = pcf.nn.PathAutoencoder.make(
    ...     normalizer, gamma_range=(0.0, 1.0), decoder=decoder_sharp, key=jr.key(2)
    ... )
    >>> isinstance(autoencoder.decoder, pcf.nn.FourierTrackNet)
    True

    """

    mlp: eqx.nn.MLP

    out_size: int = eqx.field(static=True)
    n_frequencies: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        out_size: int = 3,
        n_frequencies: int = 8,
        width_size: int = 128,
        depth: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.out_size = out_size
        self.n_frequencies = n_frequencies
        self.width_size = width_size
        self.depth = depth
        self.mlp = eqx.nn.MLP(
            in_size=1 + 2 * n_frequencies,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            scan=depth > 1,  # scan-over-layers needs >1 hidden layer
            key=key,
        )

    def features(self, gamma: RSz0, /) -> Float[Array, " {1 + 2 * self.n_frequencies}"]:
        """Fourier-feature embedding of the scalar ordering parameter."""
        gamma = jnp.asarray(gamma)
        ks = jnp.arange(1, self.n_frequencies + 1, dtype=gamma.dtype)
        ang = jnp.pi * ks * gamma
        return jnp.concatenate([jnp.atleast_1d(gamma), jnp.sin(ang), jnp.cos(ang)])

    @ft.partial(eqx.filter_jit)
    def __call__(
        self, gamma: RSz0, /, key: PRNGKeyArray | None = None
    ) -> Float[Array, " {self.out_size}"]:
        """Forward pass: Fourier-embed ``gamma``, then MLP to a position."""
        return self.mlp(self.features(gamma), key=key)


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
        lambda_p=0.1,  # set to small value in loss
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


def _track_step(
    carry: EqxTrainCarry,
    batch_inputs: tuple[Bool[Array, " B"], tuple[Array, ...]],
    *,
    optimizer: optax.GradientTransformation,
    filter_spec: Any,
) -> tuple[FSz0, EqxTrainCarry]:
    """Run one batch of TrackNet training.

    `batch_inputs` is ``(mask, (gamma, qs_mean))``.

    `filter_spec` must be the same spec the trainer used to build `opt_state`
    (see `TrackNetTrainer.init`), so that the step, the carry packing in
    `AbstractEqxScanTrainer.pack_carry_state`, and the optimizer state all
    agree on which leaves are trainable.
    """
    model, opt_state, key = carry
    mask, (gamma, qs_mean) = batch_inputs

    model_dynamic, model_static = eqx.partition(model, filter_spec)

    key, subkey = jr.split(key)
    loss, model_dynamic, opt_state = make_step(
        model_dynamic,
        model_static,
        gamma=gamma,
        qs_mean=qs_mean,
        mask=mask,
        opt_state=opt_state,
        optimizer=optimizer,
        key=subkey,
    )

    model = eqx.combine(model_dynamic, model_static)
    return loss, (model, opt_state, key)


@dataclass(frozen=True)
class TrackNetTrainer(AbstractEqxScanTrainer):
    """Scan trainer for `TrackNet`."""

    def init(  # type: ignore[override]
        self,
        model: TrackNet,
        /,
        *,
        gamma: Float[Array, " N"],
        qs_mean: Real[Array, " N D"],
        mask: Bool[Array, " N"],
        optimizer: optax.GradientTransformation,
        key: PRNGKeyArray,
    ) -> tuple[EqxTrainCarry, tuple[Bool[Array, " N"], tuple[Array, ...]]]:
        """Build the initial carry and the epoch data."""
        model_dynamic, _ = eqx.partition(model, self.filter_spec)
        opt_state = optimizer.init(model_dynamic)
        return (model, opt_state, key), (mask, (gamma, qs_mean))


def train_track_net(
    model: TrackNet,
    gamma: Float[Array, " N"],
    qs_mean: Real[Array, " N D"],
    mask: Bool[Array, " N"],
    *,
    config: TrackTrainingConfig | None = None,
    key: PRNGKeyArray,
) -> tuple[TrackNet, optax.OptState, Float[Array, " n_epochs"]]:
    """Train the TrackNet decoder using the provided data.

    Notes
    -----
    `mask` is typically sparse here -- only ordered stream members train the
    decoder. Because `shuffle_and_batch` sorts usable samples first, the
    ignorable ones cluster into whole batches that contain no usable data at
    all. Those batches are skipped, and (via `masked_mean`) excluded from the
    epoch loss rather than being averaged in as zeros.

    """
    if config is None:
        config = TrackTrainingConfig()

    optimizer = config.optimizer

    # Single source of truth for what is trainable: the step, the carry packing,
    # and `optimizer.init` must all partition the model the same way.
    filter_spec: Any = eqx.is_array

    trainer = TrackNetTrainer(
        make_step=ft.partial(_track_step, optimizer=optimizer, filter_spec=filter_spec),
        loss_agg_fn=masked_mean,
        filter_spec=filter_spec,
    )
    initial_carry, epoch_data = trainer.init(
        model, gamma=gamma, qs_mean=qs_mean, mask=mask, optimizer=optimizer, key=key
    )
    (model, opt_state, _), epoch_losses = trainer.run(
        initial_carry,
        epoch_data,
        num_epochs=config.n_epochs,
        batch_size=config.batch_size,
        key=key,
        show_pbar=config.show_pbar,
    )

    return cast("TrackNet", model), opt_state, epoch_losses
