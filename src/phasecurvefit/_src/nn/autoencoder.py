"""Autoencoder."""

__all__: tuple[str, ...] = (
    "PathAutoencoder",
    "PathAutoencoderTrainer",
    "posterior_membership",
    "train_autoencoder",
    "TrainingConfig",
)

import functools as ft
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any, ClassVar, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jtu
import optax
import plum
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from jaxmore.nn import masked_mean

from .abstractautoencoder import AbstractAutoencoder
from .externaldecoder import RunningMeanDecoder
from .membership import (
    MixtureMembershipConfig,
    WidthNet,
    membership_rampup,
    membership_responsibility,
    mixture_membership_loss,
    sigma_ceiling,
    uniform_background_density,
)
from .normalize import AbstractNormalizer
from .order_net import (
    OrderingNet,
    OrderingTrainingConfig,
    default_optimizer,
    train_ordering_net,
)
from .result import AutoencoderResult
from .track_net import (
    AbstractTrackNet,
    TrackNet,
    TrackTrainingConfig,
    decoder_loss,
    train_track_net,
)
from .trainer import AbstractEqxScanTrainer, EqxTrainCarry
from phasecurvefit._src.custom_types import FLikeSz0, FSz0, FSzN
from phasecurvefit._src.orderers.result import OrderingResult

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
    decoder: AbstractTrackNet
    normalizer: AbstractNormalizer

    width: WidthNet | None = None
    r"""Foreground half-width $\sigma(\gamma)$ (e.g. a stream's), or None.

    Present only when the autoencoder was trained with mixture-model membership
    (see `MixtureMembershipConfig`). It is kept on the model because it is needed
    at *inference* to turn the encoder's prior membership $\pi$ into a calibrated
    posterior; see `posterior_membership`.
    """

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
        decoder: AbstractTrackNet | None = None,
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
        if decoder is None:
            decoder = TrackNet(
                out_size=normalizer.n_spatial_dims,
                width_size=track_width_size,
                depth=track_depth,
                key=key_decode,
            )
        elif decoder.out_size != normalizer.n_spatial_dims:
            msg = (
                f"decoder.out_size ({decoder.out_size}) must match "
                f"normalizer.n_spatial_dims ({normalizer.n_spatial_dims})."
            )
            raise ValueError(msg)
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
    """Membership p > threshold for identifying stream members.

    Only used by the legacy (classifier) membership loss. The mixture model
    replaces this arbitrary cut with a calibrated posterior; see
    `MixtureMembershipConfig`.
    """

    membership: MixtureMembershipConfig | None = None
    r"""Opt in to mixture-model membership (outlier rejection).

    ``None`` (the default) keeps the existing classifier-style membership loss,
    bit-for-bit. Supply a `MixtureMembershipConfig` to model the data as a
    stream + background mixture in the sense of Hogg, Bovy & Lang (2010), §3, so
    that membership becomes a *posterior* driven by the reconstruction residual
    rather than a label inherited from the orderer.

    Use this if outliers survive as "members".
    """

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

    membership: MixtureMembershipConfig | None = None
    r"""Opt in to mixture-model membership (outlier rejection).

    ``None`` (the default) preserves the existing behaviour exactly. Supply a
    `MixtureMembershipConfig` to model the field as a stream + background mixture
    (Hogg, Bovy & Lang 2010, §3) so that membership is a calibrated posterior
    driven by the distance from the fitted track.

    Applies to phase 3 (joint encoder+decoder training), which is where the
    decoder -- and therefore the residual -- exists.
    """

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
            membership=self.membership,
            freeze_encoder=self.freeze_encoder_final_training,
            weight_by_density=self.weight_by_density,
        )


@eqx.filter_jit
def posterior_membership(
    model: PathAutoencoder,
    ws: Float[Array, "N TwoF"],
    /,
    *,
    background_density: float | None = None,
    key: PRNGKeyArray | None = None,
) -> FSzN:
    r"""Posterior probability that each point belongs to the foreground component.

    The foreground is whatever the model fits a track through -- a stream is the
    running example. This is the number you want when deciding which points are
    members. It is
    **not** the encoder's raw ``prob`` output: that is only the *prior* $\pi_n$,
    formed from the star's phase-space coordinates before the model has looked at
    how far the star actually landed from the fitted track. This function folds
    in the residual, giving the posterior of Hogg, Bovy & Lang (2010), §3:

    .. math::

        \hat{q}_n = \frac{\pi_n \, \mathcal{N}(r_n; 0, \sigma^2(\gamma_n))}
                         {\pi_n \, \mathcal{N}(r_n; 0, \sigma^2(\gamma_n))
                          + (1 - \pi_n) \, \rho_{\mathrm{bg}}}

    Requires a model trained with `MixtureMembershipConfig` (so that
    ``model.width`` exists).

    Parameters
    ----------
    model : PathAutoencoder
        A model trained with mixture membership.
    ws : Array, shape (N, 2D)
        Phase-space coordinates, in the same normalised frame used for training.
    background_density : float | None
        $\rho_{\mathrm{bg}}$. If None, recomputed from the extent of ``ws``.
        **Pass the same value used at training time** if you are scoring a
        different field from the one you trained on -- otherwise the posterior is
        calibrated against the wrong background.
    key : PRNGKeyArray | None
        Optional key for the (deterministic) networks.

    Returns
    -------
    responsibility : Array, shape (N,)
        Posterior membership in [0, 1]. Threshold at 0.5 for a hard cut, or keep
        it as a weight -- Hogg et al. recommend the latter.

    Raises
    ------
    ValueError
        If the model was not trained with mixture membership.

    """
    if model.width is None:
        msg = (
            "posterior_membership requires a model trained with mixture membership "
            "(TrainingConfig(membership=MixtureMembershipConfig(...))). "
            "Without it there is no fitted stream width, and the encoder's `prob` "
            "output is an uncalibrated classifier score, not a posterior."
        )
        raise ValueError(msg)

    D = ws.shape[1] // 2
    qs = ws[:, :D]

    gamma, prob = jax.vmap(model.encoder, (0, None))(ws, key)
    qs_pred = jax.vmap(model.decoder, (0, None))(gamma, key)
    sigma = jax.vmap(model.width)(gamma)

    r2 = jnp.sum(jnp.square(qs - qs_pred), axis=-1)

    rho_bg = (
        background_density
        if background_density is not None
        else uniform_background_density(qs)
    )
    return membership_responsibility(
        prob, r2, sigma, log_bg_density=jnp.log(rho_bg), n_dims=D
    )


def _mixture_decoder_loss(
    model: PathAutoencoder,
    ws: Float[Array, " B TwoF"],
    mask: Bool[Array, " B"],
    *,
    lambda_p: FLikeSz0,
    lambda_velocity: float,
    log_bg_density: float,
    sigma_ceil: FSz0,
    rampup: FSz0,
    key: PRNGKeyArray,
) -> FSz0:
    r"""Mixture-model loss: reconstruction, membership, and width, jointly.

    This is the opt-in alternative to the classifier-style membership of
    `compute_decoder_loss`. It implements the "mixture" model of Hogg, Bovy &
    Lang (2010), §3 -- see `phasecurvefit._src.nn.membership` for the full
    argument.

    Three things happen here that do not happen in the default loss:

    1. **The membership probability is driven by the residual.** A star far from
       the decoded track is explained by the background component, and its
       $\pi$ is pushed down. In the default loss no gradient connects "far from
       the track" to "not a member" at all.
    2. **The stream width is fitted**, as a function of $\gamma$, under an
       annealing ceiling that stops it inflating to absorb outliers.
    3. **The velocity-alignment term is weighted by the posterior
       responsibility**, so that a star the model has decided is background
       cannot drag the track's tangent around. This is the M-step of EM: fit the
       track using each star in proportion to how much it currently looks like a
       member.

    """
    if model.width is None:  # pragma: no cover - guarded by the caller
        msg = "mixture membership requires `model.width` to be a WidthNet."
        raise ValueError(msg)

    D = ws.shape[1] // 2
    qs = ws[:, :D]

    key, skey1, skey2 = jr.split(key, 3)
    gamma, prob = jax.vmap(model.encoder, (0, None))(ws, skey1)
    qs_pred = jax.vmap(model.decoder, (0, None))(gamma, skey2)

    # Stream half-width at each star's gamma, capped by the annealing ceiling.
    sigma = jnp.minimum(jax.vmap(model.width)(gamma), sigma_ceil)

    nll, responsibility = mixture_membership_loss(
        qs, qs_pred, prob, sigma, mask, log_bg_density=log_bg_density, rampup=rampup
    )

    if lambda_velocity == 0:
        return nll

    # ---- velocity alignment, weighted by posterior membership ----
    ps = ws[:, D:]
    ps_norm = jnp.linalg.norm(ps, axis=1, keepdims=True)
    ps_hat = jnp.where(ps_norm > 0, ps / ps_norm, jnp.zeros_like(ps))

    def decoder_tangent(g: FSz0) -> Float[Array, " D"]:
        return jax.jvp(model.decoder, (g,), (jnp.ones_like(g),))[1]

    gamma_sg = jax.lax.stop_gradient(gamma)
    dq_dgamma = jax.lax.stop_gradient(jax.vmap(decoder_tangent)(gamma_sg))
    t_norm = jnp.linalg.norm(dq_dgamma, axis=1, keepdims=True)
    t_hat = jnp.where(t_norm > 0, dq_dgamma / t_norm, jnp.zeros_like(dq_dgamma))

    sq_tangent = jnp.sum(jnp.square(t_hat - ps_hat), axis=1)

    # Responsibilities are the E-step; treat them as fixed weights in the M-step.
    resp = jax.lax.stop_gradient(responsibility) * mask
    denom = jnp.sum(resp)
    tangent_l2 = jnp.where(
        denom > 0, jnp.sum(resp * sq_tangent) / jnp.maximum(denom, 1e-12), 0.0
    )

    return nll + lambda_velocity * lambda_p * tangent_l2


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
    membership: MixtureMembershipConfig | None = None,
    log_bg_density: float = 0.0,
    sigma_ceil: FSz0 | float = 1.0,
    rampup: FSz0 | float = 1.0,
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

    Membership
    ----------
    By default membership is handled by `loss_mismember`, which penalises the
    encoder for disagreeing with the `mask` it was given. Note that this `mask`
    is itself derived from the encoder's own phase-1 predictions, so the term is
    a *fixed point*: it hardens existing beliefs and cannot discover that a star
    it called a member is in fact an outlier. See the module docstring of
    `phasecurvefit._src.nn.membership` for why that matters.

    Passing ``membership=MixtureMembershipConfig(...)`` swaps this for the
    generative mixture model of Hogg, Bovy & Lang (2010) §3, in which membership
    is a *posterior* driven by the reconstruction residual. That is the path you
    want if outliers are a problem.

    """
    # Reconstruct full model from dynamic and static parts
    model = eqx.combine(model_dynamic, model_static)

    # Get phase-space q,p separation index
    if ws.shape[1] % 2 != 0:
        msg = "ord_w has the wrong shape"
        raise ValueError(msg)
    D = ws.shape[1] // 2

    # ---- opt-in: generative mixture membership (Hogg, Bovy & Lang 2010, §3) ----
    if membership is not None:
        return _mixture_decoder_loss(
            model,
            ws,
            mask,
            lambda_p=lambda_p,
            lambda_velocity=membership.lambda_velocity,
            log_bg_density=log_bg_density,
            sigma_ceil=jnp.asarray(sigma_ceil, dtype=float),
            rampup=jnp.asarray(rampup, dtype=float),
            key=key,
        )

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
    #
    # It can, however, empty the mask entirely: the trainer only guarantees that
    # the *batch* mask has a usable sample (it skips batches where it does not),
    # and `is_member` is re-evaluated from the live encoder every step. If the
    # encoder transiently drops every star in this batch below the threshold,
    # `member_mask` is all-False, and `masked_mean` is documented to return NaN
    # on an empty mask -- which would poison the epoch loss. See `_path_loss`.
    member_mask = mask & is_member

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

    def _path_loss() -> FSz0:
        return decoder_loss(
            qs_meas=ws[:, :D],
            weights=weights,
            qs_pred=q_predict,
            t_hat=t_hat,
            p_hat=ps_hat,
            mask=member_mask,
            lambda_q=lambda_q,
            lambda_p=lambda_p,
        )

    # If the encoder currently claims no member in this batch, there is no track
    # constraint to impose, so the path loss is zero and only `loss_mismember`
    # supplies gradient (pushing the probabilities back up). `lax.cond` -- not
    # `jnp.where` -- because `where` would evaluate the NaN branch anyway and
    # its VJP would multiply that NaN by zero, giving a NaN gradient.
    loss_path = jax.lax.cond(
        jnp.any(member_mask), _path_loss, lambda: jnp.zeros_like(loss_mismember)
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
    membership: MixtureMembershipConfig | None = None,
    log_bg_density: float = 0.0,
    sigma_ceil: FSz0 | float = 1.0,
    rampup: FSz0 | float = 1.0,
) -> tuple[FSz0, PathAutoencoder, optax.OptState]:
    r"""Run a single optimization step for Phase 2 decoder training.

    Computes loss and gradients via `compute_decoder_loss`, then applies
    gradient updates to the dynamic model parameters.

    The ``membership`` / ``log_bg_density`` / ``sigma_ceil`` / ``rampup``
    arguments are forwarded to `compute_decoder_loss` and select the opt-in
    mixture-model membership. ``sigma_ceil`` and ``rampup`` are epoch-dependent
    schedules supplied by `PathAutoencoderTrainer.prepare_step_kw`.

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
        membership=membership,
        log_bg_density=log_bg_density,
        sigma_ceil=sigma_ceil,
        rampup=rampup,
    )

    # Update the dynamic components of the model
    updates, opt_state = optimizer.update(grads, opt_state, model_dynamic)
    model_dynamic = cast("PathAutoencoder", eqx.apply_updates(model_dynamic, updates))
    return loss, model_dynamic, opt_state


# ===================================================================


def _pad_to_multiple(
    mask: Bool[Array, " N"],
    *args: Float[Array, "N ..."],
    batch_size: int,
    pad_value: float,
) -> tuple[Bool[Array, " M"], tuple[Float[Array, "M ..."], ...]]:
    """Pad the dataset up to a whole number of batches, with `pad_value`.

    Why this exists
    ---------------
    `jaxmore.nn.AbstractScanNNTrainer.run` calls `shuffle_and_batch` with the
    default ``pad_value=0``; it does not (yet) expose the parameter. For this
    loss that is *not* a neutral choice: `compute_decoder_loss` penalises
    ``~mask`` rows as membership false-positives, and padding rows are ``~mask``.
    Padding with zeros would therefore plant fabricated "non-member" points at
    the origin of the normalised phase space -- i.e. right where the stream is --
    and actively train the encoder to reject its own centre.

    By padding here to an exact multiple of `batch_size`, `shuffle_and_batch`
    finds nothing left to pad and `pad_value` is never consulted. The rows we add
    carry the original ``pad_value=1`` and ``mask=False``, exactly as before.

    Delete this once `run()` accepts `pad_value`.

    """
    n = len(mask)
    n_batches = -(-n // batch_size)  # ceiling division
    pad_amount = n_batches * batch_size - n
    if pad_amount == 0:
        return mask, args

    padded_mask = jnp.pad(mask, (0, pad_amount), constant_values=False)
    padded_args = tuple(
        jnp.pad(
            arr,
            [(0, pad_amount)] + [(0, 0)] * (arr.ndim - 1),
            constant_values=pad_value,
        )
        for arr in args
    )
    return padded_mask, padded_args


def _autoencoder_step(
    carry: EqxTrainCarry,
    batch_inputs: tuple[Bool[Array, " B"], tuple[Array, ...]],
    *,
    optimizer: optax.GradientTransformation,
    filter_spec: Any,
    lambda_q: FLikeSz0,
    member_threshold: float,
    lambda_p: FLikeSz0,
    membership: MixtureMembershipConfig | None = None,
    log_bg_density: float = 0.0,
    sigma_ceil: FSz0 | float = 1.0,
    rampup: FSz0 | float = 1.0,
) -> tuple[FSz0, EqxTrainCarry]:
    """Run one batch of joint encoder+decoder training.

    `batch_inputs` is ``(mask, (ws, weights))``. The epoch-dependent schedules --
    `lambda_p`, and (for mixture membership) `sigma_ceil` and `rampup` -- arrive
    via `step_kw` and are recomputed each epoch by
    `PathAutoencoderTrainer.prepare_step_kw`.
    """
    model, opt_state, key = carry
    mask, (ws, weights) = batch_inputs

    # Gradients w.r.t. the dynamic half only. When `freeze_encoder` is set,
    # `filter_spec` excludes the encoder, so it receives no updates.
    model_dynamic, model_static = eqx.partition(model, filter_spec)

    key, subkey = jr.split(key)
    loss, model_dynamic, opt_state = make_step(
        model_dynamic,
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
        membership=membership,
        log_bg_density=log_bg_density,
        sigma_ceil=sigma_ceil,
        rampup=rampup,
    )

    model = eqx.combine(model_dynamic, model_static)
    return loss, (model, opt_state, key)


@dataclass(frozen=True)
class PathAutoencoderTrainer(AbstractEqxScanTrainer):
    r"""Scan trainer for joint encoder + decoder training.

    Owns the epoch-dependent schedules, all of which are computed in
    `prepare_step_kw` and forwarded to the step function:

    - $\lambda_p$, the velocity-alignment weight, ramped linearly from
      ``lambda_p_range[0]`` to ``lambda_p_range[1]``. Early epochs prioritise
      spatial reconstruction; later ones increasingly enforce alignment with the
      velocity field.
    - (mixture membership only) the **stream-width ceiling**, annealed
      geometrically downward, and the **membership ramp**, raised from 0 to 1.
      These two schedules bracket the mixture likelihood's two degenerate
      optima -- width inflation and membership collapse. See
      `phasecurvefit._src.nn.membership`.

    """

    lambda_p_range: tuple[float, float] = (1.0, 5.0)
    """``(start, stop)`` for the $\\lambda_p$ ramp."""

    membership: MixtureMembershipConfig | None = None
    """Mixture-model membership config, or None for the legacy classifier loss."""

    def init(  # type: ignore[override]
        self,
        model: PathAutoencoder,
        /,
        *,
        all_ws: Float[Array, "N TwoF"],
        weights: Float[Array, " N"],
        mask: Bool[Array, " N"],
        optimizer: optax.GradientTransformation,
        key: PRNGKeyArray,
    ) -> tuple[EqxTrainCarry, tuple[Bool[Array, " N"], tuple[Array, ...]]]:
        """Build the initial carry and the epoch data."""
        model_dynamic, _ = eqx.partition(model, self.filter_spec)
        opt_state = optimizer.init(model_dynamic)
        return (model, opt_state, key), (mask, (all_ws, weights))

    def prepare_step_kw(
        self, /, *, epoch_idx: Int[Array, ""], num_epochs: int, epoch_key: PRNGKeyArray
    ) -> Mapping[str, Any]:
        r"""Compute this epoch's schedules: $\lambda_p$, and the mixture ramps."""
        del epoch_key
        lambda_p_min, lambda_p_max = self.lambda_p_range
        frac = epoch_idx / (num_epochs - 1) if num_epochs > 1 else 0.0
        kw: dict[str, Any] = {
            "lambda_p": lambda_p_min + (lambda_p_max - lambda_p_min) * frac
        }

        if self.membership is not None:
            start, stop = self.membership.sigma_ceiling
            kw["sigma_ceil"] = sigma_ceiling(
                epoch_idx, num_epochs, start=start, stop=stop
            )
            kw["rampup"] = membership_rampup(
                epoch_idx, num_epochs, warmup_frac=self.membership.warmup_frac
            )

        return kw


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

    # Mixture-model membership is opt-in. When enabled the model must carry a
    # `WidthNet`, and the background density is fixed once, from the
    # field.  This must happen *before* `filter_spec` is built, since a
    # boolean-pytree filter has to match the model's structure exactly.
    membership = config.membership
    log_bg_density = 0.0
    if membership is not None:
        if model.width is None:
            key, wkey = jr.split(key)
            model = eqx.tree_at(
                lambda m: m.width,
                model,
                membership.make_width_net(key=wkey),
                is_leaf=lambda x: x is None,
            )
        D = all_ws.shape[1] // 2
        log_bg_density = float(
            jnp.log(membership.resolve_background_density(all_ws[:, :D]))
        )

    # Model surgery: decide which parts of the model are trainable.
    if config.freeze_encoder:
        # True for arrays, but False everywhere in the encoder subtree, so that
        # the encoder receives no gradient updates during this phase.
        filter_spec = jtu.map(eqx.is_array, model)
        encoder_false = jtu.map(lambda _: False, model.encoder)
        filter_spec = eqx.tree_at(lambda m: m.encoder, filter_spec, encoder_false)
    else:
        filter_spec = eqx.is_array

    # ----------------------------------------
    # Train

    optimizer = config.optimizer

    # Preserve the original ``pad_value=1``; see `_pad_to_multiple`.
    padded_mask, (padded_ws, padded_weights) = _pad_to_multiple(
        mask, all_ws, weights, batch_size=config.batch_size, pad_value=1
    )

    trainer = PathAutoencoderTrainer(
        make_step=ft.partial(
            _autoencoder_step,
            optimizer=optimizer,
            filter_spec=filter_spec,
            lambda_q=config.lambda_q,
            member_threshold=config.member_threshold,
            membership=membership,
            log_bg_density=log_bg_density,
        ),
        loss_agg_fn=masked_mean,
        filter_spec=filter_spec,
        lambda_p_range=config.lambda_p,
        membership=membership,
    )
    initial_carry, epoch_data = trainer.init(
        model,
        all_ws=padded_ws,
        weights=padded_weights,
        mask=padded_mask,
        optimizer=optimizer,
        key=key,
    )
    (model, opt_state, _), epoch_losses = trainer.run(
        initial_carry,
        epoch_data,
        num_epochs=config.n_epochs,
        batch_size=config.batch_size,
        key=key,
        show_pbar=config.show_pbar,
    )

    return cast("PathAutoencoder", model), opt_state, epoch_losses


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

    # Use the trained encoder to compute membership for all samples.
    # TODO: this happens outside of JIT. Speed up.
    # TODO: add an optional exclusion for the progenitor
    _, prob = jax.jit(jax.vmap(model.encoder, in_axes=(0, None)))(all_ws, keys[1])
    is_member = prob > config.member_threshold

    # Denoise the decoder target over the ORDERING coordinate, not the (locally
    # noisy) encoder gamma. The encoder gamma is globally monotone but jittery
    # at the running-mean window scale, so a window in encoder-gamma averages
    # spatially-scattered points and blurs the target off the track. The
    # ordering gamma (rank along the walk) is exact, so its running mean is a
    # clean centerline. The decoder is therefore trained in the ordering's own
    # gamma; at inference the encoder maps points into (approximately) that same
    # coordinate.
    gmin, gmax = model.encoder.gamma_range
    visited = ordering_indices >= 0
    ordering = ordering_indices[visited]  # point indices, in visit order
    gamma_ord = jnp.full(ordering_indices.shape, gmin, dtype=prob.dtype)
    gamma_ord = gamma_ord.at[ordering].set(
        jnp.linspace(gmin, gmax, ordering.shape[0], dtype=prob.dtype)
    )
    # Only ordered members train the decoder; unvisited tracers are gaps to fill.
    decoder_mask = is_member & visited

    D = all_ws.shape[1] // 2
    all_qs = all_ws[:, :D]
    mean_fn = RunningMeanDecoder(
        window_size=0.05,
        gamma_train=gamma_ord,
        positions_train=all_qs,
        member_train=decoder_mask,
    )
    mean_qs = jax.vmap(mean_fn, (0, None))(gamma_ord, keys[2])

    # Train the decoder to reconstruct the running-mean positions from gamma.
    config_decoder = config.decoderonly_config()
    decoder, decoder_opt_state, decoder_losses = train_track_net(
        model.decoder,
        gamma=gamma_ord,
        qs_mean=mean_qs,
        mask=decoder_mask,
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
    walk_results: OrderingResult,
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
