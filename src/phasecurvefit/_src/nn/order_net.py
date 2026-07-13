r"""Ordering Network for interpolating skipped tracers."""

__all__: tuple[str, ...] = (
    # Network components
    "OrderingNet",
    # Training functions
    "train_ordering_net",
    "make_step",
    "OrderingNetTrainer",
    "OrderingTrainingConfig",
    # Loss functions
    "encoder_loss",
    "compute_loss",
)

import functools as ft
from dataclasses import KW_ONLY, dataclass
from typing import Any, ClassVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from jaxmore.nn import masked_mean

from .trainer import AbstractEqxScanTrainer, EqxTrainCarry
from phasecurvefit._src.custom_types import FSz0, FSzN


class OrderingNet(eqx.Module):
    r"""Interpolation network:$(x, v) \;\mapsto\; (\gamma, p),$.

    This network takes N-D phase-space coordinates and outputs:

    - $\gamma \in [0, 1]$: The ordering parameter along the stream
    - $p \in [0, 1]$: The membership probability (1 = likely stream member)

    The architecture follows Appendix B.3 of Nibauer et al. (2022).

    Uses scan-over-layers for improved compilation speed. See:
    https://docs.kidger.site/equinox/tricks/#improve-compilation-speed-with-scan-over-layers

    Parameters
    ----------
    in_size : int
        Number of spatial + kinematic dimensions (6 for 3D: x, y, z, vx, vy,
        vz).
    width_size : int
        The size of each hidden layer.
    depth : int, optional
        The number of hidden layers, not include the input layer or output
        heads.  For example, `depth=2` results in an network with layers:

        ``[Linear(in_size, width_size), Linear(width_size, width_size),
        Linear(width_size, out_size), (output_heads)]``

    key : PRNGKeyArray
        JAX random key for initialization.

    """

    mlp: eqx.nn.MLP
    gamma_head: eqx.nn.Linear
    prob_head: eqx.nn.Linear

    in_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    gamma_range: tuple[float, float] = eqx.field(static=True)

    __citation__: ClassVar[str] = (
        "https://ui.adsabs.harvard.edu/abs/2022ApJ...940...22N/abstract"
    )

    def __init__(
        self,
        in_size: int = 6,
        width_size: int = 100,
        depth: int = 2,
        *,
        gamma_range: tuple[float, float] = (0.0, 1.0),
        key: PRNGKeyArray,
    ) -> None:
        # Store parameters
        self.in_size = in_size
        self.width_size = width_size
        self.depth = depth
        self.gamma_range = gamma_range

        keys = jr.split(key, 3)

        # MLP backbone: input -> hidden layers -> (hidden_size,) The output of
        # this MLP has size hidden_size, which feeds to both output heads
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            width_size=width_size,
            out_size=width_size,  # Output fed to the two heads
            depth=depth,
            activation=jax.nn.tanh,
            final_activation=jax.nn.tanh,
            use_bias=True,
            use_final_bias=True,
            scan=True,
            key=keys[0],
        )

        # Output heads: hidden_size -> 1 (to be squeezed)
        self.gamma_head = eqx.nn.Linear(width_size, 1, key=keys[1])
        self.prob_head = eqx.nn.Linear(width_size, 1, key=keys[2])

    @ft.partial(eqx.filter_jit)
    def __call__(
        self, ws: Float[Array, " {self.n_dims}"], /, key: PRNGKeyArray | None = None
    ) -> tuple[FSz0, FSz0]:
        """Forward pass through the interpolation network.

        Parameters
        ----------
        ws : Array
            Phase-space coordinates of shape (..., 2*n_dims).
            For each point: [x, y, z, vx, vy, vz] (or 2D equivalent).
        key : PRNGKeyArray | None
            Optional key.

        Returns
        -------
        gamma : Array
            Ordering parameter in [0, 1], shape (...).
        prob : Array
            Membership probability in [0, 1], shape (...).

        """
        # MLP backbone
        x = self.mlp(ws, key=key)

        # Output heads with appropriate activations
        gamma_raw = jax.nn.tanh(self.gamma_head(x)).squeeze(-1)  # [-1, 1]
        prob = jax.nn.sigmoid(self.prob_head(x)).squeeze(-1)

        # Rescale gamma from [-1, 1] to gamma_range
        gmin, gmax = self.gamma_range
        gamma = gmin + (gmax - gmin) * (gamma_raw + 1) / 2

        return gamma, prob


# ===================================================================


@eqx.filter_jit
def encoder_loss(
    gamma_true: FSzN,
    gamma_pred: FSzN,
    prob_pred_ordered: FSzN,
    prob_pred_random: FSzN,
    mask: Bool[Array, " N"],
    *,
    lambda_prob: float = 1.0,
) -> FSz0:
    r"""Compute loss for interpolation network training.

    Parameters
    ----------
    gamma_true : Array, shape (N,)
        True $\gamma$ values for ordered stream tracers.
    gamma_pred : Array, shape (N,)
        Predicted $\gamma$ values from the interpolation network.
    prob_pred_ordered : Array, shape (N,)
        Predicted membership probabilities for ordered stream tracers.
        Should be pushed toward 1.
    prob_pred_random : Array, shape (M,)
        Predicted membership probabilities for random phase-space samples.
        Should be pushed toward 0.
    mask : Array, shape (N,)
        Binary mask where True = real data, False = padding.
        Only masked positions contribute to the loss.
    lambda_prob : float, optional
        Weight for probability penalties. Default: 1.0.

    Returns
    -------
    loss : Array
        Scalar loss value.

    """
    # Compute per-element losses
    gamma_sq_error = jnp.square(gamma_true - gamma_pred)
    prob_ordered_sq_error = jnp.square(1.0 - prob_pred_ordered)
    prob_random_sq_error = jnp.square(prob_pred_random)

    # Compute masked mean only over real data
    gamma_loss = masked_mean(gamma_sq_error, mask)
    prob_ordered_penalty = masked_mean(prob_ordered_sq_error, mask)
    prob_random_penalty = masked_mean(prob_random_sq_error, mask)

    return gamma_loss + lambda_prob * (prob_ordered_penalty + prob_random_penalty)


@eqx.filter_value_and_grad
def compute_loss(
    model: OrderingNet,
    ws: Float[Array, " B TwoF"],
    gamma: Float[Array, " B"],
    rand_ws: Float[Array, "B TwoF"],
    mask: Bool[Array, " B"],
    lambda_prob: float = 1.0,
    *,
    key: PRNGKeyArray | None = None,
) -> FSz0:
    r"""Compute interpolation network loss with gradients.

    This function is decorated with ``@eqx.filter_value_and_grad`` to compute
    both the loss value and gradients with respect to the model parameters in
    a single pass. This is the recommended pattern for low-overhead training
    loops in Equinox.

    Parameters
    ----------
    model : OrderingNet
        The interpolation network being trained.
    ws : Array, shape (B, 2*n_dims)
        Batch of ordered phase-space coordinates from stream tracers.
    gamma : Array, shape (B,)
        Target $\gamma$ values for the ordered stream tracers.
    rand_ws : Array, shape (B, 2*n_dims)
        Batch of random phase-space samples (not on stream).
    mask : Array, shape (B,)
        Binary mask where True = real data, False = padding.
        Only masked positions contribute to the loss.
    lambda_prob : float, optional
        Weight for probability loss terms. Default: 1.0.
    key : PRNGKeyArray | None
        Random key.

    Returns
    -------
    loss : Array
        Scalar loss value.

    Notes
    -----
    Due to the ``@eqx.filter_value_and_grad`` decorator, calling this function
    returns a tuple ``(loss, grads)`` where ``grads`` contains gradients with
    respect to the trainable parameters of ``model``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from phasecurvefit.nn import OrderingNet

    >>> key = jax.random.key(0)
    >>> net = OrderingNet(in_size=4, key=key)
    >>> ws = jax.random.normal(key, (10, 4))
    >>> gamma = jnp.linspace(-0.5, 0.5, 10)
    >>> rand_ws = jax.random.normal(key, (10, 4))
    >>> mask = jnp.ones(10, dtype=bool)

    >>> loss, grads = compute_loss(net, ws, gamma, rand_ws, mask, lambda_prob=1.0)

    """
    # Predictions on ordered stream tracers (vectorized over batch)
    model_fn = jax.vmap(model, (0, None))
    gamma_pred, prob_pred_ordered = model_fn(ws, key)

    # Predictions on random phase-space samples (vectorized over batch)
    _, prob_pred_random = model_fn(rand_ws, key)

    # Compute loss using the loss function
    return encoder_loss(
        gamma_true=gamma,
        gamma_pred=gamma_pred,
        prob_pred_ordered=prob_pred_ordered,
        prob_pred_random=prob_pred_random,
        mask=mask,
        lambda_prob=lambda_prob,
    )


# TODO: https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops
@eqx.filter_jit
def make_step(
    model_dynamic: OrderingNet,
    model_static: OrderingNet,
    /,
    ord_ws: Float[Array, "B 2D"],
    ord_gamma: Float[Array, " B"],
    rand_ws: Float[Array, "B 2D"],
    mask: Bool[Array, " B"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    *,
    lambda_prob: float,
    key: PRNGKeyArray,
) -> tuple[FSz0, OrderingNet, optax.OptState]:
    r"""Run a single optimization step for the interpolation network.

    Parameters
    ----------
    model_dynamic, model_static : OrderingNet
        The dynamic and static components of the ordering network being trained.
    ord_ws : Array, shape (B, 2*n_dims)
        Batch of ordered phase-space coordinates from stream tracers.
    ord_gamma : Array, shape (B,)
        Target $\gamma$ values for the ordered stream tracers.
    rand_ws : Array, shape (B, 2*n_dims)
        Batch of random phase-space samples (not on stream).
    mask : Array, shape (B,)
        Binary mask where True = real data, False = padding.
        Only masked positions contribute to the loss.
    opt_state : optax.OptState
        Optimizer state.
    optimizer : optax.GradientTransformation
        Optax optimizer instance.
    lambda_prob : float, optional
        Weight for probability loss terms. Default: 1.0.
    key : PRNGKeyArray
        Ranodm key.

    Returns
    -------
    loss : Array
        Scalar loss value.
    model : OrderingNet
        Updated model after applying gradients.
    opt_state : optax.OptState
        Updated optimizer state.

    """
    # Reconstruct full model from dynamic and static parts
    model = eqx.combine(model_dynamic, model_static)

    # Compute loss and gradients
    loss, grads = compute_loss(
        model, ord_ws, ord_gamma, rand_ws, mask, lambda_prob=lambda_prob, key=key
    )
    # Update the dynamic components of the model
    updates, opt_state = optimizer.update(grads, opt_state, model_dynamic)
    model_dynamic = cast("OrderingNet", eqx.apply_updates(model_dynamic, updates))
    return loss, model_dynamic, opt_state


default_optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-7)


@dataclass
class OrderingTrainingConfig:
    """Configuration for interpolation network training."""

    _: KW_ONLY

    optimizer: optax.GradientTransformation = default_optimizer

    n_epochs: int = 800
    """Number of training epochs."""

    batch_size: int = 100
    """Batch size for training."""

    lambda_prob: float = 1.0
    """Weight for probability loss terms."""

    arclength_alpha: float = 0.8
    r"""Blend factor for arclength-like targets.

    If 0.0 (default), the training targets use uniform spacing in $\gamma$.
    If 1.0, the targets are proportional to cumulative arclength along the
    ordered tracers (computed from the position components of `all_ws`). Values
    in (0, 1) linearly blend the two targets.

    """

    show_pbar: bool = True
    """Show an epoch progress bar via tqdm."""


def _ordering_step(
    carry: EqxTrainCarry,
    batch_inputs: tuple[Bool[Array, " B"], tuple[Array, ...]],
    *,
    optimizer: optax.GradientTransformation,
    filter_spec: Any,
    lambda_prob: float,
) -> tuple[FSz0, EqxTrainCarry]:
    """Run one batch of OrderingNet training.

    `batch_inputs` is ``(mask, (ord_ws, ord_gamma, rand_ws))``, as produced by
    `shuffle_and_batch` from the epoch data assembled in `OrderingNetTrainer.init`
    and `OrderingNetTrainer.prepare_data_args`.

    `filter_spec` must be the same spec the trainer used to build `opt_state`,
    so that the step, the carry packing, and the optimizer state all agree on
    which leaves are trainable.
    """
    model, opt_state, key = carry
    mask, (ord_ws, ord_gamma, rand_ws) = batch_inputs

    # Gradients are taken w.r.t. the dynamic half only; the static half carries
    # any frozen parameters and the non-array structure.
    model_dynamic, model_static = eqx.partition(model, filter_spec)

    key, subkey = jr.split(key)
    loss, model_dynamic, opt_state = make_step(
        model_dynamic,
        model_static,
        ord_ws=ord_ws,
        ord_gamma=ord_gamma,
        rand_ws=rand_ws,
        mask=mask,
        opt_state=opt_state,
        optimizer=optimizer,
        lambda_prob=lambda_prob,
        key=subkey,
    )

    model = eqx.combine(model_dynamic, model_static)
    return loss, (model, opt_state, key)


@dataclass(frozen=True)
class OrderingNetTrainer(AbstractEqxScanTrainer):
    """Scan trainer for `OrderingNet`.

    Fresh random (off-stream) phase-space samples are drawn every epoch, so that
    the membership head sees new negatives each pass rather than memorising a
    fixed set. That is what `prepare_data_args` is for.

    """

    random_ws_shape: tuple[int, ...] = ()
    """Shape of the random negative samples: ``(n_ordered, 2 * n_dims)``."""

    ws_min: FSzN | None = None
    """Lower bound of the phase-space box the negatives are drawn from."""

    ws_max: FSzN | None = None
    """Upper bound of the phase-space box the negatives are drawn from."""

    def init(  # type: ignore[override]
        self,
        model: OrderingNet,
        /,
        *,
        ordered_ws: Float[Array, "N TwoF"],
        gamma_target: Float[Array, " N"],
        mask: Bool[Array, " N"],
        optimizer: optax.GradientTransformation,
        key: PRNGKeyArray,
    ) -> tuple[EqxTrainCarry, tuple[Bool[Array, " N"], tuple[Array, ...]]]:
        """Build the initial carry and the epoch data.

        The third data array is a placeholder for the random negatives; it is
        replaced every epoch by `prepare_data_args`. It is present here only so
        the pytree structure that `jax.lax.scan` sees is fixed from the start.
        """
        model_dynamic, _ = eqx.partition(model, self.filter_spec)
        opt_state = optimizer.init(model_dynamic)
        initial_carry = (model, opt_state, key)

        placeholder_rand_ws = jnp.zeros(self.random_ws_shape, dtype=ordered_ws.dtype)
        epoch_data = (mask, (ordered_ws, gamma_target, placeholder_rand_ws))
        return initial_carry, epoch_data

    def prepare_data_args(
        self,
        carry: EqxTrainCarry,
        data_args: tuple[Array, ...],
        /,
        *,
        epoch_idx: Int[Array, ""],
        num_epochs: int,
        epoch_key: PRNGKeyArray,
    ) -> tuple[Array, ...]:
        """Draw fresh random negatives for this epoch."""
        del carry, epoch_idx, num_epochs
        ordered_ws, gamma_target, _ = data_args
        random_ws = jr.uniform(
            epoch_key,
            shape=self.random_ws_shape,
            minval=self.ws_min,
            maxval=self.ws_max,
        )
        return (ordered_ws, gamma_target, random_ws)


def train_ordering_net(
    model: OrderingNet,
    all_ws: Float[Array, "N TwoF"],
    ordering_indices: Int[Array, "N"],
    /,
    config: OrderingTrainingConfig | None = None,
    *,
    key: PRNGKeyArray,
) -> tuple[OrderingNet, optax.OptState, Float[Array, "E"]]:
    r"""Train the interpolation network on ordered stream tracers.

    This implementation uses lax.scan for efficient batching and supports Orbax
    checkpointing. The training follows the pattern from the original
    autoencoder for maximum performance.

    Parameters
    ----------
    model : OrderingNet
        The interpolation network to train.
    all_ws : Array, shape (N, 2*n_dims)
        Phase-space coordinates from the walk algorithm.
    ordering_indices : Array, shape (N,)
        Indices representing the ordering of tracers.  Unvisited tracers have
        indices of -1.
    config : OrderingTrainingConfig | None
        Training configuration. If None, uses default config.  Set
        `arclength_alpha>0` to encourage $\gamma$ to be proportional to
        cumulative arclength along the ordered tracers (reducing local
        compression and improving rolling-mean decoders).
    key : PRNGKeyArray
        Random key for shuffling and random sampling.

    Returns
    -------
    trained_net : OrderingNet
        The trained interpolation network.
    opt_state : optax.OptState
        The optimizer state after training.
    losses : Array, shape (n_epochs,)
        Training loss per epoch.

    """
    if config is None:
        config = OrderingTrainingConfig()

    # TODO: not need to slice. Instead use a mask or something.
    # But changing this goes from ~800 epochs/s to ~30 epochs/s.
    is_ordered = ordering_indices >= 0
    ordered_ws = all_ws[ordering_indices][is_ordered]
    # ordered_ws = all_ws[is_ordered]

    # Shapes
    shape = ordered_ws.shape
    n_total = shape[0]

    # Compute phase-space bounds for random sampling
    ws_min = jnp.min(jnp.where(is_ordered[:, None], all_ws, jnp.inf), axis=0)
    ws_max = jnp.max(jnp.where(is_ordered[:, None], all_ws, -jnp.inf), axis=0)

    # Edge case: if the curve is degenerate (e.g., 1D line), some dimensions
    # will have min=max. Expand bounds to avoid random samples landing on curve.
    extent = ws_max - ws_min
    has_degenerate_dims = extent == 0
    extent = jnp.where(
        has_degenerate_dims, 1.0, extent
    )  # Use unit extent if degenerate
    ws_min = ws_min - 0.5 * extent
    ws_max = ws_max + 0.5 * extent

    # Initialize gamma targets.
    # Base target: uniform spacing in [gamma_min, gamma_max]
    gmin, gmax = model.gamma_range
    gamma_lin = jnp.linspace(gmin, gmax, n_total)

    # Optional arclength-like target (computed from positions of ordered
    # tracers) This mitigates tanh-compression in gamma by encouraging gamma to
    # track cumulative arclength rather than index.
    D = ordered_ws.shape[1] // 2
    qs_ord = ordered_ws[:, :D]

    # Cumulative arclength proxy s
    dqs = qs_ord[1:] - qs_ord[:-1]
    ds = jnp.linalg.norm(dqs, axis=1)
    s = jnp.concatenate([jnp.zeros((1,), dtype=ds.dtype), jnp.cumsum(ds)])
    s_tot = s[-1]

    # Map s -> gamma_range, guarding against degenerate s_tot
    gamma_arc = jnp.where(s_tot > 0, gmin + (gmax - gmin) * (s / s_tot), gamma_lin)

    # Blend: alpha=0 -> gamma_lin, alpha=1 -> gamma_arc
    alpha = jnp.asarray(config.arclength_alpha, dtype=gamma_lin.dtype)
    alpha = jnp.clip(alpha, 0.0, 1.0)
    gamma_target = (1.0 - alpha) * gamma_lin + alpha * gamma_arc

    # ----------------------------------------
    # Train

    optimizer = config.optimizer
    # Every row of `ordered_ws` is a real ordered tracer, so all are usable.
    # Padding introduced by batching is still masked out by `shuffle_and_batch`.
    ordered_mask = jnp.ones(n_total, dtype=bool)

    # Single source of truth for what is trainable: the step, the carry packing,
    # and `optimizer.init` must all partition the model the same way.
    filter_spec: Any = eqx.is_array

    trainer = OrderingNetTrainer(
        make_step=ft.partial(
            _ordering_step,
            optimizer=optimizer,
            filter_spec=filter_spec,
            lambda_prob=config.lambda_prob,
        ),
        loss_agg_fn=masked_mean,
        filter_spec=filter_spec,
        random_ws_shape=shape,
        ws_min=ws_min,
        ws_max=ws_max,
    )
    initial_carry, epoch_data = trainer.init(
        model,
        ordered_ws=ordered_ws,
        gamma_target=gamma_target,
        mask=ordered_mask,
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

    return cast("OrderingNet", model), opt_state, epoch_losses
