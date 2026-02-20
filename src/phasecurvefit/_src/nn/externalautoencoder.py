"""Simple autoencoder with user-provided decoder function."""

__all__: tuple[str, ...] = ("EncoderExternalDecoder",)

from typing import TypeAlias

import equinox as eqx
import plum
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from .abstractautoencoder import AbstractAutoencoder
from .autoencoder import TrainingConfig
from .externaldecoder import AbstractExternalDecoder
from .normalize import AbstractNormalizer
from .order_net import OrderingNet, OrderingTrainingConfig, train_ordering_net
from .result import AutoencoderResult
from phasecurvefit._src.custom_types import FSzN

Gamma: TypeAlias = FSzN  # noqa: UP040


class EncoderExternalDecoder(AbstractAutoencoder):
    r"""Autoencoder with trained encoder and user-provided decoder function.

    This class provides a flexible alternative to `PathAutoencoder` where the
    decoder is a user-provided function rather than a trained neural network.
    The most common use case is with a running-mean decoder that performs
    non-parametric interpolation in $\gamma$-space.

    The encoder (OrderingNet) is trained to predict $\gamma$ and membership
    probability $p$ from phase-space coordinates. The decoder function maps
    $\gamma$ values back to positions.

    Attributes
    ----------
    encoder : OrderingNet
        Neural network mapping $(x, v) \to (\gamma, p)$.
    decoder : Callable[[Array], Array]
        Function mapping $\gamma \to x$. Should be vmappable and JIT-compatible.
    normalizer : AbstractNormalizer
        Data normalizer for preprocessing phase-space coordinates.

    See Also
    --------
    PathAutoencoder : Full autoencoder with trainable decoder network.
    RunningMeanDecoder.make : Create a running-mean decoder.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> import phasecurvefit as pcf
    >>> # Create sample data
    >>> key = jr.key(0)
    >>> positions = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.0, 0.0])}
    >>> velocities = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
    >>> ordering = jnp.array([0, 1, 2])
    >>> # Create normalizer and encoder
    >>> normalizer = pcf.nn.StandardScalerNormalizer(positions, velocities)
    >>> encoder = pcf.nn.OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(1))
    >>> decoder = pcf.nn.RunningMeanDecoder(window_size=0.05)
    >>> # Create model
    >>> model = pcf.nn.EncoderExternalDecoder(
    ...     encoder=encoder, decoder=decoder, normalizer=normalizer
    ... )

    """

    encoder: OrderingNet
    decoder: AbstractExternalDecoder
    normalizer: AbstractNormalizer

    @property
    def gamma_range(self) -> tuple[float, float]:
        """Return the gamma range from the encoder."""
        return self.encoder.gamma_range


@plum.dispatch
def train_autoencoder(
    model: EncoderExternalDecoder,
    all_ws: Float[Array, " N TwoF"],
    ordering_indices: Int[Array, " N"],
    /,
    *,
    config: OrderingTrainingConfig | TrainingConfig | None = None,
    key: PRNGKeyArray,
) -> tuple[AutoencoderResult, dict[str, PyTree], Float[Array, " {config.n_epochs}"]]:
    """Train the EncoderExternalDecoder encoder and create running-mean decoder.

    This function provides a simplified training workflow:

    1. Train the encoder (OrderingNet) using supervised learning from ordering indices
    2. Create a running-mean decoder using the trained encoder and training data

    Unlike `train_autoencoder` for `PathAutoencoder`, this does not train a decoder
    network. Instead, it uses the provided (or default) decoder function.

    Parameters
    ----------
    model : EncoderExternalDecoder
        The autoencoder model to train. Its encoder will be updated.
    all_ws : Array, shape (N, 2*n_dims)
        All phase-space coordinates (positions + velocities) in normalized form.
    ordering_indices : Array, shape (N,)
        Ordering indices from walk algorithm. Valid indices (>= 0) indicate
        ordered tracers; -1 indicates skipped/unordered tracers.
    config : OrderingTrainingConfig, optional
        Training configuration for the encoder. If None, uses default config.
    decoder_kwargs : Mapping, optional
        Keyword arguments passed to decoder function creation. For running-mean
        decoder, can include 'window_size'. If None, uses defaults.
    key : PRNGKeyArray
        Random key for training.

    Returns
    -------
    result : AutoencoderResult
        Result containing the trained autoencoder and ordering data.
    opt_state : dict[str, PyTree]
        Optimizer state from encoder training (wrapped in dict for consistency).
    losses : Array, shape (n_epochs,)
        Training losses from encoder training.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> import phasecurvefit as pcf

    >>> key = jr.key(0)
    >>> N = 50
    >>> positions = {"x": jnp.arange(N, dtype=float), "y": jnp.zeros(N)}
    >>> velocities = {"x": jnp.ones(N), "y": jnp.zeros(N)}
    >>> ordering = jnp.arange(N)

    >>> model = pcf.nn.EncoderExternalDecoder(
    ...     pcf.nn.OrderingNet(in_size=4, width_size=32, depth=2, key=jr.key(1)),
    ...     pcf.nn.RunningMeanDecoder(window_size=0.05),
    ...     pcf.nn.StandardScalerNormalizer(positions, velocities),
    ... )

    Train (with minimal epochs for demonstration)

    >>> qs_norm, ps_norm = model.normalizer.transform(positions, velocities)
    >>> ws_norm = jnp.concat([qs_norm, ps_norm], axis=1)
    >>> config = pcf.nn.OrderingTrainingConfig(
    ...     n_epochs=10, batch_size=16, show_pbar=False
    ... )
    >>> result, opt_state, losses = pcf.nn.train_autoencoder(
    ...     model, ws_norm, ordering, config=config, key=jr.key(2)
    ... )
    >>> losses.shape
    (10,)

    """
    # ===========================================
    # Train Encoder

    if config is None:
        config = OrderingTrainingConfig()
    elif isinstance(config, TrainingConfig):
        config = config.encoderonly_config()

    encoder, opt_state, losses = train_ordering_net(
        model.encoder, all_ws, ordering_indices, config=config, key=key
    )

    # Update encoder in model
    model = eqx.tree_at(lambda m: m.encoder, model, encoder)

    # ===========================================
    # Update Decoder

    # Update running-mean decoder using trained encoder and training data
    decoder = model.decoder.update(model, all_ws)

    # Update decoder in model
    model = eqx.tree_at(lambda m: m.decoder, model, decoder)

    # Convert all_ws back to VectorComponents for AutoencoderResult
    D = all_ws.shape[1] // 2
    qs_norm = all_ws[:, :D]
    ps_norm = all_ws[:, D:]
    positions, velocities = model.normalizer.inverse_transform(qs_norm, ps_norm)

    # Encode to get gamma and membership_prob
    gamma, membership_prob = model.encode(positions, velocities)

    result = AutoencoderResult(
        model=model,
        positions=positions,
        velocities=velocities,
        indices=ordering_indices,
        gamma=gamma,
        membership_prob=membership_prob,
        gamma_range=model.gamma_range,
    )

    # Wrap opt_state in dict for consistency with other dispatches
    opt_states = {"encoder": opt_state}

    return result, opt_states, losses
