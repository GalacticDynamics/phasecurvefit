"""Simple autoencoder with user-provided decoder function."""

import abc

__all__: tuple[str, ...] = ("AbstractExternalDecoder", "RunningMeanDecoder")

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .normalize import AbstractNormalizer
from .order_net import OrderingNet
from phasecurvefit._src.custom_types import FSz0, FSzN, RSz0

if TYPE_CHECKING:
    import localflowwalk

Gamma: TypeAlias = FSzN  # noqa: UP040


class AbstractExternalDecoder(eqx.Module):
    """Abstract base class for external decoders."""

    def __call__(self, gamma: RSz0) -> Float[Array, " D"]:
        """Decode a single gamma value to position.

        Parameters
        ----------
        gamma : Array, shape ()
            Single gamma value to decode.

        Returns
        -------
        position : Array, shape (D,)
            Reconstructed position.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self,
        model: "localflowwalk.nn.EncoderExternalDecoder",
        all_ws: Float[Array, " N TwoF"],
    ) -> "AbstractExternalDecoder":
        """Update decoder parameters.

        This method should return a new instance of the decoder with updated
        parameters based on the provided keyword arguments.

        Parameters
        ----------
        model : EncoderExternalDecoder
            The encoder-external decoder model instance.
        all_ws : Float[Array, " N TwoF"]
            Array of all weights or other relevant data for updating.

        Returns
        -------
        AbstractExternalDecoder
            A new instance of the decoder with updated parameters.

        """


class RunningMeanDecoder(AbstractExternalDecoder):
    """Running-mean decoder for non-parametric position reconstruction.

    This decoder stores the training data and reconstructs positions
    from gamma values using a windowed running mean.

    Attributes
    ----------
    gamma_train : Array
        Precomputed gamma values for all training samples.
    positions_train : Array
        Training positions (normalized).
    window_size : float
        Window size in gamma-space.

    """

    window_size: float
    gamma_train: Float[Array, " N"] | None = None
    positions_train: Float[Array, " N D"] | None = None
    member_train: Bool[Array, " N"] | None = None

    def __call__(self, gamma: RSz0, /, key: PRNGKeyArray | None = None) -> FSz0:
        """Decode a single gamma value to position using running mean.

        Parameters
        ----------
        gamma : Array, shape ()
            Single gamma value to decode.
        key : PRNGKeyArray, optional
            Random key for any stochastic operations (not used here, but
            included for consistency).

        Returns
        -------
        position : Array, shape (D,)
            Reconstructed position.

        """
        del key  # Not used, but included for signature consistency
        (gamma_train, q_train, member_train) = eqx.error_if(
            (self.gamma_train, self.positions_train, self.member_train),
            self.gamma_train is None
            or self.positions_train is None
            or self.member_train is None,
            "Decoder not initialized with training data.",
        )

        # Error if gamma is outside the training data bounds
        gamma = eqx.error_if(
            gamma,
            gamma < jnp.min(gamma_train),
            "gamma is below minimum training gamma value",
        )
        gamma = eqx.error_if(
            gamma,
            gamma > jnp.max(gamma_train),
            "gamma is above maximum training gamma value",
        )

        # Find samples within the window
        in_window = jnp.abs(gamma_train - gamma) < (self.window_size / 2)

        # Compute weighted mean (uniform weights within window)
        weights = in_window.astype(q_train.dtype) * member_train.astype(q_train.dtype)
        total_weight = jnp.sum(weights) + 1e-10

        # Weighted mean position
        weighted_pos = jnp.sum(q_train * weights[:, None], axis=0)
        return weighted_pos / total_weight

    @classmethod
    def make(
        cls,
        encoder: OrderingNet,
        normalizer: AbstractNormalizer,
        positions: Mapping[str, Array] | Float[Array, "N D"],
        velocities: Mapping[str, Array] | Float[Array, "N D"],
        /,
        *,
        key: PRNGKeyArray | None = None,
        window_size: float = 0.05,
        member_threshold: float = 0.5,
    ) -> "RunningMeanDecoder":
        r"""Create a running-mean decoder for non-parametric position reconstruction.

        This classmethod constructs a decoder that reconstructs positions from
        $\gamma$ values using a windowed running mean in $\gamma$-space. For each
        query $\gamma$, the decoder:

        1. Predicts $\gamma$ values for all training samples using the encoder
        2. Finds samples whose $\gamma$ is within [$\gamma$ - window_size/2,
           $\gamma$ + window_size/2]
        3. Returns the mean position of those samples

        This provides a simple, non-parametric alternative to training a neural
        network decoder. It works well when the stream is smooth and well-sampled.

        Parameters
        ----------
        encoder : OrderingNet
            Trained encoder network that predicts $\gamma$ from phase-space.
        normalizer : AbstractNormalizer
            Normalizer used to preprocess the phase-space data.
        positions : Mapping[str, Array] or Array, shape (N, D)
            Training positions. If Mapping, will be transformed by normalizer.
            If Array, assumed to be already normalized.
        velocities : Mapping[str, Array] or Array, shape (N, D)
            Training velocities. Same format as positions.
        key : PRNGKeyArray, optional
            Random key for any stochastic operations (not used here, but
            included for consistency).
        window_size : float, default=0.1
            Window size in $\gamma$-space for computing running mean.  Smaller
            values give more localized (less smooth) reconstruction.
        member_threshold : float, default=0.5
            Membership probability threshold for including samples in the
            running mean.

        Returns
        -------
        decoder : RunningMeanDecoder
            Decoder module mapping $\gamma$ to positions.
            The returned decoder is JIT-compatible and vmappable.

        Notes
        -----
        The running mean is computed using a uniform (rectangular) kernel:

        $$
        \hat{x}(\gamma)
            = \frac{\sum_i x_i \cdot \mathbb{1}_{|\gamma_i - \gamma| < w/2}}
                    {\sum_i \mathbb{1}_{|\gamma_i - \gamma| < w/2}}
        $$

        where $w$ is the window size.

        The decoder stores the encoder and training data, so it can be used
        independently after creation.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import phasecurvefit as pcf

        Create sample stream data

        >>> key = jr.key(0)
        >>> N = 100
        >>> t = jnp.linspace(0, 2 * jnp.pi, N)
        >>> positions = {"x": jnp.cos(t), "y": jnp.sin(t)}
        >>> velocities = {"x": -jnp.sin(t), "y": jnp.cos(t)}

        Create and train encoder

        >>> key, key1, key2 = jr.split(key, 3)
        >>> normalizer = pcf.nn.StandardScalerNormalizer(positions, velocities)
        >>> encoder = pcf.nn.OrderingNet(
        ...     in_size=4, width_size=32, depth=2, gamma_range=(0.0, 1.0), key=key1
        ... )

        >>> # Prepare training data: concatenate normalized positions and velocities
        >>> qs_norm, ps_norm = normalizer.transform(positions, velocities)
        >>> all_ws = jnp.concatenate([qs_norm, ps_norm], axis=1)
        >>> # For this synthetic example, the ordering is just 0, 1, 2, ..., N-1
        >>> ordering_indices = jnp.arange(N)

        >>> # Train the encoder on the ordered stream data
        >>> config = pcf.nn.OrderingTrainingConfig(n_epochs=10, show_pbar=False)
        >>> trained_encoder, _, _ = pcf.nn.train_ordering_net(
        ...     encoder, all_ws, ordering_indices, config, key=key2
        ... )

        >>> # Create decoder using trained encoder
        >>> decoder = pcf.nn.RunningMeanDecoder.make(
        ...     trained_encoder, normalizer, positions, velocities, window_size=0.05
        ... )
        >>> # Reconstruct positions for gamma values within the training range.
        >>> # The decoder can only interpolate gamma values that are within the
        >>> # range of gamma predictions from the encoder on its training data.
        >>> gamma_test = jnp.array([0.4, 0.5, 0.6])
        >>> reconstructed = jax.vmap(decoder)(gamma_test)
        >>> reconstructed.shape
        (3, 2)

        """
        # Preprocess the data
        if isinstance(positions, Mapping):
            qs_norm, ps_norm = normalizer.transform(positions, velocities)
        else:
            # Assume already normalized array form
            qs_norm = positions
            ps_norm = velocities

        # Predict gamma for all training samples (only need gamma, not prob)
        ws_norm = jnp.concatenate([qs_norm, ps_norm], axis=1)
        gamma, prob = jax.vmap(encoder, in_axes=(0, None))(ws_norm, key)
        is_member = prob >= member_threshold

        # Create the decoder module
        return cls(
            gamma_train=gamma,
            positions_train=qs_norm,
            member_train=is_member,
            window_size=window_size,
        )

    def update(
        self,
        model: "localflowwalk.nn.EncoderExternalDecoder",
        all_ws: Float[Array, " N TwoF"],
    ) -> "RunningMeanDecoder":
        """Update decoder parameters.

        This method returns a new instance of the decoder with updated
        parameters based on the provided keyword arguments.

        Parameters
        ----------
        model : EncoderExternalDecoder
            The encoder-external decoder model instance.
        all_ws : Float[Array, " N TwoF"]
            Array of all weights or other relevant data for updating.

        Returns
        -------
        new_decoder : RunningMeanDecoder
            New instance of RunningMeanDecoder with updated parameters.

        """
        # Recreate decoder with trained encoder
        # Extract positions and velocities from ws
        D = all_ws.shape[1] // 2
        qs_norm = all_ws[:, :D]
        ps_norm = all_ws[:, D:]

        # Update decoder with trained encoder
        decoder = self.make(
            model.encoder,
            model.normalizer,
            qs_norm,
            ps_norm,
            window_size=self.window_size,
        )
        return decoder  # noqa: RET504
