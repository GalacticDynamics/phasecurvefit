"""Autoencoder result types with interpolation capability."""

__all__: tuple[str, ...] = ("AutoencoderResult", "fill_ordering_gaps")

from dataclasses import KW_ONLY

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .abstractautoencoder import AbstractAutoencoder
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.custom_types import FSzN, ISzN, VectorComponents


class AutoencoderResult(AbstractResult):
    r"""Result of autoencoder training and prediction with interpolation.

    Extends AbstractResult with training outputs (gamma, membership_prob) and
    provides spatial interpolation via the trained decoder network.

    Attributes
    ----------
    positions : VectorComponents
        Original position data (dict with 1D arrays).
    velocities : VectorComponents
        Original velocity data (dict with 1D arrays).
    indices : ISzN
        Indices sorted by gamma value.
    gamma_range : tuple[float, float]
        Valid range for the ordering parameter. Default: (0.0, 1.0).
    gamma : FSzN
        Ordering parameters for all tracers.
    membership_prob : Array
        Membership probabilities for all tracers.
    model : AbstractAutoencoder
        Trained autoencoder model with decoder for spatial interpolation.

    """

    positions: VectorComponents
    velocities: VectorComponents
    indices: ISzN
    gamma: FSzN
    membership_prob: FSzN
    model: AbstractAutoencoder
    _: KW_ONLY
    gamma_range: tuple[float, float]

    def __call__(
        self, gamma: Array, /, *, key: PRNGKeyArray | None = None
    ) -> VectorComponents:
        r"""Interpolate spatial positions using the decoder network.

        Parameters
        ----------
        gamma : Array
            Ordering parameter in [min_gamma, max_gamma], shape (...).
            Will be clipped to valid range and normalized for interpolation.
        key : PRNGKeyArray, optional
            JAX random key passed to the decoder for stochastic evaluation.
            If None, uses deterministic evaluation with a zero key. Default is None.

        Returns
        -------
        positions : VectorComponents
            Interpolated spatial positions from the decoder with same shape
            (...) as gamma.

        """
        # Error if gamma is out of range
        min_gamma, max_gamma = self.gamma_range
        gamma = eqx.error_if(
            gamma,
            (gamma < min_gamma) | (gamma > max_gamma),
            "gamma must be within the valid gamma_range",
        )

        # Use the decoder to interpolate positions
        qs_norm = jax.vmap(self.model.decoder, (0, None))(jnp.atleast_1d(gamma), key)

        # Handle scalar gamma case
        qs_norm = qs_norm.squeeze() if gamma.ndim == 0 else qs_norm

        # Inverse transform to get original coordinates
        qs, _ = self.model.normalizer.inverse_transform(
            qs_norm, jnp.zeros_like(qs_norm)
        )
        return qs


def fill_ordering_gaps(
    model: AbstractAutoencoder,
    result: AbstractResult,
    /,
    prob_threshold: float = 0.5,
) -> AutoencoderResult:
    r"""Use trained autoencoder to fill gaps in phase-flow walk ordering.

    This function predicts $\gamma$ values for all tracers (including those
    skipped by phase-flow walk) and returns a complete ordering.

    Parameters
    ----------
    model : PathAutoencoder
        Trained autoencoder model.
    result : AbstractResult
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
    >>> import phasecurvefit as pcf

    >>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
    >>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
    >>> result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)
    >>> keys = jax.random.split(jax.random.key(0), 2)
    >>> normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
    >>> model = pcf.nn.PathAutoencoder.make(
    ...     normalizer, gamma_range=result.gamma_range, key=keys[0]
    ... )
    >>> cfg = pcf.nn.TrainingConfig(show_pbar=False)
    >>> result, *_ = pcf.nn.train_autoencoder(model, result, config=cfg, key=keys[1])

    """
    q, p = result.positions, result.velocities

    # Predict gamma and probability for all tracers
    gamma, prob = model.encode(q, p)
    # Sort by gamma to get ordering
    sorted_indices = jnp.argsort(gamma)

    # Filter by probability threshold
    high_prob_mask = prob[sorted_indices] >= prob_threshold
    filtered_indices = sorted_indices[high_prob_mask]

    return AutoencoderResult(
        positions=q,
        velocities=p,
        indices=filtered_indices,
        gamma_range=result.gamma_range,
        gamma=gamma,
        membership_prob=prob,
        model=model,
    )
