"""Autoencoder."""

__all__: tuple[str, ...] = ("AbstractAutoencoder",)

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from .normalize import AbstractNormalizer
from phasecurvefit._src.custom_types import FSzN, VectorComponents

Gamma: TypeAlias = FSzN  # noqa: UP040


class AbstractAutoencoder(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    normalizer: AbstractNormalizer

    @property
    def gamma_range(self) -> tuple[float, float]:
        """Return the gamma range for this autoencoder."""
        raise NotImplementedError  # pragma: no cover

    def encode(
        self,
        qs: VectorComponents,
        ps: VectorComponents,
        /,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Gamma, FSzN]:
        r"""Encode phase-space coordinates to ($\gamma$, $p$).

        Parameters
        ----------
        qs, ps : VectorComponents
            Spatial / velocity coordinates of shape (N, n_dims).
        key : PRNGKeyArray, optional
            JAX random key for stochastic encoding (if applicable).

        Returns
        -------
        gamma : Array
            Ordering parameter in [0, 1], shape (N,).
        prob : Array
            Membership probability in [0, 1], shape (N,).

        """
        qs_norm, ps_norm = self.normalizer.transform(qs, ps)
        ws_norm = jnp.concatenate([qs_norm, ps_norm], axis=1)
        gamma, prob = jax.vmap(self.encoder, (0, None))(jnp.atleast_2d(ws_norm), key)
        # Remove extra dim if input was 1D
        gamma = gamma.squeeze() if qs_norm.ndim == 1 else gamma
        prob = prob.squeeze() if qs_norm.ndim == 1 else prob
        return gamma, prob

    def decode(
        self, gamma: Gamma, /, *, key: PRNGKeyArray | None = None
    ) -> VectorComponents:
        r"""Decode $\gamma$ to reconstructed position.

        Parameters
        ----------
        gamma : Array
            Ordering parameter, typically in the range defined by gamma_range,
            shape (N,).  Some decoders may support extrapolation beyond this
            range.
        key : PRNGKeyArray, optional
            JAX random key for stochastic decoding (if applicable).

        Returns
        -------
        position : VectorComponents
            Reconstructed dict of positions.

        """
        qs_norm = jax.vmap(self.decoder, (0, None))(jnp.atleast_1d(gamma), key)
        qs_norm = qs_norm.squeeze() if gamma.ndim == 0 else qs_norm
        qs, _ = self.normalizer.inverse_transform(qs_norm, jnp.zeros_like(qs_norm))
        return qs
