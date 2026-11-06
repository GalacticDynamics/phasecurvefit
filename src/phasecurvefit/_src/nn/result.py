"""Autoencoder result types with interpolation capability."""

__all__: tuple[str, ...] = ("AutoencoderResult", "fill_ordering_gaps")

from dataclasses import KW_ONLY

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .abstractautoencoder import AbstractAutoencoder
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.custom_types import BSzN, FSzN, ISzN, VectorComponents


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
        Valid range for the ordering parameter: calling the result accepts any
        $\gamma$ in it, including every tracer's own.
    track_range : tuple[float, float] | None
        The range of $\gamma$ the track covers, for drawing it:
        ``result(jnp.linspace(*result.track_range, n))``. It ends where the
        weighted number of member tracers per unit $\gamma$ falls below half
        its median, so faint tracers past the end of a stream -- the blurred
        tail of a density field -- don't carry the track past the end. Within
        ``gamma_range``. ``None`` for a result built by hand; training sets it.
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
    track_range: tuple[float, float] | None = None

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

        return self.model.decode(gamma, key=key)


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
        Result from an orderer.
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
    >>> result = pcf.order(pos, vel)
    >>> keys = jax.random.split(jax.random.key(0), 2)
    >>> normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
    >>> model = pcf.nn.PathAutoencoder.make(
    ...     normalizer, gamma_range=result.gamma_range, key=keys[0]
    ... )
    >>> cfg = pcf.nn.TrainingConfig(show_pbar=False)
    >>> result, *_ = pcf.nn.train_autoencoder(model, result, config=cfg, key=keys[1])

    """
    return _encode_to_result(
        model,
        result.positions,
        result.velocities,
        prob_threshold=prob_threshold,
        gamma_range=result.gamma_range,
        weights=getattr(result, "weights", None),
    )


def _encode_to_result(
    model: AbstractAutoencoder,
    q: VectorComponents,
    p: VectorComponents,
    /,
    *,
    prob_threshold: float,
    gamma_range: tuple[float, float],
    weights: FSzN | None = None,
) -> AutoencoderResult:
    """Encode every tracer; order by gamma, keeping those with p >= threshold."""
    gamma, prob = model.encode(q, p)
    order = jnp.argsort(gamma)
    return AutoencoderResult(
        positions=q,
        velocities=p,
        indices=order[prob[order] >= prob_threshold],
        gamma_range=gamma_range,
        track_range=_track_range(gamma, prob >= prob_threshold, weights),
        gamma=gamma,
        membership_prob=prob,
        model=model,
    )


def _track_range(
    gamma: FSzN, member: BSzN, weights: FSzN | None, /
) -> tuple[float, float]:
    """Find the span of gamma where the weighted member count is at least half typical.

    Bins the members' gamma over their own span, with one bin per 10 members up
    to 40, and keeps everything from the first to the last bin holding at least
    half the median weight of the non-empty bins. For a curve blurred by any
    symmetric kernel the along-track count is half its interior value at the
    true end, so this is the half-maximum end. Evenly sampled point clouds lose
    at most a bin at each end.
    """
    g = np.asarray(gamma, dtype=float)
    keep = np.asarray(member, dtype=bool)
    if not keep.any():  # nothing to trim against
        return float(g.min()), float(g.max())
    w = np.ones_like(g) if weights is None else np.asarray(weights, dtype=float)
    g, w = g[keep], w[keep]
    n_bins = int(np.clip(g.size // 10, 1, 40))
    edges = np.linspace(g.min(), g.max(), n_bins + 1)
    count, _ = np.histogram(g, edges, weights=w)
    ok = np.flatnonzero(count >= 0.5 * np.median(count[count > 0]))
    return float(edges[ok[0]]), float(edges[ok[-1] + 1])
