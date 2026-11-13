"""One track drawn from :func:`~phasecurvefit.sample_track`'s posterior."""

__all__: tuple[str, ...] = ("TrackDraw",)

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from phasecurvefit._src._repr import ReprMixin


class TrackDraw(ReprMixin, eqx.Module):
    r"""One track from a posterior, callable on its $\gamma$.

    Made by :func:`~phasecurvefit.sample_track`. The track and its width are
    held on a grid in $\gamma \in [-1, 1]$, in units where the positions have
    zero mean and unit spread, and are interpolated linearly between grid
    points. Calling the draw puts them back in the units of the data, so with
    `unxt` Quantity inputs it returns Quantities.

    Attributes
    ----------
    gamma : Array
        The grid, shape ``(K,)``.
    x : Array
        The track at ``gamma``, scaled, shape ``(K, D)``, with the last axis
        in the order of the position components.
    sigma : Array
        The track's width (the standard deviation of the stars about it, per
        axis) at ``gamma``, scaled, shape ``(K,)``.
    centre : dict
        Each position component's mean, which the scaling subtracted.
    scale : Array or Quantity
        The one length the scaling divided by.
    track_range : tuple[float, float]
        The $\gamma$ range the stars cover. With a latent line density it holds
        the central 99% of it; otherwise it is ``(-1, 1)``.

    """

    gamma: Float[Array, " K"]
    x: Float[Array, "K D"]
    sigma: Float[Array, " K"]
    centre: dict[str, Any]
    scale: Any
    track_range: tuple[float, float] = eqx.field(static=True)

    def __call__(self, gamma: ArrayLike, /) -> dict[str, Any]:
        r"""Evaluate the track's position components at ``gamma``."""
        g = jnp.asarray(gamma)
        return {
            k: jnp.interp(g, self.gamma, self.x[:, i]) * self.scale + c
            for i, (k, c) in enumerate(self.centre.items())
        }

    def width(self, gamma: ArrayLike, /) -> Any:
        r"""Evaluate the track's width at ``gamma``, in the units of the positions."""
        return jnp.interp(jnp.asarray(gamma), self.gamma, self.sigma) * self.scale
