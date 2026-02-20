__all__: tuple[str, ...] = ("AbstractWalkResult",)

import abc
from dataclasses import KW_ONLY

import equinox as eqx
from jaxtyping import Array

from .custom_types import ISzN, VectorComponents


class AbstractWalkResult(eqx.Module):
    """Abstract base class for walk algorithm results.

    This class defines the interface for walk result types that combine
    phase-space data with an ordering and interpolation capability.

    Subclasses must implement the `__call__` method to enable spatial
    interpolation from an ordering parameter.

    Attributes
    ----------
    positions : dict[str, Array]
        Position dictionary with keys (e.g., "x", "y", "z") and values as
        1D arrays of shape (n_obs,). These are the original positions from
        the input, not reordered.
    velocities : dict[str, Array]
        Velocity dictionary with same keys and shape as ``positions``.
        These are the original velocities from the input, not reordered.
    indices : Int[Array, " n_obs"]
        Ordered indices of visited observations. Unvisited observations are
        marked with -1. The position in the array indicates the order in the
        walk, and the value at that position is the original observation index.
    gamma_range : tuple[float, float]
        Static keyword-only argument specifying the valid range of the
        ordering parameter in `__call__`. Default is (0.0, 1.0).

    """

    positions: VectorComponents
    velocities: VectorComponents
    indices: ISzN
    _: KW_ONLY
    gamma_range: tuple[float, float] = eqx.field(static=True, default=(0.0, 1.0))

    @abc.abstractmethod
    def __call__(self, gamma: Array, /) -> VectorComponents:
        r"""Interpolate spatial positions from ordering parameter $\gamma$.

        Parameters
        ----------
        gamma : Array
            Ordering parameter in [min_gamma, max_gamma], shape (...).
            Will be clipped to valid range and normalized for interpolation.

        Returns
        -------
        positions : dict[str, Array]
            Interpolated spatial positions with same shape (...) as gamma.

        """
        ...
