"""Configuration for neighbor queries."""

__all__: tuple[str, ...] = ("WalkConfig",)

import equinox as eqx

from .metrics import AbstractDistanceMetric, AlignedMomentumDistanceMetric
from .strategies import AbstractQueryStrategy, BruteForce


class WalkConfig(eqx.Module):
    """Configuration for neighbor queries in walk_local_flow.

    Composes a distance metric with a query strategy. This is the primary
    way to configure how the algorithm selects the next point.

    Parameters
    ----------
    metric : AbstractDistanceMetric
        Distance metric for computing modified distances between points.
        Default: ``AlignedMomentumDistanceMetric()``.
    strategy : AbstractQueryStrategy
        Strategy for finding candidate neighbors.
        Default: ``BruteForce()``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    Default configuration (brute-force with full phase-space metric):

    >>> config = pcf.WalkConfig()
    >>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
    >>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}
    >>> result = pcf.walk_local_flow(
    ...     pos, vel, config=config, start_idx=0, metric_scale=1.0
    ... )

    KD-tree strategy with aligned momentum metric:

    >>> from phasecurvefit.metrics import AlignedMomentumDistanceMetric
    >>> config = pcf.WalkConfig(
    ...     metric=AlignedMomentumDistanceMetric(),
    ...     strategy=pcf.strats.KDTree(k=50),
    ... )

    """

    metric: AbstractDistanceMetric = eqx.field(
        default_factory=AlignedMomentumDistanceMetric
    )
    strategy: AbstractQueryStrategy = eqx.field(default_factory=BruteForce)
