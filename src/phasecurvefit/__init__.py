"""phasecurvefit.

This library implements algorithms for ordering phase-space observations in
stellar streams. By default, observations are ordered along an MST backbone
refined by a Self-Organizing Map; the velocity-following local-flow walk, which
uses both spatial proximity and velocity momentum to trace coherent structures
through phase-space, is available as an alternative orderer.

Phase-space data is represented as two dictionaries: - `position`: Maps
component names to position arrays (e.g., {"x": array, "y": array}) -
`velocity`: Maps component names to velocity arrays (same keys as position)

Main Components
---------------
order : function
    The primary entry point. With no orderer it runs the default pipeline,
    ``MSTOrderer() | SOMOrderer()``; pass any ``pcf.orderers`` orderer to choose
    another algorithm.
fit_track : function
    Order, refine and fit a smooth track in one call.
walk_local_flow : function
    Deprecated; use ``pcf.order`` with ``pcf.orderers.LocalFlowOrderer``.
combine_results : function
    Combine results from forward and backward walks into a single ordering.
WalkLocalFlowResult : NamedTuple
    Result container with ordered indices and original data.
WalkConfig : class
    Configuration for neighbor queries, composing a metric and strategy.
AbstractDistanceMetric : class
    Abstract base class for distance metrics.
AlignedMomentumDistanceMetric : class
    Default momentum-based distance metric.
order_w : function
    Extract reordered position and velocity arrays from results.

Submodules
----------
phasecurvefit.phasespace : module
    Low-level phase-space operations (distances, directions, similarities).
phasecurvefit.nn : module
    Neural network for interpolating skipped tracers.

Examples
--------
>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf

Create phase-space observations as dictionaries:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

Order the observations (an MST backbone, refined by a SOM when there are enough
observations):

>>> result = pcf.order(pos, vel)
>>> result.indices
Array([0, 1, 2], dtype=int32)

Configure with custom metric and strategy:

>>> config = pcf.WalkConfig(
...     metric=pcf.metrics.AlignedMomentumDistanceMetric(),
...     strategy=pcf.strats.KDTree(k=3),
... )
>>> result = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(config=config))

Citation
--------
What to cite depends on the components you use: Starkman et al. (2023) for the
SOM stage (and so the default pipeline); Nibauer et al. (2022) for momentum-
weighted ordering (``LocalFlowOrderer``) or the autoencoder (``PathAutoencoder``);
Hogg, Bovy & Lang (2010) for mixture-model membership. See the Citation page of
the documentation.

"""

__all__: tuple[str, ...] = (
    # Version
    "__version__",
    # Modules
    "nn",
    "w",
    "metrics",
    "som",
    "strats",
    "orderers",
    # Algorithm
    "walk_local_flow",
    "combine_results",
    "WalkLocalFlowResult",
    "StateMetadata",
    # Orderers
    "order",
    # End-to-end
    "fit_track",
    # Query configuration
    "WalkConfig",
    # Result accessor
    "order_w",
)

from . import metrics, nn, orderers, som, strats, w
from ._src.algorithm import (
    StateMetadata,
    WalkLocalFlowResult,
    combine_results,
    order_w,
    walk_local_flow,
)
from ._src.orderers.base import order
from ._src.pipeline import fit_track
from ._src.query_config import WalkConfig
from ._version import version as __version__

# isort: split
# Optional interop registrations (e.g., unxt)
from . import _interop  # noqa: F401
