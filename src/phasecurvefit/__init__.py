"""localflowwalk.

This library implements algorithms for ordering phase-space observations in
stellar streams. The algorithm uses both spatial proximity and velocity momentum
to trace coherent structures through phase-space.

Phase-space data is represented as two dictionaries: - `position`: Maps
component names to position arrays (e.g., {"x": array, "y": array}) -
`velocity`: Maps component names to velocity arrays (same keys as position)

Main Components
---------------
walk_local_flow : function
    The main algorithm for ordering phase-space observations.
combine_flow_walks : function
    Run forward and backward walks and combine them into a single ordering.
LocalFlowWalkResult : NamedTuple
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
localflowwalk.phasespace : module
    Low-level phase-space operations (distances, directions, similarities).
localflowwalk.nn : module
    Neural network for interpolating skipped tracers.

Examples
--------
>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf

Create phase-space observations as dictionaries:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

Order the observations:

>>> result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)
>>> result.indices
Array([0, 1, 2], dtype=int32)

Configure with custom metric and strategy:

>>> config = pcf.WalkConfig(
...     metric=pcf.metrics.AlignedMomentumDistanceMetric(),
...     strategy=pcf.strats.KDTree(k=3),
... )
>>> result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams and
Machine Learning."

"""

__all__: tuple[str, ...] = (
    # Version
    "__version__",
    # Modules
    "nn",
    "w",
    "metrics",
    "strats",
    # Algorithm
    "walk_local_flow",
    "combine_flow_walks",
    "LocalFlowWalkResult",
    "StateMetadata",
    # Query configuration
    "WalkConfig",
    # Result accessor
    "order_w",
)

from . import metrics, nn, strats, w
from ._src.algorithm import (
    LocalFlowWalkResult,
    StateMetadata,
    combine_flow_walks,
    order_w,
    walk_local_flow,
)
from ._src.query_config import WalkConfig
from ._version import version as __version__

# isort: split
# Optional interop registrations (e.g., unxt)
from . import _interop  # noqa: F401
