"""localflowwalk: Nearest Neighbors with Momentum for phase-space ordering.

This library implements the Nearest Neighbors with Momentum (NN+p) algorithm
from Nibauer et al. (2022) for ordering phase-space observations in stellar
streams. The algorithm uses both spatial proximity and velocity momentum to
trace coherent structures through phase-space.

Phase-space data is represented as two dictionaries:
- `position`: Maps component names to position arrays (e.g., {"x": array, "y": array})
- `velocity`: Maps component names to velocity arrays (same keys as position)

Main Components
---------------
walk_local_flow : function
    The main algorithm for ordering phase-space observations.
LocalFlowWalkResult : NamedTuple
    Result container with ordered indices and original data.
AbstractDistanceMetric : class
    Abstract base class for distance metrics.
AlignedMomentumDistanceMetric : class
    Default momentum-based distance metric (NN+p).
get_ordered_w : function
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
>>> import localflowwalk as lfw

Create phase-space observations as dictionaries:

>>> pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
>>> vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

Order the observations:

>>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)
>>> result.ordered_indices
Array([0, 1, 2], dtype=int32)

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning."

"""

__all__: tuple[str, ...] = (
    # Version
    "__version__",
    # Modules
    "nn",
    "w",
    "metrics",
    # Type aliases
    "ScalarComponents",
    "VectorComponents",
    # Algorithm
    "walk_local_flow",
    "LocalFlowWalkResult",
    # Strategies
    "AbstractQueryStrategy",
    "BruteForceStrategy",
    "KDTreeStrategy",
    # Result accessor
    "get_ordered_w",
)

from . import metrics, nn, w
from ._src.algorithm import (
    LocalFlowWalkResult,
    get_ordered_w,
    walk_local_flow,
)
from ._src.custom_types import ScalarComponents, VectorComponents
from ._src.strategies import AbstractQueryStrategy, BruteForceStrategy, KDTreeStrategy
from ._version import version as __version__

# isort: split
# Optional interop registrations (e.g., unxt)
from . import _interop  # noqa: F401
