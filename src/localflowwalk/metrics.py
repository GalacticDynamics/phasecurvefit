r"""Distance metrics for the ``walk_local_flow`` algorithm.

The metric system provides pluggable distance calculations that determine how
the algorithm selects the next point in a phase-space trajectory. Different
metrics enable different physical interpretations and behaviors.

Built-in Metrics
----------------

AlignedMomentumDistanceMetric
    The standard metric from Nibauer et al. (2022). Combines spatial distance
    with velocity alignment using a momentum penalty term:

SpatialDistanceMetric
    Position-only metric that computes pure Euclidean distance, ignoring
    velocity information. Equivalent to standard nearest-neighbor search.

    $$ d = d_0 + \lambda (1 - \cos\theta) $$

    where $d_0$ is the Euclidean distance between positions,
    :math:`\theta` is the angle between the current velocity and the
    direction to the candidate point, and $\lambda$ is the momentum
    weight parameter.

    This metric favors points that align with the current velocity direction,
    making it effective for tracing coherent flows in phase-space.

Custom Metrics
--------------

You can implement custom metrics by subclassing ``AbstractDistanceMetric``
and implementing the ``__call__`` method

```python
import equinox as eqx
import jax.numpy as jnp
import localflowwalk as lfw


class Full6DMetric(lfw.metrics.AbstractDistanceMetric):
    # 6D Cartesian distance with velocity weighting.

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        pass
```

See the Metrics Guide in the documentation for more details and examples.

References
----------
Nibauer, J., et al. (2022). Charting Galactic Accelerations with Stellar
Streams and Machine Learning. arXiv:2209.XXXXX

"""

__all__: tuple[str, ...] = (
    "AbstractDistanceMetric",
    "AlignedMomentumDistanceMetric",
    "SpatialDistanceMetric",
    "FullPhaseSpaceDistanceMetric",
)

from ._src.metrics import (
    AbstractDistanceMetric,
    AlignedMomentumDistanceMetric,
    FullPhaseSpaceDistanceMetric,
    SpatialDistanceMetric,
)
