# Distance Metrics Guide

The `walk_local_flow` algorithm uses distance metrics to determine how to select
the next point in a phase-space trajectory. This guide explains the metric
system and shows how to use and create custom metrics.

## Overview

A distance metric defines how the algorithm measures "closeness" between the
current point and candidate next points in phase-space. Different metrics enable
different physical interpretations and behaviors.

Metrics are configured via `WalkConfig`, which composes a metric with a query
strategy (discussed in a separate guide):

```python
import jax.numpy as jnp
import localflowwalk as lfw
from localflowwalk.metrics import FullPhaseSpaceDistanceMetric

pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

config = lfw.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

## Built-in Metrics

### SpatialDistanceMetric

A position-only metric that computes pure Euclidean distance, completely ignoring velocity information.

**Mathematical formulation:**

$$ d = d_0 $$

where $d_0$ is the Euclidean distance between positions. The `metric_scale` parameter is ignored.

**When to use:**

- Velocity information is unreliable or unavailable
- Pure spatial proximity is desired (e.g., spatial clustering)
- Comparing against baseline nearest-neighbor approaches
- Setting `metric_scale=0` with `AlignedMomentumDistanceMetric` is equivalent, but this metric is more explicit

**Usage:**

```python
from localflowwalk.metrics import SpatialDistanceMetric

# Pure nearest-neighbor search in position space
config = lfw.WalkConfig(metric=SpatialDistanceMetric())
result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=0.0)
```

### AlignedMomentumDistanceMetric

The Nearest Neighbors with Momentum (NN+p) metric from [Nibauer et al.
(2022)](https://arxiv.org/abs/2209.XXXXX).  This is the default metric.

**Mathematical formulation:**

$$ d = d_0 + \lambda (1 - \cos\theta) $$

where:

- $d_0$ is the Euclidean distance between positions
- $\theta$ is the angle between the current velocity and the direction to the candidate point
- $\lambda$ is the momentum weight parameter

**Physical interpretation:**

This metric combines spatial proximity with velocity alignment. Points that lie along the current velocity direction receive lower penalties, making the algorithm favor coherent flows in phase-space.

**Usage:**

```python
import jax.numpy as jnp
import localflowwalk as lfw
from localflowwalk.metrics import AlignedMomentumDistanceMetric

pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Aligned momentum metric
config = lfw.WalkConfig(metric=AlignedMomentumDistanceMetric())
result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

### FullPhaseSpaceDistanceMetric

A true 6D Euclidean distance metric in full phase-space, treating position and velocity symmetrically. **This is the default metric.**

**Mathematical formulation:**

$$ d = \sqrt{d_0^2 + (\tau \cdot d_v)^2} $$

where:

- $d_0$ is the Euclidean distance in position space
- $d_v$ is the Euclidean distance in velocity space
- $\tau$ is the time parameter (`metric_scale`) that converts velocity differences to position units

**Physical interpretation:**

This metric computes true Euclidean distance in the 6-dimensional phase space by combining position and velocity differences. The parameter `metric_scale` (with time units) determines the relative weighting: for example, if positions are measured in kpc and velocities in kpc/Myr, then `metric_scale` in Myr converts velocity differences to kpc, creating a uniformly scaled phase space.

Unlike `AlignedMomentumDistanceMetric`, this metric has no directional bias from momentum alignment — it treats all directions in phase space equally.

**When to use:**

- Position and velocity information are equally important
- You want true 6D proximity without momentum direction bias
- The natural time scale of the system is known
- Comparing against full phase-space clustering methods

**Usage:**

```python
from localflowwalk.metrics import FullPhaseSpaceDistanceMetric

# Full 6D phase-space distance (this is the default)
# metric_scale represents a time scale (e.g., if pos ~ kpc, vel ~ kpc/Myr, metric_scale ~ Myr)
config = lfw.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

**Comparison with momentum metric:**

- `AlignedMomentumDistanceMetric`: Directional — favors points along velocity direction
- `FullPhaseSpaceDistanceMetric`: Isotropic — treats all directions equally
- Both reduce to `SpatialDistanceMetric` when `metric_scale=0`

## Creating Custom Metrics

Custom metrics enable alternative distance calculations for specific use cases. For example, you might want:

- Full 6D Cartesian distance in phase-space
- Weighted combinations of position and velocity
- Problem-specific distance measures

### The AbstractDistanceMetric Interface

All metrics must inherit from `AbstractDistanceMetric` and implement the `__call__` method:

```python
import equinox as eqx
from localflowwalk.metrics import AbstractDistanceMetric


class CustomMetric(AbstractDistanceMetric):
    """Your custom distance metric."""

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        """Compute modified distances."""
        # Your distance calculation here
        ...
```

### Example: 6D Cartesian Metric

Here's a complete example of a metric that computes full 6D Cartesian distance:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from localflowwalk.metrics import AbstractDistanceMetric


class Full6DMetric(AbstractDistanceMetric):
    """6D Cartesian distance in phase-space.

    Treats position and velocity on equal footing, with `metric_scale` serving as
    a velocity-to-position scaling factor (units of time).

    Distance formula:
        d = sqrt(|Δr|² + (λ|Δv|)²)

    where Δr is position difference and Δv is velocity difference.
    """

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        # Compute position differences (vmap over N points)
        pos_diff = jax.tree.map(jnp.subtract, positions, current_pos)

        # Sum of squared position differences
        pos_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, pos_diff)))

        # Compute velocity differences (vmap over N points)
        vel_diff = jax.tree.map(jnp.subtract, velocities, current_vel)

        # Sum of squared velocity differences, weighted by metric_scale^2
        vel_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, vel_diff)))

        # Combined 6D distance
        return jnp.sqrt(pos_dist_sq + (metric_scale**2) * vel_dist_sq)


# Use the custom metric via WalkConfig
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

config = lfw.WalkConfig(metric=Full6DMetric())
result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

### Example: Weighted Position Metric

A metric that ignores velocity entirely and uses weighted position coordinates:

```python
class WeightedPositionMetric(AbstractDistanceMetric):
    """Position-only metric with per-component weights."""

    weights: dict[str, float] = eqx.field(static=True)

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        # Compute weighted position differences
        def weighted_diff_sq(component_name, positions_component):
            diff = positions_component - current_pos[component_name]
            weight = self.weights.get(component_name, 1.0)
            return weight * diff**2

        # Sum over all components
        weighted_dist_sq = sum(weighted_diff_sq(k, v) for k, v in positions.items())

        return jnp.sqrt(weighted_dist_sq)


# Use with custom weights (ignore y-coordinate)
metric = WeightedPositionMetric(weights={"x": 1.0, "y": 0.1})
config = lfw.WalkConfig(metric=metric)
result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=0.0)
```

## Units and Metrics

When using physical units via `unxt`, ensure your metric correctly handles unit propagation:

```python
import unxt as u
from localflowwalk.metrics import (
    AlignedMomentumDistanceMetric,
    FullPhaseSpaceDistanceMetric,
)

# Position in kpc, velocity in km/s
pos = {"x": u.Q([0.0, 1.0, 2.0], "kpc"), "y": u.Q([0.0, 0.5, 1.0], "kpc")}
vel = {"x": u.Q([1.0, 1.0, 1.0], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}

# metric_scale must have units of distance for AlignedMomentumDistanceMetric
config = lfw.WalkConfig(metric=AlignedMomentumDistanceMetric())
result = lfw.walk_local_flow(
    pos,
    vel,
    config=config,
    start_idx=0,
    metric_scale=u.Q(100.0, "kpc"),  # Momentum weight in distance units
    usys=u.unitsystems.galactic,  # Required when using Quantities
)

# For FullPhaseSpaceDistanceMetric, metric_scale has units of time
config_6d = lfw.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result_6d = lfw.walk_local_flow(
    pos,
    vel,
    config=config_6d,
    start_idx=0,
    metric_scale=u.Q(
        1.0, "Gyr"
    ),  # Time to convert velocity distance to spatial distance
    usys=u.unitsystems.galactic,  # Required when using Quantities
)
```

## Metric Comparison

| Metric                           | Position | Velocity      | Lambda Meaning                         |
| -------------------------------- | :------: | :-----------: | -------------------------------------- |
| `SpatialDistanceMetric`          | ✓        | ✗             | Ignored                                |
| `AlignedMomentumDistanceMetric`  | ✓        | ✓ (alignment) | Momentum penalty weight                |
| `FullPhaseSpaceDistanceMetric`   | ✓        | ✓ (magnitude) | Time scale (velocity → position units) |

**When to use each:**

- **FullPhaseSpaceDistanceMetric** (default): True 6D distance when position and velocity are equally important and you know the system's natural time scale. No directional preference.
- **AlignedMomentumDistanceMetric**: For coherent flows (stellar streams, winds) where velocity alignment should bias the ordering.
- **SpatialDistanceMetric**: When velocity is unreliable or you want pure spatial clustering. Good baseline for comparison.


## Metric Comparison Example

Here's a comparison of different metrics on the same data:

```python
import jax.numpy as jnp
import localflowwalk as lfw
from localflowwalk.metrics import (
    AlignedMomentumDistanceMetric,
    SpatialDistanceMetric,
)

# Sample spiral trajectory
theta = jnp.linspace(0, 4 * jnp.pi, 100)
pos = {
    "x": jnp.cos(theta) * jnp.exp(theta / 10),
    "y": jnp.sin(theta) * jnp.exp(theta / 10),
}
vel = {
    "x": jnp.gradient(pos["x"]),
    "y": jnp.gradient(pos["y"]),
}

# Compare metrics
metrics = {
    "Momentum": AlignedMomentumDistanceMetric(),
    "Spatial": SpatialDistanceMetric(),
}

for name, metric in metrics.items():
    config = lfw.WalkConfig(metric=metric)
    result = lfw.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
    n_visited = len([i for i in result.indices if i >= 0])
    print(f"{name}: {n_visited}/100 points ordered")
```

## See Also

- [Algorithm Guide](algorithm.md) - Core algorithm details
- [API Reference](../api/index.md) - Complete API documentation
