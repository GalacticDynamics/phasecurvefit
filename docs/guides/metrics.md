# Distance Metrics Guide

The `walk_local_flow` algorithm uses distance metrics to determine how to select the next point in a phase-space trajectory. This guide explains the metric system and shows how to use and create custom metrics.

## Overview

A distance metric defines how the algorithm measures "closeness" between the current point and candidate next points in phase-space. Different metrics enable different physical interpretations and behaviors.

The metric is **static** (marked with `eqx.field(static=True)`), meaning it doesn't change during the walk and is excluded from JAX's PyTree structure.

## Built-in Metrics

### AlignedMomentumDistanceMetric

The default metric implements the original Nearest Neighbors with Momentum (NN+p) algorithm from [Nibauer et al. (2022)](https://arxiv.org/abs/2209.XXXXX).

**Mathematical formulation:**

$$
d = d_0 + \lambda (1 - \cos\theta)
$$

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

# Explicit metric specification (same as default)
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, metric=AlignedMomentumDistanceMetric()
)
```

### SpatialDistanceMetric

A position-only metric that computes pure Euclidean distance, completely ignoring velocity information.

**Mathematical formulation:**

$$
d = d_0
$$

where $d_0$ is the Euclidean distance between positions. The `lam` parameter is ignored.

**When to use:**

- Velocity information is unreliable or unavailable
- Pure spatial proximity is desired (e.g., spatial clustering)
- Comparing against baseline nearest-neighbor approaches
- Setting `lam=0` with `AlignedMomentumDistanceMetric` is equivalent, but this metric is more explicit

**Usage:**

```python
from localflowwalk.metrics import SpatialDistanceMetric

# Pure nearest-neighbor search in position space
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=0.0, metric=SpatialDistanceMetric()
)
```

**Note:** Using `AlignedMomentumDistanceMetric` with `lam=0.0` gives the same result, but `SpatialDistanceMetric` makes the intent clearer and is slightly more efficient (doesn't compute velocity-related quantities).

### FullPhaseSpaceDistanceMetric

A true 6D Euclidean distance metric in full phase-space, treating position and velocity symmetrically.

**Mathematical formulation:**

$$
d = \sqrt{d_0^2 + (\tau \cdot d_v)^2}
$$

where:

- $d_0$ is the Euclidean distance in position space
- $d_v$ is the Euclidean distance in velocity space
- $\tau$ is the time parameter (`lam`) that converts velocity differences to position units

**Physical interpretation:**

This metric computes true Euclidean distance in the 6-dimensional phase space by combining position and velocity differences. The parameter `lam` (with time units) determines the relative weighting: for example, if positions are measured in kpc and velocities in kpc/Myr, then `lam` in Myr converts velocity differences to kpc, creating a uniformly scaled phase space.

Unlike `AlignedMomentumDistanceMetric`, this metric has no directional bias from momentum alignment — it treats all directions in phase space equally.

**When to use:**

- Position and velocity information are equally important
- You want true 6D proximity without momentum direction bias
- The natural time scale of the system is known
- Comparing against full phase-space clustering methods

**Usage:**

```python
from localflowwalk.metrics import FullPhaseSpaceDistanceMetric

# Full 6D phase-space distance
# lam represents a time scale (e.g., if pos ~ kpc, vel ~ kpc/Myr, lam ~ Myr)
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, metric=FullPhaseSpaceDistanceMetric()
)
```

**Comparison with momentum metric:**

- `AlignedMomentumDistanceMetric`: Directional — favors points along velocity direction
- `FullPhaseSpaceDistanceMetric`: Isotropic — treats all directions equally
- Both reduce to `SpatialDistanceMetric` when `lam=0`

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

    def __call__(self, current_pos, current_vel, positions, velocities, lam):
        """Compute modified distances.

        Parameters
        ----------
        current_pos : ScalarComponents
            Current position as dict of 0D arrays.
        current_vel : ScalarComponents
            Current velocity as dict of 0D arrays.
        positions : VectorComponents
            All positions as dict of 1D arrays (N,).
        velocities : VectorComponents
            All velocities as dict of 1D arrays (N,).
        lam : Array
            Momentum weight parameter (scalar).

        Returns
        -------
        Array
            Modified distances to all points, shape (N,).
        """
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

    Treats position and velocity on equal footing, with `lam` serving as
    a velocity-to-position scaling factor (units of time).

    Distance formula:
        d = sqrt(|Δr|² + (λ|Δv|)²)

    where Δr is position difference and Δv is velocity difference.
    """

    def __call__(self, current_pos, current_vel, positions, velocities, lam):
        # Compute position differences (vmap over N points)
        pos_diff = jax.tree.map(jnp.subtract, positions, current_pos)

        # Sum of squared position differences
        pos_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, pos_diff)))

        # Compute velocity differences (vmap over N points)
        vel_diff = jax.tree.map(jnp.subtract, velocities, current_vel)

        # Sum of squared velocity differences, weighted by lambda^2
        vel_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, vel_diff)))

        # Combined 6D distance
        return jnp.sqrt(pos_dist_sq + (lam**2) * vel_dist_sq)


# Use the custom metric
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

result = lfw.walk_local_flow(
    pos,
    vel,
    start_idx=0,
    lam=1.0,  # Now has units of time (position/velocity)
    metric=Full6DMetric(),
)
```

### Example: Weighted Position Metric

A metric that ignores velocity entirely and uses weighted position coordinates:

```python
class WeightedPositionMetric(AbstractDistanceMetric):
    """Position-only metric with per-component weights."""

    weights: dict[str, float] = eqx.field(static=True)

    def __call__(self, current_pos, current_vel, positions, velocities, lam):
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
result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.0, metric=metric)
```

## Units and Metrics

When using physical units via `unxt`, ensure your metric correctly handles unit propagation:

```python
import unxt as u
from localflowwalk.metrics import AlignedMomentumDistanceMetric

# Position in kpc, velocity in km/s
pos = {"x": u.Q([0.0, 1.0, 2.0], "kpc"), "y": u.Q([0.0, 0.5, 1.0], "kpc")}
vel = {"x": u.Q([1.0, 1.0, 1.0], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}

# Lambda must have units of distance for AlignedMomentumDistanceMetric
result = lfw.walk_local_flow(
    pos,
    vel,
    start_idx=0,
    lam=u.Q(100.0, "kpc"),  # Momentum weight in distance units
    metric=AlignedMomentumDistanceMetric(),
)

# For FullPhaseSpaceDistanceMetric, lambda has units of time
result_6d = lfw.walk_local_flow(
    pos,
    vel,
    start_idx=0,
    lam=u.Q(1.0, "Myr"),  # Time to convert velocity distance to spatial distance
    metric=FullPhaseSpaceDistanceMetric(),
)
```

## Metric Comparison

| Metric                           | Position | Velocity      | Directional Bias | Lambda Meaning                         |
| -------------------------------- | :------: | :-----------: | :--------------: | -------------------------------------- |
| `AlignedMomentumDistanceMetric`  | ✓        | ✓ (alignment) | ✓                | Momentum penalty weight                |
| `SpatialDistanceMetric`          | ✓        | ✗             | ✗                | Ignored                                |
| `FullPhaseSpaceDistanceMetric`   | ✓        | ✓ (magnitude) | ✗                | Time scale (velocity → position units) |

**When to use each:**

- **FullPhaseSpaceDistanceMetric** (default): True 6D distance when position and velocity are equally important and you know the system's natural time scale. No directional preference.
- **AlignedMomentumDistanceMetric**: For coherent flows (stellar streams, winds) where velocity alignment should bias the ordering.
- **SpatialDistanceMetric**: When velocity is unreliable or you want pure spatial clustering. Good baseline for comparison.

## Best Practices

1. **Keep metrics simple**: The metric is called many times during the walk, so keep it computationally efficient.

2. **Use JAX operations**: All operations should be JAX-compatible for `jit`, `vmap`, and `grad` support.

3. **Handle edge cases**: Consider what happens when velocity is zero, or when points coincide.

4. **Document units**: Clearly document what units `lam` should have for your metric.

5. **Test thoroughly**: Verify your metric behaves correctly with `jit`, `vmap`, and with/without units.

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
    "Full6D": Full6DMetric(),  # From custom example above
}

for name, metric in metrics.items():
    result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0, metric=metric)
    n_ordered = jnp.sum(result.ordered_indices >= 0)
    print(f"{name}: {n_ordered}/100 points ordered")
```

## See Also

- [Algorithm Guide](algorithm.md) - Core algorithm details
- [API Reference](../api/index.md) - Complete API documentation
- [Examples](examples.md) - More usage examples
