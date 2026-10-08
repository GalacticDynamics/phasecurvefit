---
file_format: mystnb
kernelspec:
  name: python3
---

# Distance Metrics Guide

The local-flow walk uses distance metrics to determine how to select the next
point in a phase-space trajectory. This guide explains the metric system and
shows how to use and create custom metrics.

## Overview

A distance metric defines how the algorithm measures "closeness" between the
current point and candidate next points in phase-space. Different metrics enable
different physical interpretations and behaviors.

```{note}
Metrics configure the {class}`~phasecurvefit.orderers.LocalFlowOrderer`. The same
velocity-alignment idea — the
`cos θ` term below — also powers the opt-in velocity mechanisms of the
{class}`~phasecurvefit.orderers.MSTOrderer` (`velocity_weight`,
`sever_cos_threshold`), which reuse `cos(v_i, v_j)` on graph edges. See the
[Orderers guide](orderers.md).
```

Metrics are configured via `WalkConfig`, which composes a metric with a query
strategy (discussed in a separate guide):

```{code-cell} python
import jax.numpy as jnp
import phasecurvefit as pcf
from phasecurvefit.metrics import FullPhaseSpaceDistanceMetric

pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

config = pcf.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=1.0),
)
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

```{code-cell} python
from phasecurvefit.metrics import SpatialDistanceMetric

# Pure nearest-neighbor search in position space
config = pcf.WalkConfig(metric=SpatialDistanceMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=0.0),
)
```

### AlignedMomentumDistanceMetric

The Nearest Neighbors with Momentum (NN+p) metric from [Nibauer et al.
(2022)](https://arxiv.org/abs/2205.11767).  This is the default metric.

**Mathematical formulation:**

$$ d = d_0 + \lambda (1 - \cos\theta) $$

where:

- $d_0$ is the Euclidean distance between positions
- $\theta$ is the angle between the current velocity and the direction to the candidate point
- $\lambda$ is the momentum weight parameter

**Physical interpretation:**

This metric combines spatial proximity with velocity alignment. Points that lie along the current velocity direction receive lower penalties, making the algorithm favor coherent flows in phase-space.

**Usage:**

```{code-cell} python
import jax.numpy as jnp
import phasecurvefit as pcf
from phasecurvefit.metrics import AlignedMomentumDistanceMetric

pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Aligned momentum metric
config = pcf.WalkConfig(metric=AlignedMomentumDistanceMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=1.0),
)
```

### FullPhaseSpaceDistanceMetric

A true 6D Euclidean distance metric in full phase-space, treating position and velocity symmetrically.

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

```{code-cell} python
from phasecurvefit.metrics import FullPhaseSpaceDistanceMetric

# Full 6D phase-space distance
# metric_scale represents a time scale (e.g., if pos ~ kpc, vel ~ kpc/Myr, metric_scale ~ Myr)
config = pcf.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=1.0),
)
```

**Comparison with momentum metric:**

- `AlignedMomentumDistanceMetric`: Directional — favors points along velocity direction
- `FullPhaseSpaceDistanceMetric`: Isotropic — treats all directions equally
- Both reduce to `SpatialDistanceMetric` when `metric_scale=0`

## Choosing `metric_scale`

`metric_scale` means something different for each metric, and it carries units,
so there is no universal good value. A way to pick it for each:

- **`AlignedMomentumDistanceMetric`**: $\lambda$ is a **length**, the extra cost
  of stepping at right angles to the current velocity. Compare it with the
  typical spacing $s$ between neighbouring points: $\lambda \ll s$ behaves like
  nearest-neighbour search, $\lambda \approx s$ weighs direction and distance
  about equally, and $\lambda \gg s$ strongly favours continuing straight ahead,
  which also means longer strides and more skipped points. The
  [stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb) uses
  $\lambda = 100$ kpc for a stream whose steps are at most a few kpc, so even a
  step $15°$ off the velocity costs more than a 3 kpc step straight ahead.
- **`FullPhaseSpaceDistanceMetric`**: $\tau$ is a **time**, converting a velocity
  difference into an equivalent distance. Choose it so that $\tau\,\Delta v$
  between points on *different* parts of the curve is much larger than the
  spacing between neighbours on the *same* part. The
  [epitrochoid autoencoder tutorial](../tutorials/epitrochoid_autoencoder.ipynb)
  uses $\tau = 4$ s: where two strands cross, their velocities differ by at least
  ~580 m/s, which puts the wrong strand over 2 km away against a ~3 m spacing.
- **`SpatialDistanceMetric`**: ignored.

In every case, check the result: if the ordering jumps between strands, the
velocity term is too weak; if the walk stops early or skips most points, it may
be too strong (or `max_dist` too small).

## Creating Custom Metrics

Custom metrics enable alternative distance calculations for specific use cases. For example, you might want:

- Full 6D Cartesian distance in phase-space
- Weighted combinations of position and velocity
- Problem-specific distance measures

### The AbstractDistanceMetric Interface

All metrics must inherit from `AbstractDistanceMetric` and implement the `__call__` method:

```{code-cell} python
import equinox as eqx
from phasecurvefit.metrics import AbstractDistanceMetric


class CustomMetric(AbstractDistanceMetric):
    """Your custom distance metric."""

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        """Compute modified distances."""
        # Your distance calculation here
        ...
```

### Example: 6D Cartesian Metric

Here's a complete example of a metric that computes full 6D Cartesian distance.
(This is what the built-in `FullPhaseSpaceDistanceMetric` does; it is written out
here to show the interface.)

```{code-cell} python
import equinox as eqx
import jax
import jax.numpy as jnp
from phasecurvefit.metrics import AbstractDistanceMetric


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

config = pcf.WalkConfig(metric=Full6DMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=1.0),
)
```

### Example: Weighted Position Metric

A metric that ignores velocity entirely and uses weighted position coordinates:

```{code-cell} python
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
config = pcf.WalkConfig(metric=metric)
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=0.0),
)
```

## Units and Metrics

When using physical units via `unxt`, ensure your metric correctly handles unit propagation:

```{code-cell} python
import unxt as u
from phasecurvefit.metrics import (
    AlignedMomentumDistanceMetric,
    FullPhaseSpaceDistanceMetric,
)

# Position in kpc, velocity in km/s
pos = {"x": u.Q([0.0, 1.0, 2.0], "kpc"), "y": u.Q([0.0, 0.5, 1.0], "kpc")}
vel = {"x": u.Q([1.0, 1.0, 1.0], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}

# metric_scale must have units of distance for AlignedMomentumDistanceMetric
config = pcf.WalkConfig(metric=AlignedMomentumDistanceMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(
        config=config,
        start_idx=0,
        metric_scale=u.Q(100.0, "kpc"),  # Momentum weight in distance units
    ),
    metadata=pcf.StateMetadata(usys=u.unitsystems.galactic),  # Required for Quantities
)

# For FullPhaseSpaceDistanceMetric, metric_scale has units of time
config_6d = pcf.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result_6d = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(
        config=config_6d,
        start_idx=0,
        metric_scale=u.Q(
            1.0, "Gyr"
        ),  # Time to convert velocity distance to spatial distance
    ),
    metadata=pcf.StateMetadata(usys=u.unitsystems.galactic),  # Required for Quantities
)
```

## Metric Comparison

| Metric                           | Position | Velocity      | Lambda Meaning                         |
| -------------------------------- | :------: | :-----------: | -------------------------------------- |
| `SpatialDistanceMetric`          | ✓        | ✗             | Ignored                                |
| `AlignedMomentumDistanceMetric`  | ✓        | ✓ (alignment) | Momentum penalty weight                |
| `FullPhaseSpaceDistanceMetric`   | ✓        | ✓ (magnitude) | Time scale (velocity → position units) |

**When to use each:**

- **FullPhaseSpaceDistanceMetric**: True 6D distance when position and velocity are equally important and you know the system's natural time scale. No directional preference.
- **AlignedMomentumDistanceMetric** (default): For coherent flows (stellar streams, winds) where velocity alignment should bias the ordering.
- **SpatialDistanceMetric**: When velocity is unreliable or you want pure spatial clustering. Good baseline for comparison.


## Metric Comparison Example

The metrics differ where the curve crosses itself. At a crossing the nearest
point is often on the *other* branch, and only a velocity-aware metric keeps
the walk on its own. An epitrochoid crosses itself many times:

```{code-cell} python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import phasecurvefit as pcf
from phasecurvefit.metrics import (
    AlignedMomentumDistanceMetric,
    SpatialDistanceMetric,
)

# Epitrochoid, ordered along t, with an open 10-degree gap
t = jnp.linspace(jnp.deg2rad(5), jnp.deg2rad(355), 300)
R, r, d = 5.0, 1.0, 4.5
k = (R + r) / r
pos = {
    "x": (R + r) * jnp.cos(t) - d * jnp.cos(k * t),
    "y": (R + r) * jnp.sin(t) - d * jnp.sin(k * t),
}
vel = {
    "x": -(R + r) * jnp.sin(t) + d * k * jnp.sin(k * t),
    "y": (R + r) * jnp.cos(t) - d * k * jnp.cos(k * t),
}

# Compare metrics. metric_scale is ~20x the point spacing (~0.57) for the
# momentum metric, and ignored by the spatial one.
metrics = {
    "Spatial": (SpatialDistanceMetric(), 0.0),
    "Aligned momentum": (AlignedMomentumDistanceMetric(), 12.0),
}

fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
for ax, (name, (metric, scale)) in zip(axs, metrics.items()):
    config = pcf.WalkConfig(metric=metric)
    result = pcf.order(
        pos,
        vel,
        pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=scale),
    )
    # Steps that jump more than 5 places along the true curve
    n_jumps = int(jnp.sum(jnp.abs(jnp.diff(result.ordering)) > 5))
    print(f"{name}: {n_jumps} jumps between branches")

    ordered_pos, _ = pcf.order_w(result)
    ax.plot(pos["x"], pos["y"], ".", c="0.75", ms=3)
    ax.plot(ordered_pos["x"], ordered_pos["y"], lw=1)
    ax.set(title=f"{name}: {n_jumps} jumps", aspect="equal")
```

The spatial walk short-cuts across the crossings; the momentum walk follows
the loops.

## See Also

- [Algorithm Guide](algorithm.md) - Core algorithm details
- [API Reference](../api/index.md) - Complete API documentation
