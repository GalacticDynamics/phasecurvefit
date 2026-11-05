---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Write a custom metric

This guide shows how to give the local-flow walk your own distance metric, for
example a weighted combination of coordinates or a problem-specific distance.

## Subclass `AbstractDistanceMetric`

A metric subclasses {class}`~phasecurvefit.metrics.AbstractDistanceMetric` and
implements `__call__`. It receives the current point's position and velocity, the
positions and velocities of all points, and `metric_scale`, and returns one distance
per point:

```{code-cell} python
import equinox as eqx
import jax
import jax.numpy as jnp

import phasecurvefit as pcf
from phasecurvefit.metrics import AbstractDistanceMetric


class CustomMetric(AbstractDistanceMetric):
    """Your custom distance metric."""

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        """Compute modified distances."""
        ...
```

Pass an instance to the walk through {class}`~phasecurvefit.WalkConfig`. The walk
steps to the unvisited point with the smallest distance.

## Example: phase-space distance

This metric adds position and velocity differences in quadrature, with
`metric_scale` converting velocity into position units. It is what the built-in
{class}`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric` does, written out to
show the interface:

```{code-cell} python
class Full6DMetric(AbstractDistanceMetric):
    """Euclidean distance in phase space: sqrt(|dq|^2 + (scale * |dp|)^2)."""

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        pos_diff = jax.tree.map(jnp.subtract, positions, current_pos)
        pos_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, pos_diff)))

        vel_diff = jax.tree.map(jnp.subtract, velocities, current_vel)
        vel_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, vel_diff)))

        return jnp.sqrt(pos_dist_sq + (metric_scale**2) * vel_dist_sq)


pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

config = pcf.WalkConfig(metric=Full6DMetric())
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=1.0),
)
```

## Example: weighted positions

A metric can hold its own settings as fields. Mark them `static` when they are not
arrays. This one ignores velocity and weights each position coordinate:

```{code-cell} python
class WeightedPositionMetric(AbstractDistanceMetric):
    """Position-only metric with per-component weights."""

    weights: dict[str, float] = eqx.field(static=True)

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        def weighted_diff_sq(name, values):
            diff = values - current_pos[name]
            return self.weights.get(name, 1.0) * diff**2

        return jnp.sqrt(sum(weighted_diff_sq(k, v) for k, v in positions.items()))


metric = WeightedPositionMetric(weights={"x": 1.0, "y": 0.1})  # down-weight y
config = pcf.WalkConfig(metric=metric)
result = pcf.order(
    pos,
    vel,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=0.0),
)
```

A metric used inside {class}`~phasecurvefit.orderers.SOMOrderer` must be symmetric in
the two points it compares ({doc}`../explanation/som`).
