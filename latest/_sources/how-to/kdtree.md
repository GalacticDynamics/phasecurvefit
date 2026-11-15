---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Restrict the walk to nearby points

This guide shows how to make the local-flow walk consider only the spatially nearest
points at each step, with a KD-tree. It needs the `kdtree` extra
({doc}`../start/install`).

By default the walk uses {class}`~phasecurvefit.strats.BruteForce`, which scores
every unvisited point at each step. {class}`~phasecurvefit.strats.KDTree` first
finds the `k` spatially nearest points with a KD-tree (from
[jaxkd](https://github.com/dodgebc/jaxkd)), then picks among them with the metric.
This keeps the walk local where strands pass close together.

Set the strategy in the {class}`~phasecurvefit.WalkConfig`:

```{code-cell} python
import jax
import jax.numpy as jnp

import phasecurvefit as pcf
from phasecurvefit.metrics import AlignedMomentumDistanceMetric

t = jax.random.permutation(jax.random.key(0), jnp.linspace(0, 4 * jnp.pi, 100))
position = {"x": jnp.cos(t), "y": jnp.sin(t), "z": t / (2 * jnp.pi)}
velocity = {"x": -jnp.sin(t), "y": jnp.cos(t), "z": jnp.ones_like(t) / (2 * jnp.pi)}
start = int(jnp.argmin(position["z"]))

config = pcf.WalkConfig(
    metric=AlignedMomentumDistanceMetric(),
    strategy=pcf.strats.KDTree(k=5),
)
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=start, metric_scale=1.0),
)
```

`k` counts usable neighbours: the current point itself is excluded, and `k` is
clamped to the number of points. If `k` is too small, the walk can miss the right
next point when it lies just outside the `k` nearest; raise `k` until the ordering
stops changing.
