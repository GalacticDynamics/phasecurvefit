---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Choose and tune the walk's metric

This guide shows how to pick the distance metric the local-flow walk uses, and how to
set its `metric_scale`. {doc}`../explanation/walk` explains what each metric
computes.

## Pick a metric

The metric goes into a {class}`~phasecurvefit.WalkConfig`, which the walk takes as
`config`:

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

- {class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` (the default): for
  coherent flows such as stellar streams, where the next point should lie ahead
  along the velocity.
- {class}`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric`: when position and
  velocity matter equally and you know a natural time scale for the system. It has
  no preferred direction.
- {class}`~phasecurvefit.metrics.SpatialDistanceMetric`: when velocities are
  unreliable, or as a position-only baseline to compare against. It ignores
  `metric_scale`.

To write your own metric, see {doc}`custom-metric`.

## Set `metric_scale`

`metric_scale` means something different for each metric and carries units, so no
single value suits every dataset:

- **Default metric:** $\lambda$ is a **length**, the extra cost of stepping at right
  angles to the current velocity. Compare it with the typical spacing $s$ between
  neighbouring points. With $\lambda \ll s$ the walk is a nearest-neighbour search.
  With $\lambda \approx s$ it weighs direction and distance about equally. With
  $\lambda \gg s$ it strongly prefers to keep going straight ahead, which also means
  longer strides and more skipped points. The
  [stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb) uses
  $\lambda = 100$ kpc for a stream whose steps are at most a few kpc, so even a step
  $15°$ off the velocity costs more than a 3 kpc step straight ahead.
- **Full phase-space metric:** $\tau$ is a **time** that turns a velocity difference
  into an equivalent distance. Choose it so that $\tau\,\Delta v$ between points on
  *different* parts of the curve is much larger than the spacing between neighbours
  on the *same* part. The
  [epitrochoid autoencoder tutorial](../tutorials/epitrochoid_autoencoder.ipynb)
  uses $\tau = 4$ s: where two strands cross, their velocities differ by at least
  about 580 m/s, which puts the wrong strand over 2 km away against a spacing of
  about 3 m.
- **Spatial metric:** ignored.

With the default metric, `metric_scale=0` switches the direction term off and gives
a plain nearest-neighbour walk:

```{code-cell} python
import jax

t = jax.random.permutation(jax.random.key(0), jnp.linspace(0, 4 * jnp.pi, 100))
position = {"x": jnp.cos(t), "y": jnp.sin(t), "z": t / (2 * jnp.pi)}
velocity = {"x": -jnp.sin(t), "y": jnp.cos(t), "z": jnp.ones_like(t) / (2 * jnp.pi)}
start = int(jnp.argmin(position["z"]))

for scale in (0.0, 1.0, 5.0):
    result = pcf.order(
        position,
        velocity,
        pcf.orderers.LocalFlowOrderer(start_idx=start, metric_scale=scale),
    )
    print(f"metric_scale={scale}: {int(result.n_visited)} visited")
```

## Check the result

Look at the ordering after every change. If it jumps between strands, the velocity
term is too weak: raise `metric_scale`. If the walk stops early or skips most points,
the term may be too strong, or `max_dist` too small ({doc}`local-flow-walk`).

To use physical units for `metric_scale`, see {doc}`units`.
