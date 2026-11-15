---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Interpolate positions along an ordering

This guide shows how to get positions at chosen values of $\gamma$ from an ordering,
without training a network.

Every orderer's result can be called with values of $\gamma$ inside
`result.gamma_range`. It interpolates linearly along the backbone when the orderer
fitted one (the MST and the SOM do), and otherwise along the ordered samples, as for
the local-flow walk:

```{code-cell} python
import jax
import jax.numpy as jnp

import phasecurvefit as pcf

pos = {"x": jnp.linspace(0, 10, 50), "y": jnp.sin(jnp.linspace(0, 2 * jnp.pi, 50))}
vel = {"x": jnp.ones(50), "y": jnp.cos(jnp.linspace(0, 2 * jnp.pi, 50))}

result = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0))
print(result.gamma_range)

gamma_values = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
interpolated = result(gamma_values)
print(interpolated["x"])
```

The interpolated track runs through the ordered samples:

```{code-cell} python
import matplotlib.pyplot as plt

track = result(jnp.linspace(*result.gamma_range, 200))

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(pos["x"], pos["y"], ".", c="0.6", label="samples")
ax.plot(track["x"], track["y"], lw=1, label=r"result($\gamma$)")
ax.plot(interpolated["x"], interpolated["y"], "o", label="gamma_values")
ax.legend();
```

## Compile, batch and differentiate

The interpolation works under `jit`, `vmap` and `grad`, whichever orderer produced
the result:

```{code-cell} python
@jax.jit
def interpolate(gamma):
    return result(gamma)


interpolate(0.5)

batch = jax.jit(jax.vmap(interpolate))(jnp.linspace(0, 1, 100))
print(batch["x"].shape)


def loss_fn(gamma):
    p = result(gamma)
    return p["x"] ** 2 + p["y"] ** 2


grads = jax.jit(jax.vmap(jax.grad(loss_fn)))(jnp.linspace(0, 1, 50))
```

Unvisited samples, marked `-1` in `result.indices`, are masked out without changing
array shapes, so the interpolation keeps working inside `jit`.

For a smooth track rather than a piecewise-linear one, fit the autoencoder
({doc}`autoencoder`).
