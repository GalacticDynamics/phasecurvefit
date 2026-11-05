---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Quick start

This page takes you from positions and velocities in an unknown order to an ordering
and a smooth track through them. It needs phasecurvefit and `matplotlib`
({doc}`install`). It takes about a minute to run.

## Your data

Here a simulation stands in for your data: 400 points along a spiral of one and a
half turns, with some scatter in position. Each point has a position and a velocity,
stored as two dictionaries with the same keys. We shuffle the points so their order
is unknown, and keep the true parameter `t` only to check the result at the end.

```{code-cell} python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import phasecurvefit as pcf

t = jnp.linspace(0.5, 0.5 + 3 * jnp.pi, 400)
r = 1 + 0.6 * t
kx, ky, kp = jax.random.split(jax.random.key(0), 3)
pos = {
    "x": r * jnp.cos(t) + 0.2 * jax.random.normal(kx, t.shape),
    "y": r * jnp.sin(t) + 0.2 * jax.random.normal(ky, t.shape),
}
vel = {"x": 0.6 * jnp.cos(t) - r * jnp.sin(t), "y": 0.6 * jnp.sin(t) + r * jnp.cos(t)}

shuffle = jax.random.permutation(kp, t.size)
t, pos, vel = jax.tree.map(lambda a: a[shuffle], (t, pos, vel))
```

Joining the points in their stored order shows that the order is lost:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(pos["x"], pos["y"], "-o", ms=2, lw=0.3)
ax.set(aspect="equal", title="Stored order");
```

## Order the points

`pcf.order` with no other arguments runs the default pipeline. It needs no start
point.

```{code-cell} python
result = pcf.order(pos, vel)
result.ordering[:10]
```

`result.ordering` lists the indices of the points from one end of the curve to the
other. `pcf.order_w` returns the positions and velocities in that order. Joined in
the new order, the points trace the spiral:

```{code-cell} python
ordered_pos, ordered_vel = pcf.order_w(result)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(ordered_pos["x"], ordered_pos["y"], "-o", ms=2, lw=0.8)
ax.set(aspect="equal", title="Recovered order");
```

The result also reports how many points the ordering reached, and the range of the
ordering coordinate $\gamma$:

```{code-cell} python
print(int(result.n_visited), result.gamma_range)
```

All 400 points are ordered, and $\gamma$ runs from $-1$ at one end to $1$ at the
other.

## Fit a smooth track

{func}`~phasecurvefit.fit_track` runs the same ordering, then trains an autoencoder
that gives every point a value of $\gamma$ and learns a smooth mean track
$x(\gamma)$. Training takes a few seconds.

```{code-cell} python
config = pcf.nn.TrainingConfig(show_pbar=False)
ordering, fitted, losses = pcf.fit_track(
    pos, vel, key=jax.random.key(1), training_config=config
)
```

Calling the fitted result with values of $\gamma$ evaluates the track:

```{code-cell} python
gamma = jnp.linspace(*fitted.gamma_range, 300)
track = fitted(gamma)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.scatter(pos["x"], pos["y"], c=fitted.gamma, s=6)
ax.plot(track["x"], track["y"], c="k", lw=1.5, label="fitted track")
fig.colorbar(im, ax=ax, label=r"$\gamma$")
ax.set(aspect="equal")
ax.legend(loc="lower left");
```

## Check the ordering

We kept the true parameter `t`. If the ordering is right, $\gamma$ rises or falls
with `t` along the whole curve, so their rank correlation is close to 1 in size:

```{code-cell} python
import numpy as np

rank = lambda a: np.argsort(np.argsort(np.asarray(a)))
print(abs(np.corrcoef(rank(fitted.gamma), rank(t))[0, 1]))
```

## What you made

You ordered 400 shuffled points without a start point, gave each a value of
$\gamma$, and fitted a smooth track you can evaluate anywhere along the curve.

- {doc}`../tutorials/index`: complete analyses of simulated stellar streams and of a
  curve that crosses itself.
- {doc}`is-it-right`: which orderer suits your data.
- {doc}`../how-to/index`: tune the ordering, the metric and the training.
