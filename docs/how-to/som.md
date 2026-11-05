---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Refine an ordering with a self-organizing map

This guide shows how to use {class}`~phasecurvefit.orderers.SOMOrderer`: chained
after another orderer, on its own, with outlier clipping, under `jit`, and as an
ensemble. {doc}`../explanation/som` explains how the map is trained and projected.

## Chain it after another orderer

The SOM smooths an ordering it is handed, so it works best after another orderer.
This is what the default pipeline does; chaining it yourself gives you control over
both stages:

```{code-cell} python
import jax.numpy as jnp

import phasecurvefit as pcf

ang = jnp.linspace(0.0, jnp.pi, 120)
pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}

chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.SOMOrderer(
    n_prototypes=12
)
result = pcf.order(pos, vel, chain)
assert int(result.n_visited) == 120
```

On noisy data the MST's backbone follows individual samples, while the SOM's
follows their average:

```{code-cell} python
import jax
import matplotlib.pyplot as plt

k1, k2 = jax.random.split(jax.random.key(1))
noisy = {
    "x": pos["x"] + 0.2 * jax.random.normal(k1, (120,)),
    "y": pos["y"] + 0.2 * jax.random.normal(k2, (120,)),
}
mst = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(noisy["x"], noisy["y"], ".", c="0.7", ms=4, label="samples")
for name, orderer in {"MST": mst, "MST | SOM": chain}.items():
    backbone = pcf.order(noisy, vel, orderer).backbone
    ax.plot(backbone["x"], backbone["y"], lw=1.5, label=f"{name} backbone")
ax.set(aspect="equal")
ax.legend();
```

Samples the earlier stage left unvisited stay unvisited, so if that stage cut the
curve short (an MST whose `jump_cap` cannot bridge a gap, say) the chain inherits the
cut. Check `n_visited` when chaining.

## Run it on its own

On its own, the SOM seeds its lattice by binning the samples along the first
principal axis of the positions:

```{code-cell} python
result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=12))
assert int(result.n_visited) == 120
```

This works only when the curve does not double back along that axis, which holds up
to about one turn. For a near-closed loop or a curve that winds further, chain the SOM
after an {class}`~phasecurvefit.orderers.MSTOrderer` instead. On its own the SOM's
direction is arbitrary; set `orient_by_velocity=True` to make $\gamma$ increase along
the mean velocity.

## Read the along-track coordinate

The SOM gives each sample a continuous arc length along its backbone, in input
order, as `result.chord`:

```{code-cell} python
assert result.chord.shape == (120,)
```

## Tune it

- **`n_prototypes`**: the number of lattice points. The setting that matters is the
  smoothing length along the curve,
  `sigma_phys = sigma_end * L / (n_prototypes - 1)`, where `L` is the track length.
  Bends tighter than `sigma_phys`, and arms closer together than it, are smoothed
  away. Raise `n_prototypes` until `sigma_phys` is below the curve's tightest turn
  and smallest gap between arms. A curve that crosses itself needs many: a six-lobed
  epitrochoid is ordered perfectly at `n_prototypes=90` and not at all at 25.
- **`sigma_start`, `sigma_end`**: the neighbourhood width in lattice units, shrunk
  geometrically during training. `sigma_start=None` uses
  `max(n_prototypes / 4, sigma_end)` on its own, and `sigma_end` when chained, so a
  chained SOM refines locally.
- **`n_epochs`**: tens, not thousands; each epoch is one pass over all the data.
- **`densify_factor`**: backbone samples per lattice segment. It has little effect
  above about 2 and its cost grows linearly, so raise it only for a strongly curved
  track.
- **`orient_by_velocity`**: flips the result so $\gamma$ increases along the mean
  velocity. It overrides a direction an earlier stage chose. Default `False`.
- **`metric`, `metric_scale`**: by default the SOM follows the previous stage. After
  an orderer that used velocity it uses a phase-space metric with a scale derived
  from the data, and otherwise it uses position alone. Set either to override. The
  metric must be symmetric, so
  {class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` does not work here.

{doc}`../reference/options` lists the defaults.

If the SOM's ordering disagrees strongly with the one it was given, it warns. Either
the earlier ordering was poor, or the lattice is too coarse and has tangled a good
ordering. If the earlier stage was doing well, raise `n_prototypes`.

## Clip outliers

The SOM has no outlier rejection by default, and contamination pulls its backbone
off the curve while the ordering still looks good. Set `outlier_clip_sigma` to reject
samples far from their nearest prototype, iterating up to `outlier_clip_max_iters`
times:

```{code-cell} python
som = pcf.orderers.SOMOrderer(n_prototypes=25, outlier_clip_sigma=3.0)
```

A rejected sample gets `-1` in `indices` and `nan` in `chord`. Clipping at the MST
stage instead, with `MSTOrderer(edge_clip_sigma=3.0) | SOMOrderer()`, also keeps the
backbone on the curve, and the two can be combined. On an arc with uniform background
mixed in, the backbone's error against the true track was:

| contamination | SOM on its own | `MSTOrderer(edge_clip_sigma=3) \| SOMOrderer()` |
|---|---|---|
| 0% | 0.025 | 0.025 |
| 10% | 0.100 | 0.026 |
| 20% | 0.159 | 0.026 |

## Use it under `jit`

On its own, the SOM can be traced:

```{code-cell} python
t = jnp.linspace(0.0, 2.0, 60)
arc = {"x": jnp.cos(t), "y": jnp.sin(t)}
arc_vel = {"x": -jnp.sin(t), "y": jnp.cos(t)}

chord = jax.jit(
    lambda q, p: pcf.order(q, p, pcf.orderers.SOMOrderer(n_prototypes=8)).chord
)(arc, arc_vel)
assert chord.shape == (60,)
```

Chained after another stage it cannot, because the number of samples it works on
depends on what the earlier stage visited. Chaining under `jit` raises a `TypeError`.
Run the chain outside `jit`, or order in two steps and pass the second stage a
concrete `init`.

## Build an ensemble

To estimate the uncertainty of the backbone, resample the data. The functions in
{mod}`phasecurvefit.som` can be `vmap`-ed over a matrix of weights, one bootstrap
resample per member:

```{code-cell} python
from phasecurvefit import som

t = jnp.linspace(0.0, 2.0, 300)
arc = {"x": jnp.cos(t) * 3, "y": jnp.sin(t) * 3}
arc_vel = {"x": -jnp.sin(t), "y": jnp.cos(t)}
metric = pcf.metrics.SpatialDistanceMetric()
proto_q, proto_p = som.init_prototypes(arc, arc_vel, n_prototypes=20)

keys = jax.random.split(jax.random.key(0), 8)
weights = jax.vmap(lambda k: som.bootstrap_weights(k, 300))(keys)


def member(w):
    res = som.fit(proto_q, proto_p, arc, arc_vel, metric=metric, weights=w)
    bq, bp = som.densify(res.prototype_positions, res.prototype_velocities, factor=5)
    return som.chord(bq, bp, arc, arc_vel, metric=metric)


chords = jax.vmap(member)(weights)
assert chords.shape == (8, 300)
assert float(jnp.std(chords, axis=0).mean()) > 0
```

{func}`~phasecurvefit.som.bootstrap_weights` draws `N` indices with replacement and
returns how often each sample came up. A weight is a multiplicity: zero means the
sample is absent, and `k` means it appears `k` times, so 0/1 weights give a
subsample. `weights` and outlier clipping cannot be combined, since clipping sets its
own weights.

Perturbing the initial lattice does not give an ensemble: the members converge to the
same answer within a few epochs ({doc}`../explanation/som`).
