---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
sd_hide_title: true
---

# phasecurvefit

<h1> <code> phasecurvefit </code> </h1>

**Order phase-space data along a curve, then fit a smooth track through it.**

You have positions and velocities for many samples that lie along a curve, such as
the stars of a stellar stream, but not the order they come in along it. Or you have
an image of the curve, such as a density map of a stream, and each pixel becomes a
sample weighted by its density. phasecurvefit finds the order, gives every sample an
ordering coordinate $\gamma$, and fits a smooth mean track $x(\gamma)$. Its default
needs no known start point, and velocity-aware orderers keep the ordering on the
right strand where a curve crosses itself. It is built on JAX: its orderers and
networks work with `jit`, `vmap` and `grad`, and everything runs on CPU or GPU.

::::{tab-set}

:::{tab-item} Point cloud

```{image} _static/hero.png
---
alt: >-
  Two panels. Left, "Input: shuffled tracers": 400 grey points along a spiral of
  one and a half turns, joined in their stored order by lines that criss-cross the
  whole figure. Right, "pcf.fit_track: ordering and track": the same points coloured
  from purple at the inner end to yellow at the outer end by the ordering
  coordinate gamma, with an orange fitted track running smoothly along the spiral.
---
```

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

# A stand-in for your data: 400 noisy points along a spiral, with velocities
t = jnp.linspace(0.5, 0.5 + 3 * jnp.pi, 400)
r = 1 + 0.6 * t
kx, ky, kp = jax.random.split(jax.random.key(0), 3)
pos = {
    "x": r * jnp.cos(t) + 0.2 * jax.random.normal(kx, t.shape),
    "y": r * jnp.sin(t) + 0.2 * jax.random.normal(ky, t.shape),
}
vel = {"x": 0.6 * jnp.cos(t) - r * jnp.sin(t), "y": 0.6 * jnp.sin(t) + r * jnp.cos(t)}
shuffle = jax.random.permutation(kp, t.size)  # hide the true order
pos, vel = jax.tree.map(lambda a: a[shuffle], (pos, vel))

config = pcf.nn.TrainingConfig(show_pbar=False)
ordering, fitted, losses = pcf.fit_track(
    pos, vel, key=jax.random.key(1), training_config=config
)
track = fitted(jnp.linspace(*fitted.track_range, 300))  # the smooth mean track
print(fitted.gamma.shape)  # an ordering coordinate for every sample: (400,)
```

:::

:::{tab-item} Density field

```{image} _static/hero-field.png
---
alt: >-
  Two panels. Left, "Input: a density image": a grey image of the same spiral,
  dark along the curve and fading out to either side. Right, "pcf.fit_track with
  weights": the bright pixels of the image coloured from purple at the outer end to
  yellow at the inner end by the ordering coordinate gamma, fainter where the
  density is lower, with an orange fitted track running along the middle of the
  spiral.
---
```

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

# A stand-in for your data: the same spiral as an image, blurred across the curve
t = jnp.linspace(0.5, 0.5 + 3 * jnp.pi, 400)
r = 1 + 0.6 * t
grid = jnp.linspace(-7.5, 7.5, 120)
X, Y = jnp.meshgrid(grid, grid)
cx, cy = r * jnp.cos(t), r * jnp.sin(t)
d2 = jnp.min((X[..., None] - cx) ** 2 + (Y[..., None] - cy) ** 2, axis=-1)
image = jnp.exp(-d2 / (2 * 0.3**2))

keep = image.ravel() > 0.05  # each bright pixel becomes a sample
pos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
vel = jax.tree.map(jnp.zeros_like, pos)  # an image has no velocities
weights = image.ravel()[keep]  # weighted by the density there

config = pcf.nn.TrainingConfig(show_pbar=False)
ordering, fitted, losses = pcf.fit_track(
    pos, vel, key=jax.random.key(1), weights=weights, training_config=config
)
track = fitted(jnp.linspace(*fitted.track_range, 300))  # the smooth mean track
print(fitted.gamma.shape == weights.shape)  # a gamma for every pixel: True
```

An image has no velocities, so nothing sets which end of the curve $\gamma$ starts
from. `fit_track` orders with the default pipeline, which is fine for a thin
stream like this one. For a wide or uniform field, order with a
{class}`~phasecurvefit.orderers.RidgeOrderer` first ({doc}`how-to/choose-orderer`).

:::

::::

The script that draws these figures runs this same code
([`docs/_static/generate_hero.py`](https://github.com/GalacticDynamics/phasecurvefit/blob/main/docs/_static/generate_hero.py)).

## Where to start

Pick the line that matches your data. Each one leads to a tutorial that runs from
simulated data to a result.

- **A stream or curve, and no known start point.** A minimum-spanning-tree backbone
  finds both ends itself: [order a stream with no start index](tutorials/stream_mst.md).
- **A stream with a known progenitor or start point.** Walk along the flow from it
  and fit the track: [the stream autoencoder tutorial](tutorials/stream_autoencoder.md),
  the best first tutorial.
- **A curve that crosses itself.** The default pipeline orders by position and cuts
  across the crossings, so use a velocity-aware orderer instead:
  [the epitrochoid autoencoder tutorial](tutorials/epitrochoid_autoencoder.md).
- **Contaminated data.** A stream-plus-background mixture model gives each sample
  a membership probability: [the outlier-rejection tutorial](tutorials/outlier_rejection.md).
- **A track that must be fast**, for example inside a likelihood evaluated at every
  step of an MCMC: [the training-free running-mean track](tutorials/stream_runningmean.md).
- **An image or density map, not a list of samples.** Each pixel becomes a sample
  weighted by its value: [the density-field tutorial](tutorials/density_fields.md).

## The documentation

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Start
:link: start/what-it-does
:link-type: doc

What the library does, whether it suits your data, installation, and a first fit.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Complete analyses from simulated data to a result, grouped by the kind of data.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

One task each: choose an orderer, tune the metric, fill gaps, reject outliers, use JAX.
:::

:::{grid-item-card} How it works
:link: explanation/index
:link-type: doc

The local-flow walk, the MST backbone, the SOM, the autoencoder and the mixture model.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Every public class and function, options and defaults, migration notes and citation.
:::

::::

## Why phasecurvefit? [![pyOpenSci Peer-Reviewed](https://pyopensci.org/badges/peer-reviewed.svg)](https://github.com/pyOpenSci/software-submission/issues/285)

Before fitting a model to a curve in phase space you need an ordering coordinate for
every sample and a smooth track through them. Doing this by hand, or with a
position-only nearest-neighbour or clustering method, breaks down in the cases that
matter:

- **Curves that cross or fold back on themselves.** Where two strands meet, the
  nearest point is often on the wrong strand. phasecurvefit's velocity-aware
  orderers use velocities as well as positions, so the ordering stays on the right
  strand.
- **No known starting point.** The default pipeline finds the two ends of the curve
  itself, so no progenitor position or hand-picked start index is needed.
- **Incomplete orderings.** An orderer may order only a reliable subset. An
  autoencoder then assigns $\gamma$ to every sample and learns a smooth mean track.
- **Contamination.** A stream-plus-background mixture model gives each sample a
  calibrated membership probability, so interlopers can be down-weighted or removed.
- **Use inside larger models.** The orderers and networks work with `jit`, `vmap`
  and `grad`, and a training-free running-mean track is available when speed matters
  more than accuracy.

phasecurvefit was built for stellar streams and applies to any ordered phase-space
data.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start

start/what-it-does
start/is-it-right
start/install
start/quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: How-to guides

how-to/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: How it works

explanation/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: About

contributing
```
