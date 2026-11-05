---
sd_hide_title: true
---

# phasecurvefit

<h1> <code> phasecurvefit </code> </h1>

**Order phase-space data along a curve, then fit a smooth track through it.**

You have positions and velocities for many samples that lie along a curve, such as
the stars of a stellar stream, but not the order they come in along it.
phasecurvefit finds that order, gives every sample an ordering coordinate
$\gamma$, and fits a smooth mean track $x(\gamma)$. It uses the velocities as well as
the positions, needs no known start point, and is built on JAX, so it works with
`jit`, `vmap` and `grad` and runs on CPU or GPU.

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
track = fitted(jnp.linspace(*fitted.gamma_range, 300))  # the smooth mean track
print(fitted.gamma.shape)  # an ordering coordinate for every sample: (400,)
```

The script that draws the figure above runs this same code
([`docs/_static/generate_hero.py`](https://github.com/GalacticDynamics/phasecurvefit/blob/main/docs/_static/generate_hero.py)).

## Where to start

Pick the line that matches your data. Each one leads to a tutorial that runs from
simulated data to a result.

- **A stream or curve, and no known start point.** A minimum-spanning-tree backbone
  finds both ends itself: [order a stream with no start index](tutorials/stream_mst.ipynb).
- **A stream with a known progenitor or start point.** Walk along the flow from it
  and fit the track: [the stream autoencoder tutorial](tutorials/stream_autoencoder.ipynb),
  the best first tutorial.
- **A curve that crosses itself.** Velocities keep the ordering on the right strand:
  [the epitrochoid autoencoder tutorial](tutorials/epitrochoid_autoencoder.ipynb).
- **Contaminated data.** A stream-plus-background mixture model gives each sample
  a membership probability: [the outlier-rejection tutorial](tutorials/outlier_rejection.ipynb).
- **A track that must be fast**, for example inside a likelihood evaluated at every
  step of an MCMC: [the training-free running-mean track](tutorials/stream_runningmean.ipynb).

## The documentation

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Start
:link: guides/quickstart
:link-type: doc

What the library does, whether it suits your data, installation, and a first fit.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Complete analyses from simulated data to a result, grouped by the kind of data.
:::

:::{grid-item-card} How-to guides
:link: guides/orderers
:link-type: doc

One task each: choose an orderer, tune the metric, fill gaps, reject outliers, use JAX.
:::

:::{grid-item-card} How it works
:link: guides/algorithm
:link-type: doc

The local-flow walk, the MST backbone, the SOM, the autoencoder and the mixture model.
:::

:::{grid-item-card} Reference
:link: api/index
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
  nearest point is often on the wrong strand. phasecurvefit uses velocities as well
  as positions, so the ordering stays on the right strand.
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
:caption: 📚 Guides

guides/quickstart
guides/metrics
guides/algorithm
guides/orderers
guides/som
guides/nn
guides/outliers
guides/jax-integration
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
:caption: 🔌 API Reference

api/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 🔀 Migration

migration/v0.3-to-v0.4
migration/v0.2-to-v0.3
migration/v0.1-to-v0.2
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: More

contributing
citation
```
