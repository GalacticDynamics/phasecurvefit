---
sd_hide_title: true
---

<h1> <code> phasecurvefit </code> </h1>

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
:caption: 🎓 Tutorials

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

# 🚀 Get Started

**phasecurvefit** is a Python library for ordering phase-space data along a
curve, and fitting a smooth track through it. It was originally built for stellar
stream simulations but is general-purpose and applies to any dataset whose
samples lie along a curve in phase space with the order along it unknown.

The default pipeline, `pcf.order(pos, vel)`, combines:

- **An MST backbone**: a minimum spanning tree of the nearest-neighbour graph,
  whose longest path runs tip to tip along the curve with no progenitor or start
  index needed
- **A SOM refinement**: a Self-Organizing Map whose prototypes average over many
  tracers, smoothing the backbone against local noise

Velocity orients the ordering along the flow, and can also steer the MST where
strands cross. The **local-flow walk**, which follows the velocity field
step by step, is available as an alternative orderer.

This is particularly useful for coherent trajectories in phase-space, such as stellar streams, but works well for many other ordering problems.

## Why phasecurvefit?

Many datasets are samples along a curve in phase space whose order along the
curve is unknown. Before fitting a model to such a curve you need two things: an
ordering coordinate for every sample, and a smooth track through them. Doing
this by hand, or with a position-only nearest-neighbor or clustering method,
breaks down in exactly the cases that matter:

- **Curves that cross or fold back on themselves.** Where two strands meet, the
  nearest point is often on the wrong strand. phasecurvefit uses velocities as
  well as positions, so the ordering stays on the right strand
  (see the [epitrochoid tutorials](tutorials/epitrochoid_autoencoder.ipynb)).
- **No known starting point.** The default MST | SOM pipeline finds the two ends of
  the curve itself, so no progenitor position or hand-picked start index is needed ([MST tutorial](tutorials/stream_mst.ipynb)).
- **Incomplete orderings.** An orderer may order only a reliable subset (the
  local-flow walk does); an
  autoencoder then assigns an ordering coordinate $\gamma$ to every sample and
  learns a smooth mean track through them ([stream autoencoder tutorial](tutorials/stream_autoencoder.ipynb)).
- **Contamination.** A stream-plus-background mixture model gives each sample a
  calibrated membership probability, so interlopers can be down-weighted or
  removed ([outlier-rejection tutorial](tutorials/outlier_rejection.ipynb)).
- **Use inside larger models.** phasecurvefit is built on JAX: the walk, the
  distance metrics and the neural networks work with `jit`, `vmap` and `grad`
  and run on CPU or GPU. A training-free running-mean track is available when
  speed matters more than accuracy, for example inside a likelihood evaluated at
  every step of an MCMC ([running-mean tutorial](tutorials/stream_runningmean.ipynb)).

phasecurvefit is a reusable, tested library for momentum-weighted ordering, with
alternative orderers, gap filling, outlier rejection and optional physical units
(via `unxt`). It was built for stellar streams but applies to any ordered
phase-space data.

---

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install phasecurvefit[all]
```

where "all" enables unit support (through `unxt`) and kdtree support through `jaxkd`.

:::

:::{tab-item} uv

```bash
uv add phasecurvefit --extra all
```

where "all" enables unit support (through `unxt`) and kdtree support through `jaxkd`.

:::

::::

To run the [tutorials](tutorials/index), install the `tutorials` extra instead,
which adds `matplotlib` (plotting) and `galax` (mock-stream generation) on top
of `[all]`:

```bash
pip install phasecurvefit[tutorials]
```

`all` intentionally excludes `tutorials`: `all` is for optional *runtime*
functionality, while `tutorials` is for packages only needed to run the
example notebooks.

::::{tab-set}

:::{tab-item} source, via uv

To install the latest development version of `phasecurvefit` directly from the
GitHub repository, use uv:

```bash
uv add git+https://github.com/GalacticDynamics/phasecurvefit.git@main
```

You can customize the branch by replacing `main` with any other branch name.

:::

:::{tab-item} building from source

To build `phasecurvefit` from source, clone the repository and install it with uv:

```bash
cd /path/to/parent
git clone https://github.com/GalacticDynamics/phasecurvefit.git
cd phasecurvefit
uv pip install -e .  # editable mode
```

:::

::::

## Quick Example

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

# Create phase-space observations as dictionaries
pos = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
}
vel = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5, 0.5]),
}

# Order observations using pcf.order: an MST backbone refined by a SOM.
# (With fewer than 15 points, as here, the SOM is skipped and the MST is used alone.)
result = pcf.order(pos, vel)

# Train autoencoder for gap filling
key = jax.random.key(0)
normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
autoencoder = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=result.gamma_range, key=key
)

train_cfg = pcf.nn.TrainingConfig(
    n_epochs_encoder=100, n_epochs_both=50, show_pbar=False
)
result, _, _ = pcf.nn.train_autoencoder(autoencoder, result, config=train_cfg, key=key)

print(result.indices)  # Array([0, 1, 2, 3, 4])
```

## Features

- ✅ **JAX-native**: The local-flow walk, the MST orderer and the neural networks support JIT compilation, vectorization, and auto-differentiation (the default MST | SOM pipeline needs an explicit orderer under `jit`/`vmap`; see the [JAX guide](guides/jax-integration))
- ✅ **High performance**: The walk is optimized with `jax.lax.while_loop`
- ✅ **No start point needed**: the default MST | SOM pipeline finds both ends of the curve itself
- ✅ **Gap filling**: Autoencoder neural network interpolates skipped tracers
- ✅ **Flexible**: Works in any number of dimensions
- ✅ **Type-safe**: Full type annotations with `jaxtyping`
- ✅ **Well-tested**: Comprehensive test suite with property-based testing

## How It Works

By default, `pcf.order` runs two stages:

1. **MST backbone**: link each tracer to its nearest neighbours, keep the minimum
   spanning tree of those links, and order the tracers along its longest path, from
   one tip of the curve to the other.
2. **SOM refinement**: train a one-dimensional Self-Organizing Map on that ordering
   and order the tracers by their projection onto it. The SOM's prototypes are
   averages over many tracers, so the result is far less sensitive to local noise
   than the MST's individual edges.

Fewer tracers than the SOM has prototypes (15 by default) cannot be refined, so the
MST ordering is returned as it is. See the [Orderers Guide](guides/orderers) and the
[SOM Guide](guides/som).

### The local-flow walk

The alternative {class}`~phasecurvefit.orderers.LocalFlowOrderer` constructs a
**single ordered walk** through your phase-space data by iteratively selecting the
nearest next point based on:

1. **Current position**: Where you are in the walk
2. **Candidate points**: Remaining unvisited observations
3. **Distance metric**: A configurable function that scores proximity
4. **Termination criteria**: Optional constraints on walk length or distance thresholds

Choose it with `pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer())`. It follows
the velocity field, so it suits open streams and curves that cross where the
velocity stays coherent; it needs a start point, and it is fully JAX-traceable.
The library ships with multiple built-in metrics (e.g., momentum-weighted,
spatial-only), and you can implement custom metrics for domain-specific use cases.
See the [Metrics Guide](guides/metrics) for full details and examples.

For the mathematical background on momentum-weighted ordering, refer to the [NN+p paper](https://arxiv.org/abs/2205.11767).

## Local-Flow Walk Options

These configure the walk, through {class}`~phasecurvefit.orderers.LocalFlowOrderer`:

- **`metric`**: Distance metric to use (default: `AlignedMomentumDistanceMetric`). Determines how "closeness" is computed. See [Metrics Guide](guides/metrics).

- **`metric_scale`**: Scale parameter for distance metrics. Interpretation depends on the metric:
  - `AlignedMomentumDistanceMetric`: momentum weight (distance units)
  - `FullPhaseSpaceDistanceMetric`: time scale for velocity-to-position conversion
  - `SpatialDistanceMetric`: unused (can be any value)

- **`max_dist`**: Maximum allowed distance to the next point. Stops the walk if no unvisited point is closer.

- **`n_max`**: Maximum number of points to include in the walk (caps walk length).

- **`start_idx`**: Starting index in the data (default: 0).

- **`terminate_indices`**: Set of indices where the walk should stop.

**`strategy`**: Neighbor query strategy instance. Options:
    - `BruteForce()` (default): compute distances to all points
    - `KDTree(k=...)`: spatial KD-tree prefiltering, then metric selection
        - Install optional dependency: `uv add phasecurvefit[kdtree]`
        - Uses [jaxkd](https://github.com/dodgebc/jaxkd)

Example using KD-tree (requires `jaxkd`):

```python
import phasecurvefit as pcf

config = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=2))
result = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(config=config))
```

## Data Format

Phase-space data uses **raw Python dictionaries** for maximum performance and JAX compatibility:

```python
import jax.numpy as jnp

# Position dictionary: coordinate names → arrays
position = {
    "x": jnp.array([0.0, 1.0, 2.0]),
    "y": jnp.array([0.0, 0.5, 1.0]),
    "z": jnp.array([0.0, 0.1, 0.2]),
}

# Velocity dictionary: same keys → velocity components
velocity = {
    "x": jnp.array([1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5]),
    "z": jnp.array([0.0, 0.0, 0.0]),
}
```

This dict-based API is designed for:

- Efficient JAX tree operations via `jax.tree.map`
- Seamless integration with JAX transformations (`jit`, `vmap`, `grad`)
- Minimal overhead in hot loops

## Next Steps

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {material-regular}`rocket_launch;2em` Quickstart
:link: guides/quickstart
:link-type: doc

Get up and running in minutes
:::

:::{grid-item-card} {material-regular}`tune;2em` Distance Metrics
:link: guides/metrics
:link-type: doc

Explore built-in and custom metrics
:::

:::{grid-item-card} {material-regular}`psychology;2em` Neural Network Gap Filling
:link: guides/nn
:link-type: doc

Interpolate skipped observations
:::

:::{grid-item-card} {material-regular}`code;2em` API Reference
:link: api/index
:link-type: doc

Full API documentation
:::

:::{grid-item-card} {material-regular}`bolt;2em` JAX Integration
:link: guides/jax-integration
:link-type: doc

Optimize with JIT, vmap, and grad
:::

:::{grid-item-card} {material-regular}`book;2em` Examples
:link: tutorials/index
:link-type: doc

Interactive tutorials with Jupyter notebooks
:::

:::{grid-item-card} {material-regular}`format_quote;2em` Citation
:link: citation
:link-type: doc

What to cite, depending on what you use
:::

::::

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
