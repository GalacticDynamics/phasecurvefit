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
guides/nn
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
:maxdepth: 2
:hidden:
:caption: More

contributing
```

# 🚀 Get Started

**phasecurvefit** is a Python library for constructing a single, ordered walk
through phase-space data. It was originally built for stellar stream
simulations but is general-purpose and applies to any dataset where you want to
order observations by proximity and momentum in phase-space.

The core approach combines:
- **Spatial proximity**: Finding nearby points in position space
- **Velocity momentum**: Preferring points that align with the current velocity direction

This is particularly useful for coherent trajectories in phase-space, such as stellar streams, but works well for many other ordered-walk problems.

---

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install phasecurvefit
```

:::

:::{tab-item} uv

```bash
uv add phasecurvefit
```

:::

:::{tab-item} source, via uv

To install the latest development version of `phasecurvefit` directly from the
GitHub repository, use uv:

```bash
uv add git+https://https://github.com/GalacticDynamics/phasecurvefit.git@main
```

You can customize the branch by replacing `main` with any other branch name.

:::

:::{tab-item} building from source

To build `phasecurvefit` from source, clone the repository and install it with uv:

```bash
cd /path/to/parent
git clone https://https://github.com/GalacticDynamics/phasecurvefit.git
cd phasecurvefit
uv pip install -e .  # editable mode
```

:::

::::

## Quick Example

```python
import jax.numpy as jnp
import phasecurvefit as pcf

# Define phase-space data as dictionaries
position = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5]),
}
velocity = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5]),
}

# Run the algorithm
result = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=1.0)

result.indices
# Array([0, 1, 2, 3])
```

## Features

- ✅ **JAX-native**: Full support for JIT compilation, vectorization, and auto-differentiation
- ✅ **High performance**: Optimized with `jax.lax.while_loop` for speed
- ✅ **Gap filling**: Autoencoder neural network interpolates skipped tracers
- ✅ **Flexible**: Works in any number of dimensions
- ✅ **Type-safe**: Full type annotations with `jaxtyping`
- ✅ **Well-tested**: Comprehensive test suite with property-based testing

## How It Works

Localflowwalk constructs a **single ordered walk** through your phase-space data by iteratively selecting the nearest next point based on:

1. **Current position**: Where you are in the walk
2. **Candidate points**: Remaining unvisited observations
3. **Distance metric**: A configurable function that scores proximity
4. **Termination criteria**: Optional constraints on walk length or distance thresholds

The library ships with multiple built-in metrics (e.g., momentum-weighted, spatial-only), and you can implement custom metrics for domain-specific use cases. See the [Metrics Guide](guides/metrics) for full details and examples.

For the mathematical background on momentum-weighted ordering, refer to the [NN+p paper](https://arxiv.org/abs/2201.12042).

## Configuration Options

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
        - Install optional dependency: `uv add localflowwalk[kdtree]`
        - Uses [jaxkd](https://github.com/dodgebc/jaxkd)

Example using KD-tree:

```python
import phasecurvefit as pcf

config = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=2))
result = pcf.walk_local_flow(position, velocity, config=config)
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

## Citation

The core algorithm originates from Nibauer et al. (2022). If you use **momentum-weighted ordering** or reference the original work in your research, please cite:

```bibtex
@article{nibauer2022charting,
  title={Charting Galactic Accelerations with Stellar Streams and Machine Learning},
  author={Nibauer, Jacob and others},
  journal={arXiv preprint arXiv:2201.12042},
  year={2022}
}
```

If you use **localflowwalk** with custom metrics or for general phase-space ordering, please cite this package directly (check the [GitHub repository](https://github.com/GalacticDynamics/localflowwalk) for the latest citation format).

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

::::

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
