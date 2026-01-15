---
sd_hide_title: true
---

<h1> <code> localflowwalk </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Guides

guides/quickstart
guides/metrics
guides/algorithm
guides/nn
guides/jax-integration
guides/examples
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

**localflowwalk** is a Python library for constructing a single, ordered walk
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
pip install localflowwalk
```

:::

:::{tab-item} uv

```bash
uv add localflowwalk
```

:::

::::

## Quick Example

```python
import jax.numpy as jnp
import localflowwalk as lfw

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
result = lfw.walk_local_flow(
    position, velocity, start_idx=0, lam=1.0
)  # momentum weight

result.ordered_indices
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

For the mathematical background on momentum-weighted ordering, refer to the [original NN+p paper](https://arxiv.org/abs/2201.12042).

## Configuration Options

- **`metric`**: Distance metric to use (default: `AlignedMomentumDistanceMetric`). Determines how "closeness" is computed. See [Metrics Guide](guides/metrics).

- **`lam`**: Momentum weight parameter (for metrics that use it).
  - `lam=0`: Spatial proximity only
  - `lam>0`: Blend spatial and momentum information
  - Not used by all metrics

- **`max_dist`**: Maximum allowed distance to the next point. Stops the walk if no unvisited point is closer.

- **`n_max`**: Maximum number of points to include in the walk (caps walk length).

- **`start_idx`**: Starting index in the data (default: 0).

- **`terminate_indices`**: Set of indices where the walk should stop.

**`strategy`**: Neighbor query strategy instance. Options:
    - `BruteForceStrategy()` (default): compute distances to all points
    - `KDTreeStrategy(k=...)`: spatial KD-tree prefiltering, then metric selection
        - Install optional dependency: `uv add localflowwalk[kdtree]`
        - Uses [jaxkd](https://github.com/dodgebc/jaxkd)

Example using KD-tree:

```python
from localflowwalk import KDTreeStrategy

result = lfw.walk_local_flow(
    position,
    velocity,
    start_idx=0,
    lam=1.0,
    strategy=KDTreeStrategy(k=2),  # query 2 spatial neighbors
)
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
:link: guides/examples
:link-type: doc

Real-world use cases
:::

::::

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
