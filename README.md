# localflowwalk: Construct Paths through Phase-Space Points

[![PyPI version](https://img.shields.io/pypi/v/localflowwalk.svg)](https://pypi.org/project/localflowwalk/)
[![Python versions](https://img.shields.io/pypi/pyversions/localflowwalk.svg)](https://pypi.org/project/localflowwalk/)

Construct paths through phase-Space points, supporting many different
algorithms.

## Installation

Install the core package:

```bash
pip install localflowwalk
```

Or with uv:

```bash
uv add localflowwalk
```

### Optional Dependencies

localflowwalk has optional dependencies for extended functionality:

- **unxt**: Physical units support for phase-space calculations
- **tree (jaxkd)**: Spatial KD-tree queries for large datasets

Install with optional dependencies:

```bash
pip install localflowwalk[interop]  # Install with unxt for unit support
pip install localflowwalk[kdtree]  # Install with jaxkd for KD-tree strategy
```

Or with uv:

```bash
uv add localflowwalk --extra interop
uv add localflowwalk --extra kdtree
```

## Quick Start

```python
import jax.numpy as jnp
import localflowwalk as lfw
from localflowwalk import KDTreeStrategy

# Create phase-space observations as dictionaries (Cartesian coordinates)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Order the observations (use KD-tree for spatial neighbor prefiltering)
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, strategy=KDTreeStrategy(k=2)
)
print(result.ordered_indices)  # Array([0, 1, 2], dtype=int32)
```

### With Physical Units

When `unxt` is installed, you can use physical units:

```python
import unxt as u

# Create phase-space observations with units
pos = {"x": u.Q([0.0, 1.0, 2.0], "kpc"), "y": u.Q([0.0, 0.5, 1.0], "kpc")}
vel = {"x": u.Q([1.0, 1.0, 1.0], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}

# Units are preserved throughout the calculation
result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=u.Q(1.0, "kpc"))
```

## Features

- **JAX-powered**: Fully compatible with JAX transformations (`jit`, `vmap`,
  `grad`)
- **Cartesian coordinates**: Works in Cartesian coordinate space (x, y, z)
- **Physical units**: Optional support via `unxt` for unit-aware calculations
- **Pluggable metrics**: Customizable distance metrics for different physical
  interpretations
- **Type-safe**: Comprehensive type hints with `jaxtyping`
- **GPU-ready**: Runs on CPU, GPU, or TPU via JAX
- **Spatial KD-tree option**: Use [jaxkd](https://github.com/dodgebc/jaxkd) to
  prefilter neighbors

# Distance Metrics

The algorithm supports pluggable distance metrics to control how points are
ordered. The default `FullPhaseSpaceDistanceMetric` uses true 6D Euclidean
distance across positions and velocities:

```python
from localflowwalk.metrics import FullPhaseSpaceDistanceMetric
import jax.numpy as jnp

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Default full phase-space metric
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, metric=FullPhaseSpaceDistanceMetric()
)
```

### Using Different Metrics

`localflowwalk` provides three built-in metrics:

1. **FullPhaseSpaceDistanceMetric** (default): True 6D Euclidean distance in
   phase space
2. **AlignedMomentumDistanceMetric**: Combines spatial distance with velocity
   alignment (original NN+p)
3. **SpatialDistanceMetric**: Pure spatial distance, ignoring velocity

```python
from localflowwalk.metrics import SpatialDistanceMetric, FullPhaseSpaceDistanceMetric
import jax.numpy as jnp

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Pure spatial ordering (ignores velocity)
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=0.0, metric=SpatialDistanceMetric()
)

# Full 6D phase-space distance
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, metric=FullPhaseSpaceDistanceMetric()
)
```

### Custom Metrics

You can define custom metrics by subclassing `AbstractDistanceMetric`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from localflowwalk.metrics import AbstractDistanceMetric


class WeightedPhaseSpaceMetric(AbstractDistanceMetric):
    """Custom weighted phase-space metric."""

    def __call__(self, current_pos, current_vel, positions, velocities, lam):
        # Compute position distance
        pos_diff = jax.tree.map(jnp.subtract, positions, current_pos)
        pos_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, pos_diff)))

        # Compute velocity distance
        vel_diff = jax.tree.map(jnp.subtract, velocities, current_vel)
        vel_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, vel_diff)))

        # Custom weighting scheme
        return jnp.sqrt(pos_dist_sq + (lam**2) * vel_dist_sq)


# Use custom metric
result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, metric=WeightedPhaseSpaceMetric()
)
```

See the
[Metrics Guide](https://localflowwalk.readthedocs.io/en/latest/guides/metrics.html)
for more details and examples.

## KD-tree Strategy

For large datasets, you can enable spatial KD-tree prefiltering to accelerate
neighbor selection:

```python
# Install optional dependency first:
# pip install localflowwalk[kdtree]

import jax.numpy as jnp

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Use KD-tree strategy and query 2 spatial neighbors per step
from localflowwalk import KDTreeStrategy

result = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=1.0, strategy=KDTreeStrategy(k=2)
)
```

## References

Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams and
Machine Learning."

## License

MIT License
