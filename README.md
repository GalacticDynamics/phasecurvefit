# phasecurvefit: Construct Paths through Phase-Space Points

[![PyPI version](https://img.shields.io/pypi/v/phasecurvefit.svg)](https://pypi.org/project/phasecurvefit/)
[![Python versions](https://img.shields.io/pypi/pyversions/phasecurvefit.svg)](https://pypi.org/project/phasecurvefit/)

Construct paths through phase-Space points, supporting many different
algorithms.

## Installation

Install the core package:

```bash
pip install phasecurvefit[all]
```

Or with uv:

```bash
uv add phasecurvefit[all]
```

<details>
  <summary>from source, using uv</summary>

```bash
uv add git+https://github.com/GalacticDynamics/phasecurvefit.git@main
```

You can customize the branch by replacing `main` with any other branch name.

</details>
<details>
  <summary>building from source</summary>

```bash
cd /path/to/parent
git clone https://github.com/GalacticDynamics/phasecurvefit.git
cd phasecurvefit
uv pip install -e .  # editable mode
```

</details>

### Optional Dependencies

phasecurvefit has optional dependencies for extended functionality:

- **unxt**: Physical units support for phase-space calculations
- **tree (jaxkd)**: Spatial KD-tree queries for large datasets

Install with optional dependencies:

```bash
# pip install phasecurvefit[all]  # Install with all extras
pip install phasecurvefit[interop]  # Install with unxt for unit support
pip install phasecurvefit[kdtree]  # Install with jaxkd for KD-tree strategy
```

Or with uv:

```bash
# uv add phasecurvefit --extra all  # installs all extras
uv add phasecurvefit --extra interop
uv add phasecurvefit --extra kdtree
```

## Quick Start

```python
import jax.numpy as jnp
import phasecurvefit as pcf

# Create phase-space observations as dictionaries (Cartesian coordinates)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Order the observations (use KD-tree for spatial neighbor prefiltering)
config = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=2))
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
print(result.indices)  # Array([0, 1, 2], dtype=int32)
```

### With Physical Units

When `unxt` is installed, you can use physical units:

```python
import unxt as u

# Create phase-space observations with units
pos = {"x": u.Q([0.0, 1.0, 2.0], "kpc"), "y": u.Q([0.0, 0.5, 1.0], "kpc")}
vel = {"x": u.Q([1.0, 1.0, 1.0], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}

# Units are preserved throughout the calculation
metric_scale = u.Q(1.0, "kpc")
result = pcf.walk_local_flow(
    pos, vel, start_idx=0, metric_scale=metric_scale, usys=u.unitsystems.galactic
)
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
from phasecurvefit.metrics import FullPhaseSpaceDistanceMetric
import jax.numpy as jnp

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Configure with full phase-space metric (the default)
config = pcf.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

### Using Different Metrics

`phasecurvefit` provides three built-in metrics:

1. **FullPhaseSpaceDistanceMetric** (default): True 6D Euclidean distance in
   phase space
2. **AlignedMomentumDistanceMetric**: Combines spatial distance with velocity
   alignment (NN+p metric)
3. **SpatialDistanceMetric**: Pure spatial distance, ignoring velocity

```python
from phasecurvefit.metrics import SpatialDistanceMetric, FullPhaseSpaceDistanceMetric
import jax.numpy as jnp

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Pure spatial ordering (ignores velocity)
config_spatial = pcf.WalkConfig(metric=SpatialDistanceMetric())
result = pcf.walk_local_flow(
    pos, vel, config=config_spatial, start_idx=0, metric_scale=0.0
)

# Full 6D phase-space distance
config_phase = pcf.WalkConfig(metric=FullPhaseSpaceDistanceMetric())
result = pcf.walk_local_flow(
    pos, vel, config=config_phase, start_idx=0, metric_scale=1.0
)
```

### Custom Metrics

You can define custom metrics by subclassing `AbstractDistanceMetric`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from phasecurvefit.metrics import AbstractDistanceMetric


class WeightedPhaseSpaceMetric(AbstractDistanceMetric):
    """Custom weighted phase-space metric."""

    def __call__(self, current_pos, current_vel, positions, velocities, metric_scale):
        # Compute position distance
        pos_diff = jax.tree.map(jnp.subtract, positions, current_pos)
        pos_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, pos_diff)))

        # Compute velocity distance
        vel_diff = jax.tree.map(jnp.subtract, velocities, current_vel)
        vel_dist_sq = sum(jax.tree.leaves(jax.tree.map(jnp.square, vel_diff)))

        # Custom weighting scheme
        return jnp.sqrt(pos_dist_sq + (metric_scale**2) * vel_dist_sq)


# Use custom metric via WalkConfig
config = pcf.WalkConfig(metric=WeightedPhaseSpaceMetric())
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

See the
[Metrics Guide](https://phasecurvefit.readthedocs.io/en/latest/guides/metrics.html)
for more details and examples.

## KD-tree Strategy

For large datasets, you can enable spatial KD-tree prefiltering to accelerate
neighbor selection:

```python
# Install optional dependency first:
# pip install phasecurvefit[kdtree]

import jax.numpy as jnp

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Use KD-tree strategy and query 2 spatial neighbors per step
config = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=2))
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```
