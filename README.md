# phasecurvefit: Construct Paths through Phase-Space Points

[![PyPI version](https://img.shields.io/pypi/v/phasecurvefit.svg)](https://pypi.org/project/phasecurvefit/)
[![Python versions](https://img.shields.io/pypi/pyversions/phasecurvefit.svg)](https://pypi.org/project/phasecurvefit/)

Construct paths through phase-Space points, supporting many different
algorithms.

## Features

- **JAX-powered**: Fully compatible with JAX transformations (`jit`, `vmap`,
  `grad`)
- **GPU-ready**: Runs on CPU, GPU, or TPU via JAX
- **Type-safe**: Comprehensive (optionally runtime checked) type hints with
  `jaxtyping`
- **Pluggable metrics**: Customizable distance metrics for different physical
  interpretations
- **Pluggable query strategies**: Flexible neighbor search strategies (e.g.,
  brute-force, KD-tree) to optimize performance
- **Highly customizable ML setup and training**: Well-chosen defaults with
  highly flexible customization for specific use-cases.
- **Physical units**: Optional support via `unxt` for unit-aware calculations

## Installation

Install the core package:

```bash
pip install phasecurvefit
```

Or with uv:

```bash
uv add phasecurvefit
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
pip install phasecurvefit[interop]  # Install with unxt for unit support
pip install phasecurvefit[kdtree]  # Install with jaxkd for KD-tree strategy
```

Or with uv:

```bash
uv add phasecurvefit --extra interop
uv add phasecurvefit --extra kdtree
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

# Create phase-space observations as dictionaries (Cartesian coordinates)
pos = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
}
vel = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5, 0.5]),
}

# Step 1: Order the observations (use KD-tree for spatial neighbor prefiltering)
config = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=3))  # k=3 for this small dataset
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
print(result.indices)  # Initial ordering

# Step 2: Create normalizer and autoencoder
key = jax.random.key(0)
normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
autoencoder = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=result.gamma_range, key=key
)

# Step 3: Configure and run training
train_config = pcf.nn.TrainingConfig(
    n_epochs_encoder=100,  # Encoder-only epochs
    n_epochs_both=50,  # Joint training epochs
    show_pbar=False,  # Disable progress bar
)

# Train the autoencoder
result, _, losses = pcf.nn.train_autoencoder(
    autoencoder, result, config=train_config, key=key
)

print(result.indices)  # Post-training ordering
```

### With Physical Units

When `unxt` is installed, you can use physical units throughout the workflow:

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf
import unxt as u

# Create phase-space observations with units
pos = {
    "x": u.Q([0.0, 1.0, 2.0, 3.0, 4.0], "kpc"),
    "y": u.Q([0.0, 0.5, 1.0, 1.5, 2.0], "kpc"),
}
vel = {
    "x": u.Q([1.0, 1.0, 1.0, 1.0, 1.0], "km/s"),
    "y": u.Q([0.5, 0.5, 0.5, 0.5, 0.5], "km/s"),
}

# Step 1: Order with units (units are preserved throughout)
metric_scale = u.Q(1.0, "kpc")
config = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=3))
result = pcf.walk_local_flow(
    pos,
    vel,
    config=config,
    start_idx=0,
    metric_scale=metric_scale,
    usys=u.unitsystems.galactic,
)

# Step 2: Create normalizer and autoencoder (handles units automatically)
key = jax.random.key(0)
normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
autoencoder = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=result.gamma_range, key=key
)
result, _, losses = pcf.nn.train_autoencoder(
    autoencoder, result, config=train_config, key=key
)
```

## Distance Metrics

The algorithm supports pluggable distance metrics to control how points are
ordered. The default metric is `AlignedMomentumDistanceMetric`, which combines
spatial proximity with velocity alignment:

```python
import jax.numpy as jnp
import phasecurvefit as pcf

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Use default metric (AlignedMomentumDistanceMetric)
config = pcf.WalkConfig()
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

### Using Different Metrics

`phasecurvefit` provides three built-in metrics:

1. **AlignedMomentumDistanceMetric** (default): Combines spatial distance with
   velocity alignment (momentum-weighted nearest neighbor)
2. **FullPhaseSpaceDistanceMetric**: True 6D Euclidean distance in phase space
3. **SpatialDistanceMetric**: Pure spatial distance, ignoring velocity

```python
import jax.numpy as jnp
import phasecurvefit as pcf

# Define simple Cartesian arrays (not quantities)
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Pure spatial ordering (ignores velocity)
config_spatial = pcf.WalkConfig(metric=pcf.metrics.SpatialDistanceMetric())
result = pcf.walk_local_flow(
    pos, vel, config=config_spatial, start_idx=0, metric_scale=0.0
)

# Full 6D phase-space distance
config_phase = pcf.WalkConfig(metric=pcf.metrics.FullPhaseSpaceDistanceMetric())
result = pcf.walk_local_flow(
    pos, vel, config=config_phase, start_idx=0, metric_scale=1.0
)
```

### Custom Metrics

You can define custom metrics by subclassing `AbstractDistanceMetric`:

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf


class WeightedPhaseSpaceMetric(pcf.metrics.AbstractDistanceMetric):
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

## Query Strategies

The algorithm supports pluggable query strategies to control how neighbors are
found. A strategy determines which points are considered as potential next steps
in the walk.

`phasecurvefit` provides two built-in strategies:

1. **BruteForce** (default): Compute distances to all remaining points and
   select the nearest one. Efficient for small to medium datasets.
2. **KDTree**: Use spatial KD-tree prefiltering to accelerate neighbor searches
   for large datasets (requires optional `jaxkd` dependency).

### Using Built-in Strategies

```python
import jax.numpy as jnp
import phasecurvefit as pcf

# Define simple Cartesian arrays
pos = {"x": jnp.array([0.0, 1.0, 2.0]), "y": jnp.array([0.0, 0.5, 1.0])}
vel = {"x": jnp.array([1.0, 1.0, 1.0]), "y": jnp.array([0.5, 0.5, 0.5])}

# Default strategy (brute-force — no configuration needed)
config_brute = pcf.WalkConfig(strategy=pcf.strats.BruteForce())
result = pcf.walk_local_flow(
    pos, vel, config=config_brute, start_idx=0, metric_scale=1.0
)

# KD-tree strategy for faster neighbor queries (large datasets)
config_kdtree = pcf.WalkConfig(strategy=pcf.strats.KDTree(k=2))
result = pcf.walk_local_flow(
    pos, vel, config=config_kdtree, start_idx=0, metric_scale=1.0
)
```

### Custom Query Strategies

You can define custom strategies by subclassing `AbstractQueryStrategy`:

```python
import jax.numpy as jnp
import phasecurvefit as pcf


class SmallestIndexStrategy(pcf.strats.AbstractQueryStrategy):
    """Custom strategy: select the smallest unvisited index.

    This is a toy example showing how to implement a custom strategy.
    By returning uniform distances, argmin selects the smallest index
    deterministically. In practice, distance-based strategies like BruteForce
    are more useful.
    """

    def init(self, positions, /, *, metadata):
        """No persistent state needed."""
        return None

    def query(
        self,
        state,
        /,
        current_pos,
        current_vel,
        positions,
        velocities,
        metric_fn,
        metric_scale,
    ):
        """Return uniform distances to all points.

        Since all distances are equal, the walk algorithm's argmin will
        deterministically select the smallest unvisited index.
        """
        # Get number of points
        n_points = len(next(iter(positions.values())))

        # Return uniform distances to all points
        # argmin will pick the smallest unvisited index
        distances = jnp.ones(n_points)

        return pcf.strats.QueryResult(distances=distances, indices=None)


# Use custom strategy via WalkConfig
config = pcf.WalkConfig(strategy=SmallestIndexStrategy())
result = pcf.walk_local_flow(pos, vel, config=config, start_idx=0, metric_scale=1.0)
```

## AI Usage Disclosure

Portions of this codebase (including tests and documentation) were refactored
and generated with the assistance of Language Models. All AI contributions have
been and will continue to be reviewed and verified by the human maintainers.
