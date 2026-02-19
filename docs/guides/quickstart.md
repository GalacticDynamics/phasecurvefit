# Quickstart Guide

Get started with localflowwalk in 5 minutes!

## Installation

Install localflowwalk using pip or uv:

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

## Basic Usage

### 1. Import the library

```python
import jax.numpy as jnp
import phasecurvefit as pcf
```

### 2. Prepare your phase-space data

Phase-space data is represented as two dictionaries:
- **position**: Maps coordinate names to position arrays
- **velocity**: Maps coordinate names to velocity arrays

```python
# Example: 2D stream
position = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
}

velocity = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5, 0.5]),
}
```

### 3. Run the algorithm

```python
result = pcf.walk_local_flow(
    position,
    velocity,
    start_idx=0,  # Start from first point
    metric_scale=1.0,  # Metric-dependent scale parameter
)

print(result.ordering)
# Array([0, 1, 2, 3, 4])
```

### 4. Extract ordered data

Use the convenience function to get reordered arrays:

```python
ordered_pos, ordered_vel = pcf.order_w(result)

print(ordered_pos["x"])
# Array([0., 1., 2., 3., 4.])
```

## Understanding the Result

The `walk_local_flow` function returns a `LocalFlowWalkResult` NamedTuple with:

- **`ordering`**: Array of indices of the discovered order
- **`position`**: Original position dictionary
- **`velocity`**: Original velocity dictionary


## Adjusting the Metric Scale

The `metric_scale` parameter controls how the algorithm weighs different aspects of the data. Its interpretation depends on which distance metric you're using:

```python
# Pure nearest neighbor (spatial only) - metric_scale ignored
result_spatial = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=0.0)

# Balanced (default)
result_balanced = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=1.0)

# Higher metric_scale value (interpretation metric-dependent)
result_momentum = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=5.0)
```

## Walking in Reverse

Use the `direction` parameter to trace streams backwards by negating the velocity vectors:

```python
# Default: forward walk following the velocity direction
result_forward = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=1.0)

# Reverse: walk against the velocity direction
result_reverse = pcf.walk_local_flow(
    position, velocity, start_idx=0, metric_scale=1.0, direction="backward"
)
```

This is useful for tracing stellar streams from the tidal tail back towards the progenitor.

## Configuring the Query

Use `WalkConfig` to configure the distance metric and query strategy:

```python
from phasecurvefit.metrics import AlignedMomentumDistanceMetric

# Configure with aligned momentum metric and KD-tree strategy
config = pcf.WalkConfig(
    metric=AlignedMomentumDistanceMetric(),
    strategy=pcf.strats.KDTree(k=5),  # Only 5 points in the fake dataset
)

result = pcf.walk_local_flow(
    position, velocity, config=config, start_idx=0, metric_scale=1.0
)
```

## Handling Gaps with max_dist

Use `max_dist` to stop when there's a gap in the data:

```python
# Stop if next nearest point is more than 2 units away
result = pcf.walk_local_flow(
    position,
    velocity,
    start_idx=0,
    metric_scale=1.0,
    max_dist=2.0,
)

# Check if any points were skipped
if result.n_skipped > 0:
    print(f"Skipped {result.n_skipped} points")
```

## Working in 3D

The algorithm works in any number of dimensions:

```python
# 3D helix
t = jnp.linspace(0, 4 * jnp.pi, 100)
position = {
    "x": jnp.cos(t),
    "y": jnp.sin(t),
    "z": t / (2 * jnp.pi),
}

velocity = {
    "x": -jnp.sin(t),
    "y": jnp.cos(t),
    "z": jnp.ones_like(t) / (2 * jnp.pi),
}

result = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=2.0)
```

## Bidirectional Walks (Forward and Reverse)

For streams that extend in both directions from a starting point, run forward and reverse walks separately, then combine them:

```python
# Run forward walk from starting point
result = pcf.walk_local_flow(
    position, velocity, start_idx=2, metric_scale=1.0, direction="both"
)

# Get the combined ordered indices
print(result.indices)  # Indices ordered from reverse tail through start to forward tail

# Extract the ordered positions and velocities
ordered_pos, ordered_vel = pcf.order_w(result)
```

This is particularly useful for:

- Tracing complete stellar streams from a central progenitor
- Exploring both tidal tails simultaneously
- Verifying stream connectivity in both directions
- Having different parameters for forward vs reverse walks

## JAX Integration

The algorithm is fully compatible with JAX transformations:

### JIT Compilation

```python
from jax import jit


@jit
def order_stream(pos, vel):
    return pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)


result = order_stream(position, velocity)
```

### Vectorization

```python
from jax import vmap

# Position and velocity for a single stream
position = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
    "y": jnp.array([0.0, 0.5, 1.0, 1.5]),
}
velocity = {
    "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5, 0.5]),
}

# To process multiple streams, use vmap with careful handling of dictionaries
# See the JAX integration guide for detailed examples
```

## Next Steps

- [Algorithm Details](algorithm.md) - Understand the math
- [API Design](api-design.md) - Learn about the dict-based API
- [JAX Integration](jax-integration.md) - Advanced JAX usage
