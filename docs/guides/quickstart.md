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
import localflowwalk as lfw
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
result = lfw.walk_local_flow(
    position,
    velocity,
    start_idx=0,  # Start from first point
    lam=1.0,  # Momentum weight
)

print(result.ordered_indices)
# Array([0, 1, 2, 3, 4])
```

### 4. Extract ordered data

Use the convenience function to get reordered arrays:

```python
ordered_pos, ordered_vel = lfw.get_ordered_w(result)

print(ordered_pos["x"])
# Array([0., 1., 2., 3., 4.])
```

## Understanding the Result

The `walk_local_flow` function returns a `LocalFlowWalkResult` NamedTuple with:

- **`ordered_indices`**: Array of indices of the discovered order
- **`position`**: Original position dictionary
- **`velocity`**: Original velocity dictionary


## Adjusting the Momentum Weight

The `lam` parameter controls how much the algorithm favors points in the velocity direction:

```python
# Pure nearest neighbor (spatial only)
result_spatial = lfw.walk_local_flow(position, velocity, start_idx=0, lam=0.0)

# Balanced (default)
result_balanced = lfw.walk_local_flow(position, velocity, start_idx=0, lam=1.0)

# Strong momentum preference
result_momentum = lfw.walk_local_flow(position, velocity, start_idx=0, lam=5.0)
```

## Handling Gaps with max_dist

Use `max_dist` to stop when there's a gap in the data:

```python
# Stop if next nearest point is more than 2 units away
result = lfw.walk_local_flow(
    position,
    velocity,
    start_idx=0,
    lam=1.0,
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

result = lfw.walk_local_flow(position, velocity, start_idx=0, lam=2.0)
```

## JAX Integration

The algorithm is fully compatible with JAX transformations:

### JIT Compilation

```python
from jax import jit


@jit
def order_stream(pos, vel):
    return lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)


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
- [Examples Gallery](examples.md) - See more use cases
- [API Design](api-design.md) - Learn about the dict-based API
- [JAX Integration](jax-integration.md) - Advanced JAX usage
