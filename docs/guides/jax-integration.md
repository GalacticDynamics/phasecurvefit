# JAX Integration

This guide shows how to use `localflowwalk` with JAX for faster computation, batching, and differentiation.

## Basic Usage

The library works seamlessly with JAX arrays—no special setup needed:

```python
import jax.numpy as jnp
import localflowwalk as lfw

# Phase-space data
position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.1, 1.2, 1.3])}

# Direct call works
result = lfw.walk_local_flow(position, velocity, start_idx=0, lam=1.0)
```

The dict-based API is JAX PyTree compatible, so it works seamlessly with JAX transformations.

## JIT Compilation

Wrap the function to enable JIT compilation for faster repeated calls:

```python
import jax
import jax.numpy as jnp
import localflowwalk as lfw


# Wrap for JIT
@jax.jit
def order_stream(position, velocity):
    return lfw.walk_local_flow(position, velocity, start_idx=0, lam=1.0)


# Data
position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.1, 1.2, 1.3])}

# First call: compiles; subsequent calls use cached version
result = order_stream(position, velocity)
```

**Note**: JIT is most beneficial when calling the same function repeatedly with similar shapes.

## Vectorization (vmap)

Process multiple streams in parallel:

```python
import jax
import jax.numpy as jnp
import localflowwalk as lfw

# Multiple streams
streams_pos = [
    {"x": jnp.array([0.0, 1.0, 2.0])},
    {"x": jnp.array([3.0, 4.0, 5.0])},
    {"x": jnp.array([6.0, 7.0, 8.0])},
]
streams_vel = [
    {"x": jnp.array([1.0, 1.1, 1.2])},
    {"x": jnp.array([1.3, 1.4, 1.5])},
    {"x": jnp.array([1.6, 1.7, 1.8])},
]

# Stack arrays
stacked_pos = {"x": jnp.stack([s["x"] for s in streams_pos])}
stacked_vel = {"x": jnp.stack([s["x"] for s in streams_vel])}

# Apply vmap over batch dimension
batched_fn = jax.vmap(
    lambda q, p: lfw.walk_local_flow(q, p, start_idx=0, lam=1.0),
    in_axes=(0, 0),
)
results = batched_fn(stacked_pos, stacked_vel)
print(f"Processed {results.ordered_indices.shape[0]} streams in parallel")
```

For a list of independent streams, use `jax.tree.map`:

```python
streams = [
    {"q": {"x": jnp.array([0.0, 1.0])}, "p": {"x": jnp.array([1.0, 1.1])}},
    {"q": {"x": jnp.array([3.0, 4.0])}, "p": {"x": jnp.array([1.3, 1.4])}},
]

results = jax.tree.map(
    lambda sd: lfw.walk_local_flow(sd["q"], sd["p"], start_idx=0, lam=1.0),
    streams,
    is_leaf=lambda x: isinstance(x, dict),
)
```

## Differentiation

Compute gradients with respect to parameters:

```python
import jax
import jax.numpy as jnp
import localflowwalk as lfw

position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.1, 1.2, 1.3])}


# Define a scalar loss
def loss_fn(lam):
    result = lfw.walk_local_flow(position, velocity, start_idx=0, lam=lam)
    return jnp.sum(result.ordered_indices.astype(jnp.float32))


# Compute gradient
grads = jax.grad(loss_fn)(jnp.array(1.5))

# Or get both value and gradient
value, grads = jax.value_and_grad(loss_fn)(jnp.array(1.5))
print(f"Loss: {value}, Gradient: {grads}")
```

## Performance Tips

**Use JAX arrays**: Convert NumPy arrays to JAX before calling:

```python
import numpy as np

# NumPy data
pos_numpy = {"x": np.array([0.0, 1.0, 2.0, 3.0])}
vel_numpy = {"x": np.array([1.0, 1.1, 1.2, 1.3])}

# Convert to JAX
pos_jax = jax.tree.map(jnp.asarray, pos_numpy)
vel_jax = jax.tree.map(jnp.asarray, vel_numpy)
result = lfw.walk_local_flow(pos_jax, vel_jax, start_idx=0, lam=1.0)
```

**Combine JIT and vmap**: For batched operations that run repeatedly, wrap both:

```python
@jax.jit
def batch_order(stacked_pos, stacked_vel):
    return jax.vmap(
        lambda q, p: lfw.walk_local_flow(q, p, start_idx=0, lam=1.0),
        in_axes=(0, 0),
    )(stacked_pos, stacked_vel)
```

## Hardware Acceleration

The library works on GPU/TPU with no code changes:

```python
import jax
import jax.numpy as jnp

position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}

# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")

# Computation automatically runs on GPU/TPU if available
result = lfw.walk_local_flow(position, velocity, start_idx=0, lam=1.0)
```

## Debugging Tips

**Disable JIT**: For easier debugging, disable JIT compilation:

```python
import jax

with jax.disable_jit():
    result = lfw.walk_local_flow(position, velocity, start_idx=0, lam=1.0)
```

**Check shapes**: Verify array shapes in dicts:

```python
import jax

print(jax.tree.map(lambda x: x.shape, position))
```

## See Also

- [Algorithm Details](algorithm.md) - Implementation specifics
- [Examples](examples.md) - More practical examples
- [JAX Documentation](https://jax.readthedocs.io/) - Official JAX guide
- [API Reference](../api/index.md) - Complete API documentation
