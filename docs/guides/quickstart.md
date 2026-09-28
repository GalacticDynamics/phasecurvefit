# Quickstart Guide

Get started with phasecurvefit in 5 minutes!

phasecurvefit takes points in phase space (positions *and* velocities) whose
order along a curve is unknown, and works out that order. This guide covers the
first step, **ordering** the points with the local-flow walk, and the settings
that control it. The [Autoencoder guide](nn.md) covers the next step: giving
every point an ordering coordinate and fitting a smooth track. For worked
examples on realistic data, see the [tutorials](../tutorials/index.md).

## Installation

Install phasecurvefit using pip or uv:

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

::::

```{note}
The plain pip/uv install above ships a **CPU-only** `jaxlib`. If you have an
NVIDIA GPU and see phasecurvefit or JAX fall back to CPU, see
[Installing for GPU](jax-integration.md#installing-for-gpu-nvidia-cuda) in
the JAX Integration guide.
```

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

The local-flow walk starts at one point and repeatedly steps to the unvisited
point that minimizes a distance metric. With the default
`AlignedMomentumDistanceMetric`, that distance combines closeness with a penalty
for the direction *to* the candidate point being misaligned with the current
point's velocity — so the walk favors candidates that are close and roughly
**ahead of it**, but a sufficiently closer point off to the side can still win.
(Other metrics have no such directional penalty; see the
[Metrics guide](metrics.md).) It needs two choices from you:

- **Where to start** (`start_idx`). Start at one end of the curve. For a stream
  that flows *away* from a central point, such as a tidal stream from its
  progenitor, start at that point and walk both ways (`direction="both"`, below).
  If you don't know either, an `MSTOrderer` can find an end for you; see
  [Letting the MST find the walk's start point](orderers.md#letting-the-mst-find-the-walks-start-point).
- **How strongly to prefer "ahead"** (`metric_scale`), explained in
  [Adjusting the Metric Scale](#adjusting-the-metric-scale).

```python
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(
        start_idx=0,  # Start from first point
        metric_scale=1.0,  # Metric-dependent scale parameter
    ),
)

print(result.ordering)
# Array([0, 1, 2, 3, 4])
```

```{note}
Above we ran the **local-flow walk** through `pcf.order` with a
{class}`~phasecurvefit.orderers.LocalFlowOrderer` — one of several pluggable
orderers. Swap in the {class}`~phasecurvefit.orderers.MSTOrderer` for near-closed
loops where the velocity reverses. See the [Orderers guide](orderers.md).
```

### 4. Extract ordered data

Use the convenience function to get reordered arrays:

```python
ordered_pos, ordered_vel = pcf.order_w(result)

print(ordered_pos["x"])
# Array([0., 1., 2., 3., 4.])
```

## Understanding the Result

`pcf.order` returns an `OrderingResult` (a `WalkLocalFlowResult` for the walk) with:

- **`ordering`**: the indices of the visited observations, in walk order
- **`indices`**: the same, padded with `-1` to the full length (a fixed shape, for JAX)
- **`n_visited`** / **`n_skipped`**: how many observations the walk reached or left out
- **`positions`**, **`velocities`**: the input data
- **`gamma_range`**: the range of the ordering coordinate $\gamma$

Calling the result, `result(gamma)`, interpolates positions along the ordering.


## Adjusting the Metric Scale

The `metric_scale` parameter controls how the algorithm weighs different aspects
of the data. Its meaning depends on the distance metric. With the default
`AlignedMomentumDistanceMetric` it is a **length**, $\lambda$: a candidate point
at angle $\theta$ from the current velocity costs an extra
$\lambda\,(1 - \cos\theta)$ on top of its distance. So compare it with the
typical spacing between neighbouring points:

- $\lambda$ much smaller than the spacing: essentially nearest-neighbour; the walk
  goes wherever the closest point is.
- $\lambda$ about the spacing: a neighbour at 90° costs as much as a point twice as
  far away straight ahead.
- $\lambda$ much larger than the spacing: strongly directional; the walk keeps
  going the way it is moving, taking longer strides and skipping points off to
  the side. (The [stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb)
  uses 100 kpc against steps of a few kpc at most.)

Raise `metric_scale` if the walk jumps between neighbouring strands; lower it if
it skips too much. See the [Metrics guide](metrics.md#choosing-metric_scale) for
the other metrics.

```python
# With the default metric, metric_scale=0 switches off the momentum penalty:
# pure nearest neighbor
result_spatial = pcf.order(
    position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=0.0)
)

# Balanced (default)
result_balanced = pcf.order(
    position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
)

# Higher metric_scale value (interpretation metric-dependent)
result_momentum = pcf.order(
    position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=5.0)
)
```

## Walking in Reverse

Use the `direction` parameter to trace streams backwards by negating the velocity vectors:

```python
# Default: forward walk following the velocity direction
result_forward = pcf.order(
    position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
)

# Reverse: walk against the velocity direction
result_reverse = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0, direction="backward"),
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

result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(config=config, start_idx=0, metric_scale=1.0),
)
```

## Handling Gaps with max_dist

Use `max_dist` to stop when there's a gap in the data. It is a plain spatial
distance: the walk stops if the step it would take next is longer than
`max_dist`, so it does not leap across a gap onto an unrelated part of the data.
Set it to several times the typical spacing between neighbouring points. The
points the walk never reaches are reported as skipped; the
[autoencoder](nn.md) can assign them an ordering afterwards.

```python
# Stop if next nearest point is more than 2 units away
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(
        start_idx=0,
        metric_scale=1.0,
        max_dist=2.0,
    ),
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

result = pcf.order(
    position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=2.0)
)
```

## Bidirectional Walks (Forward and Reverse)

For streams that extend in both directions from a starting point, walk both ways
from it with `direction="both"`:

```python
# Walk forward and backward from index 2, stitched into one ordering
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=2, metric_scale=1.0, direction="both"),
)

# Get the combined ordered indices
print(result.indices)  # Indices ordered from reverse tail through start to forward tail

# Extract the ordered positions and velocities
ordered_pos, ordered_vel = pcf.order_w(result)
```

This is particularly useful for:

- Tracing complete stellar streams from a central progenitor
- Exploring both tidal tails simultaneously

To use different parameters in each direction (e.g. different `max_dist`), run the
two walks separately and join them with `pcf.combine_results`; see the
[Algorithm guide](algorithm.md#combining-forward-and-reverse-walks).

## JAX Integration

The algorithm is fully compatible with JAX transformations:

### JIT Compilation

```python
from jax import jit


@jit
def order_stream(pos, vel):
    return pcf.order(
        pos, vel, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
    )


result = order_stream(position, velocity)
```

### Vectorization

To order many streams at once, stack them along a leading axis and `vmap` over
it; see [Vectorization](jax-integration.md#vectorization-vmap) in the JAX
Integration guide.

## Next Steps

- [Orderers](orderers.md#the-recommended-walk-then-som-chain) - `pcf.orderers.default_pipeline` refines the walk with a SOM stage, a better default than the bare walk above without changing it
- [Tutorials](../tutorials/index.md) - Worked examples, starting with a simulated stellar stream
- [Autoencoder](nn.md) - Order every point, including the ones the walk skipped, and fit a smooth track - or run the whole thing in one call with `pcf.pipeline`
- [Algorithm Details](algorithm.md) - Understand the math
- [Orderers](orderers.md) - Choose between the walk and the MST
- [JAX Integration](jax-integration.md) - Advanced JAX usage
