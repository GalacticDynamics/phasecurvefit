# Compile, batch and differentiate with JAX

This guide shows how to use the orderers under `jit`, `vmap` and `grad`.

## Which orderers trace

- {class}`~phasecurvefit.orderers.LocalFlowOrderer` and
  {class}`~phasecurvefit.orderers.MSTOrderer` work under `jit`, `vmap` and `grad`.
- {class}`~phasecurvefit.orderers.SOMOrderer` works under them on its own, but not
  chained after another stage ({doc}`som`).
- The default pipeline, `pcf.order(pos, vel)` with no orderer, does not: whether its
  SOM stage runs depends on how many samples the MST visited. It raises a
  `TypeError` saying so. Pass an explicit orderer instead.
- Calling any result to interpolate along it, `result(gamma)`, traces whichever
  orderer produced it ({doc}`interpolate`).

The examples below use the local-flow walk.

## Compile with `jit`

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf


@jax.jit
def order_stream(position, velocity):
    return pcf.order(
        position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
    )


position = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
velocity = {"x": jnp.array([1.0, 1.1, 1.2, 1.3])}

# The first call compiles; later calls with the same shapes reuse it
result = order_stream(position, velocity)
```

`start_idx` is a static field of the orderer, so to pass it as an argument, mark it
static. Each new value then compiles again:

```python
from functools import partial


@partial(jax.jit, static_argnames="start_idx")
def order_from(position, velocity, start_idx):
    return pcf.order(
        position,
        velocity,
        pcf.orderers.LocalFlowOrderer(start_idx=start_idx, metric_scale=1.0),
    )


result = order_from(position, velocity, start_idx=0)
```

## Order many streams with `vmap`

Stack streams of the same length along a leading axis and map over it. `jaxmore.vmap`
takes the orderer as a static keyword:

```python
from jaxmore import vmap

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

stacked_pos = {"x": jnp.stack([s["x"] for s in streams_pos])}
stacked_vel = {"x": jnp.stack([s["x"] for s in streams_vel])}

batched_fn = vmap(
    pcf.order,
    in_axes=(0, 0),
    static_kw={"orderer": pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)},
)
results = batched_fn(stacked_pos, stacked_vel)
print(f"Ordered {results.indices.shape[0]} streams")
```

Wrap the batched call in `jit` when you run it repeatedly:

```python
@jax.jit
def batch_order(stacked_pos, stacked_vel):
    return batched_fn(stacked_pos, stacked_vel)


results = batch_order(stacked_pos, stacked_vel)
```

For streams of different lengths, map over a list with `jax.tree.map` instead:

```python
streams = [
    {"q": {"x": jnp.array([0.0, 1.0])}, "p": {"x": jnp.array([1.0, 1.1])}},
    {"q": {"x": jnp.array([3.0, 4.0, 5.0])}, "p": {"x": jnp.array([1.3, 1.4, 1.5])}},
]

results = jax.tree.map(
    lambda sd: pcf.order(
        sd["q"], sd["p"], pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
    ),
    streams,
    is_leaf=lambda x: isinstance(x, dict) and "q" in x,
)
```

## Differentiate with `grad`

The ordering itself is a list of integer indices, so its gradient with respect to
anything is zero. What you can differentiate is what is computed from the ordering
with real numbers, such as positions interpolated along it or the autoencoder's
outputs. Here the loss depends on the interpolated track, and the gradient reaches
the input positions with the ordering held fixed:

```python
orderer = pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)


def loss_fn(pos):
    result = pcf.order(pos, velocity, orderer)
    track = result(jnp.linspace(0.0, 1.0, 5))
    return jnp.sum(track["x"] ** 2)


value, grads = jax.value_and_grad(loss_fn)(position)
print(f"Loss: {value}, d(loss)/dx: {grads['x']}")
```

## Choose the MST's neighbour search

The MST's nearest-neighbour search runs through its `neighbors` backend. By default
it uses SciPy's `cKDTree` on concrete inputs, which is fastest on CPU, and the
JAX-native {class}`~phasecurvefit.neighbors.BucketKDTree` on traced ones. SciPy
works only eagerly: it raises `TypeError` inside `jit`, `vmap` or `grad`, even on
arrays a jitted function captures. {mod}`phasecurvefit.neighbors` shows how to pin
one.

Tuning the kd-tree matters only for traced calls:

- For large `k` (32 or more), set `frontier` near `k`:
  `neighbors=pcf.neighbors.BucketKDTree(frontier=32)` measured 3–15% faster at
  `k=32`.
- In six or more dimensions the kd-tree is roughly 30–50× slower than SciPy.
  `BucketKDTree(leaf_size=32)` recovers about 1.45×; run eagerly with SciPy if you
  can.

## Debug

Turn compilation off to step through the code with ordinary Python tools:

```python
with jax.disable_jit():
    result = pcf.order(
        position, velocity, pcf.orderers.LocalFlowOrderer(start_idx=0, metric_scale=1.0)
    )
```

To run the same code on a GPU, see {doc}`gpu`.
