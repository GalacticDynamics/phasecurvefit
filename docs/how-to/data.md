---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Prepare your data

This guide shows how to pass positions and velocities to phasecurvefit.

## Use two dictionaries with the same keys

Positions and velocities are two plain Python dictionaries. Each maps a coordinate
name to a one-dimensional array with one entry per sample, and both dictionaries
use the same keys. Any number of coordinates works.

```{code-cell} python
import jax.numpy as jnp
import phasecurvefit as pcf

position = {
    "x": jnp.array([0.0, 1.0, 2.0]),
    "y": jnp.array([0.0, 0.5, 1.0]),
    "z": jnp.array([0.0, 0.1, 0.2]),
}
velocity = {
    "x": jnp.array([1.0, 1.0, 1.0]),
    "y": jnp.array([0.5, 0.5, 0.5]),
    "z": jnp.array([0.0, 0.0, 0.0]),
}

result = pcf.order(position, velocity)
```

The keys must match: `pcf.order` raises a `ValueError` naming the keys that are
missing or extra. Dictionaries are JAX PyTrees, so `jax.tree.map` applies a function
to every coordinate at once, and the orderers work under `jit` and `vmap`
({doc}`jax`).

## Convert NumPy or pandas data

Convert NumPy arrays to JAX arrays before ordering:

```{code-cell} python
import jax
import numpy as np

pos_numpy = {"x": np.array([0.0, 1.0, 2.0, 3.0])}
vel_numpy = {"x": np.array([1.0, 1.1, 1.2, 1.3])}

pos_jax = jax.tree.map(jnp.asarray, pos_numpy)
vel_jax = jax.tree.map(jnp.asarray, vel_numpy)
result = pcf.order(pos_jax, vel_jax)
```

For a pandas DataFrame `df`, build each dictionary from its columns, for example
`{k: jnp.asarray(df[k].to_numpy()) for k in ("x", "y", "z")}`.

## Missing velocities

The default pipeline accepts samples whose velocities are missing (NaN): it uses the
velocities only to orient the ordering, and skips the missing ones.

```{code-cell} python
vel_missing = {k: v.at[1].set(jnp.nan) for k, v in velocity.items()}
result = pcf.order(position, vel_missing)
print(int(result.n_visited))  # all 3 samples are ordered
```

## Weight your samples

A sample can stand for more than one star, or for the density at one pixel when your
data is an image. Pass one weight per sample as `weights`. The result keeps them,
and the autoencoder reads them from there:

```{code-cell} python
w = jnp.array([1.0, 2.0, 1.0])
result = pcf.order(position, velocity, weights=w)
print(result.weights)  # [1. 2. 1.]
```

`pcf.fit_track` and every orderer take `weights` too. A weight of 2 counts like
two copies of that sample:

- The self-organizing map fits the weighted density. The MST and the local-flow
  walk order by geometry alone and pass the weights through unchanged.
- The autoencoder weights its ordering targets and its fit to the track, but not
  its membership probability. A low-weight sample counts less in the track and is
  still a member.
- A sample with weight 0 is still ordered, but it doesn't count in the
  self-organizing map or the track fit.

Weights must be finite and non-negative, with at least one positive. They don't
yet work with the mixture-model outlier rejection
(`TrainingConfig(membership=...)`, {doc}`outliers`), which raises
`NotImplementedError`.

## Check the shapes

Every array in both dictionaries must have the same length:

```{code-cell} python
print(jax.tree.map(lambda a: a.shape, position))
```

To attach physical units, see {doc}`units`.
