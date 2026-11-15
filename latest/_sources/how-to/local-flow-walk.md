---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Use the local-flow walk

This guide shows how to set up the local-flow walk: where it starts, which way it
goes, and when it stops. {doc}`../explanation/walk` explains how the walk chooses
each step, and {doc}`tune-metric` covers `metric_scale`.

The examples use 100 shuffled points on a three-dimensional helix:

```{code-cell} python
import jax
import jax.numpy as jnp

import phasecurvefit as pcf

t = jax.random.permutation(jax.random.key(0), jnp.linspace(0, 4 * jnp.pi, 100))
position = {"x": jnp.cos(t), "y": jnp.sin(t), "z": t / (2 * jnp.pi)}
velocity = {
    "x": -jnp.sin(t),
    "y": jnp.cos(t),
    "z": jnp.ones_like(t) / (2 * jnp.pi),
}
```

## Choose the start point

The walk starts at `start_idx` and steps along the flow, so start it at one end of
the curve. Here that is the bottom of the helix:

```{code-cell} python
start = int(jnp.argmin(position["z"]))

result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=start, metric_scale=1.0),
)

ordered_pos, ordered_vel = pcf.order_w(result)
assert jnp.all(jnp.diff(ordered_pos["z"]) > 0)  # the helix climbs once ordered
```

If you know neither end, chain an {class}`~phasecurvefit.orderers.MSTOrderer` in
front of the walk and leave `start_idx` unset; the walk then starts at an end the
MST found ({doc}`chain-orderers`).

## Walk backwards

`direction="backward"` walks against the velocity field, by negating the velocities.
Use it to trace a stream from a tidal tail back towards its progenitor.

```{code-cell} python
top = int(jnp.argmax(position["z"]))
result_reverse = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=top, metric_scale=1.0, direction="backward"),
)
```

## Walk both ways from a central point

For a stream that flows away from a central point, such as a tidal stream from its
progenitor, start at that point and set `direction="both"`. The walk goes each way
and joins the two arms into one ordering, from one end through the start to the
other end:

```{code-cell} python
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=50, metric_scale=1.0, direction="both"),
)
ordered_pos, ordered_vel = pcf.order_w(result)
```

To use different settings in each direction, for example a different `max_dist`,
run the two walks separately and join them with {func}`~phasecurvefit.combine_results`:

```{code-cell} python
result_forward = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=50, metric_scale=1.0, direction="forward"),
)
result_backward = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(
        start_idx=50, metric_scale=1.0, direction="backward", max_dist=1.0
    ),
)

# Indices run from the backward tail, through the start, to the forward tail
result = pcf.combine_results(result_forward, result_backward)
```

## Stop at gaps

`max_dist` stops the walk when its next step would be longer than `max_dist` in
position, so it does not jump across a gap onto an unrelated part of the data. Set
it to several times the typical spacing between neighbouring points. The points the
walk never reaches are reported as skipped, and the autoencoder can give them a
value of $\gamma$ afterwards ({doc}`autoencoder`).

```{code-cell} python
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(start_idx=start, metric_scale=1.0, max_dist=2.0),
)

if result.n_skipped > 0:
    print(f"Skipped {result.n_skipped} points")
```

## Stop after a number of steps, or at given points

`n_max` caps the number of points in the walk, and `terminate_indices`, a
`frozenset` of indices, stops it when it reaches any of them:

```{code-cell} python
result = pcf.order(
    position,
    velocity,
    pcf.orderers.LocalFlowOrderer(
        start_idx=start, metric_scale=1.0, n_max=40, terminate_indices=frozenset({top})
    ),
)
print(int(result.n_visited))
```

{doc}`../reference/options` lists every option with its default.
