# Migration Guide

`phasecurvefit` orders phase-space tracers through a single, pluggable entry
point: `pcf.order(positions, velocities[, orderer])`. The velocity-following
local-flow walk is one orderer among several, alongside the MST-backbone orderer.

```{warning}
`walk_local_flow` is **deprecated** and will be **removed in v0.4**. It still
works — unchanged signature and `WalkLocalFlowResult` return — but calling it now
emits a `DeprecationWarning`. Use `pcf.order(positions, velocities)` instead: it
runs the same local-flow walk by default, without the warning. `combine_results`,
`order_w`, the metrics, the query strategies, and `pcf.nn` are unchanged.
```

## `order()` is the entry point

`pcf.order(positions, velocities)` runs the velocity-following walk by default
(via `LocalFlowOrderer`), exactly reproducing `walk_local_flow(positions,
velocities)`. Pass an explicit orderer to select a different algorithm.

The deprecated free function:

<!-- skip: next -->

```python
# Before (deprecated; warns, removed in v0.4):
result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)
```

The replacement:

```python
import jax.numpy as jnp
import phasecurvefit as pcf

pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
vel = {"x": jnp.ones(5)}

# The default orderer is the local-flow walk:
result = pcf.order(pos, vel)
assert result.ordering.shape == (5,)

# For non-default walk parameters, configure a LocalFlowOrderer:
tuned = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(metric_scale=0.5))
```

`start_idx`, `metric_scale`, `max_dist`, `direction`, `config`, and the other
walk knobs move from `walk_local_flow(...)` keyword arguments onto
`LocalFlowOrderer(...)`; a `usys` argument for Quantity inputs moves to
`order(..., metadata=pcf.StateMetadata(usys=...))`.

## New: the MST orderer

{class}`~phasecurvefit.orderers.MSTOrderer` orders tracers along the longest path
(graph diameter) of the minimum spanning tree of a kNN graph. It needs no
progenitor or start index, which makes it the tool of choice for near-closed
loops where the velocity field reverses and a single walk covers only one arm.

```python
import jax.numpy as jnp
import phasecurvefit as pcf

t = jnp.linspace(0.0, 1.0, 60)
pos = {"x": 10.0 * t, "y": jnp.sin(3.0 * t)}
vel = {"x": jnp.ones(60), "y": 3.0 * jnp.cos(3.0 * t)}

mst = pcf.order(pos, vel, pcf.orderers.MSTOrderer(k=8, jump_cap=2.0))
assert mst.gamma_range == (-1.0, 1.0)
```

`MSTOrderer` runs host-side (NumPy/SciPy) and requires the optional `mst` extra —
see below. Its opt-in velocity mechanisms are covered in the
[Orderers Guide](guides/orderers).

## Unified result type

`WalkLocalFlowResult` is a thin subclass of the unified
{class}`~phasecurvefit.orderers.OrderingResult`. Its public name, fields
(`positions`, `velocities`, `indices`, `gamma_range`), properties (`ordering`,
`ordered`, `n_visited`, …), and `__call__` interpolation are all unchanged:

```python
import jax.numpy as jnp
import phasecurvefit as pcf

pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
vel = {"x": jnp.ones(5)}
result = pcf.order(pos, vel)

# A walk result *is* an OrderingResult now
assert isinstance(result, pcf.orderers.OrderingResult)
# ...with all the same accessors
assert result.ordering.shape == (5,)
midpoint = result(0.5)
```

Two consequences worth knowing:

- **`train_autoencoder` accepts any `OrderingResult`** (it previously dispatched
  on `WalkLocalFlowResult`). Walk results still work unchanged, and MST results
  feed the autoencoder without any adapter.
- If you want to accept _any_ orderer's output in your own code, annotate against
  `OrderingResult` rather than `WalkLocalFlowResult`.

## Optional dependency: the `mst` extra

`MSTOrderer` depends on `scipy`, which is an **optional** dependency (it is not
required for the local-flow walk or the autoencoder):

```bash
pip install phasecurvefit[mst]      # or: uv add phasecurvefit --extra mst
pip install phasecurvefit[all]      # includes mst, interop, and kdtree
```

`import phasecurvefit` works without `scipy`; only `MSTOrderer.order()` requires
it, and raises a clear error telling you to install the extra if it is missing.

## Removed

- `phasecurvefit._src.abstract_walk_result` (and its `AbstractWalkResult`) — an
  unused internal duplicate of `AbstractResult`. It was never part of the public
  API. Target {class}`~phasecurvefit.orderers.OrderingResult` (or
  `phasecurvefit._src.abstract_result.AbstractResult`) instead.

## Compatibility summary

| Symbol                                                  | Status                                                 |
| ------------------------------------------------------- | ------------------------------------------------------ |
| `pcf.order`, `pcf.orderers`                             | Primary ordering API                                   |
| `pcf.walk_local_flow`                                   | **Deprecated**; warns, removed in **v0.4**. Use `order` |
| `pcf.combine_results`, `pcf.order_w`                    | Unchanged                                              |
| `pcf.WalkLocalFlowResult`                               | Unchanged public API; subclasses `OrderingResult`      |
| `pcf.WalkConfig`, `pcf.metrics`, `pcf.strats`, `pcf.nn` | Unchanged                                              |
| `pcf.nn.train_autoencoder`                              | Accepts any `OrderingResult` (superset)                |
| `scipy`                                                 | Optional `mst` extra                                   |
| `phasecurvefit._src.abstract_walk_result`               | Removed (internal, unused)                             |
