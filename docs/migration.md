# Migration Guide

`phasecurvefit` v0.1 introduces a pluggable **orderer** abstraction: the ordering
step (previously only `walk_local_flow`) is now one implementation behind a
common interface, alongside a new MST-backbone orderer.

**This change is additive and backward-compatible.** `walk_local_flow`,
`combine_results`, `order_w`, the metrics, the query strategies, and `pcf.nn` are
all unchanged. Existing code keeps working with no edits; the rest of this page
shows how to _adopt_ the new API.

## The orderer interface

Every orderer implements `order(positions, velocities, *, metadata=None)` and
returns an {class}`~phasecurvefit.orderers.OrderingResult`. `LocalFlowOrderer`
wraps `walk_local_flow`, so the two are exactly equivalent:

```python
import jax.numpy as jnp
import phasecurvefit as pcf

pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
vel = {"x": jnp.ones(5)}

# Before — the function:
result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

# After — the equivalent orderer:
orderer = pcf.orderers.LocalFlowOrderer(metric_scale=1.0, start_idx=0)
same = orderer.order(pos, vel)  # or: pcf.order(pos, vel, orderer)

assert jnp.array_equal(result.indices, same.indices)
```

You do **not** need to switch — `walk_local_flow` remains fully supported. The
orderer interface is useful when you want to swap ordering algorithms behind a
uniform call site.

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

mst = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
assert mst.gamma_range == (-1.0, 1.0)
```

`MSTOrderer` runs host-side (NumPy/SciPy) and requires the optional `mst` extra —
see below. Its opt-in velocity mechanisms are covered in the
[Orderers Guide](guides/orderers).

## Unified result type

`WalkLocalFlowResult` is now a thin subclass of the unified
{class}`~phasecurvefit.orderers.OrderingResult`. Its public name, fields
(`positions`, `velocities`, `indices`, `gamma_range`), properties (`ordering`,
`ordered`, `n_visited`, …), and `__call__` interpolation are all unchanged:

```python
import jax.numpy as jnp
import phasecurvefit as pcf

pos = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
vel = {"x": jnp.ones(5)}
result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

# A walk result *is* an OrderingResult now
assert isinstance(result, pcf.orderers.OrderingResult)
# ...with all the same accessors
assert result.ordering.shape == (5,)
midpoint = result(0.5)
```

Two consequences worth knowing:

- **`train_autoencoder` now accepts any `OrderingResult`** (it previously
  dispatched on `WalkLocalFlowResult`). Walk results still work unchanged, and
  MST results now feed the autoencoder without any adapter.
- If you want to accept _any_ orderer's output in your own code, annotate against
  `OrderingResult` rather than `WalkLocalFlowResult`.

## Optional dependency: the `mst` extra

`MSTOrderer` depends on `scipy`, which is an **optional** dependency (it is not
required for `walk_local_flow` or the autoencoder):

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

## Backward-compatibility summary

| Symbol                                                      | Status in v0.1                                        |
| ----------------------------------------------------------- | ----------------------------------------------------- |
| `pcf.walk_local_flow`, `pcf.combine_results`, `pcf.order_w` | Unchanged                                             |
| `pcf.WalkLocalFlowResult`                                   | Unchanged public API; now subclasses `OrderingResult` |
| `pcf.WalkConfig`, `pcf.metrics`, `pcf.strats`, `pcf.nn`     | Unchanged                                             |
| `pcf.nn.train_autoencoder`                                  | Now accepts any `OrderingResult` (superset)           |
| `pcf.orderers`, `pcf.order`                                 | **New**                                               |
| `scipy`                                                     | Moved to the optional `mst` extra                     |
| `phasecurvefit._src.abstract_walk_result`                   | **Removed** (internal, unused)                        |
