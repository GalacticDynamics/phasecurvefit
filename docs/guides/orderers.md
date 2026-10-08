---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Orderers

An **orderer** turns phase-space tracers `(positions, velocities)` into an
ordered result that the autoencoder consumes unchanged. All orderers share one
interface — {class}`~phasecurvefit.orderers.AbstractOrderer` — and return a
unified {class}`~phasecurvefit.orderers.OrderingResult`, so they are
interchangeable at call sites. With no orderer, `pcf.order(pos, vel)` runs the
[default pipeline](#the-default-pipeline-mst-then-som): an MST backbone refined by
a SOM.

<!-- skip: next -->
```python
import phasecurvefit as pcf

orderer = pcf.orderers.MSTOrderer(k=10, jump_cap=3.0)
result = orderer.order(qs, ps)  # or: pcf.order(qs, ps, orderer)
model = pcf.nn.PathAutoencoder.make(normalizer, gamma_range=result.gamma_range, key=key)
ae, *_ = pcf.nn.train_autoencoder(model, result, config=cfg, key=key)
```

## Choosing an orderer

You usually do not need to: the default pipeline, `MSTOrderer | SOMOrderer`, needs
no start point and suits most curves. Reach for another orderer when its mechanism
fits your data better:

| Orderer | Best for | Mechanism |
|---|---|---|
| {class}`~phasecurvefit.orderers.LocalFlowOrderer` | open streams; multi-petal / self-intersecting curves where a coherent velocity field can be *followed* | velocity-following greedy walk from a start point |
| {class}`~phasecurvefit.orderers.MSTOrderer` | **near-closed loops** and streams with no known starting point, e.g. where the velocity field *reverses* at an unknown progenitor | kNN graph → minimum spanning tree → longest-path (diameter) backbone → arc-length ordering |
| {class}`~phasecurvefit.orderers.SOMOrderer` | **refining** any initial ordering; producing a continuous chord parameter for the fit | 1-D self-organizing map → smooth backbone → arc-length projection |

The two are complementary. The walk needs a start point and follows the flow;
where the velocity reverses at a progenitor it must start *at* the progenitor and
walk both ways (`direction="both"`). The MST needs no progenitor — the graph
diameter finds the two tips itself — and orders tip-to-tip with bounded per-step
jumps, which is exactly what a near-closed loop needs.

## LocalFlowOrderer

The {class}`~phasecurvefit.orderers.LocalFlowOrderer` is the velocity-following
greedy walk — the original `phasecurvefit` ordering algorithm, now behind the
orderer interface. From `start_idx` it repeatedly steps to the nearest unvisited tracer
under a pluggable phase-space **metric**, tracing the coherent flow of the
velocity field. Unlike the MST it is **fully JAX-traceable** (jit / vmap / grad).

```{code-cell} python
import jax.numpy as jnp

import phasecurvefit as pcf

pos = {"x": jnp.linspace(0.0, 5.0, 20), "y": jnp.zeros(20)}
vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}

walk = pcf.orderers.LocalFlowOrderer(metric_scale=1.0, start_idx=0)
res = walk.order(pos, vel)
assert int(res.n_visited) == 20
```

The hyperparameters (carried by the orderer object) are:

- **`metric_scale`** — the metric's scale parameter (e.g. the momentum weight for
  the default {class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric`).
- **`config`** — a {class}`~phasecurvefit.WalkConfig` composing the distance
  **metric** with the neighbor-query **strategy** (brute force, or
  {class}`~phasecurvefit.strats.KDTree` to restrict candidates to spatial
  neighbors). See the
  [Metrics guide](metrics.md).
- **`start_idx`** — index of the starting tracer.
- **`direction`** — `"forward"` follows the velocity field, `"backward"` traces
  against it, and `"both"` walks each way from `start_idx` and stitches the two
  arms into one tip-to-tip ordering.
- **`max_dist`** — gap detection: stop when the nearest unvisited tracer is
  farther than this (the rest are left unvisited for the autoencoder to fill).
- **`terminate_indices`**, **`n_max`** — optional stopping conditions.

Because the walk *follows* a coherent flow, it is the right choice for open
streams and for self-intersecting curves where the velocity stays coherent
through the crossings. Its one requirement is a start point: on a near-closed
loop whose velocity **reverses** at a progenitor, the walk has to start at the
progenitor and use `direction="both"`. When that point is unknown, the
[MSTOrderer](#mstorderer) orders the loop without one. For the walk's
mathematics, the metric internals, and the `direction="both"` / `combine_results`
machinery, see the [Algorithm guide](algorithm.md).

## MSTOrderer

The MST's graph algorithms are **host-side** (NumPy/SciPy); `order()` runs them
through `jax.pure_callback` when traced, so it works under `jit` and `vmap`.
Pure-spatial is the default:

```{code-cell} python
import jax.numpy as jnp

import phasecurvefit as pcf

pos = {"x": jnp.linspace(0.0, 5.0, 20), "y": jnp.zeros(20)}
vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
res = pcf.orderers.MSTOrderer(k=5, jump_cap=1.0).order(pos, vel)
assert res.gamma_range == (-1.0, 1.0)
assert int(res.n_visited) == 20
```

`jump_cap` severs edges longer than its value before building the MST; it should
exceed the typical inter-tracer spacing but stay below the loop-opening /
arm-separation scale. If the kNN graph is disconnected (e.g. `jump_cap` too
small, or a gap in the stream wider than the `k` neighbours reach), `on_disconnected`
controls the response: `"raise"` (default), `"warn"` (order the largest component,
leave the rest unvisited), `"largest"` (same, silently), or `"connect"` (join the
pieces along their shortest links and order everything; the bridge links ignore
`jump_cap` and velocity severing, which is what split the graph).

### Velocity is opt-in

Three mechanisms bring velocity into the MST (all off by default), each reusing
the phase-space notion of velocity alignment `cos(v_i, v_j)`:

- **`velocity_weight`** — edge weights become
  `||dq|| + velocity_weight * (1 - cos(v_i, v_j))`, so spatially-close arms that
  move oppositely are not bridged. Set it on the scale of the inter-tracer
  spacing.
- **`sever_cos_threshold`** — drop edges with `cos(v_i, v_j)` below the
  threshold, cutting the reversal seam of a near-closed loop (or cross-branch
  edges at a self-intersection).
- **`orient_by_velocity`** — flip the ordering so `gamma` increases along the
  mean velocity, giving a deterministic, physically-meaningful direction.

For heavily self-intersecting curves (many crossings), velocity-awareness is
*necessary* to stop the spatial MST from short-circuiting across branches — but
such multi-petal curves are usually better served by the momentum
{class}`~phasecurvefit.orderers.LocalFlowOrderer`. The MST's sweet spot is the
near-closed single loop.

## Physical units (unxt)

Orderers accept `unxt.Quantity` inputs and return Quantities, given a unit
system:

<!-- skip: next -->
```python
result = orderer.order(qs, ps, metadata=pcf.StateMetadata(usys=usys))
```

Because the MST is host-side, unit handling is a simple strip-in / reattach-out:
`positions`/`velocities` keep their input units and `backbone` is returned in the
position units. `velocity_weight` and `jump_cap` are interpreted in the `usys`
length units.

## Result: `OrderingResult`

Every built-in orderer returns this one type. Its `__call__` interpolates
positions from the ordering parameter `gamma`: along the `backbone` polyline when
one is present -- {class}`~phasecurvefit.orderers.MSTOrderer` and
{class}`~phasecurvefit.orderers.SOMOrderer` both fit one -- and otherwise along
the ordered visited observations, which is what the walk does. The historical
`WalkLocalFlowResult` is a thin subclass of `OrderingResult`.

## Chaining orderers

Orderers compose with `|`. Each stage receives the previous stage's result as
`init`; a stage that can use a prior ordering does, and one that cannot ignores
the value.

Accepting the parameter is what makes a stage chainable beyond the first
position. An orderer written before `init` existed can still *lead* a chain,
because the head is called without it, but raises `TypeError` if placed after
another stage — loudly, rather than silently dropping the ordering it was
handed.

```{code-cell} python
import jax.numpy as jnp

import phasecurvefit as pcf

ang = jnp.linspace(0.0, jnp.pi, 60)
pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}

chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.LocalFlowOrderer()
result = pcf.order(pos, vel, chain)
assert int(result.n_visited) == 60
```

`|` builds a {class}`~phasecurvefit.orderers.ChainOrderer` and flattens, so
`a | b | c` is one three-stage chain rather than a nest. The explicit form is
equivalent:

```{code-cell} python
chain = pcf.orderers.ChainOrderer(
    pcf.orderers.MSTOrderer(k=8, jump_cap=3.0),
    pcf.orderers.LocalFlowOrderer(),
)
assert len(chain.stages) == 2
```

### The default pipeline: MST, then SOM

`pcf.order(q, p)` with no orderer runs
{func}`~phasecurvefit.orderers.default_pipeline`, which is
`MSTOrderer(...) | SOMOrderer(...)`: the MST finds the curve's two ends and a
tip-to-tip ordering, and the SOM averages over many tracers to smooth it. Its
chain differs from one written by hand in three ways:

- it falls back to the MST alone when there are fewer visited tracers than
  `n_prototypes` (15 by default), rather than raising;
- its MST stage is configured to work at any data scale and to lose nothing --
  `jump_cap=inf`, `orient_by_velocity=True`, `on_disconnected="connect"` -- where
  `MSTOrderer()`'s own `jump_cap=3.0` is an absolute length that severs every edge
  of a sparse or large-scale dataset, and its `"raise"` policy would stop on the
  first gap in a stream (the neighbour graph disconnects there; `"connect"` bridges
  the pieces rather than ordering only the larger one);
- it cannot run under `jit` or `vmap`: whether the SOM stage runs depends on the
  number of tracers the MST visited.

```{code-cell} python
result = pcf.orderers.default_pipeline(pos, vel, n_prototypes=12)
assert int(result.n_visited) == 60
assert result.gamma_range == (-1.0, 1.0)
```

A gap in the stream wider than the neighbours reach does not cost you a side of it:

```{code-cell} python
t = jnp.concatenate([jnp.linspace(0.0, 1.0, 60), jnp.linspace(1.2, 2.2, 60)])
gappy = {"x": t, "y": jnp.zeros(120)}
flow = {"x": jnp.ones(120), "y": jnp.zeros(120)}
assert int(pcf.order(gappy, flow).n_visited) == 120
```

Any {class}`~phasecurvefit.orderers.SOMOrderer` keyword (`sigma_end`,
`metric`, ...) passes through as a keyword to `default_pipeline` itself. To tune the
MST stage instead -- a finite `jump_cap`, velocity-aware edges, outlier rejection
with `edge_clip_sigma` -- build the chain and pass it to `pcf.order`:

```{code-cell} python
chain = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0, edge_clip_sigma=3.0) | (
    pcf.orderers.SOMOrderer(n_prototypes=12)
)
result = pcf.order(pos, vel, chain)
```

The default used to be the walk alone (`LocalFlowOrderer()`); the change, and how
to keep the walk, is covered in the
[v0.3 → v0.4 migration guide](../migration/v0.3-to-v0.4.md).

### Letting the MST find the walk's start point

The walk has to begin at an end of the curve. Naming that index by hand means
knowing the answer before you have ordered anything, and `start_idx=0` is only
right when the input happens to arrive already ordered.

The MST has no such problem: it orders along the graph diameter, tip to tip, so
its first observation *is* an endpoint. Chained, the walk takes its start from
there. On a shuffled 400-point arc, the rank correlation `|rho|` between the
walk's order and the true order goes from about 0.6 walking from index 0 to
1.00 walking from the MST's tip:

```{code-cell} python
import jax
import matplotlib.pyplot as plt
import numpy as np

ang = jax.random.permutation(jax.random.key(0), jnp.linspace(0.0, jnp.pi, 400))
arc = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
arc_vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}

walk = pcf.orderers.LocalFlowOrderer(metric_scale=1.0)
runs = {
    "walk from index 0": walk,
    "MST | walk": pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | walk,
}

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (name, orderer) in zip(axs, runs.items()):
    res = pcf.order(arc, arc_vel, orderer)
    rho = np.corrcoef(np.arange(400), np.asarray(ang)[res.ordering])[0, 1]
    q, _ = pcf.order_w(res)
    ax.plot(arc["x"], arc["y"], ".", c="0.75", ms=3)
    ax.plot(q["x"], q["y"], lw=1)
    ax.plot(q["x"][0], q["y"][0], "*", ms=12, label="start")
    ax.set(title=f"{name}: |rho| = {abs(rho):.2f}", aspect="equal")
axs[0].legend()

assert abs(rho) > 0.999  # the chained walk recovers the order
```

From index 0 (a random point, after shuffling) the walk runs to one end, then
jumps back to cover the rest; from the MST's tip it runs end to end.

An explicit `start_idx` always wins, so this changes nothing for callers who
already pass one; `start_idx=None` is the default and means "ask `init`, else
start at 0".

### What a stage inherits

Chaining does **not** narrow the data. Every stage is handed the full
`(positions, velocities)`; `init` is additional context, and what a stage does
with it is that stage's own business.
{class}`~phasecurvefit.orderers.LocalFlowOrderer` reads one thing from it — the
index to start walking from — and otherwise orders every observation it is
given.

So a stage does not inherit an upstream stage's rejections unless it is written
to. An outlier that {class}`~phasecurvefit.orderers.MSTOrderer`'s
`edge_clip_sigma` dropped is visited again by the next stage unless that stage
restricts itself to `init.indices`. Check `n_visited` on the final result if
rejection is meant to stick.

{class}`~phasecurvefit.orderers.SOMOrderer` is a stage that does restrict
itself: it works on exactly the set the previous stage visited, so rejections
do stick through it.

## SOMOrderer

See the dedicated {doc}`som` guide.
