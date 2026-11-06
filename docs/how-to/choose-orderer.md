---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Choose an orderer

This guide shows how to pick the orderer that suits your curve. {doc}`../explanation/mst`,
{doc}`../explanation/walk` and {doc}`../explanation/som` explain how each one works.

## Start with the default

`pcf.order(pos, vel)` with no orderer runs the default pipeline, an
{class}`~phasecurvefit.orderers.MSTOrderer` refined by a
{class}`~phasecurvefit.orderers.SOMOrderer`. It needs no start point and suits most
curves that don't cross themselves. Pass another orderer as the third argument when
its mechanism fits your data better:

| Orderer | Best for | Mechanism |
|---|---|---|
| {class}`~phasecurvefit.orderers.LocalFlowOrderer` | open streams, and curves that cross themselves where the velocity field stays coherent through the crossings | a greedy walk that follows the velocity field from a start point |
| {class}`~phasecurvefit.orderers.MSTOrderer` | near-closed loops, and streams with no known start point, for example where the velocity reverses at an unknown progenitor | k-nearest-neighbour graph, minimum spanning tree, its longest path as the backbone, then arc length along it |
| {class}`~phasecurvefit.orderers.SOMOrderer` | refining any initial ordering, and a continuous along-track coordinate | a one-dimensional self-organizing map, a smooth backbone through its prototypes, then arc length along it |
| {class}`~phasecurvefit.orderers.RidgeOrderer` | density fields, such as an image of a stream, and wide or weighted point clouds | a kernel density estimate of the weighted samples, the path along its ridge as the backbone, then arc length along it |

The walk and the MST complement each other. The walk needs a start point and
follows the flow. Where the velocity reverses at a progenitor it has to start at the
progenitor and walk both ways ({doc}`local-flow-walk`). The MST needs no start point,
because the longest path of the tree finds the two ends itself.

## Order with the local-flow walk

Give the walk a start index and a `metric_scale` ({doc}`tune-metric`):

```{code-cell} python
import jax.numpy as jnp

import phasecurvefit as pcf

pos = {"x": jnp.linspace(0.0, 5.0, 20), "y": jnp.zeros(20)}
vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}

walk = pcf.orderers.LocalFlowOrderer(metric_scale=1.0, start_idx=0)
res = pcf.order(pos, vel, walk)
assert int(res.n_visited) == 20
```

{doc}`local-flow-walk` covers its options: direction, gaps and stopping conditions.

## Order with the MST alone

The MST orders by position by default:

```{code-cell} python
res = pcf.order(pos, vel, pcf.orderers.MSTOrderer(k=5, jump_cap=1.0))
assert res.gamma_range == (-1.0, 1.0)
assert int(res.n_visited) == 20
```

- **`k`** is the number of neighbours each sample links to.
- **`jump_cap`** cuts edges longer than its value before the tree is built. Set it
  above the typical spacing between neighbouring samples and below the distance
  between separate arms or the two ends of a near-closed loop. It is an absolute
  length, and its default of 3.0 cuts every edge of a dataset whose spacing is
  larger than that.
- **`on_disconnected`** chooses what happens when the neighbour graph falls into
  pieces, for example when `jump_cap` is too small or a gap in the stream is wider
  than the `k` neighbours reach: `"raise"` (the default) raises, `"warn"` orders the
  largest piece and warns, `"largest"` does the same silently, and `"connect"` joins
  the pieces along their shortest links and orders everything.

### Use velocities in the MST

Three options bring velocity into the MST. All are off by default.

- **`velocity_weight`** adds `velocity_weight * (1 - cos(v_i, v_j))` to each edge
  length, so arms that are close in position but move in opposite directions are not
  joined. Set it on the scale of the spacing between neighbouring samples.
- **`sever_cos_threshold`** drops edges whose velocity cosine is below the
  threshold. This cuts the seam where a near-closed loop's velocity reverses, and the
  edges between branches at a crossing.
- **`orient_by_velocity`** flips the ordering so that $\gamma$ increases along the
  mean velocity.

```{code-cell} python
mst = pcf.orderers.MSTOrderer(
    k=5, jump_cap=1.0, sever_cos_threshold=0.0, orient_by_velocity=True
)
res = pcf.order(pos, vel, mst)
```

On a curve that crosses itself many times, the MST needs these options to stay on
its own branch, and the local-flow walk is usually the better choice
({doc}`../tutorials/epitrochoid_mst`).

## Refine with the SOM

A {class}`~phasecurvefit.orderers.SOMOrderer` smooths an ordering it is handed.
Chain it after another orderer ({doc}`chain-orderers`); {doc}`som` covers its
options.

## Order a density field

For an image of a stream, make each pixel a sample weighted by its density
({doc}`data`) and order with a {class}`~phasecurvefit.orderers.RidgeOrderer`. It
follows the ridge of the weighted density, so it finds the middle of a wide or
blurred stream where the MST's longest path wanders across it. It works out the
stream's width from the data. Chain a SOM to refine it:

```{code-cell} python
g = jnp.linspace(-1.5, 1.5, 61)
X, Y = jnp.meshgrid(g, g)
image = jnp.exp(-((jnp.hypot(X, Y) - 1.0) ** 2) / (2 * 0.1**2)) * (Y > 0)
keep = image.ravel() > 0.05  # each bright pixel becomes a sample
pix = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
pix_vel = {"x": jnp.zeros_like(pix["x"]), "y": jnp.zeros_like(pix["y"])}

ridge = pcf.orderers.RidgeOrderer() | pcf.orderers.SOMOrderer(
    n_prototypes=20, metric=pcf.metrics.SpatialDistanceMetric()
)
res = pcf.order(pix, pix_vel, ridge, weights=image.ravel()[keep])
```

It doesn't use velocities, so the direction of the ordering is arbitrary. The width
it uses, `bandwidth`, is the RMS distance of the samples from the ridge, found by
iterating from a wide first guess. Pass `bandwidth=` to set it yourself, for
example to smooth over clumps in a noisy image. It finds the ends to within about
one `bandwidth`.
