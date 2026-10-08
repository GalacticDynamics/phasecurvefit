---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Self-Organizing Maps

A **Self-Organizing Map** learns a 1-D lattice of *prototype* vectors in phase
space. Because the lattice is one-dimensional, the only information in a
prototype's lattice coordinate is its index — prototypes `i` and `j` are
neighbours exactly when `|i - j| = 1` — so a trained SOM is a string of beads
threaded through the data, and the order of the beads is the order of the
curve.

{class}`~phasecurvefit.orderers.SOMOrderer` trains such a map and orders the
tracers by projecting them onto it.

## Why use it

The SOM is a **smoothing** stage. Its prototypes are averages over many
tracers, so the backbone it produces is far less sensitive to local noise than
a greedy walk's step-by-step decisions or an MST's individual graph edges. It is
most useful as a refinement *after* an initial ordering, and before fitting --
which is exactly what the default pipeline does: `pcf.order(pos, vel)` with no
orderer runs an {class}`~phasecurvefit.orderers.MSTOrderer` chained into a
`SOMOrderer` (see
[The default pipeline](orderers.md#the-default-pipeline-mst-then-som)). Chaining
it explicitly gives you control over both stages:

```{code-cell} python
import jax.numpy as jnp

import phasecurvefit as pcf

ang = jnp.linspace(0.0, jnp.pi, 120)
pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}

chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.SOMOrderer(
    n_prototypes=12
)
result = pcf.order(pos, vel, chain)
assert int(result.n_visited) == 120
```

The smoothing shows on noisy data: the MST's backbone follows individual
tracers, while the SOM's follows their average.

```{code-cell} python
import jax
import matplotlib.pyplot as plt

k1, k2 = jax.random.split(jax.random.key(1))
noisy = {
    "x": pos["x"] + 0.2 * jax.random.normal(k1, (120,)),
    "y": pos["y"] + 0.2 * jax.random.normal(k2, (120,)),
}
mst = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(noisy["x"], noisy["y"], ".", c="0.7", ms=4, label="tracers")
for name, orderer in {"MST": mst, "MST | SOM": chain}.items():
    backbone = pcf.order(noisy, vel, orderer).backbone
    ax.plot(backbone["x"], backbone["y"], lw=1.5, label=f"{name} backbone")
ax.set(aspect="equal")
ax.legend();
```

It also runs standalone, initializing itself by binning along the first
principal axis of the positions:

```{code-cell} python
result = pcf.order(pos, vel, pcf.orderers.SOMOrderer(n_prototypes=12))
assert int(result.n_visited) == 120
```

## Standalone initialization: know your curve

When `SOMOrderer` runs standalone (no prior ordering supplied), its only way to
seed the lattice is to bin observations along the first principal axis of the
positions — the N-D replacement for the paper's "bin in an observational
longitude". This is a valid proxy for curve order only when the curve does not
double back along that axis, which holds up to about one turn. Beyond that the
initial lattice is tangled, and no `sigma` or epoch count repairs it: batch
Kohonen refines the neighbourhood structure it is handed and cannot undo a bad
global topology.

If you are ordering a near-closed loop or a phase-wrapped curve — exactly the
case {class}`~phasecurvefit.orderers.MSTOrderer` exists for — chain after it
instead of running the SOM standalone:

```{code-cell} python
chain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0) | pcf.orderers.SOMOrderer()
```

which supplies a valid ordering for the SOM to refine. When a prior ordering is
supplied the SOM holds its neighbourhood at `sigma_end` rather than starting
wide, so it refines that ordering locally instead of re-deriving a global one
and discarding it.

Chaining also buys **outlier robustness**. The SOM has no rejection of its own,
so field contamination drags the backbone toward the contaminants — and more
prototypes makes that worse, not better, because they chase the interlopers.
Chaining after an orderer that does reject, such as
{class}`~phasecurvefit.orderers.MSTOrderer` with `edge_clip_sigma`, keeps the
backbone on the curve. Measured on an arc with uniform background mixed in,
backbone error against the true track:

| contamination | SOM standalone | `MSTOrderer(edge_clip_sigma=3) \| SOMOrderer()` |
|---|---|---|
| 0% | 0.025 | 0.025 |
| 10% | 0.100 | 0.026 |
| 20% | 0.159 | 0.026 |

Note the member *ordering* stays good throughout: the damage is entirely in
the track.

One caveat: points a prior stage left unvisited stay unvisited, so if that
stage truncates the curve (an `MSTOrderer` whose `jump_cap` cannot bridge a
gap, say) the chain inherits the truncation. Check `n_visited` when chaining.

## The chord parameter

Unlike the other orderers, the SOM produces a *continuous* along-track
coordinate, not merely a permutation. `result.chord` is the arc length of each
observation's projection onto the backbone, in input order:

```{code-cell} python
assert result.chord.shape == (120,)
```

This is the natural affine parameter for the downstream fit — it is what §3.1 of
the reference below uses as the Kalman filter's time step.

## How the projection works

1. The trained prototypes are interpolated by a **centripetal Catmull-Rom**
   spline and resampled to vertices equally spaced in arc length. This
   backbone is what `result.backbone` holds and what
   {meth}`~phasecurvefit.orderers.OrderingResult.__call__` interpolates along.
2. Each datum is assigned to its nearest backbone vertex **under `metric`**.
   Chained after a velocity-aware stage that is a phase-space metric; standalone
   it is pure position distance, and anti-parallel arms *can* then capture each
   other (on a hairpin with arms 0.10 apart, 55 of 600 stars land on the wrong
   arm at `metric_scale=0.0`, none at `0.2`). See [Tuning](#tuning).
3. The assignment is refined to sub-vertex resolution by projecting onto the
   two adjacent segments **in position space**, clamped to each segment — except
   at the two tips, where the clamp is released so that data beyond the ends
   extrapolate rather than piling up.

The split in steps 2 and 3 is deliberate. A metric such as
{class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` is not induced by
an inner product, so "project onto a line segment" is undefined under it: the
metric chooses *which* segment, and Euclidean position geometry chooses *where*
within it. The chord is therefore a genuine physical arc length.

Step 1 is also what lets this generalize beyond two dimensions. On a
piecewise-linear polyline, every point whose nearest polyline point is a vertex
receives the same arc length; the original 2-D formulation broke that tie with
an `arctan2` angle sweep that has no N-D analogue. On a smooth curve those
wedges shrink to nothing, so no angle machinery is needed.

## Tuning

- **`n_prototypes`** — more prototypes track finer structure but begin to follow
  noise. The paper reports results insensitive to the exact value above roughly
  10 per distinct segment; that does not hold for curves that cross themselves.
  A 6-lobed self-intersecting epitrochoid is ordered perfectly at
  `n_prototypes=90` and not at all at the default 25 — |rho| 1.00 against 0.23 —
  and no metric or `sigma` setting rescues the coarse lattice, because the
  structure it must represent is finer than one lattice step.

  The number that actually matters is the **physical smoothing length**

  ```
  sigma_phys = sigma_end * L / (n_prototypes - 1)
  ```

  where `L` is the track length. The backbone is pulled toward the centre of
  curvature by roughly `sigma_phys**2 / (2R)` for a track of radius `R`, so any
  curvature radius or self-approach distance **below `sigma_phys` is smoothed
  away** —
  the two arms of a hairpin merge into one line, silently and plausibly. At the
  defaults `sigma_phys` is about 3% of the whole track. Size `n_prototypes` so
  that `sigma_phys` sits below the curve's tightest turn and smallest
  self-approach.
- **`sigma_start` / `sigma_end`** — the neighbourhood width in lattice units,
  annealed geometrically across training. Large early values fix the global
  ordering; small late values refine local detail. `sigma_start=None` uses
  `max(n_prototypes / 4, sigma_end)` — `n_prototypes / 4` floored at
  `sigma_end` so the anneal is never inverted on a very small lattice.
- **`n_epochs`** — batch-Kohonen epochs. Tens, not thousands: each epoch is a
  single matmul over all the data.
- **`densify_factor`** — backbone samples per prototype segment. Little effect
  above ~2, and it multiplies the dominant cost of `chord` linearly, so raise it
  only for a strongly curved track.
- **`orient_by_velocity`** — the SOM has no progenitor anchor, so a standalone
  ordering's *direction* is arbitrary: it follows the sign of the initializer's
  principal-axis eigenvector, which is stable but meaningless, and in practice
  often runs against the flow. Set this to flip the result so the chord
  increases along the mean velocity, as
  {class}`~phasecurvefit.orderers.MSTOrderer`'s option of the same name does.
  Note it **overrides** an inherited direction rather than deferring to it:
  chained after a stage that already fixed one, setting this can silently
  reverse that choice. Default `False`.
- **`metric` / `metric_scale`** — both default to `None`, meaning *follow the
  previous stage*. Chained after an orderer that used velocity — an
  {class}`~phasecurvefit.orderers.MSTOrderer` with `sever_cos_threshold` or
  `velocity_weight`, or a {class}`~phasecurvefit.orderers.LocalFlowOrderer` with
  a non-zero scale — the SOM resolves to
  {class}`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric` with

  ```
  metric_scale = sigma_phys / (2 |v|)
  ```

  which makes the velocity term separate anti-parallel branches by about the
  distance the lattice can already resolve. Standalone, or after a
  position-only stage, it resolves to
  {class}`~phasecurvefit.metrics.SpatialDistanceMetric`.

  This matters because ordering on position alone after a stage that used
  velocity *undoes that stage's work*: at a crossing the two branches are
  spatially coincident and differ only in velocity. On the epitrochoid above, a
  velocity-aware MST scoring |rho| = 0.94 drops to 0.59 under a position-only
  SOM and rises to 0.99 under the default.

  Set either explicitly to override. A non-zero `metric_scale` on its own
  selects the phase-space metric, since handing it to
  `SpatialDistanceMetric` — which discards it — would silently do nothing;
  passing that metric *and* a scale is rejected at construction. `metric_scale`
  is a *time* converting velocity differences into position units, so its right
  value is unit-system dependent. Too large and "nearest prototype" becomes
  "nearest in velocity", which on a winding curve conflates points a whole turn
  apart.

  The metric must be **symmetric** in the two points it compares, because it
  answers "which prototype is this datum nearest".
  {class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` is not: it
  scores *forward along the direction of travel*, which is what a greedy walk
  step needs, and under it the lattice collapses toward the curve's head.
- **`outlier_clip_sigma` / `outlier_clip_max_iters`** — see
  [Outlier rejection](#outlier-rejection) below. `None` (default) disables it.

## Outlier rejection

`SOMOrderer` has no outlier rejection of its own by default, so field
contamination drags the backbone off the curve. **The failure is silent**: the
*ordering* stays good — rank correlation against truth stays above 0.99 even at
20% contamination, because the ordering only needs the backbone roughly right
along-track — but the *reconstructed track*, the actual deliverable, degrades
up to 4x over that range. A user sees a smooth, plausible backbone that is
simply in the wrong place, with nothing in the ordering quality to reveal it.

Set `outlier_clip_sigma` to enable robust, iterated rejection by quantization
error — each datum's distance to its own best-matching prototype:

```{code-cell} python
pcf.orderers.SOMOrderer(n_prototypes=25, outlier_clip_sigma=3.0)
```

Mirrors {attr}`~phasecurvefit.orderers.MSTOrderer.edge_clip_sigma` /
`edge_clip_max_iters` in mechanism and naming: robust median/MAD clipping in
log space (distances are positive and heavy-tailed), floored so a well-fit
lattice — where the spread collapses to ~0 — is not shredded by microscopic
variation, iterated (refit, reclip) up to `outlier_clip_max_iters` times or
until a round rejects nothing new. Unlike `MSTOrderer`'s edge clipping there is
no graph here and so no component-size veto: a quantization error belongs to
one datum alone, not to an edge that could fragment a graph. A rejected datum
gets `-1` in `indices` and `nan` in `chord`, the same contract as a point a
prior stage never visited — so it composes with, rather than overrides, a
prior stage's own rejections.

Chaining after an orderer with its own rejection ({class}`~phasecurvefit.orderers.MSTOrderer`'s
`edge_clip_sigma`) remains a complete alternative to this option, and the two
compose: `MSTOrderer(edge_clip_sigma=3.0) | SOMOrderer()` rejects at the MST
stage, before the SOM ever sees the contamination.

## When the SOM makes things worse

A refinement stage normally keeps most of the order it is handed. When the
lattice is too coarse for the curve it does the opposite — it tangles a good
ordering — and the result looks plausible. `SOMOrderer` warns when its ordering
disagrees wholesale with the one it was given, naming the smoothing length:

```
SOMOrderer's ordering disagrees with the one it was given (rank correlation
0.28). Either the prior ordering was poor and this is a genuine overhaul, or
the lattice is too coarse for the curve and has tangled a good ordering ...
```

The warning cannot tell those two apart, so it asks you to compare. If the
prior stage was doing well, raise `n_prototypes` until `sigma_phys` sits below
the curve's tightest turn.

## JAX tracing: standalone yes, chained no

Standalone, `SOMOrderer.order()` is traceable — `jit` it, or build the orderer
inside one:

```{code-cell} python
import jax
import jax.numpy as jnp

import phasecurvefit as pcf

t = jnp.linspace(0.0, 2.0, 60)
pos = {"x": jnp.cos(t), "y": jnp.sin(t)}
vel = {"x": -jnp.sin(t), "y": jnp.cos(t)}

chord = jax.jit(
    lambda q, p: pcf.order(q, p, pcf.orderers.SOMOrderer(n_prototypes=8)).chord
)(pos, vel)
assert chord.shape == (60,)
```

**Chained after another stage it is not.** The working set is whatever the
prior stage visited, so its size depends on that stage's *values*; under a
transform `init.indices` is a tracer and no shape fixed at trace time can hold
the result. Chaining under `jit` raises a `TypeError` saying so, rather than
failing deep inside JAX. Run the chain outside `jit`, or order in two steps and
hand the second stage a concrete `init`.

This is a limit of selecting the working set, not of the algorithm: the SOM
core ({func}`~phasecurvefit.som.fit`, {func}`~phasecurvefit.som.densify`,
{func}`~phasecurvefit.som.chord`) is traceable either way, as the ensemble
section below relies on.

## Ensembles

{mod}`phasecurvefit.som` is `vmap`-able, but that alone does not buy you an
ensemble. Batch Kohonen is strongly contractive: each epoch replaces every
prototype outright with a neighbourhood-weighted mean of the data, so a
prototype survives only through *which datum it wins*. Once two members agree
on their assignments — which happens within a few epochs on a smooth, well
sampled curve — every later epoch is byte-identical and they stay merged.

So perturbing a shared initialization does not work. Jittered members converge
to *exactly* zero separation, which is pinned as a test rather than left as a
warning.

The diversity has to come from the data, which is what `weights` is for:

```{code-cell} python
import jax
import jax.numpy as jnp

import phasecurvefit as pcf
from phasecurvefit import som

t = jnp.linspace(0.0, 2.0, 300)
pos = {"x": jnp.cos(t) * 3, "y": jnp.sin(t) * 3}
vel = {"x": -jnp.sin(t), "y": jnp.cos(t)}
metric = pcf.metrics.SpatialDistanceMetric()
proto_q, proto_p = som.init_prototypes(pos, vel, n_prototypes=20)

# One bootstrap resample per member, as counts.
keys = jax.random.split(jax.random.key(0), 8)
weights = jax.vmap(lambda k: som.bootstrap_weights(k, 300))(keys)


def member(w):
    res = som.fit(proto_q, proto_p, pos, vel, metric=metric, weights=w)
    bq, bp = som.densify(res.prototype_positions, res.prototype_velocities, factor=5)
    return som.chord(bq, bp, pos, vel, metric=metric)


chords = jax.vmap(member)(weights)
assert chords.shape == (8, 300)
assert float(jnp.std(chords, axis=0).mean()) > 0  # a real posterior
```

{func}`~phasecurvefit.som.bootstrap_weights` draws `N` indices with replacement
and returns how often each datum came up, so roughly `1/e` of the data is left
out of any given member. Weights rather than gathered indices keeps every array
at shape `(N,)`: the ensemble is a `vmap` over an `(M, N)` weight matrix with
the data passed once, not `M` copies of the data.

The weight is a multiplicity: a weight of zero is the datum being absent, and
an integer weight of `k` is the datum appearing `k` times. That is what makes
multinomial counts a genuine bootstrap rather than a perturbation that
resembles one, and it means 0/1 weights are a subsample if you want the
cheaper option. `weights` is the same parameter
[outlier rejection](#outlier-rejection) uses; the two are mutually exclusive,
since that one computes its own weights round by round.

What the spread then *means* depends on what you varied. Resampling the data
propagates sampling noise, which is usually the thing worth reporting. Varying
only the initial lattice measures initialization sensitivity instead — a
different question, and rarely the one being asked.

## References

Starkman, N., Bovy, J., Webb, J. J., Calvetti, D., & Somersalo, E. (2023).
*On the Fast Track: Rapid construction of stellar stream paths.* MNRAS
**522**(4), 5022–5036. [arXiv:2212.00949](https://arxiv.org/abs/2212.00949),
doi:[10.1093/mnras/stad1166](https://doi.org/10.1093/mnras/stad1166).

**If you use this SOM stage in published work, please cite that paper.** It is
the source of the method: the 1-D lattice, the equi-frequency initialization,
and the projection-and-order procedure are all from §2.2 and Appendix A.

This implementation deviates from the paper in the places listed below;
published numbers will not reproduce bit-for-bit:

1. **Batch Kohonen** replaces the paper's online update (A9)/(A10). The batch
   form is the fixed point of the conventional online update; it shares that
   fixed point but not the trajectory to it. Note that (A9) *as printed*
   increments by the best-matching unit's residual `w - p^(c(n))`, whose
   equilibrium does not contain `p^(k)` at all and is not solved by the batch
   form; the equivalence holds under the conventional reading, whose increment
   is proportional to `w - p^(k)`.
2. **`sigma` is annealed** geometrically across training, where (A8) uses a
   fixed, user-chosen coupling constant.
3. **The backbone is a C1 spline** through the prototypes, where §2.2.2
   connects them by line segments — a change of method. §2.2.2 rejects
   projecting onto a smooth curve:
   "not only is projection onto a curve challenging, it is not correct for this
   problem". The objection is a metric mismatch, not a cost: training couples
   prototypes through a piecewise-linear lattice, so the fit is *to* a polyline,
   and projecting the fitted prototypes onto a spline measures against a
   different curve than the one that was fit. We accept that mismatch. It buys
   the N-dimensional projection: a C1 backbone has no vertex-tie wedges, which
   is what removes the 2-D-only angle sweep (see
   [How the projection works](#how-the-projection-works)). The mismatch is
   small when `sigma_phys` is small relative to the curve's radius, since the
   spline then departs from the polyline by less than the smoothing bias
   already present. Closing it means coupling the prototypes smoothly during
   training too, which is not implemented here.
4. **The projection generalizes to N dimensions**, replacing the angle sweep
   §2.2.2 uses to order data in convexity regions.
5. **Each datum is compared against two segments** — those adjacent to its
   nearest backbone vertex — rather than against every node and every
   projection, as step (4) of §2.2.2 specifies. Restricting the candidate set
   is what keeps the projection performant at catalogue scale.
6. **There is no progenitor/origin anchor** (step (1)), so a standalone
   ordering has no defined direction. Chain after another orderer, or set
   `orient_by_velocity`.
7. **Velocity enters the best-matching-unit search only when chained.** (A7)
   takes the nearest prototype under a metric over all `D` features, with `D`
   explicitly including velocities. Here velocity participates when the previous
   stage's metric used it, or when you pass a metric and scale yourself; a standalone SOM
   orders on position alone. The scale is also derived from the data rather than
   user-chosen.
8. **The binning coordinate is different.** The paper bins along `phi_1`, the
   longitude produced by its §2.1 great-circle frame fit.
   {func}`~phasecurvefit.som.init_prototypes` bins along the first principal
   axis of the positions instead.
9. **Outlier rejection is an addition**, not in the paper: optional, robust
   rejection by quantization error (see [Outlier rejection](#outlier-rejection)
   above), mirroring {class}`~phasecurvefit.orderers.MSTOrderer`'s edge-length
   clipping.

The BibTeX entry is on the [Citation page](../citation.md).

## See also

- {doc}`orderers` — the orderer interface and how stages chain.
- {doc}`algorithm` — the local-flow walk.
