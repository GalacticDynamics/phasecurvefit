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
most useful as a refinement *after* an initial ordering, and before fitting:

```python
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

It also runs standalone, initializing itself by binning along the first
principal axis of the positions:

```python
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

```python
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

```python
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

## A note on ensembles

{mod}`phasecurvefit.som` is `vmap`-able. An ensemble additionally needs a
diversity source — batch Kohonen is strongly contractive, so jittered
initializations alone converge to bit-identical members.

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

The BibTeX entry is in the [Citation](../index.md#citation) section.

## See also

- {doc}`orderers` — the orderer interface and how stages chain.
- {doc}`algorithm` — the local-flow walk.
