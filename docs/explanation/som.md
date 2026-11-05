# The self-organizing map

{class}`~phasecurvefit.orderers.SOMOrderer` refines an ordering with a
one-dimensional self-organizing map (SOM), following Starkman et al. (2023). It is
the second stage of the default pipeline. {doc}`../how-to/som` shows how to use and
tune it.

## A string of beads

A SOM learns a lattice of prototype vectors in phase space. Because the lattice is
one-dimensional, prototypes $i$ and $j$ are neighbours exactly when $|i - j| = 1$.
A trained map is a string of beads threaded through the data, and the order of the
beads is the order along the curve.

Each prototype is an average over many samples. So the backbone through the
prototypes is much less sensitive to local noise than a walk's step-by-step choices or
an MST's individual edges. That is why the SOM is a refinement stage: given a rough
ordering, it smooths it before the fit.

## Seeding the lattice

Chained after another orderer, the SOM starts from that ordering and holds its
neighbourhood narrow (at `sigma_end`) from the start, so it refines the ordering
locally rather than re-deriving a global one.

On its own, its only way to seed the lattice is to bin the samples along the first
principal axis of the positions. That is a fair proxy for order along the curve only
while the curve does not double back along that axis, up to about one turn. Beyond
that the initial lattice is tangled, and training cannot untangle it: the batch
algorithm refines the neighbourhood structure it is given and cannot undo a wrong
global order.

## Projecting the samples

After training, every sample is placed along the map:

1. The prototypes are joined by a centripetal Catmull-Rom spline, resampled at points
   equally spaced in arc length. This smooth curve is `result.backbone`, and calling
   the result interpolates along it.
2. Each sample is assigned to its nearest backbone point under the stage's `metric`.
3. That position is refined by projecting onto the two adjacent segments in
   position, clamped to each segment, except at the two ends, where data beyond the
   ends extrapolate rather than piling up.

Steps 2 and 3 use different geometry on purpose. A metric such as
{class}`~phasecurvefit.metrics.AlignedMomentumDistanceMetric` is not built from an
inner product, so projecting onto a segment is undefined under it. The metric chooses
which segment, and position chooses where on it. The result, `result.chord`, is a
true arc length along the backbone, a continuous coordinate rather than only a
ranking.

The smooth spline is also what makes the projection work in any number of
dimensions. On a piecewise-linear polyline, every sample whose nearest point is a
vertex gets the same arc length; the original two-dimensional method broke that tie
with an angle sweep that has no analogue in higher dimensions. On a smooth curve those
ties vanish.

## The smoothing length

The setting that matters is the smoothing length along the curve,

$$\sigma_{\text{phys}} = \sigma_{\text{end}} \, \frac{L}{n_{\text{prototypes}} - 1},$$

where $L$ is the track length. The backbone is pulled towards the centre of curvature
by about $\sigma_{\text{phys}}^2 / 2R$ on a bend of radius $R$, so bends tighter than
$\sigma_{\text{phys}}$, and arms closer together than it, are smoothed away. The two
arms of a hairpin then merge into one line, and the result still looks plausible. At
the defaults $\sigma_{\text{phys}}$ is about 3% of the track. A curve that crosses
itself needs a much finer lattice than a simple arc: the structure it must represent
is finer than one lattice step.

## Which metric it uses

After a stage that used velocity, ordering on position alone would undo that stage's
work: at a crossing the two branches coincide in position and differ only in velocity.
On an epitrochoid, a velocity-aware MST with a rank correlation of 0.94 to the truth
dropped to 0.59 under a position-only SOM, and rose to 0.99 under the default. So by
default the SOM follows the previous stage. After a velocity-aware stage it uses
{class}`~phasecurvefit.metrics.FullPhaseSpaceDistanceMetric` with
`metric_scale = sigma_phys / (2 |v|)`, which separates opposite-moving branches by
about the distance the lattice can resolve. On its own, or after a position-only
stage, it uses {class}`~phasecurvefit.metrics.SpatialDistanceMetric`.

The metric must be symmetric in the two points it compares, because it answers
"which prototype is this sample nearest". The default walk metric is not: it scores
forward along the direction of travel, and under it the lattice collapses towards the
head of the curve.

## Contamination

The SOM has no outlier rejection of its own by default, and contamination pulls its
backbone towards the contaminants. More prototypes make this worse, because they chase
the interlopers. The failure is silent: the ordering's rank correlation against the
truth stays above 0.99 even at 20% contamination, while the reconstructed track, which
is what you use downstream, degrades up to four times. Chaining after an orderer that
rejects outliers, or turning on `outlier_clip_sigma`, keeps the backbone on the curve
({doc}`../how-to/som`).

## Why perturbing the initialization gives no ensemble

The batch algorithm is strongly contractive: each epoch replaces every prototype with a
neighbourhood-weighted mean of the data, so a prototype survives only through which
samples it wins. Once two ensemble members agree on their assignments, every later
epoch is identical and they stay merged, so jittered initializations converge to
exactly zero spread. The spread has to come from the data, by resampling it. Resampling
propagates sampling noise, which is usually what is worth reporting; varying only the
initial lattice measures sensitivity to the initialization, a different question.

## Differences from the paper

If you use the SOM stage in published work, please cite Starkman et al. (2023). The
1-D lattice, the equal-frequency initialization and the projection-and-ordering
procedure come from its §2.2 and Appendix A. This implementation differs in these ways,
so published numbers will not reproduce exactly:

1. **Batch updates** replace the paper's online update (A9)/(A10). The batch form is
   the fixed point of the conventional online update; it shares that fixed point but
   not the path to it. (A9) as printed increments by the best-matching unit's residual
   `w - p^(c(n))`, whose equilibrium does not contain `p^(k)` and is not solved by the
   batch form; the equivalence holds under the conventional reading, whose increment
   is proportional to `w - p^(k)`.
2. **The neighbourhood width shrinks** geometrically during training, where (A8) uses
   a fixed coupling constant.
3. **The backbone is a smooth (C1) spline** through the prototypes, where §2.2.2 joins
   them with line segments and argues against projecting onto a smooth curve, because
   training fits a polyline and the spline measures against a different curve. That
   mismatch is accepted here in exchange for the N-dimensional projection, and it is
   small when $\sigma_{\text{phys}}$ is small compared with the curve's radius.
4. **The projection works in any number of dimensions**, replacing the angle sweep
   §2.2.2 uses in convex regions.
5. **Each sample is compared with two segments**, those next to its nearest backbone
   point, rather than with every node and projection as step (4) of §2.2.2 specifies.
   This keeps the projection fast on large catalogues.
6. **There is no progenitor anchor** (step (1)), so a standalone ordering has no
   defined direction. Chain after another orderer, or set `orient_by_velocity`.
7. **Velocity enters the best-matching-unit search only when chained**, or when you
   pass a metric and scale yourself. (A7) uses all features including velocities. The
   scale is derived from the data rather than chosen by the user.
8. **The binning coordinate differs.** The paper bins along `phi_1`, the longitude of
   its fitted great-circle frame (§2.1); here the samples are binned along the first
   principal axis of the positions.
9. **Outlier clipping is an addition** not in the paper.

## References

Starkman, N., Bovy, J., Webb, J. J., Calvetti, D., & Somersalo, E. (2023). *On the
Fast Track: Rapid construction of stellar stream paths.* MNRAS **522**(4), 5022–5036.
[arXiv:2212.00949](https://arxiv.org/abs/2212.00949),
doi:[10.1093/mnras/stad1166](https://doi.org/10.1093/mnras/stad1166). The BibTeX entry
is on the {doc}`../reference/citation` page.
