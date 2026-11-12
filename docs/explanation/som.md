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

If the earlier stage left some samples unvisited, the SOM keeps them in its arrays
and gives them zero weight. They take no part in seeding or training, and they stay
unvisited (`-1`) in the result. Since no array changes size, the chained SOM runs
under `jit`, `vmap` and `grad`.

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
is what you use downstream, degrades up to four times. Turning on `outlier_mixture`
keeps the backbone on the curve ({doc}`../how-to/som`).

## The outlier mixture

`outlier_mixture` uses the model {doc}`outliers` describes for the autoencoder, after
Hogg, Bovy & Lang (2010), §3. Each sample's position comes either from the curve or
from a flat background:

$$
\mathcal{L}_n = \pi\, f(q_n) + (1 - \pi)\, \rho_{\mathrm{bg}} ,
$$

where $\pi$ is the fraction of members, $f$ the curve's density and
$\rho_{\mathrm{bg}}$ one over the volume of the field. Batch Kohonen training is close
to fitting a Gaussian mixture: Gepperth & Pfülb (2020) show that an energy-based SOM
does gradient descent on an approximation of a Gaussian mixture's log-likelihood. So
the mixture is fitted by EM inside the training loop. Each epoch computes every
sample's membership $r_n$ from the current prototypes, refits $\pi$ and the curve's
width $s$ in closed form, and then counts each sample by $\omega_n r_n$ in the update.
With the prototypes standing in for the track, nothing has to be ordered before
membership can be computed.

### The curve's density

$f$ is a Gaussian tube of width $s$ about the polyline through the prototypes. Each
segment carries a share of the members, spread evenly along it. A Gaussian about each
prototype would be simpler, but it is lumpy along the track wherever $s$ is smaller
than the spacing between prototypes, and samples between two prototypes then look like
background: on a clean arc it rejected 45 of 300 members.

A segment's share is the weighted fraction of all the samples it explains, refitted
each epoch. With equal shares, an outlier that drew an end prototype out to itself,
as one does in a plain SOM, stayed a member, because the long segment out to it still
claimed its share of the members. Weighting the shares by membership instead fed back
on itself, and stretches that lost members lost their share.

The tube is open at its two ends, as {func}`~phasecurvefit.som.chord` is: past the
end prototypes the end segments carry on. Early in training the wide neighbourhood
draws the lattice in from the ends of the curve. With closed ends, the samples left
outside looked like background, stopped pulling the ends back out, and on a clean arc
half of the members were rejected.

### Widening the curve early in training

Early in training the lattice is a rough copy of the curve, and the samples scatter
about it by more than they will about the final one. While the neighbourhood is wide
the lattice is a smoothed copy of the curve. Chained after another orderer, the
neighbourhood starts narrow, but the lattice starts from the means of bins along the
earlier ordering, which cut inside the curve's bends. So the memberships are computed
with $s^2$ widened by
$(0.1\,\sigma_t\,\Delta)^2 - (0.1\,\sigma_{\mathrm{end}}\,\Delta)^2$, where
$\Delta$ is the mean spacing between prototypes and $\sigma_t$ shrinks geometrically
from $\max(\sigma_{\mathrm{start}}, n_{\mathrm{prototypes}}/4)$ to
$\sigma_{\mathrm{end}}$ over the epochs. The widening is zero at the last epoch.

Without it, members the early lattice missed were called background and stayed lost:
on a clean arc trained for 40 epochs, 256 of 300. Tied to the lattice's own
neighbourhood, which stays at `sigma_end` when chained, the widening was zero there,
and after an MST on a near-closed loop with three lobes and 15 prototypes, 306 of 450
members were rejected. A widening of 0.17 or more let a background of 240 samples
among 540 drag the backbone off at 20 epochs.

### Weights

A weight multiplies a sample's terms in every sum, the M-steps included, and never
enters its membership, the rule {doc}`density-fields` derives for the autoencoder. A
sample of weight 2 trains exactly as two coincident copies of it.

### Limits

At 240 background samples among 540, the backbone stayed within 0.16 of the arc for
10 epochs, and failed at 40 epochs, keeping almost all the background. Chained after
another orderer the SOM's neighbourhood stays narrow, so it can't move a lattice the
earlier stage started in the background; {doc}`../how-to/som` shows the setting that
helps.

The background is flat over the samples' bounding box, so it needs room outside the
curve. When the samples fill that box, as the pixels of a mask or an image do, the
background explains them better than the curve does. On a straight mask of half-width
0.3, the flat background beat the fitted tube by 136 in total log-likelihood, and EM
ended with a member fraction of 0.33, calling 974 of the 1071 pixels background.

So a fit whose member fraction $\pi$ ends below 0.5 keeps every sample, as if the
mixture were off. The SOM's input is meant to be a sample of a curve, and a fit that
calls most of it background has fitted the box instead. On the mask this keeps all
1071 pixels. The contaminated arc ended with $\pi$ between 0.76 and 1 at every level
of background in the table of {doc}`../how-to/som`, and the four streams of
{doc}`autoencoder` between 0.96 and 1, so none of them changed. I tried two other
fixes. A flat background over a field 1.5 times wider on each axis kept every pixel
of the mask, but at 120 and 240 background samples the arc's backbone error grew from
0.038 to 0.14 and from 0.16 to 0.71. Holding $\pi$ at 0.5 or more in the M-step still
rejected 472 pixels. The cost of the rule is a field where interlopers outnumber the
members: there the mixture rejects nothing.

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
9. **The outlier mixture is an addition** not in the paper.

## References

Starkman, N., Bovy, J., Webb, J. J., Calvetti, D., & Somersalo, E. (2023). *On the
Fast Track: Rapid construction of stellar stream paths.* MNRAS **522**(4), 5022–5036.
[arXiv:2212.00949](https://arxiv.org/abs/2212.00949),
doi:[10.1093/mnras/stad1166](https://doi.org/10.1093/mnras/stad1166). The BibTeX entry
is on the {doc}`../reference/citation` page.

Hogg, D. W., Bovy, J., & Lang, D. (2010). *Data analysis recipes: Fitting a model to
data.* [arXiv:1008.4686](https://arxiv.org/abs/1008.4686).

Gepperth, A., & Pfülb, B. (2020). *A Rigorous Link Between Self-Organizing Maps and
Gaussian Mixture Models.* [arXiv:2009.11710](https://arxiv.org/abs/2009.11710).
