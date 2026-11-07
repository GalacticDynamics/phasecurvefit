# Outlier rejection with a mixture model

The autoencoder's default membership is bad at finding outliers. Stars that are
plainly not on the stream survive as members, and because they survive they pull the
fitted track towards themselves, worst at the ends, where there is least data to hold
it in place. This page explains why, and how modelling the background fixes it.
{doc}`../how-to/outliers` shows how to turn it on.

## Why the default membership cannot reject anything

### It is trained to keep the outliers

The encoder's first phase trains the membership head as a classifier:

```text
prob_ordered_penalty = masked_mean((1.0 - prob_pred_ordered) ** 2, mask)  # p -> 1
prob_random_penalty  = masked_mean(prob_pred_random ** 2, mask)           # p -> 0
```

The positive set is every sample the orderer visited. An MST, like any spanning
structure, has no way to reject a sample: it threads outliers into the ordering along
their cheapest edge. Those outliers then land in the positive set, and the network is
trained to call them members. The membership head reproduces the orderer's blind spot,
and training longer does not fix it.

### The negatives are too easy

The negatives are drawn uniformly from the bounding box of positions and velocities. A
uniform draw in that box almost never has a plausible pairing of position and velocity,
because real stream stars have tightly correlated $(x, v)$. So the classifier reaches a
low loss by learning a coarse boundary around correlated $(x, v)$, and never has to
resolve the structure across the track, which is where outliers are found.

### The membership never sees the residual

What defines an outlier is its distance from the fitted track,

$$r_n = \lVert q_n - x_\theta(\gamma_\theta(w_n)) \rVert ,$$

and the decoder computes it at every step. But no gradient connects "this star is far
from the track" to "lower its membership". The membership is badly calibrated as a
result: it saturates near 1 for anything near the data, so a threshold of 0.5 cuts
nothing.

## Modelling the background

Instead of classifying, the mixture model describes the data. Following Hogg, Bovy &
Lang (2010), §3, each star comes either from the stream or from a smooth background:

$$
\mathcal{L}_n = \pi_n \, \mathcal{N}\!\left(q_n \,;\, x_\theta(\gamma_n),\,
                \sigma^2(\gamma_n)\,\mathbb{I}\right)
              + (1 - \pi_n)\,\rho_{\mathrm{bg}} ,
$$

and training minimizes $-\sum_n \log \mathcal{L}_n$.

- $\pi_n$ is the encoder's membership output. It plays the role of $1 - P_b$ in Hogg
  et al., except that a network predicts a weight for each star instead of one global
  fraction of bad data.
- $\mathcal{N}(\cdot)$ is the stream: a Gaussian of width $\sigma(\gamma)$ about the
  decoded track.
- $\rho_{\mathrm{bg}}$ is a flat background density over the field. Hogg et al. use a
  broad Gaussian; a uniform density is the natural analogue when the field is a bounded
  footprint.

The gradient with respect to $\pi_n$ now does the right thing. Close to the track the
Gaussian dominates and $\pi_n \to 1$; far from it the background dominates and
$\pi_n \to 0$. Outlier rejection comes out of the likelihood instead of a
self-referential label. The crossover is set by $\rho_{\mathrm{bg}}$, a background
surface density with a physical meaning, rather than a hand-picked cut. Hogg et al.
argue against sigma-clipping and hand-tuned cuts for the same reason.

With per-sample weights, each sample's term in the sum is multiplied by its weight,
and its posterior membership doesn't change. {doc}`density-fields` derives this.

The encoder's output is the prior $\pi_n$. The posterior membership, which
{func}`~phasecurvefit.nn.posterior_membership` returns, also accounts for how far the
star landed from the track, and is the number to use.

## Two ways the fit can fail

A plain mixture likelihood has two degenerate optima, and training can reach both.
Each is closed off by a schedule.

**Width inflation.** If a few outliers sit only a few widths off the track, the
likelihood prefers to widen the stream to include them over lowering their $\pi_n$, so
more training does not help. `sigma_ceiling` caps the width with a ceiling that shrinks
during training. Early on, while the track is still poor, everything should look like a
member. As the track sharpens, the ceiling tightens and stars that stay far from it are
forced into the background. The width may fall below the ceiling, but not rise above
it.

**Membership collapse.** At the start the decoded track is meaningless, so every star
has a large residual, and the cheapest move is to call the whole dataset background,
$\pi_n \to 0$. Once $\pi_n = 0$ the stream component carries no weight, no gradient
reaches the decoder, the track drifts, residuals grow, and $\pi$ stays at zero. With
outliers 20 widths away, a mixture without a warm-up pushed the median member residual
to about 8, against a true width of 0.15, and rejected all 400 genuine members.
`warmup_frac` prevents this the way EM is usually initialized: it fits the track first,
with membership held near 1, and only then lets the model reject stars. In the loss,
$\pi^{\mathrm{eff}}_n = 1 - w(t)\,(1 - \pi_n)$, so at $w = 0$ it is a plain Gaussian fit
to the track.

## What it does not fix

The encoder's negatives are still uniform in the box, so its prior $\pi_n$ remains a
weak discriminator. The mixture works anyway, because the posterior is dominated by the
residual. Harder negatives, such as real stars displaced across the track or with
shuffled velocities, would sharpen the prior too; that is not implemented.

## References

Hogg, D. W., Bovy, J., & Lang, D. (2010). *Data analysis recipes: Fitting a model to
data.* [arXiv:1008.4686](https://arxiv.org/abs/1008.4686). §3, "Pruning outliers", is
the mixture model implemented here.

```bibtex
@article{hogg2010data,
  title={Data analysis recipes: Fitting a model to data},
  author={Hogg, David W. and Bovy, Jo and Lang, Dustin},
  journal={arXiv preprint arXiv:1008.4686},
  year={2010},
  eprint={1008.4686},
  archivePrefix={arXiv},
  primaryClass={astro-ph.IM}
}
```
