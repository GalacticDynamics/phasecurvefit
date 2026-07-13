# Outlier rejection

The default autoencoder is bad at finding outliers. Stars that are obviously not
on the stream survive as "members", and because they survive, they drag the
fitted track towards themselves — worst at the ends, where there is least data to
hold it down.

This guide explains why, and how the mixture-model membership fixes it.

## Why the default membership can't reject anything

### It is trained to keep the outliers

Phase 1 trains the encoder's membership head as a *classifier*:

```{code-block} python
prob_ordered_penalty = masked_mean((1.0 - prob_pred_ordered) ** 2, mask)  # p -> 1
prob_random_penalty  = masked_mean(prob_pred_random ** 2, mask)           # p -> 0
```

The positive set is `ordered_ws` — every tracer the orderer visited. And an MST
(or any spanning structure) has **no reject option**: it threads outliers into
the ordering along their cheapest edge. Those outliers then land in the positive
set, and the network is explicitly supervised to call them members.

The membership head is a *student of the orderer*. It faithfully reproduces the
orderer's blind spot. You cannot fix this by training longer.

### The negatives are too easy

The negatives are drawn uniformly from the phase-space bounding box — positions
*and* velocities. A uniform draw in that 4D/6D box essentially never has a
physically plausible $(x, v)$ pairing, because real stream stars have tightly
correlated position and velocity. So the classifier reaches near-zero loss by
learning a coarse "is this $(x, v)$ correlated at all" boundary, and never has to
resolve the *transverse* structure of the track — which is exactly where outlier
detection lives.

### `p` never sees the residual

The quantity that actually *defines* an outlier is its distance from the fitted
track,

$$
r_n = \lVert q_n - x_\theta(\gamma_\theta(w_n)) \rVert ,
$$

and the decoder computes it on every step. But no gradient connects "this star is
far from the track" to "lower its membership probability". The autoencoder has
precisely the signal you want and throws it away.

The symptom is that `p` is badly **calibrated**: it saturates near 1 for anything
remotely near the data manifold, so `member_threshold=0.5` slices through
nothing.

## The fix: model the background

Rather than classifying, *model* the data. Following Hogg, Bovy & Lang (2010),
§3 ("Pruning outliers"), treat each star as drawn either from the stream or from
a smooth background:

$$
\mathcal{L}_n = \pi_n \, \mathcal{N}\!\left(q_n \,;\, x_\theta(\gamma_n),\,
                \sigma^2(\gamma_n)\,\mathbb{I}\right)
              + (1 - \pi_n)\,\rho_{\mathrm{bg}}
$$

and minimise $-\sum_n \log \mathcal{L}_n$.

- $\pi_n$ is the encoder's membership output. It plays the role of $(1 - P_b)$ in
  Hogg et al., except **amortised**: instead of one global bad-data fraction, a
  network predicts a per-star mixture weight.
- $\mathcal{N}(\cdot)$ is the stream: a Gaussian of width $\sigma(\gamma)$ about
  the decoded track.
- $\rho_{\mathrm{bg}}$ is a flat background density over the field. (Hogg et al.
  use a broad Gaussian $(Y_b, V_b)$; a uniform density is the natural analogue
  when the field is a bounded footprint.)

Now the gradient w.r.t. $\pi_n$ is automatic and *correct*: close to the track,
the Gaussian dominates and $\pi_n \to 1$; far from it, the background dominates
and $\pi_n \to 0$. **Outlier rejection falls out of the likelihood** instead of
being imposed by a self-referential label.

It also retires `member_threshold`. The crossover is now set by
$\rho_{\mathrm{bg}}$, which is a physically interpretable background surface
density rather than an arbitrary cut. Hogg et al. are blunt about why that
matters: sigma-clipping and hand-tuned cuts are *"a procedure and not the outcome
of justifiable modeling"*.

## Using it

Membership modelling is **opt-in**. Passing `membership=None` (the default)
leaves the existing behaviour bit-for-bit unchanged.

```{code-block} python
import phasecurvefit as pcf

membership = pcf.nn.MixtureMembershipConfig(
    sigma_init=0.1,           # your guess at the stream width
    sigma_ceiling=(0.5, 0.1), # anneal the width ceiling down
    warmup_frac=0.3,          # fit the track before rejecting anything
)

config = pcf.nn.TrainingConfig(
    n_epochs_both=2000,
    membership=membership,
)
result, model, *_ = pcf.nn.train_autoencoder(model, walkresult, config=config, key=key)
```

Then get **calibrated** membership out — note this is *not* the encoder's raw
`prob`:

```{code-block} python
q = pcf.nn.stream_membership(model, ws)   # posterior, in [0, 1]
members = q > 0.5
```

`prob` is only the *prior* $\pi_n$: what the encoder believes from the star's
coordinates alone, before seeing how far it landed from the track.
`stream_membership` folds in the residual and returns the posterior $\hat{q}_n$.
That is the number you want. Hogg et al. recommend keeping it as a weight rather
than thresholding at all.

## The two things that will bite you

A naive mixture likelihood has **two degenerate optima**, and both are reachable.
Each has a schedule that closes it off. These are not tuning knobs to fiddle
with; they are structural.

### 1. Width inflation

If a few outliers sit only a few stream-widths off the track, the likelihood is
*happier widening the stream* to swallow them than lowering their $\pi_n$. It
genuinely prefers this, so no amount of extra training helps.

`sigma_ceiling=(start, stop)` closes this off. The width is capped from above by
a ceiling that anneals geometrically downward. Early on it is generous — while
the track is still bad, everything *should* look like a member. As the track
sharpens, the ceiling squeezes and stars that stay far from it are progressively
forced to explain themselves as background. The width may go *below* the ceiling
if the data want it to; it may not go above.

### 2. Membership collapse

Worse, and self-reinforcing. At initialisation the decoded track is meaningless,
so *every* star has a huge residual. The likelihood's cheapest move is to declare
the whole dataset background: $\pi_n \to 0$ everywhere. But once $\pi_n = 0$, the
stream component carries no weight, **no gradient reaches the decoder**, the track
drifts freely, residuals grow, and $\pi$ is pinned at zero forever. Training
collapses to "there is no stream".

This is not hypothetical. With outliers at 20 stream-widths, an un-ramped mixture
drove the median inlier residual to ~8 (true stream width: 0.15) and rejected
**all 400 genuine members**.

`warmup_frac` closes this off, via the standard EM initialisation: fit the track
*first*, with membership pinned near 1, and only then let the model start
disowning stars. Concretely $\pi^{\mathrm{eff}}_n = 1 - w(t)\,(1 - \pi_n)$, so at
$w=0$ the loss degenerates to a plain Gaussian reconstruction NLL.

## What to expect

On a synthetic arc — 400 members of width $\sigma = 0.15$, plus 30 outliers
displaced perpendicular to the track — thresholding the posterior at 0.5:

| outlier offset | caught (no warm-up) | members lost | **caught (warm-up 0.3)** | **members lost** |
| -------------: | ------------------: | -----------: | -----------------------: | ---------------: |
|           3 σ  |               0 / 30 |            0 |                   0 / 30 |            **0** |
|           5 σ  |               2 / 30 |            0 |                  21 / 30 |            **0** |
|          10 σ  |              30 / 30 |            0 |              **30 / 30** |            **0** |
|          20 σ  |              30 / 30 |    **400** ✗ |              **30 / 30** |            **0** |

Three things worth reading off this:

- **Zero false positives everywhere.** The model does not cut genuine members.
- **0/30 at 3σ is the right answer, not a failure.** A star 3σ from the track is
  genuinely consistent with membership — the stream has Gaussian tails, and you
  would *expect* some members out there. Rejecting them would be wrong.
- **The warm-up is load-bearing.** Without it, the 20σ case collapses completely.

The fitted width came out at 0.150 against a true 0.15 — you get the stream width
for free, as a by-product.

## What this does not fix

The negatives in phase 1 are still uniform-in-box, so the encoder's *prior*
$\pi_n$ remains a weak discriminator. The mixture works anyway, because the
posterior is dominated by the residual. But hardening the negatives — displacing
real stars perpendicular to the track, or shuffling velocities to break the
$x$–$v$ correlation — would sharpen the prior too, and is a natural follow-up.

## References

Hogg, D. W., Bovy, J., & Lang, D. (2010). *Data analysis recipes: Fitting a model
to data.* [arXiv:1008.4686](https://arxiv.org/abs/1008.4686). §3, "Pruning
outliers", is the mixture model implemented here.

```{code-block} bibtex
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

## See also

- {doc}`nn` — the neural-network architecture and training phases.
- {doc}`/tutorials/index` — a worked example on a mock stream with injected
  outliers.
