# Track uncertainty

A fitted track comes from one draw of the data, and a different draw from the same
stream would give a slightly different track. This page explains where that
uncertainty comes from, the two ways phasecurvefit estimates it (a bootstrap of the
whole pipeline, and the posterior of a probabilistic model of the track), and how
well the resulting bands are calibrated.
{doc}`../how-to/track-uncertainty` shows how to compute them.

## Where the uncertainty enters

The track passes through four stages, and each adds its own uncertainty.

- The ordering. The MST and the SOM are discrete. A small change in the data can
  change which samples are linked, which are called background, and where the
  ordering starts and stops.
- Membership. A sample counted as a member pulls the track towards it, and a sample
  counted as background doesn't.
- The encoder's $\gamma$. Where a sample lands along the track decides which part of
  the track it pulls on.
- The decoder. Even with the ordering, membership and $\gamma$ fixed, the track is
  fitted to noisy positions.

The first three are discrete choices or come from networks trained on them, so no
likelihood connects them to the data. A posterior over the decoder's weights alone
holds them fixed.

## Methods considered

| method | cost | captures | measured here |
|---|---|---|---|
| bootstrap of the whole pipeline (Efron 1979) | one fit per member | all four stages | yes |
| deep ensemble, seeds only (Lakshminarayanan et al. 2017) | one fit per member | training randomness | yes |
| last-layer Laplace (Kristiadi et al. 2020; Daxberger et al. 2021) | about one fit | decoder, with $\gamma$ and features fixed | yes |
| last-layer HMC or NUTS (Hoffman & Gelman 2014; BlackJAX, Cabezas et al. 2024) | about one fit | the same as last-layer Laplace | no, see below |
| full-network HMC (Izmailov et al. 2021) | many fits | decoder | no |
| variational inference, Bayes by Backprop (Blundell et al. 2015) | a few fits | decoder | no |
| MC dropout (Gal & Ghahramani 2016) | one fit, retrained with dropout | decoder | no |
| SWAG (Maddox et al. 2019) | about one fit | decoder, near one optimum | no |
| Gaussian process on the residuals | about one fit | decoder, with $\gamma$ fixed | no |

Every method below the bootstrap puts a posterior on the decoder only, given one
ordering, one membership and one set of $\gamma$. A posterior also needs a likelihood.
The default joint loss is a weighted sum of a squared distance and a velocity-alignment
term, with no noise model, so a Gaussian noise level has to be added for any of them.
The mixture model of {doc}`outliers` supplies one, $\sigma(\gamma)$, but only when it
is turned on. With per-sample weights that are not counts, such as image intensities,
the weighted loss is a pseudo-likelihood and its curvature isn't an uncertainty
({doc}`density-fields`).

The decoder's last layer is linear, so with a Gaussian likelihood and prior the
posterior over it is Gaussian, and the Laplace approximation is exact. HMC on the
last layer would sample the same distribution, so it was not run separately.
Full-network HMC, VI, MC dropout, SWAG and a GP on the residuals were not run. They
all leave the ordering, membership and $\gamma$ fixed, and the measurements below
show that the decoder alone carries only part of the variance.

## Measured calibration

The test data are a sine-shaped arc, $x = 10 s$, $y = 2 \sin(1.5 \pi s)$, with 400
samples, a Gaussian width of 0.2 across the track, and velocities along it. There
were 12 independent datasets and 20 members per ensemble, with the default
{class}`~phasecurvefit.nn.TrainingConfig`. At 46 points along the true curve, I took
each member's signed offset along the true normal, and counted the true curve as
covered when the offset 0 lay inside the central 68% or 95% of the members. The
prototypes are in `prototypes/track_uncertainty/` in the repository.

| method | 68% coverage | 95% coverage | 68% half-width | cost per member |
|---|---|---|---|---|
| {func}`~phasecurvefit.bootstrap_track` | 0.56 | 0.84 | 0.027 | 8.6 s |
| seed ensemble | 0.36 | 0.56 | 0.015 | 8.8 s |
| last-layer Laplace | 0.38 | 0.66 | 0.018 | 1.9 s on top of one fit |

The bootstrap row is {func}`~phasecurvefit.bootstrap_track` as it ships. The other two
rows were measured on an earlier version of the pipeline, before
{func}`~phasecurvefit.fit_track` used a {class}`~phasecurvefit.som.SOMMixture` by
default. On that version a bootstrap built from weights instead of rows covered 0.59
and 0.85, so the comparison between methods holds.

The error of a single fit was 0.038 rms. The tube about the reference fit that
{meth}`~phasecurvefit.TrackEnsemble.radius` returns covered the truth 0.58 and 0.88 of
the time, with a median 68% radius of 0.032.

Training randomness alone gives about 30% of the bootstrap's variance, and the
last-layer posterior about the same. Most of the variance comes from the data and the
stages that depend on it, so a method that fixes those stages gives bands that are
too narrow.

## What the bootstrap misses

The bootstrap gets the variance right. Its spread at each point averaged 0.028, and
the scatter of a single fit across the 12 datasets averaged 0.030. The coverage falls
short because of bias. The fit's mean offset from the true curve had an rms of 0.022,
close to the spread, and it was largest where the track bends, 0.047 at $s = 0.83$,
and at the far tip, 0.074 at $s = 0.95$. The 68% coverage dropped to 0.25–0.33 at those
points. A track fitted to a resample is smoothed in the same way as the reference, so
the bootstrap cannot see this bias. Treat the band as a lower limit on the
uncertainty where the track curves sharply.

On the earlier pipeline, with 4% interlopers spread over the field, the bands were
wider (68% half-width 0.048) and covered 0.77 and 0.97 of the time, because which
interlopers are drawn and clipped changes from member to member. That run changed two
things at once, so it doesn't separate the two effects, and it has not been repeated
with the {class}`~phasecurvefit.som.SOMMixture` default.

## Why repeated samples are passed as rows

A bootstrap resample can be written as integer weights.
{func}`~phasecurvefit.bootstrap_track` passes repeated samples as repeated rows
instead, so `weights` keep their own meaning, such as the density at each pixel, and a
resampled sample keeps its weight.

## A posterior for the track

{func}`~phasecurvefit.sample_track` takes another route. It writes down a likelihood
for the samples and draws from the posterior of the track under it, with NumPyro
(Phan et al. 2019). There's no network, so the band holds the noise in the data and
the unknown position of each sample along the track, given the ordering it starts
from.

The model, {func}`~phasecurvefit.track_model`:

- The track is a Gaussian process in $\gamma \in [-1, 1]$, one per position
  component, in the Hilbert-space approximation (Solin & Särkkä 2014;
  Riutort-Mayol et al. 2020). Its amplitude and length scale are latent.
- Each sample's $\gamma$ is summed out on a grid, so no sample is pinned to one
  $\gamma$.
- Samples scatter about the track with a Gaussian. By default its width changes
  along the track, and $\gamma$ has a latent density along it, so a sparse tail
  isn't squeezed into a sliver of $\gamma$.
- A flat background over the samples' bounding box takes interlopers, with a latent
  fraction (Hogg, Bovy & Lang 2010, sec. 3).
- The velocities' mean is the speed times the track's unit tangent, plus an offset of
  up to about 0.05 of the speed. A track that runs the wrong way, or turns the wrong
  way round a loop, then misfits every velocity. Stream stars move a few degrees off
  their track, and a hard tie (`velocity_tie=0`) pulled the track of a mock stream
  away from its stars.

The fit starts from the ordering of {func}`~phasecurvefit.orderers.default_pipeline`
and keeps its topology. On a stream that crosses itself, a start that turns onto the
other branch stays there, and NUTS chains don't leave it either. In the prototypes
NUTS chains often settled in different modes, so SVI with a full-rank Gaussian is the
default, and `method="nuts"` warns when its two chains disagree.

`weights` act as counts in the likelihood. A band from image pixels is right when the
weights are the counts in each pixel, and too wide or too narrow when they are a
density scaled to some peak ({doc}`density-fields`).

### Measured calibration of the posterior

I ran {func}`~phasecurvefit.sample_track` with its defaults on four cases with a known
true track, 8 datasets each and 200 draws per fit. Velocities are the unit tangent
plus Gaussian noise of 0.1 per component.

- helix: the sine arc above with a third component $z = 1.5 \cos(\pi s)$, 400
  samples, width 0.2.
- gap: the sine arc with no samples at $0.45 < s < 0.55$.
- loop: a prolate trochoid with one loop that crosses itself, 600 samples, width
  0.15, run with `n_basis=32, n_grid=512`.
- varying: the sine arc with 800 samples, a dense clump near $s = 0.35$ and sparse
  tails, and a width that grows from 0.06 to 0.36 along it.

Coverage is counted as above, at 46 points along the true curve, from the draws'
offsets across it.

| case | 68% coverage | 95% coverage | 68% half-width | error of the band's centre (rms) | time per fit |
|---|---|---|---|---|---|
| helix | 0.75 | 0.97 | 0.025 | 0.022 | 29 s |
| gap | 0.68 | 0.99 | 0.030 | 0.028 | 37 s |
| loop | 0.85 | 0.98 | 0.026 | 0.018 | 42 s |
| varying | 0.88 | 1.00 | 0.027 | 0.018 | 76 s |

On the varying case the fitted width matched the true one with a median ratio of
1.01, and the true width lay inside the draws' central 68% at 0.63 of the points. The
68% coverage in the sparse far tail ($s > 0.8$) was 0.84. The times are on a shared
machine and include the ordering. The scripts are in `prototypes/numpyro/` in the
repository.

The bootstrap was measured on the arc alone and the posterior on the helix, which adds
a third component to it, so the two aren't the same test. On them the posterior
covered 0.75 and 0.97 and the bootstrap 0.56 and 0.84.

## References

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). "Weight
Uncertainty in Neural Networks." ICML. [arXiv:1505.05424](https://arxiv.org/abs/1505.05424).

Cabezas, A., Corenflos, A., Lao, J., Louf, R., et al. (2024). "BlackJAX: Composable
Bayesian inference in JAX." [arXiv:2402.10797](https://arxiv.org/abs/2402.10797).

Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., et al. (2021). "Laplace
Redux -- Effortless Bayesian Deep Learning." NeurIPS.
[arXiv:2106.14806](https://arxiv.org/abs/2106.14806).

Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." Annals of
Statistics 7, 1.

Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing
Model Uncertainty in Deep Learning." ICML. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142).

Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path
Lengths in Hamiltonian Monte Carlo." [arXiv:1111.4246](https://arxiv.org/abs/1111.4246).

Hogg, D. W., Bovy, J., & Lang, D. (2010). "Data analysis recipes: Fitting a model to
data." [arXiv:1008.4686](https://arxiv.org/abs/1008.4686).

Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. (2021). "What Are Bayesian
Neural Network Posteriors Really Like?" [arXiv:2104.14421](https://arxiv.org/abs/2104.14421).

Kristiadi, A., Hein, M., & Hennig, P. (2020). "Being Bayesian, Even Just a Bit, Fixes
Overconfidence in ReLU Networks." ICML. [arXiv:2002.10118](https://arxiv.org/abs/2002.10118).

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.
[arXiv:1612.01474](https://arxiv.org/abs/1612.01474).

Maddox, W., Garipov, T., Izmailov, P., Vetrov, D., et al. (2019). "A Simple Baseline
for Bayesian Uncertainty in Deep Learning." NeurIPS.
[arXiv:1902.02476](https://arxiv.org/abs/1902.02476).

Phan, D., Pradhan, N., & Jankowiak, M. (2019). "Composable Effects for Flexible and
Accelerated Probabilistic Programming in NumPyro."
[arXiv:1912.11554](https://arxiv.org/abs/1912.11554).

Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A., & Vehtari, A. (2020).
"Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic
programming." [arXiv:2004.11408](https://arxiv.org/abs/2004.11408).

Solin, A., & Särkkä, S. (2014). "Hilbert Space Methods for Reduced-Rank Gaussian
Process Regression." [doi:10.1007/s11222-019-09886-w](https://doi.org/10.1007/s11222-019-09886-w),
[arXiv:1401.5508](https://arxiv.org/abs/1401.5508).
