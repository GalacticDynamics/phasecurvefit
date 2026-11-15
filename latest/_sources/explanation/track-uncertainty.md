# Track uncertainty

A fitted track comes from one draw of the data, and a different draw from the same
stream would give a slightly different track. This page explains where that
uncertainty comes from, why phasecurvefit estimates it with a bootstrap of the whole
pipeline, and how well the resulting bands are calibrated.
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
