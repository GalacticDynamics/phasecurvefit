# phasecurvefit and NumPyro: what fits where

Prototypes only, not library code. Versions checked: NumPyro 0.22.0, JAX 0.8.2,
Equinox 0.13.x. Every NumPyro fact below was checked in the installed source
(`numpyro/contrib/module.py`, `numpyro/contrib/hsgp/`,
`numpyro/distributions/mixtures.py`, `numpyro/infer/autoguide.py`,
`numpyro/handlers.py`) or in the published wheels on PyPI.

## API facts

- `numpyro.contrib.module` exports `flax_module`, `random_flax_module`,
  `nnx_module`, `random_nnx_module`, `eqx_module` and `random_eqx_module`.
  `eqx_module` and `random_eqx_module` first appear in 0.19.0 (Aug 2025).
  `haiku_module` is in 0.20.0 and gone from 0.21.0 (May 2026) on.
- `random_eqx_module(name, module, prior)` partitions the module with
  `eqx.is_inexact_array`, names each leaf by its JAX key path, and samples it
  from `prior` (one distribution, a dict by key path, or a callable of name and
  shape). So a phasecurvefit `TrackNet`, `FourierTrackNet` or `OrderingNet` goes
  in as it is, with no wrapper.
- `numpyro.contrib.hsgp` gives the Hilbert-space GP approximation (Solin &
  Särkkä 2020; Riutort-Mayol et al. 2023): `eigenfunctions`,
  `diag_spectral_density_squared_exponential`, `hsgp_squared_exponential`,
  `hsgp_matern`. `hsgp_*` samples a site named `"beta"`, so two of them in one
  model need `handlers.scope`; the prototypes call the pieces directly.
- `MixtureGeneral` and `MixtureSameFamily` assert that each component's support
  is parameter-free, so a `Uniform(low, high)` background is refused. The Hogg
  et al. background is written as `numpyro.factor` with `logaddexp` instead.
- Autoguides: `AutoNormal`, `AutoDelta`, `AutoDiagonalNormal`,
  `AutoMultivariateNormal`, `AutoLowRankMultivariateNormal`,
  `AutoLaplaceApproximation`, `AutoIAFNormal`, `AutoBNAFNormal`, `AutoDAIS`, and
  batched variants.
- In `spectral_density_squared_exponential` the `alpha` argument enters
  linearly, so it is the kernel variance, though `hsgp_squared_exponential`'s
  docstring calls it the amplitude. Taking `jnp.sqrt` of
  `diag_spectral_density_squared_exponential` underflows to 0 at short length
  scales or high frequencies, and its gradient there is NaN. On the first run
  this NaN gave 837 and 930 divergences in 1000 NUTS draws, and SVI ended in
  NaN. The prototypes write sqrt(S) in closed form instead
  (`tracklik.sqrt_spd`).
- `AutoLaplaceApproximation` returned an all-zero `scale_tril` when the Hessian
  at the SVI end point was not positive definite. That gives a zero-width
  posterior with no warning or error.
- `handlers.scale(scale=w)` multiplies a site's log-density by `w`, broadcast
  over a plate, which is how per-star weights enter (option g). Checked: weights
  (2, 1, 1, 1, 1) give the same log-density as the data with star 0 repeated
  (−45.5364 both). A weight of exactly 0 raises
  `ValueError: 'scale' argument should be positive`, so a density field with
  empty pixels needs `handlers.mask` for those, or the weight multiplied into
  the `factor` directly.

## What traces today (option e)

Checked with `jax.jit(jax.jacrev(...))` with respect to the x positions, on an
80-point arc:

| orderer                                           | traces under jit+grad                                              | outputs with a gradient         |
| ------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------- |
| `MSTOrderer()`                                    | yes                                                                | `chord`, `backbone`, `__call__` |
| `SOMOrderer()`                                    | yes                                                                | `chord`, `backbone`, `__call__` |
| `LocalFlowOrderer()`                              | yes                                                                | `chord`, `__call__`             |
| `MSTOrderer() \| SOMOrderer()` (default pipeline) | no: "SOMOrderer cannot be traced when chained after another stage" | —                               |
| `RidgeOrderer()`                                  | no: host SciPy graph, by design                                    | —                               |

`indices` is piecewise constant, so its gradient is zero almost everywhere. The
ordering is a function of the observed data only, so inside a model it is a
constant: it needs to be traced only if positions are themselves latent (e.g.
noisy positions).

## Units (option h)

Quantities do not flow through NumPyro. `dist.Normal(Q, Q).log_prob(Q)` fails in
the support check, a Quantity passed as `obs=` fails to convert to
dimensionless, and `quax.quaxify` of a whole model leaks tracers through
NumPyro's effect handlers. A log-density of a length has units, so this is not a
missing dispatch. The workable design strips at the boundary in the
`StateMetadata` unit system and re-attaches units to the track draws. That goes
against AGENTS.md's rule that Quantities flow through the algorithm, so it needs
the user's decision.

## The options

Measured numbers are in `README.md`.

**(a) Bayesian NN decoder or encoder.** `random_eqx_module` takes a
phasecurvefit `TrackNet` or `OrderingNet` directly, so no bridge code is needed.
NUTS over a 338-weight `TrackNet` did not converge (R̂ 3.8–4.5), and mean-field
VI was biased. The default decoder has about 20 000 weights. This is not worth
library code beyond a docs example.

**(b) Mixture membership as a NumPyro likelihood.** Hogg et al. §3 with a global
background fraction f_bg and the width σ as latents, written as
`factor(logaddexp(...))`, because `MixtureGeneral` refuses a `Uniform`
component. It recovered σ = 0.201 (true 0.2) with 4% interlopers and kept the
track as accurate as without them. Compared with the library's loss: the library
predicts a per-star prior π_n with the encoder, anneals a width ceiling, and
warms up membership to avoid collapse. In the NumPyro version the priors do that
job. Nothing collapsed in 12 datasets, from a start at the ordering's
least-squares fit. The per-star π_n could come back as π(γ) with its own HSGP if
the background fraction varies along the stream.

**(c) Bayesian basis-function decoder.** The HSGP track is exactly this: 32
basis coefficients plus 4 kernel hyperparameters and σ. With γ fixed it is cheap
(8 s) but undercovers (0.49, 0.82), because γ's errors are left out. A Bayesian
last layer on the trained decoder is the sibling's method C (0.38, 0.66). For a
linear-Gaussian last layer the Laplace answer is exact, so HMC would add
nothing.

**(d) SVI versus NUTS.** On the marginal model: `AutoMultivariateNormal` is 30
times faster than NUTS and a little conservative. `AutoNormal` is wider still.
`AutoLaplace` fails silently. NUTS is the reference.

**(e) The pipeline inside a model.** The ordering depends only on observed data,
so in a model it is a constant, used to start the sampler. Marginalising γ on a
grid makes the ordering unnecessary inside the likelihood. It only fixes which
end is γ = −1 and gives the starting point, so the default chain's `jit`
limitation does not matter here. Treating positions as latent (measurement
errors) would need the orderer to trace; the `som-chain-jit` session is making
the chain trace.

**(f) phasecurvefit pieces as NumPyro factors.** The marginal-γ likelihood is a
natural drop-in. Given any track function x(γ) (an HSGP, a decoder, or an orbit
integrated from a potential and progenitor), it gives log p(stars | track, σ,
f_bg). That is the piece a user needs for joint inference with a potential.

**(g) Per-sample weights.** `handlers.scale` or `w * loglik` inside the factor.
Weight 2 is identical to two copies (checked). So the posterior is right only
when the weights are counts. For image data the proper model is a Poisson
likelihood for the pixel counts with rate ∝ the model's surface density, or a
Gaussian with the pixel noise. A weighted point likelihood with arbitrary
normalisation gives bands whose width depends on that normalisation. A library
helper should say so and take a `weights` argument documented as counts.

**(h) Units.** Strip at the boundary in the `StateMetadata` unit system, and
give back the draws as Quantities. See "Units" above.
