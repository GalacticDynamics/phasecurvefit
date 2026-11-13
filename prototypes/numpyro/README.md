# NumPyro track posteriors: what was measured

Prototypes only, not library code. `SURVEY.md` has the NumPyro API facts, what
traces, and the options that were not prototyped. `common.py` is copied
unchanged from the sibling session's `prototypes/track_uncertainty/` (branch
`claude/bayesian-track-uncertainty`), so its mock data and coverage metric are
identical and the numbers can be compared directly.

## Setup

- Same mock as the sibling prototypes: a sine-shaped arc, x = 10 s, y = 2
  sin(1.5 π s), 400 stars uniform in arc length, Gaussian scatter across the
  track of width 0.2, velocity along the tangent. 12 datasets (seeds 0–11). The
  interloper runs add 16 stars (4%) uniform over the field, the same ones as the
  sibling's `run_interlopers.py`.
- Coverage: at 46 points along the true track, the signed offset of each
  posterior track draw along the true normal. A band covers when 0 lies between
  its 16th and 84th percentiles (68%) or its 2.5th and 97.5th (95%). Coverage is
  a frequency over the 12 datasets and 46 points. The points are correlated, so
  the coverage numbers are good to a few hundredths, not to the last digit.
- Track model (`tracklik.py`): each coordinate x_d(γ) is a Hilbert-space GP
  (`numpyro.contrib.hsgp` eigenfunctions, 16 basis functions, box 1.3, squared
  exponential) with latent amplitude and length scale per coordinate. Stars
  scatter about it with an isotropic Gaussian of latent width σ. Positions are
  centred and scaled by one scalar.
  - **fixed**: γ_n fixed at the encoder's values from one `fit_track`, members
    only.
  - **marginal**: γ_n latent, uniform on [−1, 1], summed out on a 256-point
    grid. No per-star γ and no neural network. It is started from the default
    pipeline's ordering (rank γ, then ridge least squares).
  - With interlopers, the marginal model adds the Hogg, Bovy & Lang (2010) §3
    background: flat density `uniform_background_density` with a latent fraction
    f_bg ~ Beta(1, 9).
- NUTS: dense mass matrix, target acceptance 0.9, 500 warmup + 500 draws, 2
  chains in parallel. SVI: 4000 Adam steps (lr 2e-3, gradient clip 10) from the
  same start, 200 posterior draws.
- Apple-silicon CPU, shared with other sessions; load average was 3–40 during
  the runs, so times are rough.

## Results, clean arc (12 datasets)

| method                                          | 68% cov | 95% cov | 68% half-width | 95% half-width | time per dataset      |
| ----------------------------------------------- | ------- | ------- | -------------- | -------------- | --------------------- |
| NUTS, marginal γ                                | 0.64    | 0.93    | 0.031          | 0.060          | 200 s                 |
| SVI `AutoMultivariateNormal`, marginal γ        | 0.73    | 0.98    | 0.038          | 0.076          | 6 s                   |
| SVI `AutoNormal`, marginal γ                    | 0.85    | 0.99    | 0.054          | 0.100          | 10 s                  |
| SVI `AutoLaplaceApproximation`, marginal γ      | 0.34    | 0.42    | 0.000          | 0.000          | 3 s                   |
| NUTS, fixed encoder γ                           | 0.49    | 0.82    | 0.023          | 0.045          | 8 s + one `fit_track` |
| _sibling:_ bootstrap of whole pipeline, 20 fits | 0.59    | 0.85    | 0.028          | 0.053          | 20 × 13.7 s           |
| _sibling:_ seed ensemble, 20 fits               | 0.36    | 0.56    | 0.015          | 0.028          | 20 × 8.8 s            |
| _sibling:_ last-layer Laplace                   | 0.38    | 0.66    | 0.018          | 0.036          | 1.9 s + one fit       |

Error of the band centre (rms over datasets and points) was 0.034 for NUTS
marginal, 0.033 for SVI MVN, and 0.036 for a single `fit_track`.

NUTS diagnostics, marginal model: R̂ max 1.06 over all 12 datasets. Track ESS
(ESS of x(γ) at 12 points along the track, out of 1000 draws) had a
median-over-datasets minimum of 102 and a worst of 31. The median dataset had 20
divergent transitions out of 1000. The fixed model mixed worse on some datasets:
track ESS down to 2, R̂ up to 1.50.

## Results, arc with 4% interlopers (12 datasets)

| method                                 | 68% cov | 95% cov | 68% half-width | centre rms | time       |
| -------------------------------------- | ------- | ------- | -------------- | ---------- | ---------- |
| NUTS, marginal γ + background          | 0.63    | 0.93    | 0.031          | 0.034      | 330 s      |
| SVI MVN, marginal γ + background       | 0.75    | 0.99    | 0.040          | 0.033      | 11 s       |
| NUTS, fixed encoder γ (no background)  | 0.50    | 0.81    | 0.024          | 0.036      | 12 s + fit |
| single `fit_track`                     | —       | —       | —              | 0.065      | 18 s       |
| _sibling:_ bootstrap, index resampling | 0.77    | 0.97    | 0.048          | 0.045      | 20 × 9.3 s |

With the background term the posterior median width was 0.201 against a true 0.2
(0.180 for the fixed model, which has no background). The marginal model's track
is as accurate with interlopers as without (centre rms 0.034), while the single
fit's error nearly doubled, to 0.065. Worst NUTS chain: R̂ 1.09, track ESS 9.

## Full Bayesian NN decoder (`run_bnn.py`, 3 datasets)

NUTS over all 338 weights of a `TrackNet(width_size=16, depth=2)` via
`random_eqx_module`, Normal(0, 1) priors, γ fixed at the encoder's values. It
did not converge on any dataset: R̂ max 3.8–4.5, minimum ESS 1, about 100 s each.
Mean-field VI (`AutoNormal`, 20 000 steps, about 20 s) gave a band centre 0.068
rms off the truth and 68% and 95% coverage of 0.49 and 0.83. The library's
default decoder, `TrackNet(100, 3)`, has about 20 000 weights.

## What the numbers say

- The marginal-γ model with NUTS is close to calibrated: 0.64 and 0.93 against
  0.68 and 0.95, with and without interlopers. It is the only method here that
  is.
- It is also the only one that uses no neural network and no fixed γ. The
  uncertainty in where each star sits along the track is integrated over instead
  of being fixed by the encoder, and the smoothness is a latent length scale
  instead of a training choice. Fixing γ at the encoder's values cut coverage to
  0.49 and 0.82.
- SVI with a full-rank Gaussian guide costs 6–11 s and overcovers a little
  (0.73, 0.98). Mean-field `AutoNormal` overcovers more. Its bands are 1.7 times
  NUTS's width, which points to an unconverged guide, not a better posterior.
- `AutoLaplaceApproximation` failed every time: the Hessian at the SVI end point
  was not positive definite, and NumPyro returned an all-zero covariance with no
  warning.
- A full BNN decoder under NUTS does not mix even at 338 weights.
- Remaining bias: the band centre is 0.034 rms off the truth, about the 68%
  half-width. That is why coverage sits a little under nominal.

## Validation on harder cases (`cases.py`, `validate.py`, `mock.py`)

Truth metric: offsets of each draw from the true curve in the true curve's
normal plane at 46 points (s = 0.05 to 0.95). Projections follow the curve
continuously (`cases._track_s`), so at a self-crossing a point is matched on its
own branch. A point the draw never reaches counts as not covered. SVI is
`AutoMultivariateNormal` (4000 steps); NUTS is 2 chains of 500 + 500. "single"
is one `fit_track`.

| case (seeds)                    | single rms | SVI 68% / 95% | SVI rms | NUTS 68% / 95% (2 seeds) |
| ------------------------------- | ---------- | ------------- | ------- | ------------------------ |
| helix, 3-D (8)                  | 0.028      | 0.69 / 0.95   | 0.033   | 0.75 / 0.94              |
| helix + velocities (8)          | 0.028      | 0.66 / 0.95   | 0.032   | 0.66 / 0.91              |
| helix, velocity tied (8)        | 0.028      | 0.82 / 0.99   | 0.019   | running                  |
| arc with a gap (7)              | 0.042      | 0.68 / 0.92   | 0.043   | 0.70 / 0.83              |
| image, weights = counts (7)     | 0.035      | 0.89 / 0.99   | 0.014   | 0.55 / 0.88              |
| image, weights / max (7)        | 0.038      | 0.97 / 1.00   | 0.019   | not run                  |
| image, weights × 5 (7)          | 0.030      | 0.86 / 0.99   | 0.013   | not run                  |
| loop, positions only (8)        | 0.47       | 0.48 / 0.63   | 0.047   | 0.60 / 0.78              |
| loop + velocities (8)           | 0.47       | 0.41 / 0.60   | 0.037   | 0.46 / 0.57              |
| loop, velocity tied (8)         | 0.47       | 0.77 / 0.86   | 0.017   | running                  |
| loop + vel., velocity-MST start | 0.47       | 0.78 / 0.96   | 0.025   | not run                  |
| loop, tied, velocity-MST start  | 0.47       | 0.89 / 0.99   | 0.016   | not run                  |
| mock stream, 1000 of 8000 (4)   | —          | 0.86 / 0.98   | 0.024   | did not mix              |

SVI rms is the rms of the band centre over all seeds, except the loop rows,
which give the median over seeds: one untied loop seed with velocities is off by
1.5 and would swamp the rest.

The loop rows are the main finding. Every orderer tried (default MST | SOM, MST,
LocalFlow, MST | LocalFlow) turns at the self-crossing instead of crossing it,
on 7 or 8 of 8 seeds. The model starts there and stays. Two changes fix it:

- _Velocity mean tied to the tangent_ (`model(..., vel_tangent=True)`): the
  velocity mean is exp(v(γ)) times the unit tangent of the track, v a GP
  log-speed. A track that runs the wrong way round the loop misfits every
  velocity. From the bad start this rescues 5 of 8 seeds. On the helix it cuts
  the centre error from 0.032 to 0.019.
- _Velocity-weighted MST start_: `MSTOrderer(velocity_weight=2.0)` (off by
  default, and `default_pipeline` does not set it). It gets the crossing right
  on 8 of 8 loop seeds, with or without the SOM after it, and changes nothing on
  the helix. With it, 8 of 8 seeds come out right, tied or not; tied is better
  calibrated.

`velocity_weight` adds to `‖dq‖`, so it has position units. 2.0 suits this toy
(neighbour spacing about 0.1); a default would need to scale with the data.

Other findings:

- NUTS often fails to mix on this model: R̂ 2 to 3.5 on 5 of 14 runs. The two
  chains' median tracks differ by about the band width, so they sit in different
  modes; it is not a direction flip. On the mock stream with the latent line
  density and width it never mixed (R̂ 2.2 and 3.2, about 1.5 h each).
- The model's centre beats a single `fit_track` on images by 2 to 3 times.
- Image weights: SVI's band barely narrows when the weights are multiplied by 5
  (0.023 to 0.020), while NUTS on counts gives 0.010. SVI is too wide there. The
  weights act as counts, so `weights / max` widens the band as it should.
- Mock stream: positions only, a latent line density and a γ-dependent width
  were needed to reach the sparse tails. Coverage is against the same model
  fitted to all 8000 stars, so it checks consistency under subsampling, not
  bias.

## PR B checks: varying width and density, and the mock with tied velocities

`vary` case (`cases.py`): a sine arc of 800 stars, a dense clump near s = 0.35
and tails about 7 times sparser, width 0.06 + 0.3 s² (0.06 to 0.36). SVI, 8
seeds. "Tail" is the 68% coverage over s > 0.8.

| model                                   | 68% / 95%   | tail 68% | centre rms | width in 68% / 95% band | fitted / true width |
| --------------------------------------- | ----------- | -------- | ---------- | ----------------------- | ------------------- |
| fixed width, uniform γ                  | 0.72 / 0.91 | 0.31     | 0.029      | —                       | —                   |
| line density + varying width (`--v2`)   | 0.80 / 0.99 | 0.75     | 0.025      | 0.61 / 0.93             | 1.02 (median)       |
| `--v2` + velocities tied to the tangent | 0.90 / 0.99 | 0.84     | 0.018      | 0.55 / 0.91             | 1.03                |

One `fit_track` has a centre rms of 0.043 here. The two channels do what they
are for: the width profile comes back within a few per cent, and the sparse tail
is covered. The width band is somewhat narrow.

Mock stream with tied velocities: the subsets are consistent with their own
full-data reference (68% / 95% = 0.87 / 1.00 over 4 subsets, centre rms
0.006–0.017), but that reference is not the untied one. The two differ by 0.09
rms, rising to 0.3 near one end, against band half-widths of 0.02. The tied
reference is 21.5 long against 27.1, its stars sit further from it (median 0.097
against 0.072), and its tangent matches the stars' velocities to 0.6–1.1°
against 1–4° for the untied track. So with 8000 stars the velocity term wins:
the tie pulls the track onto the direction of the velocities and away from the
positions. In the toy cases the velocities are the tangent plus noise, so the
tie was exact there. In a real stream the stars' velocities are not parallel to
the track by a few degrees, and the hard tie biases the track.

The tie is still needed for the direction (the loop). A version that keeps the
direction and allows a small misalignment would add a free offset δ(γ) to the
velocity mean, exp(v) t̂ + δ, with a prior on δ of a few per cent of the speed:
running the wrong way still costs about twice the speed, while a few degrees of
misalignment costs almost nothing.

Tried as `model(..., vel_slack=0.05)` (`--slack`): the mean is exp(v) t̂ + 0.05
d(γ), d a GP per velocity component.

| full-data mock fit | track length | median star distance | median angle (v, track) | rms offset from untied |
| ------------------ | ------------ | -------------------- | ----------------------- | ---------------------- |
| untied             | 27.1         | 0.072                | 2.0°                    | —                      |
| hard tie           | 21.5         | 0.097                | 0.9°                    | 0.093                  |
| loose tie (0.05)   | 21.4         | 0.073                | 2.0°                    | 0.039                  |

The loose tie fits the positions as well as the untied model, and agrees with it
to 0.01–0.04 along most of the track (0.15 at the far end). Its subsets cover
their own reference at 0.98 / 1.00 (hw 0.020–0.022, 61–66 s). Both tied fits are
shorter: the central 99% of their line density ends earlier. Not looked into.

On the known-truth cases the loose tie keeps what the hard tie gave (8 seeds
each, velocity-aware default start):

| case          | hard tie: right, 68% / 95%, rms | loose tie: right, 68% / 95%, rms |
| ------------- | ------------------------------- | -------------------------------- |
| loop          | 8 of 8, 0.89 / 0.99, 0.016      | 8 of 8, 0.83 / 0.99, 0.020       |
| vary (`--v2`) | 8 of 8, 0.90 / 0.99, 0.014      | 8 of 8, 0.88 / 1.00, 0.019       |

The loop result here comes mostly from the start: with the velocity-aware
ordering of #270 the untied model already gets 8 of 8 (0.78 / 0.96).

## Not measured

- A known-truth test of the line-density and varying-width channels.
- The mock stream with velocities tied to the tangent.
- Scaling with N: the marginal likelihood costs N × K per evaluation (400 × 256
  here).

## The library's `sample_track` (PR B)

`validate_lib.py` and `run_lib.sh` run `pcf.sample_track` with its defaults
(background, line density, varying width, velocity tie 0.05) on the known-truth
cases, 8 seeds each, 200 draws. Loop uses `n_basis=32, n_grid=512`.

| case  | 68%  | 95%  | hw68  | centre rms | time |
| ----- | ---- | ---- | ----- | ---------- | ---- |
| helix | 0.75 | 0.97 | 0.025 | 0.022      | 29 s |
| gap   | 0.68 | 0.99 | 0.030 | 0.028      | 37 s |
| loop  | 0.85 | 0.98 | 0.026 | 0.018      | 42 s |
| vary  | 0.88 | 1.00 | 0.027 | 0.018      | 76 s |

Vary: width ratio 1.01, true width inside the draws' 68% at 0.63 of points, tail
(s > 0.8) 68% coverage 0.84. Times on a shared machine; the later vary seeds ran
alongside the test suite.
