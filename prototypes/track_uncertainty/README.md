# Track uncertainty: what was measured

Prototypes only, not library code. `common.py` makes the mock data and measures
coverage. `run_dataset.py` runs methods A, B and C on one dataset.
`run_interlopers.py` runs the bootstrap on data with interlopers. `analyze.py`
prints the summary. `run_shipped.py` and `analyze_shipped.py` measure
`pcf.bootstrap_track` as it ships (see the last results section).

## Setup

- True track: a sine-shaped arc, x = 10 s, y = 2 sin(1.5 π s), s in [0, 1]. 400
  samples, uniform in arc length, Gaussian scatter across the track of width σ =
  0.2, velocities along the tangent with noise 0.1.
- 12 independent datasets (seeds 0–11). Coverage is a frequency over datasets,
  so one dataset can't measure it.
- Each sampled track is compared with the truth at 46 points along it (s from
  0.05 to 0.95). At each point I take the sampled track's signed offset along
  the true normal. This doesn't depend on each fit's own γ, which differs from
  fit to fit. A band covers the truth when offset 0 lies between its 16th and
  84th percentiles (68%), or between its 2.5th and 97.5th (95%).
- `fit_track` with default `TrainingConfig` (800 + 100 + 200 epochs), no mixture
  membership. Apple-silicon CPU, 4 runs in parallel, so the times are per
  process under load.

## Methods

- **A, bootstrap of the whole pipeline.** Each member refits MST → SOM →
  autoencoder on a multinomial resample (`som.bootstrap_weights` passed as
  `weights`) with a fresh training key. 20 members. Passing weights turned off
  `fit_track`'s outlier clipping on this version of the pipeline, so A is not
  quite what `bootstrap_track` does.
- **B, seed ensemble (deep ensemble).** Same data, a fresh training key per
  member. The ordering is deterministic, so only training randomness varies. 20
  members.
- **C, last-layer Laplace** around member 0 of B. Gaussian posterior over the
  decoder's final linear layer, with γ fixed at the encoder's values, noise
  variance from the member residuals, and prior precision chosen by the
  evidence. For a linear-Gaussian last layer this posterior is exact, so
  last-layer HMC would give the same answer. 200 draws.

## Results, clean arc (no interlopers)

| method               | 68% coverage | 95% coverage | 68% half-width | 95% half-width | cost                    |
| -------------------- | ------------ | ------------ | -------------- | -------------- | ----------------------- |
| A bootstrap          | 0.59         | 0.85         | 0.028          | 0.053          | 13.7 s per member       |
| B seed ensemble      | 0.36         | 0.56         | 0.015          | 0.028          | 8.8 s per member        |
| C last-layer Laplace | 0.38         | 0.66         | 0.018          | 0.036          | 1.9 s on top of one fit |

The error of a single fit is 0.037 rms, which is σ/5.4. Coverage is about the
same at the ends (first and last 20% of the grid) as in the middle for all three
methods.

The bootstrap gets the variance right. Its spread at each point (mean 0.031)
matches the scatter of the single fit across the 12 datasets (mean 0.028). What
it misses is bias. The single fit's mean offset from the truth has rms 0.023
along the track, close to the spread. It is largest at the bends (+0.022 at s =
0.35, +0.046 at s = 0.83) and at the far tip (−0.076 at s = 0.95), and that is
where coverage drops to 0.2–0.3. A resample of a smoothed fit is smoothed in the
same way, so a bootstrap cannot see this bias. The reflected ("basic") bootstrap
interval was worse, at 0.44 and 0.70, and θ̂ ± z·sd gave 0.59 and 0.89.

B and C see only part of the variance. Training randomness alone (B) gives about
30% of the bootstrap's variance, and C gives a similar amount. C also leaves out
the uncertainty in γ and in the hidden-layer features.

## Results, arc with 4% interlopers

Same arc plus 16 interlopers uniform over the field. The bootstrap resamples
indices instead of passing weights, because `fit_track` turns off its outlier
clipping when it is given weights. 12 datasets, 20 members, 9.3 s per member.

|                                | 68% coverage | 95% coverage | 68% half-width | 95% half-width |
| ------------------------------ | ------------ | ------------ | -------------- | -------------- |
| A bootstrap (index resampling) | 0.77         | 0.97         | 0.048          | 0.091          |

The bands are wider, because which interlopers are drawn and clipped changes
from member to member. The bias is still there (band-centre rms 0.045). The
bands are wide enough to cover it here, except at the far tip (0.3 at s = 0.95).
This run changes two things at once: interlopers, and index resampling with
clipping on. So it doesn't separate their effects.

## Results, the shipped `bootstrap_track`

Re-run after rebasing on #268, which made `fit_track` use `SOMMixture()` by
default. `run_shipped.py` calls
`pcf.bootstrap_track(pos, vel, key=key(seed), n_samples=20)` with the default
`TrainingConfig`, on the same 12 clean datasets. It scores the members as above
(the per-point interval), and also scores the tube that `TrackEnsemble.radius`
returns: covered when the reference's offset from the truth along the true
normal is at most the radius at the reference point nearest the true point.

|                    | 68% coverage | 95% coverage | 68% width      | 95% width      |
| ------------------ | ------------ | ------------ | -------------- | -------------- |
| per-point interval | 0.56         | 0.84         | 0.027 (half)   | 0.048 (half)   |
| tube from `radius` | 0.58         | 0.88         | 0.032 (radius) | 0.057 (radius) |

8.6 s per fit (4 processes at once). The radius was never NaN at the 46 points,
which lie inside s = 0.05–0.95. The reference's error is 0.038 rms. The members'
spread (0.028) matches the scatter of the reference across datasets (0.030). The
bias has rms 0.022 and is largest at the bend near s = 0.83 (+0.047) and at the
far tip (−0.074 at s = 0.95), where 68% coverage is 0.25–0.33. The conclusions
above hold.

The 0.62/0.89 tube coverage quoted in an earlier draft of the docs had no script
behind it and is replaced by this table.

## Not measured

- The mixture-membership path and the real mock stream in
  `docs/tutorials/mockstream.npz`. It has no known true track, so a truth would
  have to come from a fit to all 8000 stars.
- VI, MC dropout, SWAG, full-network HMC and GP decoders. See
  `docs/explanation/track-uncertainty.md` for why.
