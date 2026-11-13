# Refactor plan: a posterior for the track

Based on the prototypes in this folder; numbers are in `README.md`.

## What the validation changed

The earlier recommendation was to make the probabilistic track the backend of
`fit_track`. The validation supports the model but not that wiring:

- The model never uses the autoencoder. It starts from the orderer's ordering,
  and the ordering is what decides whether it succeeds (the loop).
- `fit_track` returns `(ordering, AutoencoderResult, losses)`, and the encoder
  that maps any star to γ is useful on its own. A `method=` switch that changes
  the return type would break every caller.
- `bootstrap_track` (#265) already gives a band as a `TrackEnsemble`, whose
  `offsets` and `radius` only call each member on γ. A posterior fits that
  shape: each draw is a track you call on γ.

So the plan is two PRs. `fit_track` stays as it is.

## PR A: velocity-aware default ordering (no NumPyro)

The loop failed because `default_pipeline` builds a position-only MST.
`MSTOrderer(velocity_weight=2.0)` fixed the crossing on 8 of 8 loop seeds and
changed nothing on the helix. This also fixes the point estimate: one
`fit_track` failed on 5 of 8 loop seeds.

1. `velocity_weight` adds `w (1 - cos)` to `‖dq‖`, so `w` has position units and
   no one default fits kpc and toy data. Add a unit-free form: multiply instead,
   `‖dq‖ (1 + w (1 - cos))`, under a new keyword (proposed `velocity_factor`),
   so the existing keyword keeps its meaning.
2. Turn it on in `default_pipeline`. Choose the value by checking the loop,
   helix, sine arc, gap and mock stream, and the docs tutorials' figures.
3. Tests: a small loop where `default_pipeline` must cross straight over; the
   factor with Quantities; jit/grad still trace. Migration note, and the
   `explanation/mst.md` and `how-to/choose-orderer.md` pages.

## PR B: `sample_track` (NumPyro, optional extra)

```python
post = pcf.sample_track(pos, vel, key=key)  # SVI by default
post = pcf.sample_track(pos, vel, key=key, method="nuts")
post.offsets(gamma)
post.radius(gamma)  # as TrackEnsemble
pcf.track_model  # the NumPyro model, to use inside a user's own model
```

- **Model** (`tracklik.model` made proper): HSGP track, γ summed out on a grid,
  Hogg, Bovy & Lang (2010) flat background with latent fraction. When velocities
  are given, their mean is speed × unit tangent of the track. That tie improved
  the helix centre error from 0.032 to 0.019 and rescued the loop.
- **Start**: the orderer's ordering (PR A's default), ranked γ by cumulative
  weight, least-squares basis fit.
- **Inference**: SVI with a full-rank Gaussian guide by default (12–60 s per
  run, calibrated on helix, gap, loop and mock). NUTS as an option. It warns
  when R̂ > 1.05, which happened on 5 of 14 runs.
- **Result**: `TrackPosterior`, holding the draws as callable tracks with their
  own γ range. `TrackEnsemble`'s `offsets` and `radius` are reused by typing
  their members as anything callable on γ, not by copying them.
- **Line density and γ-dependent width**: needed on the mock stream to reach the
  sparse tails. Not yet tested against a known truth, so they ship off by
  default until that test passes (below).
- **Weights**: documented as counts. Scaling them changes the band.
- **Units**: decided at PR B, from the unxt investigation (`unxt-numpyro.md` in
  the shared notes). Each position's mean and one length scale stay Quantities;
  the model sees only dimensionless ratios, stripped with `ustrip("", ...)`, and
  draws return Quantities. No unxt change needed.
- **Dependency**: `phasecurvefit[numpyro]`, imported lazily; `sample_track`
  raises a clear error without it.
- **Not doing**: a shared likelihood with `mixture_membership_loss` (they share
  only a Gaussian and `uniform_background_density`, which is reused); Bayesian
  NN decoders (NUTS did not mix at 338 weights); `AutoLaplaceApproximation`
  (silently zero width).
- **Tests**: weights equal repeated rows; SVI band covers a known arc; tied
  velocities fix a small loop; Quantity round trip; R̂ warning; missing-extra
  error.
- **Docs**: `how-to/track-uncertainty.md` gets a posterior section next to the
  bootstrap; `explanation/track-uncertainty.md` gets the model; API reference
  and migration note.

## Before PR B: two more checks (prototype)

1. A known-truth stream whose width and density vary along it, to test the line
   density and width channels.
2. The mock stream with the velocities tied to the tangent.

Large N: the likelihood costs N × K per step. 8000 stars × 512 grid points is
fine; for much larger catalogues SVI can subsample stars per step. That is left
until someone needs it.

## Decisions for you

1. A new `sample_track` beside `fit_track`, not a `fit_track` backend.
2. NumPyro as an optional extra, not a core dependency.
3. Units: Quantity means and scale, only dimensionless ratios stripped (decided
   at PR B).
4. A new unit-free `velocity_factor` keyword rather than redefining
   `velocity_weight`.
