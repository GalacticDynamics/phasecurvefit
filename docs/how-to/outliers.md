# Reject outliers with the mixture model

This guide shows how to train the autoencoder with a stream-plus-background mixture
model, and read a calibrated membership probability for each sample.
{doc}`../explanation/outliers` explains why the default membership cannot reject
outliers and how the mixture model fixes it. The
[outlier-rejection tutorial](../tutorials/outlier_rejection.ipynb) runs it end to end
on a mock stream with injected interlopers.

## Turn it on

Membership modelling is opt-in: with `membership=None`, the default, training is
unchanged. Pass a {class}`~phasecurvefit.nn.MixtureMembershipConfig` in the
{class}`~phasecurvefit.nn.TrainingConfig`:

<!-- skip: next -->
```python
import phasecurvefit as pcf

membership = pcf.nn.MixtureMembershipConfig(
    sigma_init=0.1,  # your guess at the stream width, in normalised units
    sigma_ceiling=(0.5, 0.1),  # shrink the width ceiling during training
    warmup_frac=0.3,  # fit the track before rejecting anything
)

config = pcf.nn.TrainingConfig(n_epochs_both=2000, membership=membership)
result, model, *_ = pcf.nn.train_autoencoder(model, ordering, config=config, key=key)
```

## Read the membership

Use {func}`~phasecurvefit.nn.posterior_membership`, not the encoder's raw
probability:

<!-- skip: next -->
```python
q = pcf.nn.posterior_membership(model, ws)  # posterior probability, in [0, 1]
members = q > 0.5
```

The encoder's probability is only the prior: what it believes from a sample's
coordinates before seeing how far the sample landed from the track.
`posterior_membership` folds in that distance. Hogg, Bovy & Lang (2010) recommend
keeping the posterior as a weight rather than thresholding it.

## Keep the two schedules

Both settings below close off a way the fit can fail. Keep them on.

- **`sigma_ceiling=(start, stop)`** caps the stream width from above, with a ceiling
  that shrinks during training. Without it, outliers a few widths from the track are
  absorbed by widening the stream instead of being rejected.
- **`warmup_frac`** fits the track first, with every sample held as a member, and
  only then lets the model reject samples. Without it, training can declare the
  whole dataset background and never recover.

## What to expect

On a synthetic arc of 400 members with width $\sigma = 0.15$, plus 30 outliers
displaced across the track, thresholding the posterior at 0.5 gave:

| outlier offset | caught (no warm-up) | members lost | caught (warm-up 0.3) | members lost |
| -------------: | ------------------: | -----------: | -------------------: | -----------: |
|           3 σ  |              0 / 30 |            0 |               0 / 30 |            0 |
|           5 σ  |              2 / 30 |            0 |              21 / 30 |            0 |
|          10 σ  |             30 / 30 |            0 |              30 / 30 |            0 |
|          20 σ  |             30 / 30 |          400 |              30 / 30 |            0 |

- With the warm-up, no genuine members were lost at any offset.
- Catching none at 3σ is correct: a sample 3σ from the track is consistent with
  membership, since the stream has Gaussian tails.
- Without the warm-up, the 20σ case lost every member.

The fitted width came out at 0.150, against a true 0.15.

To clip outliers while ordering, before any training, use the MST's
`edge_clip_sigma` or the SOM's `outlier_clip_sigma` ({doc}`som`).
