---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

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
{class}`~phasecurvefit.nn.TrainingConfig`. Here the data are 200 points on an arc
with 10 interlopers scattered around it. The epoch counts are small to keep the
example fast; use the defaults, or more, on real data.

```{code-cell} python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

ang = jnp.linspace(0.0, jnp.pi, 200)
k_bg, k_model, k_train = jax.random.split(jax.random.key(0), 3)
bg = jax.random.uniform(k_bg, (2, 10), minval=-6.0, maxval=6.0)
pos = {
    "x": jnp.concatenate([5 * jnp.cos(ang), bg[0]]),
    "y": jnp.concatenate([5 * jnp.sin(ang), jnp.abs(bg[1])]),
}
vel = {
    "x": jnp.concatenate([-jnp.sin(ang), jnp.zeros(10)]),
    "y": jnp.concatenate([jnp.cos(ang), jnp.ones(10)]),
}

ordering = pcf.order(pos, vel)
model = pcf.nn.PathAutoencoder.make(
    pcf.nn.StandardScalerNormalizer(pos, vel),
    gamma_range=ordering.gamma_range,
    key=k_model,
)

membership = pcf.nn.MixtureMembershipConfig(
    sigma_init=0.1,  # your guess at the stream width, in normalised units
    sigma_ceiling=(0.5, 0.1),  # shrink the width ceiling during training
    warmup_frac=0.3,  # fit the track before rejecting anything
)
config = pcf.nn.TrainingConfig(
    n_epochs_encoder=100,
    n_epochs_decoder=50,
    n_epochs_both=200,
    membership=membership,
    show_pbar=False,
)
result, _, losses = pcf.nn.train_autoencoder(
    model, ordering, config=config, key=k_train
)
```

## Read the membership

Use {func}`~phasecurvefit.nn.posterior_membership` on the trained network,
`result.model`, not the encoder's raw probability. It takes the samples as one array
of positions and velocities in the normalised coordinates the network was trained
in:

```{code-cell} python
qn, pn = result.model.normalizer.transform(pos, vel)
ws = jnp.concatenate([qn, pn], axis=1)

q = pcf.nn.posterior_membership(result.model, ws)  # posterior, in [0, 1]
members = q > 0.5
print(q.shape)
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

With `membership` set, the encoder's $\gamma$ is also frozen during the joint phase,
because an encoder that keeps training can make the same collapse happen. Leave
`freeze_encoder_final_training` at its default ({doc}`../explanation/outliers`
explains why).

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

## Reject outliers in an image

For an image, order the pixels with their values as weights
([order a density field](choose-orderer.md#order-a-density-field)), then train with
the mixture and the two settings a wide curve needs ({doc}`autoencoder`):

```{code-cell} python
config = pcf.nn.TrainingConfig(
    freeze_encoder_final_training=True,
    batch_size=1000,
    membership=pcf.nn.MixtureMembershipConfig(),
    show_pbar=False,
)
```

Pass the weighted ordering to `train_autoencoder` as above, and it reads the weights
from there. A weight sets how much a pixel counts in the fit, not whether it is a
member, so the faint pixels at the edges of the curve stay members
({doc}`../explanation/density-fields` derives why).

On the density-field tutorial's image, with pixels of 0.1 and three bright blobs
added, ordered with `RidgeOrderer() | SOMOrderer(n_prototypes=30)` and trained with
these settings, thresholding the posterior at 0.5 kept every pixel on the curve. It
rejected every pixel of the two blobs 2.0 and 3.2 from the curve. The third blob's
centre is 0.59 from the curve, inside its half-width of 0.63 there, and the blob
stayed a member.
