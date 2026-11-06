---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Fill gaps and fit a track with the autoencoder

This guide shows how to give every sample a value of $\gamma$, including samples an
orderer left out, and fit a smooth mean track through them.
{doc}`../explanation/autoencoder` explains how the network is trained.

## In one call

{func}`~phasecurvefit.fit_track` runs the default pipeline, builds an autoencoder,
and trains it:

```{code-cell} python
import jax
import jax.numpy as jnp

import phasecurvefit as pcf

pos = {"x": jnp.linspace(0, 5, 50), "y": jnp.sin(jnp.linspace(0, jnp.pi, 50))}
vel = {"x": jnp.ones(50), "y": jnp.cos(jnp.linspace(0, jnp.pi, 50))}

config = pcf.nn.TrainingConfig(show_pbar=False)
ordering, result, losses = pcf.fit_track(
    pos, vel, key=jax.random.key(0), training_config=config
)
```

`ordering` is the default pipeline's result, `result` the trained
{class}`~phasecurvefit.nn.AutoencoderResult`, and `losses` the training loss per
epoch.

## Step by step

Build the pieces yourself to use a different orderer, a pre-built model, or another
decoder. `train_autoencoder` accepts any orderer's result:

```{code-cell} python
ordering = pcf.order(pos, vel)

key = jax.random.key(0)
normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
autoencoder = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=ordering.gamma_range, key=key
)

result, _, losses = pcf.nn.train_autoencoder(
    autoencoder, ordering, config=config, key=key
)
```

## Read the result

`result.gamma` holds $\gamma$ for every sample, and `result.membership_prob` the
encoder's membership probability. Calling the result with values of $\gamma$
evaluates the track. Draw it over `result.track_range`, which stops where the
samples thin out at each end:

```{code-cell} python
import matplotlib.pyplot as plt

track = result(jnp.linspace(*result.track_range, 200))

fig, ax = plt.subplots(figsize=(6, 3))
im = ax.scatter(pos["x"], pos["y"], c=result.gamma, s=15)
ax.plot(track["x"], track["y"], c="k", lw=1, label="track")
fig.colorbar(im, ax=ax, label=r"$\gamma$")
ax.legend();
```

Training runs in three phases with different objectives, so compare `losses` only
within a phase. The
[stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb) shows how to
read the loss curve and check the final model.

## Change the training settings

{class}`~phasecurvefit.nn.TrainingConfig` sets the epochs and loss weights of each
phase:

```{code-cell} python
config = pcf.nn.TrainingConfig(
    n_epochs_encoder=800,  # encoder only
    n_epochs_decoder=100,  # decoder only
    n_epochs_both=200,  # both networks together
    batch_size=100,
    lambda_prob=1.0,  # membership loss weight
    lambda_q=1.0,  # spatial reconstruction loss weight
    lambda_p=(1.0, 150.0),  # velocity alignment weight, ramped over phase 3
    show_pbar=False,
)
result, _, losses = pcf.nn.train_autoencoder(
    autoencoder, ordering, config=config, key=key
)
```

- `lambda_p` is the `(start, stop)` of the velocity-alignment ramp in the joint
  phase. A higher stop value, 100 to 150, enforces stronger alignment.
- `n_epochs_encoder` defaults to 800. A few hundred gives a quicker, rougher fit.
- `batch_size`: larger batches are more stable but use more memory.

{doc}`../reference/options` lists every setting with its default.

## Use a faster, training-free track

If speed matters more than accuracy, for example inside a likelihood evaluated at
every step of an MCMC, a {class}`~phasecurvefit.nn.RunningMeanDecoder` can replace
the trained decoder. The
[stream running-mean tutorial](../tutorials/stream_runningmean.ipynb) shows how, and
when it is accurate enough.

To reject interlopers while training, see {doc}`outliers`.
