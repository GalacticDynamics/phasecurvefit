# Autoencoder for Gap Filling

An orderer can leave tracers unvisited — the local-flow walk skips some because of its momentum condition, and the default MST | SOM pipeline leaves out any the MST graph does not connect. This guide explains how to use an autoencoder to assign ordering values ($\gamma$) to these skipped tracers, and to fit a smooth track through the lot.

```{note}
The examples below start from the default pipeline, but `train_autoencoder` accepts **any**
orderer's output — it dispatches on the unified
{class}`~phasecurvefit.orderers.OrderingResult`. An
{class}`~phasecurvefit.orderers.MSTOrderer` result feeds the autoencoder the same
way (see the [Orderers guide](orderers.md)); the MST even supplies a `backbone`
that the decoder can be trained against.
```

## Problem and Solution

**Problem**: an ordering is not yet a smooth track, and may be incomplete. The local-flow walk
follows a single thread through the data, so it leaves out
many tracers: those off to the side of its path, and everything beyond a gap
larger than `max_dist`. On the simulated stream in the
[stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb), the walk
orders 248 of 8000 stars. It also gives only an ordering, not a smooth track
through the data.

**Solution**: An autoencoder with two networks:
- **Encoder**: $(x, v) \rightarrow (\gamma, p)$ — predicts ordering and membership probability
- **Decoder**: $\gamma \rightarrow x$ — reconstructs position from ordering

The encoder learns from the ordered tracers and generalizes to predict $\gamma$ for skipped tracers.
The decoder gives the smooth mean track $x(\gamma)$, which you can evaluate at
any $\gamma$. If training time matters more than accuracy, a running-mean
decoder can replace the trained one; see the
[stream running-mean tutorial](../tutorials/stream_runningmean.ipynb).

## Quick Start

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

# Get an initial ordering from the default pipeline (MST backbone + SOM)
pos = {"x": jnp.linspace(0, 5, 50), "y": jnp.sin(jnp.linspace(0, jnp.pi, 50))}
vel = {"x": jnp.ones(50), "y": jnp.cos(jnp.linspace(0, jnp.pi, 50))}
ordering = pcf.order(pos, vel)

# Create normalizer and autoencoder
key = jax.random.key(0)
normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
autoencoder = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=ordering.gamma_range, key=key
)

# Train autoencoder
config = pcf.nn.TrainingConfig(show_pbar=False)
result, _, losses = pcf.nn.train_autoencoder(
    autoencoder, ordering, config=config, key=key
)

gamma = result.gamma
ordered_all = result.indices
```

These four steps -- order, normalize, build, train -- collapse into one call
via {func}`~phasecurvefit.fit_track`:

```python
_fast_config = pcf.nn.TrainingConfig(  # to make the examples fast.
    n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
)
ordering, result, losses = pcf.fit_track(
    pos, vel, key=jax.random.key(0), training_config=_fast_config
)
```

Use the pieces directly, as above, for control over any individual step --
a different orderer, a pre-built model, a decoder swap.

## How It Works

1. **Initialization**: The orderer assigns $\gamma$ (over its `gamma_range`, $[-1, 1]$ for the default pipeline) to ordered tracers
2. **Phase 1 (encoder)**: Encoder learns to predict $\gamma$ from phase-space
   coordinates, and a membership probability $p$ to distinguish stream from
   background
3. **Phase 2 (decoder)**: With the encoder frozen, the decoder is fit to a running
   mean of the ordered member tracers' positions — a warm start for the track
4. **Phase 3 (joint)**: Both networks train together on spatial reconstruction plus
   velocity alignment, with the alignment weight ramped linearly from
   `lambda_p[0]` to `lambda_p[1]` over the phase

Each phase optimizes a different objective, so the concatenated `losses` returned
by `train_autoencoder` can only be compared within a phase, and the returned model
holds the final-epoch weights. The
[stream autoencoder tutorial](../tutorials/stream_autoencoder.ipynb) walks through
reading the loss curve and checking the final model with a fixed measure.

## Customizing Training

The default settings appear to work for most cases,
but can be set by the user.

```python
config = pcf.nn.TrainingConfig(
    n_epochs_encoder=800,  # Encoder-only epochs
    n_epochs_decoder=100,  # Decoder-only epochs
    n_epochs_both=200,  # En+Decoder epochs
    batch_size=100,  # Batch size for training
    lambda_prob=1.0,  # Probability loss weight
    lambda_q=1.0,  # Spatial reconstruction loss weight
    lambda_p=(1.0, 150.0),  # Velocity alignment loss weight range
    show_pbar=False,
)

result, _, losses = pcf.nn.train_autoencoder(
    autoencoder, ordering, config=config, key=key
)
```

**Key parameters**:
- `lambda_p`: `(start, stop)` of the Phase 3 alignment ramp; a higher `lambda_p[1]` (100-150) enforces stronger velocity alignment
- `n_epochs_encoder`: Default 800; fewer (a few hundred) gives a quicker, rougher fit
- `batch_size`: Larger batches are more stable but require more memory
- `lambda_q`: Weight for spatial reconstruction loss in Phase 3
