# Autoencoder for Gap Filling

The walk algorithm skips some tracers due to the momentum condition. This guide explains how to use an autoencoder to assign ordering values ($\gamma$) to these skipped tracers.

## Problem and Solution

**Problem**: walk inevitably skips tracers that don't align with the velocity direction.

**Solution**: An autoencoder with two networks:
- **Encoder**: $(x, v) \rightarrow (\gamma, p)$ — predicts ordering and membership probability
- **Decoder**: $\gamma \rightarrow x$ — reconstructs position from ordering

The encoder learns from the walk-ordered tracers and generalizes to predict $\gamma$ for skipped tracers.

## Quick Start

```python
import jax
import jax.numpy as jnp
import phasecurvefit as pcf

# Get initial ordering from walk
pos = {"x": jnp.linspace(0, 5, 50), "y": jnp.sin(jnp.linspace(0, jnp.pi, 50))}
vel = {"x": jnp.ones(50), "y": jnp.cos(jnp.linspace(0, jnp.pi, 50))}
walkresult = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

# Create normalizer and autoencoder
key = jax.random.key(0)
normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
autoencoder = pcf.nn.PathAutoencoder.make(normalizer, key=key)

# Train autoencoder
config = pcf.nn.TrainingConfig(show_pbar=False)
trained, _, losses = pcf.nn.train_autoencoder(
    autoencoder, walkresult, config=config, key=key
)

# Fill gaps
result = pcf.nn.fill_ordering_gaps(trained, walkresult)

gamma = result.gamma
ordered_all = result.indices
```

## How It Works

1. **Initialization**: Walk assigns $\gamma \in [-1, 1]$ to ordered tracers
2. **Phase 1**: Encoder learns to predict $\gamma$ from phase-space coordinates
3. **Phase 2**: Both networks train together with momentum constraint — ensures velocity alignment
4. **Membership**: Network outputs probability $p$ to distinguish stream from background

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

trained, _, losses = pcf.nn.train_autoencoder(
    autoencoder, walkresult, config=config, key=key
)
```

**Key parameters**:
- `lambda_p`: Higher maximum (100-150) enforces stronger velocity alignment in Phase 2
- `n_epochs_encoder`: Should be ~200-500 for good initial interpolation
- `batch_size`: Larger batches are more stable but require more memory
- `lambda_q`: Weight for spatial reconstruction loss in Phase 2
