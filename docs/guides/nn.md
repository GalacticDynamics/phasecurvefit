# Autoencoder for Gap Filling

The NN+p algorithm skips some tracers due to the momentum condition. This guide explains how to use an autoencoder to assign ordering values ($\gamma$) to these skipped tracers.

## Problem and Solution

**Problem**: NN+p inevitably skips tracers that don't align with the velocity direction.

**Solution**: An autoencoder with two networks:
- **Encoder**: $(x, v) \rightarrow (\gamma, p)$ — predicts ordering and membership probability
- **Decoder**: $\gamma \rightarrow x$ — reconstructs position from ordering

The encoder learns from the NN+p-ordered tracers and generalizes to predict $\gamma$ for skipped tracers.

## Quick Start

```python
import jax
import jax.numpy as jnp
import localflowwalk as lfw
import localflowwalk as lfw

# Get initial ordering from NN+p
pos = {"x": jnp.linspace(0, 5, 50), "y": jnp.sin(jnp.linspace(0, jnp.pi, 50))}
vel = {"x": jnp.ones(50), "y": jnp.cos(jnp.linspace(0, jnp.pi, 50))}
result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

# Train autoencoder
autoencoder = lfw.nn.Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=2)
config = lfw.nn.TrainingConfig(n_epochs=500)
trained, losses = lfw.nn.train_autoencoder(autoencoder, result, config=config)

# Fill gaps
ae_result = lfw.nn.fill_ordering_gaps(trained, result)

gamma = ae_result.gamma
ordered_all = ae_result.ordered_indices
```

## How It Works

1. **Initialization**: NN+p assigns $\gamma \in [-1, 1]$ to ordered tracers
2. **Phase 1**: Encoder learns to predict $\gamma$ from phase-space coordinates
3. **Phase 2**: Both networks train together with momentum constraint — ensures velocity alignment
4. **Membership**: Network outputs probability $p$ to distinguish stream from background

## Customizing Training

```python
config = lfw.nn.TrainingConfig(
    learning_rate=1e-3,  # Optimizer learning rate
    n_epochs=500,  # Total epochs
    batch_size=32,  # Batch size
    lambda_momentum=100.0,  # Momentum loss weight
    n_random_samples=100,  # Background samples
    phase1_epochs=200,  # Phase 1 duration
)

trained, losses = lfw.nn.train_autoencoder(autoencoder, result, config=config)
```

**Key parameters**:
- `lambda_momentum`: Higher (100-1000) enforces stronger velocity alignment
- `phase1_epochs`: Should be ~200-500 for good initial interpolation
- `n_random_samples`: More samples improve membership discrimination
