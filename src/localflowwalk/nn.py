r"""Autoencoder neural network for gap filling.

This module implements the autoencoder from Appendix A.2 of Nibauer et al. (2022)
for assigning $\gamma$ values to stream tracers that were skipped by NN+p.

The autoencoder consists of:
1. **Interpolation Network** (Encoder): Maps $(x, v) \to (\gamma, p)$
2. **Param-Net** (Decoder): Maps $\gamma \to x$

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import localflowwalk as lfw

>>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
>>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
>>> result = lfw.walk_local_flow(pos, vel, start_idx=0, lam=1.0)

>>> autoencoder = lfw.nn.Autoencoder(rngs=jax.random.PRNGKey(0), n_dims=2)
>>> cfg = lfw.nn.TrainingConfig(n_epochs=100)
>>> trained, losses = lfw.nn.train_autoencoder(autoencoder, result, config=cfg)

"""

__all__: tuple[str, ...] = (
    # Network components
    "InterpolationNetwork",
    "ParamNet",
    "Autoencoder",
    # Training
    "train_autoencoder",
    "TrainingConfig",
    # Results
    "AutoencoderResult",
    # Convenience functions
    "fill_ordering_gaps",
)

from ._src.autoencoder import (
    Autoencoder,
    AutoencoderResult,
    InterpolationNetwork,
    ParamNet,
    TrainingConfig,
    fill_ordering_gaps,
    train_autoencoder,
)
