r"""Autoencoder neural network for gap filling.

This module implements the autoencoder from Appendix A.2 of Nibauer et al.
(2022) for assigning $\gamma$ values to stream tracers that were skipped by
phase-flow walk.

The autoencoder consists of:
1. **Interpolation Network** (Encoder): Maps $(x, v) \to (\gamma, p)$
2. **Param-Net** (Decoder): Maps $\gamma \to x$

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf

>>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.zeros(20)}
>>> vel = {"x": jnp.ones(20), "y": jnp.zeros(20)}
>>> result = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

>>> keys = jax.random.split(jax.random.key(0), 2)
>>> normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
>>> model = pcf.nn.PathAutoencoder.make(
...     normalizer=normalizer, gamma_range=result.gamma_range, key=keys[0]
... )
>>> cfg = pcf.nn.TrainingConfig(n_epochs_both=100, show_pbar=False)
>>> result, *_ = pcf.nn.train_autoencoder(model, result, config=cfg, key=keys[1])

"""

__all__: tuple[str, ...] = (
    # Network components
    "OrderingNet",
    "TrackNet",
    "AbstractAutoencoder",
    "PathAutoencoder",
    "AbstractExternalDecoder",
    "EncoderExternalDecoder",
    "RunningMeanDecoder",
    # Training
    "train_autoencoder",
    "train_ordering_net",
    "TrainingConfig",
    "OrderingTrainingConfig",
    # Loss functions
    "encoder_loss",
    # Results
    "AutoencoderResult",
    # Convenience functions
    "fill_ordering_gaps",
    # Normalizers
    "AbstractNormalizer",
    "StandardScalerNormalizer",
)

from ._src.nn import (
    AbstractAutoencoder,
    AbstractExternalDecoder,
    AbstractNormalizer,
    AutoencoderResult,
    EncoderExternalDecoder,
    OrderingNet,
    OrderingTrainingConfig,
    PathAutoencoder,
    RunningMeanDecoder,
    StandardScalerNormalizer,
    TrackNet,
    TrainingConfig,
    encoder_loss,
    fill_ordering_gaps,
    train_autoencoder,
    train_ordering_net,
)
