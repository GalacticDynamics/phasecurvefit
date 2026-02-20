r"""Autoencoder Neural Network for interpolating skipped tracers.

This module implements the autoencoder neural network from Appendix A.2 of
Nibauer et al. (2022) for assigning $\gamma$ values to stream tracers that were
skipped by the phase-flow walk algorithm.

The autoencoder consists of two parts:

1. **Interpolation Network**: Maps phase-space coordinates $(x, v) \to (\gamma,
   p)$ where $\gamma \in [0, 1]$ is the ordering parameter and $p \in [0, 1]$
   is the membership probability.
2. **Param-Net (Decoder)**: Maps $\gamma \to x$, reconstructing the position
   from the ordering parameter.

Training follows a two-step process:

1. Train the interpolation network on ordered tracers from phase-flow walk.
2. Jointly train both networks with a momentum condition to refine $\gamma$
   values.

References
----------
Nibauer et al. (2022). "Charting Galactic Accelerations with Stellar Streams
and Machine Learning." Appendix A.2.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf

Create phase-space data and run phase-flow walk:

>>> pos = {"x": jnp.linspace(0, 5, 20), "y": jnp.sin(jnp.linspace(0, jnp.pi, 20))}
>>> vel = {"x": jnp.ones(20), "y": jnp.cos(jnp.linspace(0, jnp.pi, 20))}
>>> walkresult = pcf.walk_local_flow(pos, vel, start_idx=0, metric_scale=1.0)

Train autoencoder to interpolate skipped tracers:

>>> keys = jax.random.split(jax.random.key(0), 2)
>>> normalizer = pcf.nn.StandardScalerNormalizer(pos, vel)
>>> model = pcf.nn.PathAutoencoder.make(
...     normalizer, gamma_range=walkresult.gamma_range, key=keys[0]
... )
>>> cfg = pcf.nn.TrainingConfig(n_epochs_both=100, show_pbar=False)
>>> result, *_ = pcf.nn.train_autoencoder(model, walkresult, config=cfg, key=keys[1])

>>> list(result.positions.keys())
['x', 'y']
>>> list(result.velocities.keys())
['x', 'y']

"""

from .abstractautoencoder import *
from .autoencoder import *
from .externalautoencoder import *
from .externaldecoder import *
from .normalize import *
from .order_net import *
from .result import *
from .scanmlp import *
from .track_net import *
from .utils import *
