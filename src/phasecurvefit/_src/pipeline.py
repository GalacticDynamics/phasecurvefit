"""One-call, end-to-end path fitting: order, refine, and train.

Collapses the multi-step recipe in the quickstart and autoencoder guides
(:func:`~phasecurvefit.orderers.default_pipeline` feeding a
:class:`~phasecurvefit.nn.PathAutoencoder`) into a single function, for
callers who want a fitted track without assembling the steps themselves.
"""

__all__: tuple[str, ...] = ("fit_track",)

import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.nn import (
    AutoencoderResult,
    PathAutoencoder,
    StandardScalerNormalizer,
    TrainingConfig,
    train_autoencoder,
)
from phasecurvefit._src.orderers.default_pipeline import default_pipeline


def fit_track(
    positions: VectorComponents,
    velocities: VectorComponents,
    *,
    key: PRNGKeyArray,
    metadata: StateMetadata | None = None,
    n_prototypes: int = 15,
    training_config: TrainingConfig | None = None,
) -> tuple[AbstractResult, AutoencoderResult, Float[Array, " n_epochs"]]:
    """Order, refine, and fit a smooth track through phase-space tracers.

    Runs :func:`~phasecurvefit.orderers.default_pipeline` (the velocity-flow
    walk refined by a SOM stage), builds a
    :class:`~phasecurvefit.nn.PathAutoencoder` sized to match, and trains it
    with :func:`~phasecurvefit.nn.train_autoencoder` -- the same three steps
    as the quickstart and autoencoder guides, in one call.

    For control over any individual step (a different orderer, a
    pre-built model, a decoder swap, ...), call the pieces directly instead;
    this is the convenience path, not a replacement for them.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers.
    key
        Split once for model initialization and once for training.
    metadata
        Passed through to :func:`~phasecurvefit.orderers.default_pipeline`.
    n_prototypes
        Passed to :func:`~phasecurvefit.orderers.default_pipeline`.
    training_config
        Passed to :func:`~phasecurvefit.nn.train_autoencoder`. ``None`` uses
        its own defaults.

    Returns
    -------
    ordering_result : AbstractResult
        What :func:`~phasecurvefit.orderers.default_pipeline` produced --
        useful on its own for diagnosing the ordering stage independently of
        the fit.
    autoencoder_result : AutoencoderResult
        The trained result: call it with a ``gamma`` value to interpolate
        along the fitted track, or read ``.model`` for the network itself.
    losses : Array
        Training losses, for diagnosing convergence.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> ang = jnp.linspace(0.0, jnp.pi, 60)
    >>> pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    >>> vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    >>> cfg = pcf.nn.TrainingConfig(
    ...     n_epochs_encoder=5, n_epochs_decoder=5, n_epochs_both=5, show_pbar=False
    ... )
    >>> ordering, fitted, losses = pcf.fit_track(
    ...     pos, vel, key=jax.random.key(0), n_prototypes=12, training_config=cfg
    ... )
    >>> int(ordering.n_visited)
    60
    >>> fitted.model.normalizer.n_spatial_dims
    2

    """
    key_model, key_train = jr.split(key)
    ordering_result = default_pipeline(
        positions, velocities, metadata=metadata, n_prototypes=n_prototypes
    )
    normalizer = StandardScalerNormalizer(positions, velocities)
    model = PathAutoencoder.make(
        normalizer=normalizer, gamma_range=ordering_result.gamma_range, key=key_model
    )
    autoencoder_result, _opt_state, losses = train_autoencoder(
        model, ordering_result, config=training_config, key=key_train
    )
    return ordering_result, autoencoder_result, losses
