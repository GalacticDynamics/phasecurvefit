"""The pluggable orderer abstraction.

An *orderer* consumes phase-space tracers ``(positions, velocities)`` and returns
an :class:`~phasecurvefit._src.abstract_result.AbstractResult` (in practice an
:class:`~phasecurvefit._src.orderers.result.OrderingResult`) that the
autoencoder consumes unchanged.

Contract
--------
``order()`` is a one-shot, **host-side** preprocessing step: it is *not* required
to be jit/vmap-traceable (this lets ``MSTOrderer`` use plain NumPy/SciPy). It
returns arrays the AE consumes directly: plain ``jnp`` arrays for array inputs,
or unit-aware ``unxt`` Quantities when given Quantity inputs (via the ``unxt``
interop). ``indices`` stores visited observation indices in visit order as a
prefix of length ``n_visited``, with all remaining entries set to ``-1``;
``gamma_range`` is static.
"""

__all__: tuple[str, ...] = ("AbstractOrderer", "order")

import abc

import equinox as eqx

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents


class AbstractOrderer(eqx.Module):
    """Base class for ordering algorithms.

    Subclasses carry their own hyperparameters and implement :meth:`order`.
    """

    @abc.abstractmethod
    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,
    ) -> AbstractResult:
        """Order the tracers and return a result the autoencoder can consume."""
        ...


def order(
    positions: VectorComponents,
    velocities: VectorComponents,
    orderer: AbstractOrderer,
    *,
    metadata: StateMetadata | None = None,
) -> AbstractResult:
    """Functional façade mirroring ``walk_local_flow``: run ``orderer`` on data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> q = {"x": jnp.array([0.0, 1.0, 2.0])}
    >>> p = {"x": jnp.array([1.0, 1.0, 1.0])}
    >>> res = pcf.order(q, p, pcf.orderers.LocalFlowOrderer(metric_scale=1.0))
    >>> res.indices
    Array([0, 1, 2], dtype=int32)

    """
    return orderer.order(positions, velocities, metadata=metadata)
