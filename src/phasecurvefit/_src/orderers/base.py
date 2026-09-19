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

__all__: tuple[str, ...] = ("AbstractOrderer", "chord_along_ordering", "order")

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents


def chord_along_ordering(
    positions: VectorComponents, indices: Int[Array, " n_obs"]
) -> Float[Array, " n_obs"]:
    """Arc length along the ordered observations, in input order.

    The chord contract is "distance travelled along the curve this result
    represents". For an orderer whose curve *is* the path through the visited
    observations -- the walk, the MST backbone -- that is the cumulative
    distance between consecutive ordered points, which is what this computes.

    An orderer that fits a separate smooth curve and projects onto it should not
    use this: projecting is not the same as summing along the ordering, and such
    an orderer has already paid for the projection.

    Parameters
    ----------
    positions
        Phase-space positions, full arrays in input order.
    indices
        The result's ``indices``: visited observations in order, ``-1``-padded.

    Returns
    -------
    Array, shape (n_obs,)
        Arc length per observation in **input order**. Unvisited observations
        carry ``nan``, matching the ``chord`` field's contract.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phasecurvefit._src.orderers.base import chord_along_ordering

    >>> pos = {"x": jnp.array([0.0, 1.0, 2.0, 9.0]), "y": jnp.zeros(4)}
    >>> chord_along_ordering(pos, jnp.array([0, 1, 2, -1]))
    Array([ 0.,  1.,  2., nan], dtype=float32)

    """
    indices = jnp.asarray(indices)
    n_obs = indices.shape[0]
    valid = indices >= 0
    # Gather with the padding parked on index 0 rather than compacting: a
    # boolean mask is not traceable, and `order()` is called under `jit`.
    gather = jnp.where(valid, indices, 0)
    comps = sorted(positions)
    q = jnp.stack([jnp.asarray(positions[c])[gather] for c in comps], axis=-1)
    step = jnp.linalg.norm(jnp.diff(q, axis=0), axis=-1)
    # A step counts only between two genuinely visited observations, so the
    # parked padding contributes nothing to the cumulative length.
    step = jnp.where(valid[1:] & valid[:-1], step, 0.0)
    s = jnp.concatenate([jnp.zeros(1, dtype=step.dtype), jnp.cumsum(step)])
    # Scatter back to input order. Unvisited entries are sent to a scratch slot
    # past the end, which is then dropped, so they cannot clobber index 0.
    scatter = jnp.where(valid, indices, n_obs)
    out = jnp.full(n_obs + 1, jnp.nan, dtype=s.dtype).at[scatter].set(s)
    return out[:n_obs]


def _check_component_keys(
    positions: VectorComponents, velocities: VectorComponents
) -> None:
    """Reject mismatched component keys, naming both sides of the difference."""
    if set(positions) != set(velocities):
        missing = sorted(set(positions) - set(velocities))
        extra = sorted(set(velocities) - set(positions))
        msg = (
            "positions and velocities must have the same component keys; "
            f"missing={missing}, extra={extra}."
        )
        raise ValueError(msg)


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
        init: AbstractResult | None = None,
    ) -> AbstractResult:
        """Order the tracers and return a result the autoencoder can consume.

        ``init`` is an optional result from a previous ordering stage. Orderers
        that can refine a prior ordering (e.g. ``SOMOrderer``) use it; the rest
        accept and ignore it, so every orderer is chainable.
        """
        ...

    def __or__(self, other: "AbstractOrderer") -> "AbstractOrderer":
        """Compose two orderers into a :class:`ChainOrderer`."""
        # Lazy import: chain imports AbstractOrderer from this module.
        from phasecurvefit._src.orderers.chain import ChainOrderer  # noqa: PLC0415

        return ChainOrderer(self, other)


def order(
    positions: VectorComponents,
    velocities: VectorComponents,
    orderer: AbstractOrderer | None = None,
    *,
    metadata: StateMetadata | None = None,
    init: AbstractResult | None = None,
) -> AbstractResult:
    """Order tracers with ``orderer`` -- the primary ordering entry point.

    ``orderer`` defaults to :class:`~phasecurvefit.orderers.LocalFlowOrderer`, so
    ``order(positions, velocities)`` runs the velocity-following local-flow walk
    (equivalent to the deprecated ``walk_local_flow(positions, velocities)``).
    Pass any :class:`AbstractOrderer` (e.g. ``MSTOrderer``) to select a different
    algorithm.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> q = {"x": jnp.array([0.0, 1.0, 2.0])}
    >>> p = {"x": jnp.array([1.0, 1.0, 1.0])}

    The default orderer is the local-flow walk:

    >>> pcf.order(q, p).indices
    Array([0, 1, 2], dtype=int32)

    Or pass an orderer explicitly:

    >>> res = pcf.order(q, p, pcf.orderers.LocalFlowOrderer(metric_scale=1.0))
    >>> res.indices
    Array([0, 1, 2], dtype=int32)

    """
    if orderer is None:
        # Lazy import: localflow imports AbstractOrderer from this module.
        from phasecurvefit._src.orderers import localflow  # noqa: PLC0415

        orderer = localflow.LocalFlowOrderer()
    if init is None:
        # Omit ``init`` entirely so orderers predating it still work.
        return orderer.order(positions, velocities, metadata=metadata)
    return orderer.order(positions, velocities, metadata=metadata, init=init)
