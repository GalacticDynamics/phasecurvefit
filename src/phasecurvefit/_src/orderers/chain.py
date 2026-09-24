"""Sequential composition of orderers.

``ChainOrderer`` runs its stages in order, threading each stage's result into
the next as ``init``::

    pcf.order(q, p, MSTOrderer(k=10) | LocalFlowOrderer())

A stage must *accept* ``init`` to sit anywhere but the head, though it is free
to ignore the value. An orderer predating the parameter can still lead a chain
-- the head is called without ``init`` -- but raises ``TypeError`` if placed
after another stage, rather than silently discarding what came before.

``order`` must stay a plain method, not a ``plum`` dispatch: the chain forwards
its arguments untouched so each stage's own dispatch selects itself.
"""

__all__: tuple[str, ...] = ("ChainOrderer",)

from .base import AbstractOrderer
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents


class ChainOrderer(AbstractOrderer):
    """Run several orderers in sequence, threading results forward.

    Parameters
    ----------
    *stages
        The orderers to run, in order. Nested chains are flattened, so
        ``a | b | c`` yields one three-stage chain rather than a nest.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    >>> pos = {"x": jnp.linspace(0.0, 3.0, 4), "y": jnp.zeros(4)}
    >>> vel = {"x": jnp.ones(4), "y": jnp.zeros(4)}

    >>> chain = pcf.orderers.ChainOrderer(
    ...     pcf.orderers.LocalFlowOrderer(), pcf.orderers.LocalFlowOrderer()
    ... )
    >>> chain.order(pos, vel).indices
    Array([0, 1, 2, 3], dtype=int32)

    The ``|`` operator builds the same thing, and flattens:

    >>> chain = pcf.orderers.LocalFlowOrderer() | pcf.orderers.LocalFlowOrderer()
    >>> len(chain.stages)
    2

    """

    stages: tuple[AbstractOrderer, ...]

    def __init__(self, *stages: AbstractOrderer) -> None:
        flat = tuple(
            s
            for stage in stages
            for s in (stage.stages if isinstance(stage, ChainOrderer) else (stage,))
        )
        if not flat:
            msg = "ChainOrderer requires at least one stage."
            raise ValueError(msg)
        self.stages = flat

    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,
        init: AbstractResult | None = None,
    ) -> AbstractResult:
        """Run each stage in turn, feeding its result to the next as ``init``."""
        result = init
        for stage in self.stages:
            if result is None:
                # Omit ``init`` entirely so a stage written against v0.3.1's
                # ``order()`` still works as the head of a chain.
                result = stage.order(positions, velocities, metadata=metadata)
            else:
                result = stage.order(
                    positions, velocities, metadata=metadata, init=result
                )
        return result
