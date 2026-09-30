"""The library's recommended, non-breaking default ordering pipeline.

A separate module rather than living in ``base``: this chains
:class:`~phasecurvefit.orderers.LocalFlowOrderer` with
:class:`~phasecurvefit.orderers.SOMOrderer`, both of which import
``AbstractOrderer`` *from* ``base`` -- ``base`` importing them back at module
scope would cycle. Downstream of all three, this module can import both at the
top level instead of needing ``base``'s lazy, ``noqa: PLC0415`` imports.
"""

__all__: tuple[str, ...] = ("default_pipeline",)

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.orderers.localflow import LocalFlowOrderer
from phasecurvefit._src.orderers.som import SOMOrderer


def default_pipeline(
    positions: VectorComponents,
    velocities: VectorComponents,
    *,
    metadata: StateMetadata | None = None,
    n_prototypes: int = 15,
    **som_kwargs: object,
) -> AbstractResult:
    """Order tracers with the library's recommended two-stage pipeline.

    Chains the velocity-following local-flow walk with a SOM refinement
    stage -- ``LocalFlowOrderer() | SOMOrderer(n_prototypes=n_prototypes,
    **som_kwargs)``. The SOM's prototypes average over many tracers, so the
    backbone it produces is far less sensitive to local noise than the walk's
    step-by-step decisions alone; see :doc:`/guides/som`.

    This is *not* what :func:`order` runs by default: making the SOM part of
    the primary entry point's default would be a breaking change (a different
    result type, a different ``gamma_range``, and a real per-call cost --
    `GalacticDynamics/phasecurvefit#57
    <https://github.com/GalacticDynamics/phasecurvefit/issues/57>`_). This
    function is the non-breaking alternative: call it explicitly to get the
    better default without touching :func:`order`'s own.

    Falls back to the walk alone when fewer tracers were visited than
    ``n_prototypes`` -- the SOM's own minimum
    (:func:`~phasecurvefit.som.init_prototypes` needs at least that many
    points to bin). Rather than raise on the library's own small examples,
    the SOM stage simply cannot help there, so it is skipped instead of
    failing.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers, as for :func:`order`.
    metadata
        Passed through to both stages.
    n_prototypes
        Passed to :class:`~phasecurvefit.orderers.SOMOrderer`; also the
        threshold below which the SOM stage is skipped.
    **som_kwargs
        Any other :class:`~phasecurvefit.orderers.SOMOrderer` keyword
        (``metric``, ``sigma_end``, ...).

    Returns
    -------
    AbstractResult
        An :class:`~phasecurvefit.orderers.OrderingResult` from the SOM stage
        when there was enough data, otherwise the walk's own
        :class:`~phasecurvefit.WalkLocalFlowResult`. The two carry different
        ``gamma_range``: ``(0, 1)`` for the bare walk, ``(-1, 1)`` after the
        SOM -- callers reading ``gamma_range`` off the result rather than
        assuming a fixed one are unaffected by which stage actually ran.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> ang = jnp.linspace(0.0, jnp.pi, 60)
    >>> pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    >>> vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    >>> result = pcf.orderers.default_pipeline(pos, vel, n_prototypes=12)
    >>> int(result.n_visited)
    60

    Too few tracers for the SOM stage falls back to the walk alone rather
    than raising:

    >>> q = {"x": jnp.array([0.0, 1.0, 2.0])}
    >>> p = {"x": jnp.array([1.0, 1.0, 1.0])}
    >>> pcf.orderers.default_pipeline(q, p).gamma_range
    (0.0, 1.0)

    """
    walk_result = LocalFlowOrderer().order(positions, velocities, metadata=metadata)
    if int(walk_result.n_visited) < n_prototypes:
        return walk_result
    som = SOMOrderer(n_prototypes=n_prototypes, **som_kwargs)
    return som.order(positions, velocities, metadata=metadata, init=walk_result)
