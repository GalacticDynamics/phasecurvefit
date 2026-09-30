"""The library's default ordering pipeline: an MST backbone refined by a SOM.

A separate module rather than living in ``base``: this chains
:class:`~phasecurvefit.orderers.MSTOrderer` with
:class:`~phasecurvefit.orderers.SOMOrderer`, both of which import
``AbstractOrderer`` *from* ``base`` -- ``base`` importing them back at module
scope would cycle. Downstream of all three, this module can import both at the
top level instead of needing ``base``'s lazy, ``noqa: PLC0415`` imports.
"""

__all__: tuple[str, ...] = ("default_pipeline",)

import jax

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.orderers.mst import MSTOrderer
from phasecurvefit._src.orderers.som import SOMOrderer


def default_pipeline(
    positions: VectorComponents,
    velocities: VectorComponents,
    *,
    metadata: StateMetadata | None = None,
    init: AbstractResult | None = None,
    n_prototypes: int = 15,
    **som_kwargs: object,
) -> AbstractResult:
    """Order tracers with the library's default two-stage pipeline.

    Chains an MST backbone ordering with a SOM refinement stage --
    ``MSTOrderer(...) | SOMOrderer(n_prototypes=n_prototypes, **som_kwargs)``.
    This is what :func:`~phasecurvefit.order` runs when it is given no
    ``orderer``. The MST needs no progenitor or start index and orders a stream
    tip to tip, including near-closed loops whose velocity field reverses; the
    SOM's prototypes then average over many tracers, so the final backbone is
    far less sensitive to local noise than the MST's individual graph edges. See
    :doc:`/guides/som`.

    The MST stage is configured to work on data of any scale and to fail soft,
    not to be tuned: ``jump_cap=inf`` (the default ``jump_cap`` is an absolute
    length, which severs every edge of a sparse or large-scale dataset),
    ``orient_by_velocity=True`` (so ``gamma`` increases along the flow rather
    than in an arbitrary tip-to-tip direction), and ``on_disconnected="warn"``
    (order the largest connected piece and warn, rather than raise). For
    anything else -- a finite ``jump_cap``, velocity-aware edges, outlier
    clipping -- build the chain yourself and pass it to
    :func:`~phasecurvefit.order`.

    Falls back to the MST alone when fewer tracers were visited than
    ``n_prototypes`` -- the SOM's own minimum
    (:func:`~phasecurvefit.som.init_prototypes` needs at least that many
    points to bin). Rather than raise on the library's own small examples,
    the SOM stage simply cannot help there, so it is skipped instead of
    failing.

    Unlike its MST stage alone, this cannot run under ``jit`` or ``vmap``: the
    fallback test and the SOM stage both need the concrete visited count, and
    a SOM stage chained after another cannot be traced. To trace an ordering,
    pass :func:`~phasecurvefit.order` an explicit orderer (``MSTOrderer()`` and
    ``LocalFlowOrderer()`` both trace; see :doc:`/guides/jax-integration`).

    Parameters
    ----------
    positions, velocities
        Phase-space tracers, as for :func:`~phasecurvefit.order`.
    metadata
        Passed through to both stages.
    init
        A prior ordering result, passed to the first (MST) stage, which -- like
        any orderer that cannot refine a prior ordering -- accepts and ignores
        it.
    n_prototypes
        Passed to :class:`~phasecurvefit.orderers.SOMOrderer`; also the
        threshold below which the SOM stage is skipped.
    **som_kwargs
        Any other :class:`~phasecurvefit.orderers.SOMOrderer` keyword
        (``metric``, ``sigma_end``, ...).

    Returns
    -------
    AbstractResult
        An :class:`~phasecurvefit.orderers.OrderingResult`, from the SOM stage
        when there was enough data and from the MST otherwise. Both carry
        ``gamma_range == (-1.0, 1.0)``.

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

    Too few tracers for the SOM stage falls back to the MST alone rather
    than raising:

    >>> q = {"x": jnp.array([0.0, 1.0, 2.0])}
    >>> p = {"x": jnp.array([1.0, 1.0, 1.0])}
    >>> pcf.orderers.default_pipeline(q, p).indices
    Array([0, 1, 2], dtype=int32)

    """
    mst = MSTOrderer(
        jump_cap=float("inf"), orient_by_velocity=True, on_disconnected="warn"
    )
    mst_result = mst.order(positions, velocities, metadata=metadata, init=init)
    try:
        n_visited = int(mst_result.n_visited)
    except jax.errors.ConcretizationTypeError as exc:
        msg = (
            "The default ordering pipeline cannot run under jit or vmap: "
            "whether to run the SOM stage depends on the number of tracers "
            "visited, which is only known once the MST has run. Pass an "
            "explicit orderer instead, e.g. `pcf.orderers.MSTOrderer(...)` or "
            "`pcf.orderers.LocalFlowOrderer()`, both of which can be traced."
        )
        raise TypeError(msg) from exc
    if n_visited < n_prototypes:
        return mst_result
    som = SOMOrderer(n_prototypes=n_prototypes, **som_kwargs)
    return som.order(positions, velocities, metadata=metadata, init=mst_result)
