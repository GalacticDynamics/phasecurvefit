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
from jaxtyping import Array, Float

from zeroth import zeroth

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
    weights: Float[Array, " n_obs"] | None = None,
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
    :doc:`/explanation/som`.

    The MST stage is configured to work on data of any scale and to lose
    nothing, not to be tuned: ``jump_cap=inf`` (the default ``jump_cap`` is an
    absolute length, which severs every edge of a sparse or large-scale
    dataset), ``velocity_factor=30`` (an edge between tracers moving in
    different directions costs up to 60 median kNN edge lengths more, so where a
    stream crosses itself the tree crosses straight over instead of turning
    onto the other branch), ``orient_by_velocity=True`` (so ``gamma`` increases
    along the flow rather than in an arbitrary tip-to-tip direction), and
    ``on_disconnected="connect"`` (a gap in the stream splits the neighbour
    graph; the pieces are joined along their shortest links instead of keeping
    only the largest). Because the MST used velocities, the SOM does too (see
    :class:`~phasecurvefit.orderers.SOMOrderer`). For anything else -- a finite
    ``jump_cap``, other velocity settings, outlier clipping, a different ``k``
    -- build the chain yourself and pass it to :func:`~phasecurvefit.order`.

    Falls back to the MST alone when fewer tracers were visited than
    ``n_prototypes`` -- the SOM's own minimum
    (:func:`~phasecurvefit.som.init_prototypes` needs at least that many
    points to bin). Rather than raise on the library's own small examples,
    the SOM stage simply cannot help there, so it is skipped instead of
    failing.

    It runs under ``jit``, ``vmap`` and ``grad`` (see :doc:`/how-to/jax`).
    Under ``jit`` and ``vmap`` the fallback test uses the number of tracers,
    which is the number this MST visits: it joins every piece and clips no
    edges.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers, as for :func:`~phasecurvefit.order`.
    metadata
        Passed through to both stages.
    weights
        Per-tracer weights, passed through to both stages: the SOM fits with
        them and the MST carries them on its result.
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
    # ``nan_policy="omit"``: the default entry point must not reject a
    # catalogue for lacking some radial velocities. A tracer with a missing
    # velocity joins the tree as a leaf and is skipped when orienting.
    mst = MSTOrderer(
        jump_cap=float("inf"),
        velocity_factor=30.0,
        orient_by_velocity=True,
        on_disconnected="connect",
        nan_policy="omit",
    )
    mst_result = mst.order(positions, velocities, metadata=metadata, weights=weights)
    try:
        n_visited = int(mst_result.n_visited)
    except jax.errors.ConcretizationTypeError:
        # Under ``jit``/``vmap`` the visited count has no value yet. This MST
        # joins every piece (``on_disconnected="connect"``) and clips no edges,
        # so it visits every tracer and the static count stands in for it.
        n_visited = len(zeroth(positions.values()))  # type: ignore[arg-type]  # components are 1-D
    if n_visited < n_prototypes:
        return mst_result
    # The MST used velocities, so the SOM follows them: it must omit a missing
    # one too, unless the caller says otherwise.
    som_kwargs.setdefault("nan_policy", "omit")
    som = SOMOrderer(n_prototypes=n_prototypes, **som_kwargs)  # type: ignore[arg-type]  # SOMOrderer validates its own keywords
    return som.order(
        positions, velocities, metadata=metadata, init=mst_result, weights=weights
    )
