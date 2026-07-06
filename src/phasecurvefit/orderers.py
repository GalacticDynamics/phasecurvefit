r"""Pluggable ordering algorithms for phasecurvefit.

An *orderer* turns phase-space tracers ``(positions, velocities)`` into an
ordered result the autoencoder consumes. All orderers share the
:class:`AbstractOrderer` interface and return an :class:`OrderingResult`, so they
are interchangeable at call sites::

    import phasecurvefit as pcf

    orderer = pcf.orderers.LocalFlowOrderer(metric_scale=1.0)
    result = orderer.order(qs, ps)  # or pcf.order(qs, ps, orderer)

Built-in orderers
-----------------
LocalFlowOrderer
    Velocity-following greedy walk (wraps :func:`~phasecurvefit.walk_local_flow`).
MSTOrderer
    MST longest-path backbone ordering for near-closed-loop / self-overlapping
    streams. Host-side (NumPy/SciPy).

See Also
--------
phasecurvefit.order : functional façade mirroring ``walk_local_flow``.

"""

__all__: tuple[str, ...] = (
    "AbstractOrderer",
    "LocalFlowOrderer",
    "MSTOrderer",
    "OrderingResult",
)

from ._src.orderers.base import AbstractOrderer
from ._src.orderers.localflow import LocalFlowOrderer
from ._src.orderers.mst import MSTOrderer
from ._src.orderers.result import OrderingResult
