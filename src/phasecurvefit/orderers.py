r"""Pluggable ordering algorithms for phasecurvefit.

An *orderer* turns phase-space tracers ``(positions, velocities)`` into an
ordered result the autoencoder consumes. All orderers share the
:class:`AbstractOrderer` interface and return an :class:`OrderingResult`, so they
are interchangeable at call sites::

    import phasecurvefit as pcf

    orderer = pcf.orderers.LocalFlowOrderer(metric_scale=1.0)
    result = orderer.order(qs, ps)  # or pcf.order(qs, ps, orderer)

With no orderer, :func:`phasecurvefit.order` runs :func:`default_pipeline`: an
MST backbone refined by a SOM, ``MSTOrderer() | SOMOrderer()``.

Built-in orderers
-----------------
MSTOrderer
    MST longest-path backbone ordering for near-closed-loop / self-overlapping
    streams. Host-side graph algorithms (NumPy/SciPy).
SOMOrderer
    Self-Organizing Map refinement: trains a 1-D SOM and orders by arc-length
    projection onto its backbone. Cite Starkman et al. (2023).
LocalFlowOrderer
    Velocity-following greedy walk.
    Momentum-weighted ordering: cite Nibauer et al. (2022).
ChainOrderer
    Runs orderers in sequence, threading each result into the next as
    ``init``. Also built by ``a | b``.

See Also
--------
phasecurvefit.order : the primary entry point; defaults to ``default_pipeline``.

"""

__all__: tuple[str, ...] = (
    "AbstractOrderer",
    "ChainOrderer",
    "LocalFlowOrderer",
    "MSTOrderer",
    "OrderingResult",
    "SOMOrderer",
    "default_pipeline",
)

from ._src.orderers.base import AbstractOrderer
from ._src.orderers.chain import ChainOrderer
from ._src.orderers.default_pipeline import default_pipeline
from ._src.orderers.localflow import LocalFlowOrderer
from ._src.orderers.mst import MSTOrderer
from ._src.orderers.result import OrderingResult
from ._src.orderers.som import SOMOrderer
