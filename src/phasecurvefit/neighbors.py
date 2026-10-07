"""Exact k-nearest-neighbour backends, selectable per orderer.

``BucketKDTree`` (default) and ``BruteForce`` are JAX-native and trace under
``jax.jit``/``vmap``/``grad``; ``JaxKD`` wraps the optional jaxkd package;
``SciPy`` is scipy's ``cKDTree`` -- fastest on CPU, eager-only::

    import phasecurvefit as pcf

    orderer = pcf.orderers.MSTOrderer(neighbors=pcf.neighbors.SciPy())
"""

__all__: tuple[str, ...] = (
    "AbstractNeighborSearch",
    "BruteForce",
    "BucketKDTree",
    "JaxKD",
    "SciPy",
)

from ._src.neighbors import (
    AbstractNeighborSearch,
    BruteForce,
    BucketKDTree,
    JaxKD,
    SciPy,
)
