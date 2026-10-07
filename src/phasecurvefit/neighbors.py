"""Exact k-nearest-neighbour backends, selectable per orderer.

``BucketKDTree`` (default) and ``BruteForce`` are JAX-native and trace under
``jax.jit``/``vmap``/``grad``; ``Jaxkd`` wraps the optional jaxkd package;
``Scipy`` is scipy's ``cKDTree`` -- fastest on CPU, eager-only::

    import phasecurvefit as pcf

    orderer = pcf.orderers.MSTOrderer(neighbors=pcf.neighbors.Scipy())
"""

__all__: tuple[str, ...] = (
    "AbstractNeighborSearch",
    "BruteForce",
    "BucketKDTree",
    "Jaxkd",
    "Scipy",
)

from ._src.neighbors import (
    AbstractNeighborSearch,
    BruteForce,
    BucketKDTree,
    Jaxkd,
    Scipy,
)
