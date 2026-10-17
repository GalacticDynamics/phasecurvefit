"""Regression tests for ``max_dist`` edge values in the local-flow walk (#166).

Visited points used to be masked with ``inf * max_dist``, which is NaN for
``max_dist == 0`` (and ``-inf`` for negative values), so the walk kept
re-selecting the start point instead of stopping. The duplicate-point case is
the one the stop checks alone don't catch: the neighbour sits at distance
exactly ``0 <= max_dist``, so only a correct mask stops ``argmin`` from
landing on the (NaN-masked) start point.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf

CASES = [
    # (positions, max_dist, expected indices)
    ([0.0, 1.0, 2.0], 0.0, [0, -1, -1]),
    ([0.0, 1.0, 2.0], -1.0, [0, -1, -1]),
    ([0.0, 1.0, 2.0], 0.5, [0, -1, -1]),
    ([0.0, 0.0, 2.0], 0.0, [0, 1, -1]),
]


@pytest.mark.parametrize(("xs", "max_dist", "expected"), CASES)
def test_walk_respects_max_dist_edge_values(xs, max_dist, expected):
    """Only neighbours within ``max_dist`` are visited; never the start twice."""
    q = {"x": jnp.array(xs)}
    p = {"x": jnp.ones(3)}
    orderer = pcf.orderers.LocalFlowOrderer(metric_scale=0.0, max_dist=max_dist)

    result = pcf.order(q, p, orderer)

    np.testing.assert_array_equal(result.indices, expected)


@pytest.mark.parametrize(("xs", "max_dist", "expected"), CASES)
def test_walk_respects_max_dist_edge_values_quantity(xs, max_dist, expected):
    """Same as above, with ``unxt.Quantity`` inputs."""
    u = pytest.importorskip("unxt")
    q = {"x": u.Q(xs, "kpc")}
    p = {"x": u.Q([1.0, 1.0, 1.0], "km/s")}
    orderer = pcf.orderers.LocalFlowOrderer(
        metric_scale=u.Q(0.0, "kpc"), max_dist=u.Q(max_dist, "kpc")
    )
    meta = pcf.StateMetadata(usys=u.unitsystems.galactic)

    result = pcf.order(q, p, orderer, metadata=meta)

    np.testing.assert_array_equal(result.indices, expected)
