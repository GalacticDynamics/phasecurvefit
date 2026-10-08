"""Regression tests for ``max_dist`` edge values in the local-flow walk (#166).

Visited points used to be masked with ``inf * max_dist``, which is NaN for
``max_dist == 0`` (and ``-inf`` for negative values), so the walk kept
re-selecting the start point instead of stopping.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf


@pytest.mark.parametrize("max_dist", [0.0, -1.0, 0.5])
def test_walk_stops_at_start_when_no_point_within_max_dist(max_dist):
    """No neighbour within ``max_dist``: only the start point is visited."""
    q = {"x": jnp.array([0.0, 1.0, 2.0])}
    p = {"x": jnp.array([1.0, 1.0, 1.0])}
    orderer = pcf.orderers.LocalFlowOrderer(metric_scale=0.0, max_dist=max_dist)

    result = pcf.order(q, p, orderer)

    np.testing.assert_array_equal(result.indices, [0, -1, -1])
