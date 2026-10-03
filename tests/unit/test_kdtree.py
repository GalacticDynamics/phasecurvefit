"""Tests for KD-tree strategy in the local-flow walk."""

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf


@pytest.mark.parametrize(
    "config",
    [
        pcf.WalkConfig(strategy=pcf.strats.BruteForce()),
        pcf.WalkConfig(strategy=pcf.strats.KDTree(k=2)),
    ],
)
def test_walk_local_flow_kdtree_matches_bruteforce(config):
    pos = {
        "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
        "y": jnp.array([0.0, 0.5, 0.8, 1.2]),
    }
    vel = {
        "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
        "y": jnp.array([0.2, 0.2, 0.2, 0.2]),
    }

    res = pcf.order(
        pos, vel, pcf.orderers.LocalFlowOrderer(metric_scale=0.5, config=config)
    )
    # For this simple dataset, both strategies should visit all points in order
    assert res.all_visited
    assert (res.indices == jnp.array([0, 1, 2, 3])).all()


def test_kdtree_duplicate_points_match_bruteforce():
    """Coincident points must not cost a candidate slot or be skipped.

    The tree may return a duplicate of the current point before the point
    itself, so self must be excluded by index, not by dropping slot 0.
    """
    x = jnp.array([0.0, 0.5, 1.5, 1.5, 2.0, 2.5])
    pos = {"x": x, "y": jnp.zeros_like(x)}
    vel = {"x": jnp.ones_like(x), "y": jnp.zeros_like(x)}

    def run(strategy):
        orderer = pcf.orderers.LocalFlowOrderer(
            metric_scale=0.5, config=pcf.WalkConfig(strategy=strategy)
        )
        return pcf.order(pos, vel, orderer)

    bf = run(pcf.strats.BruteForce())
    kd = run(pcf.strats.KDTree(k=2))
    assert bf.all_visited
    assert kd.all_visited
    # Duplicates (2, 3) are tied, so their relative order is arbitrary.
    assert kd.indices[:2].tolist() == [0, 1]
    assert set(kd.indices[2:4].tolist()) == {2, 3}
    assert kd.indices[4:].tolist() == [4, 5]


def test_walk_local_flow_kdtree_k_parameter():
    pos = {
        "x": jnp.linspace(0.0, 9.0, 10),
        "y": jnp.linspace(0.0, 9.0, 10) * 0.1,
    }
    vel = {
        "x": jnp.ones(10),
        "y": jnp.ones(10) * 0.1,
    }

    # Run with KD-tree using small k
    res_small = pcf.order(
        pos,
        vel,
        pcf.orderers.LocalFlowOrderer(
            metric_scale=0.5, config=pcf.WalkConfig(strategy=pcf.strats.KDTree(k=3))
        ),
    )
    # Run with KD-tree using larger k
    res_large = pcf.order(
        pos,
        vel,
        pcf.orderers.LocalFlowOrderer(
            metric_scale=0.5, config=pcf.WalkConfig(strategy=pcf.strats.KDTree(k=8))
        ),
    )

    # Both should produce valid ordered results
    assert res_small.n_visited >= 1
    assert res_large.n_visited >= 1

    # Larger k should be at least as thorough as small k
    assert res_large.n_visited >= res_small.n_visited
