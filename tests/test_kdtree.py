"""Tests for KD-tree strategy in walk_local_flow."""

import jax.numpy as jnp
import pytest

import localflowwalk as lfw


@pytest.mark.parametrize(
    "strategy",
    [lfw.BruteForceStrategy(), lfw.KDTreeStrategy(k=2)],
)
def test_walk_local_flow_kdtree_matches_bruteforce(strategy):
    pos = {
        "x": jnp.array([0.0, 1.0, 2.0, 3.0]),
        "y": jnp.array([0.0, 0.5, 0.8, 1.2]),
    }
    vel = {
        "x": jnp.array([1.0, 1.0, 1.0, 1.0]),
        "y": jnp.array([0.2, 0.2, 0.2, 0.2]),
    }

    res = lfw.walk_local_flow(pos, vel, start_idx=0, lam=0.5, strategy=strategy)
    # For this simple dataset, both strategies should visit all points in order
    assert res.all_visited
    assert (res.ordered_indices == jnp.array([0, 1, 2, 3])).all()


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
    res_small = lfw.walk_local_flow(
        pos, vel, start_idx=0, lam=0.5, strategy=lfw.KDTreeStrategy(k=3)
    )
    # Run with KD-tree using larger k
    res_large = lfw.walk_local_flow(
        pos, vel, start_idx=0, lam=0.5, strategy=lfw.KDTreeStrategy(k=8)
    )

    # Both should produce valid ordered results
    assert res_small.n_visited >= 1
    assert res_large.n_visited >= 1

    # Larger k should be at least as thorough as small k
    assert res_large.n_visited >= res_small.n_visited
