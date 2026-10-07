"""Benchmarks for exact all-points kNN: our kd-tree vs jaxkd vs scipy.

The interloper case is the gate's thin margin (it must stay within 2x of our
clean-stream time); jaxkd is skipped there because it takes minutes.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree

from phasecurvefit._src import kdtree as kd

K = 10


def _data(name, n):
    rng = np.random.default_rng(0)
    if name == "blob":
        return rng.normal(size=(n, 3)).astype(np.float32)
    t = np.linspace(0, 1, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    if name == "interlopers":
        i = rng.choice(n, n // 100, replace=False)
        pad = np.array([0.0, 3.0, 3.0])
        p[i] = rng.uniform(p.min(0) - pad, p.max(0) + pad, (len(i), 3))
    return p[rng.permutation(n)].astype(np.float32)


NAMES = ["stream", "blob", "interlopers"]
SIZES = [
    pytest.param(1_000, id="n1e3"),
    pytest.param(10_000, id="n1e4"),
    pytest.param(100_000, id="n1e5"),
]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", NAMES)
def test_ours(benchmark, name, n):
    """Build + all-points query, compiled."""
    p = jnp.asarray(_data(name, n))
    f = jax.jit(functools.partial(kd.all_knn, k=K))
    run = lambda: jax.block_until_ready(f(p))
    run()
    benchmark(run)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", ["stream", "blob"])
def test_jaxkd(benchmark, name, n):
    """Jaxkd build + query, compiled (the gate's reference)."""
    jaxkd = pytest.importorskip("jaxkd")
    p = jnp.asarray(_data(name, n))
    f = jax.jit(lambda x: jaxkd.query_neighbors(jaxkd.build_tree(x), x, k=K + 1))
    run = lambda: jax.block_until_ready(f(p))
    run()
    benchmark(run)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", NAMES)
def test_scipy(benchmark, name, n):
    """Scipy cKDTree with every core, for reference."""
    p = _data(name, n)
    benchmark(lambda: cKDTree(p).query(p, k=K + 1, workers=-1))
