"""Benchmarks for the MST graph stage: pure JAX vs the host (SciPy) stage.

Same kNN indices for both, so only stage (b) is timed. The spec's gate: the JAX
stage within ~3x of the host stage at n = 1e5.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasecurvefit._src.neighbors import BucketKDTree
from phasecurvefit._src.orderers.mst import _host_graph, _jax_graph

K = 10
CASES = {
    "stream": {},
    "clip": {"edge_clip_sigma": 3.0},
    "connect": {"connect": True, "jump_cap": 0.05},
}
SIZES = [
    pytest.param(1_000, id="n1e3"),
    pytest.param(10_000, id="n1e4"),
    pytest.param(100_000, id="n1e5"),
]


def _setup(case, n):
    rng = np.random.default_rng(0)
    t = np.sort(rng.random(n))
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    if case == "clip":
        i = rng.choice(n, n // 100, replace=False)
        p[i] = rng.uniform(p.min(0), p.max(0), (len(i), 3))
    v = np.c_[np.ones(n), 3 * np.cos(3 * t), -0.6 * np.sin(2 * t)]
    p, v = p.astype(np.float32), v.astype(np.float32)
    nbr = np.asarray(BucketKDTree().knn(jnp.asarray(p), K)[0])
    cfg = {
        "jump_cap": np.inf,
        "velocity_weight": 0.0,
        "sever_cos_threshold": None,
        "orient_by_velocity": True,
        "connect": False,
        "edge_clip_sigma": None,
        "edge_clip_max_iters": 5,
    } | CASES[case]
    return p, v, nbr, cfg


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("case", list(CASES))
def test_jax(benchmark, case, n):
    """The pure-JAX graph stage, compiled."""
    p, v, nbr, cfg = _setup(case, n)
    f = jax.jit(functools.partial(_jax_graph, **cfg))
    nb = jnp.asarray(nbr)  # no split (no velocity mechanism): placeholder tables
    one = jnp.ones((), jnp.float32)  # scale: the data is already in range
    args = (jnp.asarray(p), jnp.asarray(v), nb, nb, nb[:, 0], jnp.ones(n, bool), one)
    run = lambda: jax.block_until_ready(f(*args))
    run()
    benchmark(run)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("case", list(CASES))
def test_host(benchmark, case, n):
    """The host (SciPy) graph stage, for reference."""
    p, v, nbr, cfg = _setup(case, n)
    od = "connect" if cfg.pop("connect") else "largest"
    benchmark(
        lambda: _host_graph(
            p,
            v,
            nbr,
            nbr,
            nbr[:, 0],
            k=K,
            nan_policy="raise",
            on_disconnected=od,
            workers=-1,
            **cfg,
        )
    )
