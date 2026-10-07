"""Benchmarks for the MST-backbone orderer.

End-to-end ``MSTOrderer.order`` on a noisy 3-D stream, eagerly -- the path a
user takes for one-shot ordering. ``edge_clip_sigma`` is benchmarked separately
because sigma-clipping is a sizeable share of the cost when it is on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf


def _stream(n, interloper_frac=0.0, seed=0):
    """Build a shuffled noisy 3-D stream, optionally with scattered interlopers."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    pos = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)]
    pos += rng.normal(0.0, 0.02, (n, 3))
    vel = np.c_[np.ones(n), 3 * np.cos(3 * t), -0.6 * np.sin(2 * t)]
    m = int(interloper_frac * n)
    if m:
        idx = rng.choice(n, m, replace=False)
        pad = np.array([0.0, 3.0, 3.0])
        lo, hi = pos.min(0) - pad, pos.max(0) + pad
        pos[idx] = rng.uniform(lo, hi, (m, 3))
    perm = rng.permutation(n)
    to = lambda a: {c: jnp.asarray(a[perm, i]) for i, c in enumerate("xyz")}
    return to(pos.astype(np.float32)), to(vel.astype(np.float32))


@pytest.mark.parametrize("n", [1_000, 10_000], ids=["n1e3", "n1e4"])
@pytest.mark.parametrize(
    ("frac", "kw"),
    [(0.0, {"jump_cap": 2.0}), (0.01, {"jump_cap": 50.0, "edge_clip_sigma": 3.0})],
    ids=["clean", "edge_clip"],
)
@pytest.mark.parametrize(
    "neighbors",
    [pcf.neighbors.BucketKDTree(), pcf.neighbors.SciPy()],
    ids=["bucket", "scipy"],
)
def test_order(benchmark, n, frac, kw, neighbors):
    """Benchmark ``order``; ``edge_clip`` adds 1% interlopers and sigma-clipping."""
    pos, vel = _stream(n, interloper_frac=frac)
    orderer = pcf.orderers.MSTOrderer(k=10, neighbors=neighbors, **kw)
    run = lambda: jax.block_until_ready(orderer.order(pos, vel).indices)
    run()
    assert benchmark(run).shape == (n,)
