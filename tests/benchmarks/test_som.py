"""Benchmarks for the SOM ordering stage.

`fit` and `chord` dominate `SOMOrderer.order`. Both are measured at the
**shipped defaults** and through `jax.jit` -- the path `SOMOrderer` takes.
"""

import jax
import jax.numpy as jnp
import pytest

import phasecurvefit as pcf
from phasecurvefit import som

#: The shipped configuration, read off the orderer so the benchmarks follow any
#: future retune instead of pinning yesterday's defaults.
DEFAULTS = pcf.orderers.SOMOrderer()


def _helix(n):
    """Build an n-point one-turn helix with tangent velocities."""
    t = jnp.linspace(0.0, 1.0, n)
    ang = 2 * jnp.pi * t
    pos = {"x": jnp.cos(ang), "y": jnp.sin(ang), "z": 2.0 * t}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang), "z": jnp.full(n, 2.0)}
    return pos, vel


class TestSOMCoreBenchmarks:
    """The two functions that dominate the cost, jitted as they ship."""

    @pytest.mark.parametrize("n", [1_000, 10_000], ids=["n1e3", "n1e4"])
    @pytest.mark.parametrize("k", [15, 50], ids=["k15", "k50"])
    def test_fit(self, benchmark, n, k):
        """Benchmark `fit`: `n_epochs` passes over all the data."""
        pos, vel = _helix(n)
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=k)
        fit = jax.jit(
            lambda a, b: som.fit(
                a,
                b,
                pos,
                vel,
                metric=DEFAULTS.metric,
                metric_scale=DEFAULTS.metric_scale,
                n_epochs=DEFAULTS.n_epochs,
            )
        )
        run = lambda: jax.block_until_ready(fit(pq, pp))
        run()  # compile outside the measurement
        out = benchmark(run)
        assert out[0]["x"].shape == (k,)

    @pytest.mark.parametrize("n", [1_000, 10_000], ids=["n1e3", "n1e4"])
    @pytest.mark.parametrize("k", [15, 50], ids=["k15", "k50"])
    def test_chord(self, benchmark, n, k):
        """Benchmark `chord`: the N x M nearest-vertex search."""
        pos, vel = _helix(n)
        pq, pp = som.init_prototypes(pos, vel, n_prototypes=k)
        bq, bp = som.densify(pq, pp, factor=DEFAULTS.densify_factor)
        chord = jax.jit(
            lambda: som.chord(
                bq,
                bp,
                pos,
                vel,
                metric=DEFAULTS.metric,
                metric_scale=DEFAULTS.metric_scale,
            )
        )
        run = lambda: jax.block_until_ready(chord())
        run()
        out = benchmark(run)
        assert out.shape == (n,)


class TestSOMOrdererBenchmarks:
    """End-to-end, the path a user actually takes."""

    @pytest.mark.parametrize("n", [1_000, 10_000], ids=["n1e3", "n1e4"])
    def test_order_standalone(self, benchmark, n):
        """Benchmark `pcf.order` with a standalone SOMOrderer at its defaults."""
        pos, vel = _helix(n)
        run = lambda: jax.block_until_ready(pcf.order(pos, vel, DEFAULTS).indices)
        run()
        out = benchmark(run)
        assert out.shape == (n,)
