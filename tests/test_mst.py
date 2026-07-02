"""Tests for the MST-backbone orderer (``pcf.orderers.MSTOrderer``)."""

import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.abstract_result import AbstractResult

pytest.importorskip("scipy")


def _open_arc(n=200, seed=0):
    """Points sampled along a known open 1-D curve, then shuffled.

    Evenly spaced along the true parameter so the kNN graph stays connected
    (random-uniform sampling can leave gaps wider than ``k`` can bridge).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    x = 10.0 * t
    y = np.sin(3.0 * t)
    noise = rng.normal(0.0, 0.02, size=(n, 2))
    pos = {"x": jnp.asarray(x + noise[:, 0]), "y": jnp.asarray(y + noise[:, 1])}
    vel = {"x": jnp.ones(n), "y": jnp.asarray(3.0 * np.cos(3.0 * t))}
    # shuffle so input order carries no ordering information
    perm = rng.permutation(n)
    pos = {k: v[perm] for k, v in pos.items()}
    vel = {k: v[perm] for k, v in vel.items()}
    t_shuffled = t[perm]
    return pos, vel, jnp.asarray(t_shuffled)


def _open_ring(n=240, gap_deg=20.0, seed=0):
    """Points on a near-closed ring with a small angular gap (an open loop).

    Evenly spaced in angle so the along-ring kNN graph stays connected while the
    gap chord exceeds ``jump_cap`` and is severed.
    """
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, (360.0 - gap_deg), n) * np.pi / 180.0
    x, y = np.cos(ang), np.sin(ang)
    perm = rng.permutation(n)
    pos = {"x": jnp.asarray(x[perm]), "y": jnp.asarray(y[perm])}
    vel = {"x": jnp.asarray(-np.sin(ang)[perm]), "y": jnp.asarray(np.cos(ang)[perm])}
    return pos, vel, jnp.asarray(ang[perm])


class TestMSTNamespace:
    """Tests for MST namespace."""

    def test_exported(self):
        """Exported."""
        assert hasattr(pcf.orderers, "MSTOrderer")


class TestMSTConformance:
    """Tests for MST conformance."""

    def test_returns_orderingresult_with_backbone(self):
        """Returns orderingresult with backbone."""
        pos, vel, _ = _open_arc()
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        assert isinstance(res, AbstractResult)
        assert res.backbone is not None
        assert res.gamma_range == (-1.0, 1.0)
        # every tracer ordered (no -1) for a connected graph
        assert jnp.all(res.indices >= 0)
        assert set(res.indices.tolist()) == set(range(len(res.indices)))
        out = res(jnp.linspace(-1.0, 1.0, 11))
        assert jnp.all(jnp.isfinite(out["x"]))


class TestMSTOrderingCorrectness:
    """Tests for MST ordering correctness."""

    def test_monotone_in_true_arclength(self):
        """Monotone in true arclength."""
        pos, vel, t = _open_arc(n=300)
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        t_ordered = np.asarray(t)[np.asarray(res.ordering)]
        # rank correlation with position-in-order is ~ +/-1 for a clean arc
        rho = np.corrcoef(np.arange(t_ordered.size), t_ordered)[0, 1]
        assert abs(rho) > 0.99

    def test_endpoints_are_extremes(self):
        """Endpoints are extremes."""
        pos, vel, t = _open_arc(n=300)
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(pos, vel)
        order = np.asarray(res.ordering)
        t = np.asarray(t)
        # the two tips sit at the two ends of the true parameter range
        tip_t = sorted([t[order[0]], t[order[-1]]])
        assert tip_t[0] < 0.02
        assert tip_t[1] > 0.98


class TestMSTLoop:
    """Tests for MST loop."""

    def test_full_coverage_and_tips_at_gap(self):
        """Full coverage and tips at gap."""
        pos, vel, ang = _open_ring(n=240, gap_deg=20.0)
        res = pcf.orderers.MSTOrderer(k=10, jump_cap=0.1).order(pos, vel)
        assert jnp.all(res.indices >= 0)  # ~100% coverage
        order = np.asarray(res.ordering)
        ang = np.asarray(ang)
        # the two ends of the ordering sit at the two sides of the gap
        tip_angs = sorted([ang[order[0]], ang[order[-1]]])
        assert tip_angs[0] < np.deg2rad(20.0)  # near angle 0
        assert tip_angs[1] > np.deg2rad(320.0)  # near angle 340 (other gap edge)


class TestMSTDeterminism:
    """Tests for MST determinism."""

    def test_deterministic(self):
        """Deterministic."""
        pos, vel, _ = _open_arc()
        o = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)
        assert jnp.array_equal(o.order(pos, vel).indices, o.order(pos, vel).indices)


def _two_clusters(n_a=80, n_b=30):
    """Two well-separated line clusters -> a disconnected kNN graph."""
    xa = np.linspace(0.0, 1.0, n_a)
    xb = np.linspace(0.0, 1.0, n_b) + 100.0
    x = np.concatenate([xa, xb])
    y = np.zeros_like(x)
    pos = {"x": jnp.asarray(x), "y": jnp.asarray(y)}
    vel = {"x": jnp.ones_like(jnp.asarray(x)), "y": jnp.zeros_like(jnp.asarray(x))}
    return pos, vel


class TestMSTDisconnected:
    """Tests for MST disconnected."""

    def test_raise_on_disconnected(self):
        """Raise on disconnected."""
        pos, vel, _ = _open_arc(n=100)
        # jump_cap far below inter-point spacing -> disconnected graph
        with pytest.raises(ValueError, match=r"disconnected|connected"):
            pcf.orderers.MSTOrderer(k=8, jump_cap=1e-6, on_disconnected="raise").order(
                pos, vel
            )

    def test_largest_orders_bigger_component(self):
        """Largest orders bigger component."""
        pos, vel = _two_clusters(n_a=80, n_b=30)
        res = pcf.orderers.MSTOrderer(
            k=5, jump_cap=0.5, on_disconnected="largest"
        ).order(pos, vel)
        assert int((res.indices >= 0).sum()) == 80  # bigger cluster ordered
        assert int((res.indices < 0).sum()) == 30  # smaller left unvisited

    def test_warn_on_disconnected(self):
        """Warn on disconnected."""
        pos, vel = _two_clusters()
        with pytest.warns(UserWarning, match="disconnected"):
            pcf.orderers.MSTOrderer(k=5, jump_cap=0.5, on_disconnected="warn").order(
                pos, vel
            )
