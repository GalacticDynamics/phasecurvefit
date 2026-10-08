"""Tests for unxt (physical-units) support in the orderers.

Mirrors ``walk_local_flow``'s Quantity-in / Quantity-out UX. Because MST is
host-side, unit handling is a simple strip-in / reattach-out.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.algorithm import StateMetadata

pytest.importorskip("scipy")
u = pytest.importorskip("unxt")


def _arc_quantity(n=120):
    t = np.linspace(0.0, 1.0, n)
    x = 10.0 * t
    y = np.sin(3.0 * t)
    q = {"x": u.Q(jnp.asarray(x), "kpc"), "y": u.Q(jnp.asarray(y), "kpc")}
    p = {
        "x": u.Q(jnp.ones(n), "km/s"),
        "y": u.Q(jnp.asarray(3.0 * np.cos(3.0 * t)), "km/s"),
    }
    return q, p


class TestMSTUnxt:
    """Tests for MST unxt."""

    def test_quantity_matches_stripped_run(self):
        """Quantity matches stripped run."""
        q, p = _arc_quantity()
        usys = u.unitsystems.galactic
        o = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)
        res_q = o.order(q, p, metadata=StateMetadata(usys=usys))

        q_plain = {k: u.ustrip(usys, v) for k, v in q.items()}
        p_plain = {k: u.ustrip(usys, v) for k, v in p.items()}
        res_plain = o.order(q_plain, p_plain)

        assert jnp.array_equal(res_q.indices, res_plain.indices)

    def test_quantity_out(self):
        """Quantity out."""
        q, p = _arc_quantity()
        usys = u.unitsystems.galactic
        res = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(
            q, p, metadata=StateMetadata(usys=usys)
        )
        assert isinstance(res.positions["x"], u.AbstractQuantity)
        assert isinstance(res.velocities["x"], u.AbstractQuantity)
        assert isinstance(res.backbone["x"], u.AbstractQuantity)
        assert res.positions["x"].unit == u.unit("kpc")

    def test_missing_usys_errors(self):
        """Missing usys errors."""
        q, p = _arc_quantity()
        with pytest.raises((TypeError, RuntimeError), match="usys"):
            pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(q, p)


class TestLocalFlowUnxt:
    """Tests for local flow unxt."""

    def test_localflow_quantity_delegates_to_walk(self):
        """Localflow quantity delegates to walk."""
        q = {"x": u.Q(jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), "kpc")}
        p = {"x": u.Q(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]), "km/s")}
        usys = u.unitsystems.galactic
        orderer = pcf.orderers.LocalFlowOrderer(
            metric_scale=u.Q(1.0, "kpc"), start_idx=0
        )
        res = orderer.order(q, p, metadata=StateMetadata(usys=usys))
        direct = pcf.order(
            q,
            p,
            pcf.orderers.LocalFlowOrderer(metric_scale=u.Q(1.0, "kpc")),
            metadata=pcf.StateMetadata(usys=usys),
        )
        assert jnp.array_equal(res.indices, direct.indices)
        assert isinstance(res.positions["x"], u.AbstractQuantity)
