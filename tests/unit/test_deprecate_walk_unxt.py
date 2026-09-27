"""Quantity-support tests for ``order()`` vs ``walk_local_flow``.

Split out from ``test_deprecate_walk.py`` so the whole module skips cleanly
when ``unxt`` isn't installed, mirroring ``test_orderers_unxt.py``.
"""

import warnings

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf

u = pytest.importorskip("unxt")


def _walk(pos, vel, **kwargs):
    """Call the deprecated ``walk_local_flow``, suppressing its warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pcf.walk_local_flow(pos, vel, **kwargs)


class TestUnxtEquivalence:
    """The Quantity path behaves the same: order() equals the (warning) walk."""

    def _quantity_data(self):
        q = {
            "x": u.Q(jnp.array([0.0, 1.0, 2.0]), "m"),
            "y": u.Q(jnp.array([0.0, 0.5, 1.0]), "m"),
        }
        p = {"x": u.Q(jnp.ones(3), "m/s"), "y": u.Q(jnp.full(3, 0.5), "m/s")}
        return q, p, u.unitsystems.si

    def test_quantity_order_equals_walk(self):
        """Quantity ``order()`` reproduces the deprecated Quantity walk."""
        q, p, usys = self._quantity_data()
        got = pcf.order(
            q,
            p,
            pcf.orderers.LocalFlowOrderer(metric_scale=u.Q(1.0, "m")),
            metadata=pcf.StateMetadata(usys=usys),
        ).indices
        exp = _walk(q, p, start_idx=0, metric_scale=u.Q(1.0, "m"), usys=usys).indices
        assert jnp.array_equal(got, exp)

    def test_quantity_walk_warns(self):
        """The deprecated Quantity walk also warns."""
        q, p, usys = self._quantity_data()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            pcf.walk_local_flow(
                q, p, start_idx=0, metric_scale=u.Q(1.0, "m"), usys=usys
            )
