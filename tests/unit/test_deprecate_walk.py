"""Deprecation and equivalence tests for ``order()`` vs ``walk_local_flow``.

Covers issue #36: ``pcf.order`` is the primary entry point (default orderer is the
local-flow walk), and ``walk_local_flow`` is a deprecated alias that emits a
``DeprecationWarning`` while producing identical results.
"""

import warnings

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf

POS = {
    "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    "y": jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]),
}
VEL = {"x": jnp.ones(5), "y": jnp.full(5, 0.1)}


def _walk(pos, vel, **kwargs):
    """Call the deprecated ``walk_local_flow``, suppressing its warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pcf.walk_local_flow(pos, vel, **kwargs)


class TestOrderDefault:
    """``order()`` defaults to the local-flow walk and equals the deprecated func."""

    def test_default_orderer_is_local_flow(self):
        """``order(pos, vel)`` with no orderer == an explicit ``LocalFlowOrderer``."""
        got = pcf.order(POS, VEL).indices
        exp = pcf.order(POS, VEL, pcf.orderers.LocalFlowOrderer()).indices
        assert jnp.array_equal(got, exp)

    def test_default_equals_deprecated_walk(self):
        """``order(pos, vel)`` reproduces ``walk_local_flow(pos, vel)`` exactly."""
        assert jnp.array_equal(pcf.order(POS, VEL).indices, _walk(POS, VEL).indices)

    def test_nondefault_params_via_orderer(self):
        """Non-default walk params on ``LocalFlowOrderer`` match the walk kwargs."""
        lfo = pcf.orderers.LocalFlowOrderer(
            start_idx=4, metric_scale=0.5, direction="backward"
        )
        got = pcf.order(POS, VEL, lfo).indices
        exp = _walk(
            POS, VEL, start_idx=4, metric_scale=0.5, direction="backward"
        ).indices
        assert jnp.array_equal(got, exp)


class TestDeprecation:
    """``walk_local_flow`` warns; the ``order()``/orderer path is silent."""

    def test_walk_local_flow_warns(self):
        """Directly calling ``walk_local_flow`` emits a ``DeprecationWarning``."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            pcf.walk_local_flow(POS, VEL)

    def test_order_path_is_silent(self):
        """``order()`` and orderers emit no warning."""
        # filterwarnings=error is set project-wide; simplefilter("error") makes
        # any emitted warning raise, so a clean pass proves silence.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pcf.order(POS, VEL)
            pcf.orderers.LocalFlowOrderer().order(POS, VEL)


class TestUnxtEquivalence:
    """The Quantity path behaves the same: order() equals the (warning) walk."""

    def _quantity_data(self):
        import unxt as u

        q = {
            "x": u.Q(jnp.array([0.0, 1.0, 2.0]), "m"),
            "y": u.Q(jnp.array([0.0, 0.5, 1.0]), "m"),
        }
        p = {"x": u.Q(jnp.ones(3), "m/s"), "y": u.Q(jnp.full(3, 0.5), "m/s")}
        return q, p, u.unitsystems.si

    def test_quantity_order_equals_walk(self):
        """Quantity ``order()`` reproduces the deprecated Quantity walk."""
        import unxt as u

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
        import unxt as u

        q, p, usys = self._quantity_data()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            pcf.walk_local_flow(
                q, p, start_idx=0, metric_scale=u.Q(1.0, "m"), usys=usys
            )
