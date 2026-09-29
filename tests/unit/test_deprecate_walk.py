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
