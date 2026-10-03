"""Deprecation and equivalence tests for ``order()`` vs ``walk_local_flow``.

Covers issue #36: ``pcf.order`` is the primary entry point, and ``walk_local_flow``
is a deprecated alias that emits a ``DeprecationWarning`` while producing results
identical to ``order(..., LocalFlowOrderer(...))``. ``order``'s own default is no
longer the walk (it is the MST | SOM pipeline); see ``test_orderers.py``.
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


class TestOrderEqualsWalk:
    """``order(..., LocalFlowOrderer(...))`` equals the deprecated function."""

    def test_local_flow_orderer_equals_deprecated_walk(self):
        """``order(pos, vel, LocalFlowOrderer())`` reproduces ``walk_local_flow``."""
        lfo = pcf.orderers.LocalFlowOrderer()
        assert jnp.array_equal(
            pcf.order(POS, VEL, lfo).indices, _walk(POS, VEL).indices
        )

    def test_bare_order_is_not_the_walk(self):
        """``order(pos, vel)`` is no longer the walk: it is the MST | SOM default.

        A walk result has ``gamma_range == (0, 1)``; the default's is ``(-1, 1)``.
        """
        assert _walk(POS, VEL).gamma_range == (0.0, 1.0)
        assert pcf.order(POS, VEL).gamma_range == (-1.0, 1.0)

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
            pcf.order(POS, VEL, pcf.orderers.LocalFlowOrderer())
            pcf.orderers.LocalFlowOrderer().order(POS, VEL)
