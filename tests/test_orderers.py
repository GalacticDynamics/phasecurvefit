"""Tests for the pluggable orderer abstraction (``pcf.orderers``)."""

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.abstract_result import AbstractResult


class TestOrdererNamespace:
    """The public ``pcf.orderers`` surface."""

    def test_exports(self):
        """Exports."""
        assert hasattr(pcf, "orderers")
        for name in ("AbstractOrderer", "LocalFlowOrderer", "OrderingResult"):
            assert hasattr(pcf.orderers, name), name

    def test_order_facade_exists(self):
        """Order facade exists."""
        assert callable(pcf.order)


class TestOrderingResultUnification:
    """OrderingResult is unified; WalkLocalFlowResult subclasses it."""

    def test_walklocalflowresult_subclasses_orderingresult(self):
        """Walklocalflowresult subclasses orderingresult."""
        assert issubclass(pcf.WalkLocalFlowResult, pcf.orderers.OrderingResult)
        assert issubclass(pcf.orderers.OrderingResult, AbstractResult)

    def test_backbone_aware_call_interpolates_along_backbone(self):
        """Backbone aware call interpolates along backbone."""
        # A straight backbone from (0,0) to (1,0), sampled at the two tips only.
        positions = {"x": jnp.array([0.0, 0.5, 1.0]), "y": jnp.array([0.0, 0.0, 0.0])}
        velocities = {"x": jnp.zeros(3), "y": jnp.zeros(3)}
        backbone = {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0, 0.0])}
        res = pcf.orderers.OrderingResult(
            positions=positions,
            velocities=velocities,
            indices=jnp.array([0, 1, 2]),
            gamma_range=(-1.0, 1.0),
            backbone=backbone,
        )
        mid = res(jnp.array(0.0))  # midpoint of gamma_range -> midpoint of backbone
        assert jnp.allclose(mid["x"], 0.5)
        assert jnp.allclose(mid["y"], 0.0)

    def test_degenerate_gamma_range_raises(self):
        """A zero-width gamma_range is rejected at construction (no inf/NaN)."""
        with pytest.raises(ValueError, match="gamma_range"):
            pcf.orderers.OrderingResult(
                positions={"x": jnp.array([0.0, 1.0])},
                velocities={"x": jnp.zeros(2)},
                indices=jnp.array([0, 1]),
                gamma_range=(1.0, 1.0),
            )

    def test_empty_backbone_call_raises_clear_error(self):
        """__call__ on an empty backbone gives a clear error, not an index crash."""
        res = pcf.orderers.OrderingResult(
            positions={"x": jnp.array([])},
            velocities={"x": jnp.array([])},
            indices=jnp.array([], dtype=jnp.int32),
            gamma_range=(-1.0, 1.0),
            backbone={"x": jnp.array([])},
        )
        with pytest.raises(ValueError, match="backbone"):
            res(jnp.array(0.0))

    def test_no_backbone_falls_back_to_ordered_points(self):
        """No backbone falls back to ordered points."""
        # Without a backbone, __call__ matches the legacy walk interpolation.
        positions = {"x": jnp.array([0.0, 1.0, 2.0])}
        velocities = {"x": jnp.zeros(3)}
        res = pcf.orderers.OrderingResult(
            positions=positions,
            velocities=velocities,
            indices=jnp.array([0, 1, 2]),
        )
        out = res(jnp.array(0.5))
        assert jnp.allclose(out["x"], 1.0)


class TestLocalFlowOrdererRegression:
    """LocalFlowOrderer.order reproduces walk_local_flow bit-for-bit."""

    @pytest.fixture
    def data(self):
        """Return sample phase-space data."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}
        return q, p

    @pytest.mark.parametrize("direction", ["forward", "backward", "both"])
    def test_matches_walk_local_flow(self, data, direction):
        """Matches walk local flow."""
        q, p = data
        direct = pcf.order(q, p, pcf.orderers.LocalFlowOrderer(direction=direction))
        orderer = pcf.orderers.LocalFlowOrderer(
            metric_scale=1.0, start_idx=0, direction=direction
        )
        res = orderer.order(q, p)
        assert jnp.array_equal(res.indices, direct.indices)
        assert res.gamma_range == direct.gamma_range
        assert jnp.array_equal(res.positions["x"], direct.positions["x"])

    def test_order_facade_matches_method(self, data):
        """Order facade matches method."""
        q, p = data
        orderer = pcf.orderers.LocalFlowOrderer(metric_scale=1.0, start_idx=0)
        assert jnp.array_equal(
            pcf.order(q, p, orderer).indices, orderer.order(q, p).indices
        )


class TestConformance:
    """Interface conformance shared by every orderer."""

    def test_localflow_conformance(self):
        """Localflow conformance."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0])}
        res = pcf.orderers.LocalFlowOrderer(metric_scale=1.0).order(q, p)
        assert isinstance(res, AbstractResult)
        # indices: valid permutation-with--1
        vis = res.indices[res.indices >= 0]
        assert len(set(vis.tolist())) == len(vis)  # no duplicates
        # gamma_range static tuple of two floats
        assert isinstance(res.gamma_range, tuple)
        lo, hi = res.gamma_range
        # __call__ finite over the range
        out = res(jnp.linspace(lo, hi, 7))
        assert jnp.all(jnp.isfinite(out["x"]))
