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


class TestVelocityAwareness:
    """An orderer's result says whether velocity informed the ordering."""

    def test_metrics_declare_whether_they_read_velocity(self):
        """It is a property of the metric type, not of any scale parameter."""
        assert pcf.metrics.SpatialDistanceMetric.uses_velocity is False
        assert pcf.metrics.AlignedMomentumDistanceMetric.uses_velocity is True
        assert pcf.metrics.FullPhaseSpaceDistanceMetric.uses_velocity is True

    def test_a_custom_metric_defaults_to_velocity_aware(self):
        """The signature takes velocities, so ignoring them is the special case.

        Defaulting the other way would silently drop a third-party
        velocity-aware metric back to position-only inside a chain.
        """

        class _Custom(pcf.metrics.AbstractDistanceMetric):
            def __call__(self, q, p, qs, ps, scale):  # noqa: ARG002
                return jnp.zeros(next(iter(qs.values())).shape[0])

        assert _Custom.uses_velocity is True

    def test_localflow_reports_its_metrics_answer(self, arc):
        """The flag follows the metric, not ``metric_scale``.

        A zero scale makes a phase-space metric numerically position-only, but
        the walk is still configured to follow the flow. Keying on the metric
        also keeps the flag static: ``metric_scale`` is a differentiable leaf
        and is a tracer under ``jit``/``grad``.
        """
        pos, vel, _ = arc()
        assert pcf.orderers.LocalFlowOrderer().order(pos, vel).velocity_aware is True
        zero = pcf.orderers.LocalFlowOrderer(metric_scale=0.0)
        assert zero.order(pos, vel).velocity_aware is True
        spatial = pcf.orderers.LocalFlowOrderer(
            config=pcf.WalkConfig(metric=pcf.metrics.SpatialDistanceMetric())
        )
        assert spatial.order(pos, vel).velocity_aware is False

    def test_the_deprecated_walk_reports_it_too(self, arc):
        """The flag is set by the walk, not by the orderer wrapping it.

        Direct callers of the walk -- including the deprecated
        ``walk_local_flow`` -- get the same answer as ``order``, so a
        downstream stage reading ``init.velocity_aware`` is not misled by
        which entry point produced the result.
        """
        pos, vel, _ = arc()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            assert pcf.walk_local_flow(pos, vel).velocity_aware is True

    def test_mst_reports_velocity_awareness(self, arc):
        """Only settings that steer the ordering count, not orient_by_velocity."""
        pos, vel, _ = arc()
        plain = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0)
        assert plain.order(pos, vel).velocity_aware is False
        for kw in ({"sever_cos_threshold": 0.9}, {"velocity_weight": 0.5}):
            orderer = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, **kw)
            assert orderer.order(pos, vel).velocity_aware is True
        oriented = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, orient_by_velocity=True)
        assert oriented.order(pos, vel).velocity_aware is False


def test_mst_rejects_a_negative_velocity_weight():
    """Only ``> 0`` engages the phase-space edge weights.

    A negative value silently did nothing, and reported ``velocity_aware``
    False on an orderer the caller believed was using velocity.
    """
    with pytest.raises(ValueError, match="velocity_weight must be >= 0"):
        pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, velocity_weight=-1.0)
