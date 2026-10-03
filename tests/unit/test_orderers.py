"""Tests for the pluggable orderer abstraction (``pcf.orderers``)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import _local_flow_walk


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

    def test_order_facade_rejects_both_empty_with_a_clear_message(self):
        """An empty dict satisfies every orderer's dispatch signature equally.

        #129: ``plum`` cannot pick between the plain-array and Quantity
        overloads for ``{}``, so it raises ``AmbiguousLookupError`` naming its
        own internals rather than the caller's mistake -- before any
        orderer's own body runs. ``pcf.order``, the primary documented entry
        point, catches it here with a message about the actual problem.
        """
        with pytest.raises(ValueError, match="cannot both be empty"):
            pcf.order({}, {})

    def test_order_facade_leaves_non_dict_falsy_input_to_dispatch(self):
        """The guard is about the empty-*dict* ambiguity, not falsiness.

        ``None`` is also falsy, but it is a different mistake -- a type
        error, not "nothing to order" -- and deserves the message dispatch
        itself gives, not this guard's misleading "both empty" one.
        """
        with pytest.raises(LookupError) as exc_info:
            pcf.order(None, None)
        assert "cannot both be empty" not in str(exc_info.value)


class TestDefaultPipeline:
    """``order()``'s default: an MST backbone refined by a SOM (#57)."""

    @staticmethod
    def _explicit_chain(n_prototypes):
        """Build the chain ``default_pipeline`` documents itself as running."""
        mst = pcf.orderers.MSTOrderer(
            jump_cap=float("inf"), orient_by_velocity=True, on_disconnected="warn"
        )
        return mst | pcf.orderers.SOMOrderer(n_prototypes=n_prototypes)

    def test_chains_the_mst_and_the_som(self, arc):
        """Enough tracers: the result is exactly the MST-then-SOM chain."""
        pos, vel, _ = arc(n=60)
        result = pcf.orderers.default_pipeline(pos, vel, n_prototypes=12)
        expected = self._explicit_chain(12).order(pos, vel)
        assert isinstance(result, pcf.orderers.OrderingResult)
        assert result.gamma_range == (-1.0, 1.0)
        assert eqx.tree_equal(result, expected)

    def test_bare_order_runs_the_default_pipeline(self, arc):
        """``order(pos, vel)`` with no orderer *is* ``default_pipeline``."""
        pos, vel, _ = arc(n=60)
        assert eqx.tree_equal(
            pcf.order(pos, vel), pcf.orderers.default_pipeline(pos, vel)
        )

    def test_recovers_the_arc_order(self, arc, spearman):
        """The whole point: the chained result should track the truth well."""
        curve = arc(n=60)
        result = pcf.order(curve.positions, curve.velocities)
        assert spearman(result, curve) > 0.99

    def test_orders_along_the_velocity(self):
        """``gamma`` increases along the flow, whichever way the data are stored."""
        t = jnp.linspace(jnp.pi, 0.0, 60)  # the flow runs toward *lower* t
        pos = {"x": 5.0 * jnp.cos(t), "y": 5.0 * jnp.sin(t)}
        vel = {"x": jnp.sin(t), "y": -jnp.cos(t)}
        idx = pcf.order(pos, vel).indices
        # The flow runs from index 0 toward index 59, so the ordering does too.
        assert int(idx[0]) < int(idx[-1])

    @pytest.mark.parametrize("scale", [1.0, 1e3, 1e-3])
    def test_is_independent_of_the_data_scale(self, arc, spearman, scale):
        """No absolute length scale is baked in.

        ``MSTOrderer``'s own ``jump_cap`` default (3.0) severs every edge of a
        dataset whose spacing exceeds it, so a default built on it would raise
        on sparse or large-scale data.
        """
        curve = arc(n=60)
        pos = {k: v * scale for k, v in curve.positions.items()}
        result = pcf.order(pos, curve.velocities)
        assert int(result.n_visited) == 60
        assert spearman(result, curve) > 0.99

    def test_falls_back_to_the_mst_alone_below_n_prototypes(self):
        """#57's own landmine: the library's small examples must keep working.

        3 tracers, the package's own headline example size, with the SOM's
        default n_prototypes=15 -- far too few for init_prototypes, which
        would otherwise raise ``binning needs at least n_prototypes points``.
        """
        pos = {"x": jnp.array([0.0, 1.0, 2.0])}
        vel = {"x": jnp.array([1.0, 1.0, 1.0])}
        result = pcf.order(pos, vel)
        assert isinstance(result, pcf.orderers.OrderingResult)
        assert not isinstance(result, pcf.WalkLocalFlowResult)
        assert result.gamma_range == (-1.0, 1.0)
        assert jnp.array_equal(result.indices, jnp.array([0, 1, 2]))

    def test_the_fallback_threshold_is_n_prototypes_itself(self, arc):
        """Exactly at the threshold and one above it, not just far below."""
        pos, vel, _ = arc(n=12)
        below = pcf.orderers.default_pipeline(pos, vel, n_prototypes=13)
        at = pcf.orderers.default_pipeline(pos, vel, n_prototypes=12)
        # The MST stage records its backbone length; the SOM stage does not.
        assert below.backbone_size is not None
        assert at.backbone_size is None

    def test_som_kwargs_reach_the_som_stage(self, arc):
        """A caller can still tune the SOM through this entry point."""
        pos, vel, _ = arc(n=60)
        tight = pcf.orderers.default_pipeline(pos, vel, n_prototypes=12, sigma_end=0.1)
        loose = pcf.orderers.default_pipeline(pos, vel, n_prototypes=12, sigma_end=5.0)
        # Different smoothing must produce a different backbone -- otherwise
        # sigma_end silently never reached the SOM.
        assert not jnp.array_equal(tight.backbone["x"], loose.backbone["x"])

    def test_a_gap_in_the_stream_does_not_drop_a_side(self):
        """Two arcs with a gap wider than ``k`` neighbours reach: both are ordered.

        Without bridging, the kNN graph disconnects at the gap and only the
        larger half would be ordered (and warned about).
        """
        t = np.concatenate([np.linspace(0, 1, 60), np.linspace(1.2, 2.2, 60)])
        pos = {"x": jnp.asarray(t), "y": jnp.zeros(t.size)}
        vel = {"x": jnp.ones(t.size), "y": jnp.zeros(t.size)}
        result = pcf.order(pos, vel)  # `filterwarnings = error`: no warning either
        assert int(result.n_visited) == 120
        x = np.asarray(pos["x"])[np.asarray(result.ordering)]
        assert np.all(np.diff(x) >= 0)  # along the flow, across the gap

    def test_default_n_prototypes_is_15(self, arc):
        """The documented threshold: 14 tracers fall back, 15 run the SOM."""
        for n, som_ran in ((14, False), (15, True)):
            pos, vel, _ = arc(n=n)
            result = pcf.order(pos, vel)
            assert (result.backbone_size is None) is som_ran, n

    def test_orientation_follows_the_flow_on_shuffled_data(self, arc):
        """Signed, not just |rho|: ``gamma`` rises with the true parameter.

        ``arc``'s velocity runs along increasing ``t``, and the tracers arrive
        shuffled, so a correct orientation is not a storage-order accident.
        """
        curve = arc(n=80)
        result = pcf.order(curve.positions, curve.velocities)
        t_along = np.asarray(curve.t)[np.asarray(result.ordering)]
        assert np.corrcoef(np.arange(t_along.size), t_along)[0, 1] > 0.99

    def test_one_nan_velocity_does_not_flip_the_orientation(self):
        """A NaN velocity must not turn the orientation test into a coin flip."""
        n = 30
        x = jnp.linspace(0.0, 3.0, n)
        vel_x = jnp.ones(n).at[7].set(jnp.nan)
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(n)
            pos = {"x": x[perm], "y": jnp.zeros(n)}
            vel = {"x": vel_x[perm], "y": jnp.zeros(n)}
            ordered_x = np.asarray(pos["x"])[np.asarray(pcf.order(pos, vel).ordering)]
            assert ordered_x[0] < ordered_x[-1], seed

    def test_init_is_rejected_rather_than_silently_dropped(self, arc):
        """The default has no use for ``init``: say so, do not discard it."""
        pos, vel, _ = arc(n=30)
        prior = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer())
        with pytest.raises(ValueError, match="init"):
            pcf.order(pos, vel, init=prior)
        # An explicit orderer still takes ``init``.
        pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(), init=prior)

    def test_tracing_the_default_says_to_pass_an_orderer(self, arc):
        """Under jit/vmap the default cannot decide whether to run the SOM: say so."""
        pos, vel, _ = arc(n=30)
        with pytest.raises(TypeError, match="explicit orderer"):
            jax.jit(lambda q, p: pcf.order(q, p).chord)(pos, vel)
        batched = jax.tree.map(lambda a: a[None], (pos, vel))
        with pytest.raises(TypeError, match="explicit orderer"):
            jax.vmap(lambda q, p: pcf.order(q, p).chord)(*batched)
        # ...while the explicit orderers it points to do trace.
        for orderer in (
            pcf.orderers.MSTOrderer(jump_cap=float("inf")),
            pcf.orderers.LocalFlowOrderer(),
        ):
            jax.jit(lambda q, p, o=orderer: pcf.order(q, p, o).chord)(pos, vel)

    def test_explicit_orderer_bypasses_the_default(self, arc):
        """Passing an orderer is unaffected by the default changing."""
        pos, vel, _ = arc(n=30)
        lfo = pcf.orderers.LocalFlowOrderer()
        walk = pcf.order(pos, vel, lfo)
        assert isinstance(walk, pcf.WalkLocalFlowResult)
        # The walk's gamma_range is (0, 1); the default pipeline's is (-1, 1).
        assert walk.gamma_range == (0.0, 1.0)
        assert pcf.order(pos, vel).gamma_range == (-1.0, 1.0)


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
    """``LocalFlowOrderer.order`` and ``order(..., LocalFlowOrderer)`` agree."""

    @pytest.fixture
    def data(self):
        """Return sample phase-space data."""
        q = {"x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        p = {"x": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])}
        return q, p

    @pytest.mark.parametrize("direction", ["forward", "backward", "both"])
    def test_order_matches_orderer_method(self, data, direction):
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

    def test_velocity_aware_stays_concrete_under_grad_and_jit(self, arc):
        """Deriving the flag from the metric keeps it out of tracer territory.

        #77: an earlier design (never merged past #56's own review) derived
        ``velocity_aware`` from ``bool(metric_scale != 0.0)``, which is a
        differentiable leaf and a tracer under ``jit``/``grad`` -- forcing a
        ``try/except TypeError`` that silently always resolved to ``True``
        while tracing. Since the flag instead reads ``config.metric.uses_velocity``,
        a plain Python attribute of the metric type, ``metric_scale`` being
        traced never enters into it: no exception to catch, and the answer is
        the same traced or not.
        """
        pos, vel, _ = arc()

        def run(scale: float, *, metric, expected: bool) -> float:
            orderer = pcf.orderers.LocalFlowOrderer(
                metric_scale=scale, config=pcf.WalkConfig(metric=metric)
            )
            result = orderer.order(pos, vel)
            assert result.velocity_aware is expected
            return jnp.sum(result.chord)

        aligned = pcf.metrics.AlignedMomentumDistanceMetric()
        spatial = pcf.metrics.SpatialDistanceMetric()

        # Neither raises -- the historical failure mode was a TypeError from
        # `bool()` on a tracer, masked by the try/except into a silent `True`.
        # Checking both metrics, not just the velocity-aware default, is what
        # actually distinguishes this from the old derivation: a try/except
        # forcing `True` under any trace would still pass a True-only check,
        # metric_scale traced or not, since it never sees `False` to get wrong.
        # ``float(...)`` rather than discarding the return value: on an async
        # backend an unconsumed result can defer a runtime failure past this
        # test, silently. Forcing it is what makes "neither raises" a claim
        # about actually running the computation, not just tracing it.
        float(jax.grad(lambda s: run(s, metric=aligned, expected=True))(1.0))
        float(jax.jit(lambda s: run(s, metric=aligned, expected=True))(1.0))
        float(jax.grad(lambda s: run(s, metric=spatial, expected=False))(1.0))
        float(jax.jit(lambda s: run(s, metric=spatial, expected=False))(1.0))

    def test_the_bare_walk_reports_it_too(self, arc):
        """The flag is set by the walk, not by the orderer wrapping it.

        Direct callers of the walk get the same answer as ``order``, so a
        downstream stage reading ``init.velocity_aware`` is not misled by
        which entry point produced the result.
        """
        pos, vel, _ = arc()
        assert _local_flow_walk(pos, vel).velocity_aware is True

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
