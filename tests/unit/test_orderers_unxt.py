"""Tests for unxt (physical-units) support in the orderers.

Mirrors ``walk_local_flow``'s Quantity-in / Quantity-out UX. Because MST is
host-side, unit handling is a simple strip-in / reattach-out.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import pytest

import dataclassish

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


def test_quantity_localflow_takes_its_start_from_init():
    """The Quantity path must resolve ``start_idx`` like the plain one.

    It reaches ``_local_flow_walk`` directly, so an unresolved ``None`` would
    arrive as a start index rather than being taken from ``init``.
    """
    q, p = _arc_quantity()
    usys = u.unitsystems.galactic
    md = StateMetadata(usys=usys)
    prior = pcf.orderers.MSTOrderer(k=8, jump_cap=5.0, on_disconnected="largest").order(
        q, p, metadata=md
    )
    chained = (
        pcf.orderers.MSTOrderer(k=8, jump_cap=5.0, on_disconnected="largest")
        | pcf.orderers.LocalFlowOrderer()
    ).order(q, p, metadata=md)
    assert int(np.asarray(chained.ordering)[0]) == int(np.asarray(prior.ordering)[0])


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (pcf.metrics.AlignedMomentumDistanceMetric(), True),
        (pcf.metrics.FullPhaseSpaceDistanceMetric(), True),
        (pcf.metrics.SpatialDistanceMetric(), False),
    ],
    ids=["aligned-momentum", "full-phase-space", "spatial"],
)
def test_quantity_localflow_reports_velocity_awareness(metric, expected):
    """The Quantity path must set ``velocity_aware`` like the plain one.

    It reaches ``_local_flow_walk`` directly rather than through the plain-array
    dispatch, so the flag has to be set in both places or a later stage silently
    drops back to position-only on unit-ful input. The value follows the metric,
    not ``metric_scale``.
    """
    q, p = _arc_quantity()
    usys = u.unitsystems.galactic
    orderer = pcf.orderers.LocalFlowOrderer(config=pcf.WalkConfig(metric=metric))
    result = orderer.order(q, p, metadata=StateMetadata(usys=usys))
    assert result.velocity_aware is expected


def test_localflow_quantity_agrees_with_plain_field_by_field():
    """Every field of the result must agree between the plain and Quantity paths.

    #71: ``velocity_aware`` (#56), ``chord`` (#67) and the ``init``-derived
    start index (#70) each silently diverged here in turn, because the
    Quantity dispatch reaches ``_local_flow_walk`` directly instead of
    delegating to the plain one. Comparing the whole result with
    ``eqx.tree_equal`` -- which checks static fields (``gamma_range``,
    ``velocity_aware``) as well as array leaves -- rather than naming a fixed
    set of fields, means a *future* field that goes missing from the Quantity
    path fails this test instead of shipping silently, which is the actual
    acceptance criterion in #71.
    """
    q, p = _arc_quantity()
    usys = u.unitsystems.galactic
    md = StateMetadata(usys=usys)
    q_plain = {k: u.ustrip(usys, v) for k, v in q.items()}
    p_plain = {k: u.ustrip(usys, v) for k, v in p.items()}

    prior = pcf.orderers.MSTOrderer(k=8, jump_cap=5.0, on_disconnected="largest")
    init_plain = prior.order(q_plain, p_plain)
    init_q = prior.order(q, p, metadata=md)
    # If MST's own Quantity dispatch ever diverged from its plain one, this
    # test's real target (LocalFlowOrderer's dispatches) would fail on a
    # difference it did not cause -- assert the shared premise explicitly so
    # a failure below can be trusted to be about #71's invariant.
    assert jnp.array_equal(init_plain.indices, init_q.indices)

    orderer = pcf.orderers.LocalFlowOrderer(
        config=pcf.WalkConfig(metric=pcf.metrics.FullPhaseSpaceDistanceMetric())
    )
    result_plain = orderer.order(q_plain, p_plain, init=init_plain)
    result_q = orderer.order(q, p, metadata=md, init=init_q)

    # Strip result_q's Quantity-valued fields back to plain arrays so both
    # results share one pytree structure -- only then can eqx.tree_equal walk
    # them leaf-by-leaf together.
    result_q_stripped = dataclassish.replace(
        result_q,
        positions={k: u.ustrip(usys, v) for k, v in result_q.positions.items()},
        velocities={k: u.ustrip(usys, v) for k, v in result_q.velocities.items()},
        chord=u.ustrip(usys, result_q.chord),
    )

    # tree_equal has no equal_nan option, and chord's own contract is nan for
    # unvisited observations, so a matching nan on both sides must not read
    # as a disagreement -- but nan_to_num alone would also let a genuine nan
    # on one side and a coincidentally-equal finite value (e.g. 0.0) on the
    # other silently pass. Checking the nan masks agree first, separately,
    # is what tells those two cases apart: only once every nan is known to be
    # in the same place on both sides is it safe to neutralise them and
    # compare the rest numerically.
    def _isnan(tree):
        is_float = eqx.is_inexact_array
        return jt.map(lambda x: jnp.isnan(x) if is_float(x) else False, tree)

    def _denan(tree):
        is_float = eqx.is_inexact_array
        return jt.map(lambda x: jnp.nan_to_num(x) if is_float(x) else x, tree)

    # Not under jit here, so these are concrete bools -- ``bool(...)`` is fine
    # (the ``is True`` idiom in eqx.tree_equal's own docs guards against a
    # tracer under jit, which does not apply outside one).
    assert bool(eqx.tree_equal(_isnan(result_plain), _isnan(result_q_stripped)))
    agree = eqx.tree_equal(
        _denan(result_plain), _denan(result_q_stripped), rtol=1e-5, atol=1e-8
    )
    assert bool(agree)


def test_chord_value_matches_the_stripped_pipeline():
    """The unit-ful chord must be the right *number*, not just the right label.

    Every other assertion here is on ``.unit``. A dispatch that relabelled
    instead of converting would return a chord 1000x too small while still
    tagged ``pc``, and pass all of them.
    """
    q, p = _arc_quantity()
    q = {k: u.uconvert("pc", v) for k, v in q.items()}
    # galactic's length is kpc, deliberately not the data's pc, so a dispatch
    # that ignored the unit system could not pass by coincidence.
    usys = u.unitsystems.galactic
    orderer = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0)

    got = orderer.order(q, p, metadata=StateMetadata(usys=usys)).chord

    stripped = orderer.order(
        {k: u.ustrip(usys, v) for k, v in q.items()},
        {k: u.ustrip(usys, v) for k, v in p.items()},
    ).chord

    assert got.unit == u.unit("pc")
    np.testing.assert_allclose(
        np.asarray(u.ustrip("kpc", got)), np.asarray(stripped), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize(
    ("x_unit", "y_unit", "expected"),
    [("pc", "pc", "pc"), ("pc", "kpc", "kpc")],
    ids=["shared", "mixed"],
)
def test_chord_unit_falls_back_when_components_disagree(x_unit, y_unit, expected):
    """``chord`` must not take its unit from whichever component sorts first.

    It is one length for all components, so there is no component to take a
    unit from. With a shared position unit that unit is used -- ``pc`` here,
    which is not the unit system's, so the assertion cannot pass by
    coincidence. With mixed units there is no defensible choice among them, so
    it comes back in the unit system's length (``kpc`` for galactic).
    """
    q, p = _arc_quantity()
    q["x"] = u.uconvert(x_unit, q["x"])
    q["y"] = u.uconvert(y_unit, q["y"])
    result = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0).order(
        q, p, metadata=StateMetadata(usys=u.unitsystems.galactic)
    )
    assert result.chord.unit == u.unit(expected)


@pytest.mark.parametrize(
    ("x_unit", "y_unit", "expected"),
    [("pc", "pc", "pc"), ("pc", "kpc", "kpc")],
    ids=["shared", "mixed"],
)
def test_chord_unit_falls_back_for_localflow_too(x_unit, y_unit, expected):
    """The same ``_chord_unit`` fallback must hold on the LocalFlowOrderer path.

    That dispatch reaches ``_local_flow_walk`` directly rather than going
    through the same code as MSTOrderer/SOMOrderer, so the fallback has to be
    exercised here separately -- the MST-only coverage above would not catch a
    regression specific to this path.
    """
    q, p = _arc_quantity()
    q["x"] = u.uconvert(x_unit, q["x"])
    q["y"] = u.uconvert(y_unit, q["y"])
    orderer = pcf.orderers.LocalFlowOrderer(metric_scale=u.Q(1.0, "kpc"))
    result = orderer.order(q, p, metadata=StateMetadata(usys=u.unitsystems.galactic))
    assert result.chord.unit == u.unit(expected)


def test_som_quantity_metric_scale_accepts_a_time_unit():
    """``SOMOrderer``'s ``metric_scale`` must be stripped like everything else.

    #85: ``FullPhaseSpaceDistanceMetric`` multiplies ``metric_scale`` by a
    velocity difference to get a position difference, so it is dimensionally
    a time -- the metric's own docstring says so. The Quantity dispatch
    stripped ``positions``/``velocities`` but forwarded ``self`` (and so
    ``self.metric_scale``) untouched into the plain, non-``quax`` SOM core:
    a genuine time Quantity there reached ``metric_scale * d_vel`` still
    unit-ful while ``d_vel`` was already a bare number, raising
    ``UnitConversionError: 'Myr2' and '' (dimensionless) are not
    convertible`` from inside the squared term.
    """
    q, p = _arc_quantity()
    usys = u.unitsystems.galactic
    result = pcf.orderers.SOMOrderer(
        n_prototypes=12,
        metric=pcf.metrics.FullPhaseSpaceDistanceMetric(),
        metric_scale=u.Q(50.0, "Myr"),
    ).order(q, p, metadata=StateMetadata(usys=usys))
    assert int(result.n_visited) == 120

    # The Quantity path must agree with the stripped one, not just avoid
    # raising -- a dispatch that silently dropped metric_scale to 0 would
    # also pass the assertion above.
    q_plain = {k: u.ustrip(usys, v) for k, v in q.items()}
    p_plain = {k: u.ustrip(usys, v) for k, v in p.items()}
    stripped = pcf.orderers.SOMOrderer(
        n_prototypes=12,
        metric=pcf.metrics.FullPhaseSpaceDistanceMetric(),
        metric_scale=float(u.ustrip(usys, u.Q(50.0, "Myr"))),
    ).order(q_plain, p_plain)
    assert jnp.array_equal(result.indices, stripped.indices)
