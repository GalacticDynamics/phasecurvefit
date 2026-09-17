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


def test_som_orderer_with_quantities():
    """SOMOrderer accepts Quantity input and returns unit-aware output."""
    import unxt as u

    import phasecurvefit as pcf

    # Positions in "pc", the "galactic" usys length unit in "kpc" -- different,
    # so the unit assertions exercise `u.uconvert` rather than coinciding.
    ang = jnp.linspace(0.0, jnp.pi, 60)
    pos = {
        "x": u.Quantity(5000.0 * jnp.cos(ang), "pc"),
        "y": u.Quantity(5000.0 * jnp.sin(ang), "pc"),
    }
    vel = {
        "x": u.Quantity(-jnp.sin(ang), "km/s"),
        "y": u.Quantity(jnp.cos(ang), "km/s"),
    }
    usys = u.unitsystem("galactic")
    result = pcf.order(
        pos,
        vel,
        pcf.orderers.SOMOrderer(n_prototypes=10),
        metadata=pcf.StateMetadata(usys=usys),
    )
    assert int(result.n_visited) == 60
    assert result.backbone["x"].unit == pos["x"].unit
    assert result.chord.unit == pos["x"].unit


def test_chained_mst_som_with_quantities():
    """A chained MST -> SOM orderer threads ``init`` through the Quantity path.

    This is the path where the prior stage's ``OrderingResult`` (itself
    Quantity-valued from the MST dispatch) must survive the strip/reattach
    round trip and reach SOM's dispatch as a usable ``init``.
    """
    import unxt as u

    import phasecurvefit as pcf

    q, p = _arc_quantity()
    usys = u.unitsystems.galactic
    chain = pcf.orderers.MSTOrderer(k=8, jump_cap=2.0) | pcf.orderers.SOMOrderer(
        n_prototypes=10
    )
    result = pcf.order(q, p, chain, metadata=pcf.StateMetadata(usys=usys))

    # Parity check: the same chain on unit-stripped arrays. A Quantity
    # dispatch that dropped `init` would run SOM standalone here and diverge.
    q_plain = {k: u.ustrip(usys, v) for k, v in q.items()}
    p_plain = {k: u.ustrip(usys, v) for k, v in p.items()}
    result_plain = pcf.order(q_plain, p_plain, chain)

    assert int(result.n_visited) == 120
    assert jnp.array_equal(result.indices, result_plain.indices)
    assert isinstance(result.backbone["x"], u.AbstractQuantity)
    assert result.backbone["x"].unit == q["x"].unit
    assert isinstance(result.chord, u.AbstractQuantity)
    assert result.chord.unit == q["x"].unit


@pytest.mark.parametrize(
    "orderer",
    [
        pcf.orderers.MSTOrderer(k=8, jump_cap=3.0),
        pcf.orderers.SOMOrderer(n_prototypes=8),
        pcf.orderers.MSTOrderer(k=8, jump_cap=3.0)
        | pcf.orderers.SOMOrderer(n_prototypes=8),
    ],
    ids=["mst", "som", "chain"],
)
def test_quantity_without_usys_raises_a_helpful_error(orderer):
    """Omitting ``metadata`` must surface the intended message, not AttributeError.

    ``None`` reaches the interop layer, which has to handle it.
    """
    q, p = _arc_quantity()
    with pytest.raises(TypeError, match="`usys` must be provided"):
        pcf.order(q, p, orderer)


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
