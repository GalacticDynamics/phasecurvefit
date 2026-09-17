"""Tests for ChainOrderer and the ``init=`` ordering contract."""

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf


def _line(n: int = 20):
    pos = {"x": jnp.linspace(0.0, 5.0, n), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    return pos, vel


def test_chain_of_two_runs():
    pos, vel = _line()
    chain = pcf.orderers.ChainOrderer(
        pcf.orderers.LocalFlowOrderer(), pcf.orderers.LocalFlowOrderer()
    )
    result = chain.order(pos, vel)
    assert int(result.n_visited) == 20


def test_or_operator_equals_explicit_chain():
    a, b, c = (
        pcf.orderers.LocalFlowOrderer(),
        pcf.orderers.MSTOrderer(k=5, jump_cap=3.0),
        pcf.orderers.LocalFlowOrderer(),
    )
    # Identity comparison: eqx.Module __eq__ can yield arrays, not bools.
    sugar = (a | b | c).stages
    explicit = pcf.orderers.ChainOrderer(a, b, c).stages
    assert len(sugar) == len(explicit)
    assert all(x is y for x, y in zip(sugar, explicit, strict=True))


def test_or_operator_flattens():
    a, b, c = (
        pcf.orderers.LocalFlowOrderer(),
        pcf.orderers.LocalFlowOrderer(),
        pcf.orderers.LocalFlowOrderer(),
    )
    assert len((a | b | c).stages) == 3


def test_three_stage_chain_orders():
    pos, vel = _line()
    chain = (
        pcf.orderers.LocalFlowOrderer()
        | pcf.orderers.MSTOrderer(k=5, jump_cap=3.0)
        | pcf.orderers.LocalFlowOrderer()
    )
    result = pcf.order(pos, vel, chain)
    assert int(result.n_visited) == 20


def test_init_is_threaded_to_stages():
    pos, vel = _line()
    seen = []

    class Recorder(pcf.orderers.AbstractOrderer):
        def order(self, positions, velocities, *, metadata=None, init=None):  # noqa: ARG002
            seen.append(init)
            return pcf.orderers.LocalFlowOrderer().order(positions, velocities)

    pcf.orderers.ChainOrderer(Recorder(), Recorder()).order(pos, vel)
    assert seen[0] is None
    assert isinstance(seen[1], pcf.orderers.OrderingResult)


def test_empty_chain_raises():
    with pytest.raises(ValueError, match="at least one stage"):
        pcf.orderers.ChainOrderer()


def test_existing_orderers_accept_and_ignore_init():
    pos, vel = _line()
    prior = pcf.orderers.LocalFlowOrderer().order(pos, vel)
    plain = pcf.orderers.MSTOrderer(k=5, jump_cap=3.0).order(pos, vel)
    with_init = pcf.orderers.MSTOrderer(k=5, jump_cap=3.0).order(pos, vel, init=prior)
    assert jnp.array_equal(plain.indices, with_init.indices)


class _LegacyOrderer(pcf.orderers.AbstractOrderer):
    """An orderer written against v0.3.1's contract, with no ``init`` parameter.

    ``init`` was added to ``AbstractOrderer.order`` after v0.3.1, so third-party
    orderers released against that API do not accept it. They must keep working
    where no prior result exists to forward.
    """

    def order(self, positions, velocities, *, metadata=None):  # noqa: ARG002
        return pcf.orderers.LocalFlowOrderer().order(positions, velocities)


def test_legacy_orderer_without_init_still_works_via_the_facade():
    pos, vel = _line()
    assert int(pcf.order(pos, vel, _LegacyOrderer()).n_visited) == 20


def test_legacy_orderer_works_as_the_head_of_a_chain():
    pos, vel = _line()
    chain = _LegacyOrderer() | pcf.orderers.MSTOrderer(k=5, jump_cap=3.0)
    assert int(pcf.order(pos, vel, chain).n_visited) == 20


def test_legacy_orderer_fails_clearly_when_asked_to_refine():
    """It cannot consume a prior result, so chaining it second must not be silent."""
    pos, vel = _line()
    chain = pcf.orderers.LocalFlowOrderer() | _LegacyOrderer()
    with pytest.raises(TypeError, match="init"):
        pcf.order(pos, vel, chain)
