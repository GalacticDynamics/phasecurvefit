"""Tests for ChainOrderer and the ``init=`` ordering contract."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

import phasecurvefit as pcf
from phasecurvefit._src.orderers.localflow import _resolve_start_idx


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


def _shuffled_arc(n=400, seed=3):
    """Build an arc whose input order carries no ordering information.

    Index 0 is somewhere in the middle, so a walk that starts there runs out of
    curve in one direction and has to be told otherwise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    ang = np.pi * t
    x = 5 * np.cos(ang) + rng.normal(0, 0.03, n)
    y = 5 * np.sin(ang) + rng.normal(0, 0.03, n)
    perm = rng.permutation(n)
    pos = {"x": jnp.asarray(x)[perm], "y": jnp.asarray(y)[perm]}
    vel = {"x": jnp.asarray(-np.sin(ang))[perm], "y": jnp.asarray(np.cos(ang))[perm]}
    return pos, vel, t[perm]


def _rho(result, truth):
    idx = np.asarray(result.ordering)
    return abs(spearmanr(truth[idx], np.arange(idx.size)).statistic)


class TestLocalFlowStartFromInit:
    """The walk should take its start point from a prior stage."""

    def test_chaining_after_mst_beats_the_arbitrary_default(self):
        """Starting from a real tip is the whole point of chaining here."""
        pos, vel, truth = _shuffled_arc()
        walk = pcf.orderers.LocalFlowOrderer(direction="forward")
        mst = pcf.orderers.MSTOrderer(k=10, jump_cap=3.0, on_disconnected="largest")

        alone = _rho(walk.order(pos, vel), truth)
        chained = _rho((mst | walk).order(pos, vel), truth)

        assert alone < 0.9, "fixture no longer exercises a bad default start"
        assert chained > 0.99
        assert chained > alone

    def test_start_idx_is_taken_from_the_prior_orderings_first_point(self):
        """``init`` supplies the index; MST orders tip-to-tip so it is an end."""
        pos, vel, _ = _shuffled_arc()
        prior = pcf.orderers.MSTOrderer(
            k=10, jump_cap=3.0, on_disconnected="largest"
        ).order(pos, vel)
        resolved = _resolve_start_idx(None, prior)
        assert resolved == int(np.asarray(prior.ordering)[0])

    @pytest.mark.parametrize(
        ("start_idx", "indices", "expected"),
        [
            (7, [0, 1, 2], 7),
            (7, None, 7),
            (None, None, 0),
            (None, [-1, -1, 7, 3, 5], 7),
            (None, [-1, -1, -1], 0),
        ],
        ids=["explicit-wins", "explicit-unchained", "unchained", "padded", "nothing"],
    )
    def test_resolve_start_idx(self, start_idx, indices, expected):
        """The index is the first *visited* entry, and an explicit one wins.

        ``_resolve_start_idx`` reads only ``init.indices``, so the cases are
        stated as index arrays rather than by running an orderer to make one.
        """
        init = None
        if indices is not None:
            n = len(indices)
            init = pcf.orderers.OrderingResult(
                positions={"x": jnp.zeros(n)},
                velocities={"x": jnp.zeros(n)},
                indices=jnp.asarray(indices, dtype=jnp.int32),
            )
        assert _resolve_start_idx(start_idx, init) == expected


def test_order_facade_forwards_init():
    """``pcf.order`` accepts ``init`` and passes it to the orderer.

    The facade has two branches -- with and without ``init`` -- because an
    orderer predating the parameter cannot be handed it at all.
    """
    pos, vel, _ = _shuffled_arc(n=120)
    prior = pcf.orderers.MSTOrderer(k=8, jump_cap=3.0, on_disconnected="largest").order(
        pos, vel
    )
    chained = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer(), init=prior)
    assert int(np.asarray(chained.ordering)[0]) == int(np.asarray(prior.ordering)[0])

    # ... and without it, the walk falls back to its own default.
    plain = pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer())
    assert int(np.asarray(plain.ordering)[0]) == 0
