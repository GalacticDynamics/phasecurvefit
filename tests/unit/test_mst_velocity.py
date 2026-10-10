"""Tests for the opt-in velocity mechanisms of ``MSTOrderer``.

Fixture: a *hairpin* — two spatially-adjacent arms (vertical gap far smaller than
in-arm spacing) with **opposite** velocities. Pure-spatial ordering zigzags
between the arms; velocity information should keep them apart or orient them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.orderers import mst as mst_module

pytest.importorskip("scipy")


def _hairpin(n_a=50, n_b=40, gap=0.05):
    """Two close, anti-parallel arms. Arm A: +x velocity; arm B: -x velocity."""
    xa = np.linspace(0.0, 10.0, n_a)
    xb = np.linspace(0.0, 10.0, n_b)
    x = np.concatenate([xa, xb])
    y = np.concatenate([np.zeros(n_a), np.full(n_b, gap)])
    vx = np.concatenate([np.ones(n_a), -np.ones(n_b)])
    vy = np.zeros(n_a + n_b)
    pos = {"x": jnp.asarray(x), "y": jnp.asarray(y)}
    vel = {"x": jnp.asarray(vx), "y": jnp.asarray(vy)}
    return pos, vel


def _sign_flips(vals):
    s = np.sign(np.asarray(vals))
    return int(np.sum(s[1:] != s[:-1]))


class TestPhaseSpaceEdgeWeights:
    """Mechanism 1: velocity_weight penalises anti-parallel edges."""

    def test_velocity_weight_reduces_zigzag(self):
        """Velocity weight reduces zigzag."""
        pos, vel = _hairpin()
        o_spatial = pcf.orderers.MSTOrderer(k=6, jump_cap=1.0).order(pos, vel)
        o_vel = pcf.orderers.MSTOrderer(k=6, jump_cap=1.0, velocity_weight=5.0).order(
            pos, vel
        )
        vx = np.asarray(vel["x"])
        flips_spatial = _sign_flips(vx[np.asarray(o_spatial.ordering)])
        flips_vel = _sign_flips(vx[np.asarray(o_vel.ordering)])
        assert flips_vel < flips_spatial

    def test_weight_zero_is_spatial(self):
        """Weight zero is spatial."""
        pos, vel = _hairpin()
        a = pcf.orderers.MSTOrderer(k=6, jump_cap=1.0).order(pos, vel)
        b = pcf.orderers.MSTOrderer(k=6, jump_cap=1.0, velocity_weight=0.0).order(
            pos, vel
        )
        assert jnp.array_equal(a.indices, b.indices)


class TestVelocitySevering:
    """Mechanism 2: sever_cos_threshold cuts anti-parallel edges."""

    def test_severing_separates_arms(self):
        """Severing separates arms."""
        pos, vel = _hairpin(n_a=50, n_b=40)
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=1.0, sever_cos_threshold=0.0, on_disconnected="largest"
        ).order(pos, vel)
        # cross-arm edges (cos ~ -1) severed -> arms are separate components;
        # the larger arm (A, 50 pts, +x velocity) is ordered, the rest unvisited.
        assert int((res.indices >= 0).sum()) == 50
        vx_ordered = np.asarray(vel["x"])[np.asarray(res.ordering)]
        assert np.all(vx_ordered > 0)


class TestTipOrientation:
    """Mechanism 3: orient_by_velocity fixes the gamma sign along velocity."""

    def test_orientation_follows_velocity(self):
        """Orientation follows velocity."""
        # single arm from x=0..10, velocity pointing in -x
        x = np.linspace(0.0, 10.0, 60)
        pos = {"x": jnp.asarray(x), "y": jnp.zeros(60)}
        vel_neg = {"x": -jnp.ones(60), "y": jnp.zeros(60)}
        vel_pos = {"x": jnp.ones(60), "y": jnp.zeros(60)}

        o_neg = pcf.orderers.MSTOrderer(
            k=6, jump_cap=2.0, orient_by_velocity=True
        ).order(pos, vel_neg)
        o_pos = pcf.orderers.MSTOrderer(
            k=6, jump_cap=2.0, orient_by_velocity=True
        ).order(pos, vel_pos)

        xo_neg = np.asarray(pos["x"])[np.asarray(o_neg.ordering)]
        xo_pos = np.asarray(pos["x"])[np.asarray(o_pos.ordering)]
        # gamma increases along velocity: -x velocity -> ordering high-x to low-x
        assert xo_neg[0] > xo_neg[-1]
        # flipping the velocity flips the orientation
        assert xo_pos[0] < xo_pos[-1]


_MECHANISMS = {
    "velocity_weight": {"velocity_weight": 5.0},
    "sever_cos_threshold": {"sever_cos_threshold": 0.0},
    "orient_by_velocity": {"orient_by_velocity": True},
}


def _line(n=60):
    """Build a straight line moving in +x, which every mechanism must order."""
    x = np.linspace(0.0, 10.0, n)
    pos = {"x": jnp.asarray(x), "y": jnp.zeros(n)}
    vel = {"x": jnp.ones(n), "y": jnp.zeros(n)}
    return x, pos, vel


class TestMissingVelocities:
    """NaN velocities are missing data, policed by ``nan_policy``; inf is an error.

    Every silent stand-in for a missing velocity (cos = 0, cos = 1, neighbour
    imputation) lets one tracer reconnect the anti-parallel arms of the
    hairpin, so skipping them has to be asked for.
    """

    @pytest.mark.parametrize("mechanism", sorted(_MECHANISMS))
    def test_nan_raises_by_default(self, mechanism):
        """Skipping a missing velocity has to be asked for."""
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[20].set(jnp.nan), "y": vel["y"]}
        orderer = pcf.orderers.MSTOrderer(
            k=6, jump_cap=1.0, on_disconnected="largest", **_MECHANISMS[mechanism]
        )
        with pytest.raises(ValueError, match="nan_policy='omit'") as err:
            orderer.order(pos, vel)
        assert mechanism in str(err.value)
        for other in set(_MECHANISMS) - {mechanism}:  # names only what is on
            assert other not in str(err.value)

    @pytest.mark.parametrize("policy", ["raise", "omit"])
    @pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["+inf", "-inf"])
    @pytest.mark.parametrize("mechanism", sorted(_MECHANISMS))
    def test_inf_raises_under_either_policy(self, mechanism, bad, policy):
        """Infinity is not a missing measurement, so ``omit`` does not cover it."""
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[20].set(bad), "y": vel["y"]}
        orderer = pcf.orderers.MSTOrderer(
            k=6,
            jump_cap=1.0,
            on_disconnected="largest",
            nan_policy=policy,
            **_MECHANISMS[mechanism],
        )
        with pytest.raises(ValueError, match="infinite"):
            orderer.order(pos, vel)

    def test_raises_under_jit(self):
        """Traced, the same error surfaces from the host stage at run time."""
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[20].set(jnp.nan), "y": vel["y"]}
        orderer = pcf.orderers.MSTOrderer(k=6, jump_cap=1.0, velocity_weight=5.0)
        with pytest.raises(jax.errors.JaxRuntimeError, match="nan_policy='omit'"):
            jax.block_until_ready(
                jax.jit(lambda p, v: orderer.order(p, v).indices)(pos, vel)
            )

    def test_rejects_an_unknown_nan_policy(self):
        """A typo fails at construction, not at the first NaN."""
        with pytest.raises(ValueError, match="nan_policy"):
            pcf.orderers.MSTOrderer(nan_policy="propagate")

    @pytest.mark.parametrize("mechanism", [None, *sorted(_MECHANISMS)])
    def test_omit_accepts_nan(self, mechanism):
        """Under ``omit`` (or with no mechanism on) a NaN velocity is tolerated.

        The tracers with a velocity stay in exact order. A run of missing ones
        joins as leaves, each ordered at a backbone vertex beside the run
        (projection is to the nearest vertex), so within the run the order is
        only as good as the gap it fills.
        """
        _, pos, vel = _line()
        vel = {"x": vel["x"].at[10:15].set(jnp.nan), "y": vel["y"]}
        kw = {} if mechanism is None else _MECHANISMS[mechanism]
        res = pcf.orderers.MSTOrderer(k=6, jump_cap=2.0, nan_policy="omit", **kw).order(
            pos, vel
        )
        order = np.asarray(res.ordering)
        assert len(order) == 60
        run = (order >= 10) & (order < 15)
        steps = np.diff(order[~run])
        assert np.all(np.sign(steps) == np.sign(steps[0]))  # monotone along x
        ranks = np.flatnonzero(run)
        rank_9, rank_15 = (int(np.flatnonzero(order == i)[0]) for i in (9, 15))
        # In the gap between 9 and 15, give or take one: a leaf ties with the
        # vertex it projects to and may sort just past it.
        assert np.all(np.abs(ranks - (rank_9 + rank_15) / 2) <= 4)

    def test_spatial_default_never_reads_velocities(self):
        """No mechanism on: even the default ``raise`` accepts NaN."""
        _, pos, vel = _line()
        vel = {"x": vel["x"].at[10:15].set(jnp.nan), "y": vel["y"]}
        res = pcf.orderers.MSTOrderer(k=6, jump_cap=2.0).order(pos, vel)
        assert int(res.n_visited) == 60

    def test_a_stationary_tracer_is_not_severed(self):
        """A zero velocity is data: it joins as a leaf rather than being cut off.

        Read as perpendicular (cos = 0), any threshold above 0 cut it off and
        ``on_disconnected="raise"`` raised.
        """
        _, pos, vel = _line()
        vel = {"x": vel["x"].at[30].set(0.0), "y": vel["y"]}
        res = pcf.orderers.MSTOrderer(k=6, jump_cap=2.0, sever_cos_threshold=0.5).order(
            pos, vel
        )
        assert int(res.n_visited) == 60

    @pytest.mark.parametrize("where", [20, 70], ids=["arm-A", "arm-B"])
    @pytest.mark.parametrize("bad", [0.0, jnp.nan], ids=["stationary", "omitted-nan"])
    def test_a_directionless_tracer_does_not_bridge_the_arms(self, bad, where):
        """It joins as a leaf, so severing still separates the arms.

        Given any cosine (0 or 1), one such tracer merged the hairpin's arms:
        all 90 tracers kept instead of arm A's 50.
        """
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[where].set(bad), "y": vel["y"]}
        res = pcf.orderers.MSTOrderer(
            k=6,
            jump_cap=1.0,
            sever_cos_threshold=0.0,
            on_disconnected="largest",
            nan_policy="omit",
        ).order(pos, vel)
        assert int(res.n_visited) <= 51  # arm A, plus at most that one leaf

    def test_one_directed_tracer_still_makes_every_other_a_leaf(self):
        """The split holds down to a single directed tracer: a star around it.

        Skipping it there left the directionless tracers in the full kNN
        graph -- the arm-bridging behaviour the split exists to avoid.
        """
        _, pos, vel = _line(n=6)
        vel = {"x": jnp.zeros(6).at[0].set(1.0), "y": vel["y"]}
        P = np.stack([np.asarray(pos[c]) for c in "xy"], axis=1)
        V = np.stack([np.asarray(vel[c]) for c in "xy"], axis=1)
        split = mst_module._directionless_as_leaves(P, V, k=3, workers=1)
        assert split is not None
        rows, cols, _ = split
        assert sorted(rows.tolist()) == [1, 2, 3, 4, 5]  # one edge each...
        assert set(cols.tolist()) == {0}  # ...to the directed tracer

    @pytest.mark.parametrize("mechanism", ["velocity_weight", "sever_cos_threshold"])
    def test_a_stationary_tracer_does_not_raise(self, mechanism):
        """Under the default ``raise``, a zero velocity is accepted as data."""
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[20].set(0.0), "y": vel["y"]}
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=1.0, on_disconnected="largest", **_MECHANISMS[mechanism]
        ).order(pos, vel)
        assert int(res.n_visited) > 0

    @pytest.mark.parametrize("bad", [jnp.nan, 0.0], ids=["omitted-nan", "stationary"])
    def test_orientation_skips_tracers_without_a_direction(self, bad):
        """One missing or stationary velocity cannot decide the orientation."""
        x, pos, vel = _line()
        vel = {"x": vel["x"].at[30].set(bad), "y": vel["y"]}
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=2.0, orient_by_velocity=True, nan_policy="omit"
        ).order(pos, vel)
        steps = np.diff(x[np.asarray(res.ordering)])
        assert np.all(steps > 0)

    def test_orientation_skips_partly_non_finite_velocities(self):
        """A velocity with any NaN component is skipped whole.

        On a diagonal, (NaN, -1e6) used to keep its finite y component, whose
        huge term against the flow flipped the ordering.
        """
        t = np.linspace(0.0, 10.0, 60)
        pos = {"x": jnp.asarray(t), "y": jnp.asarray(t)}
        vel = {
            "x": jnp.ones(60).at[30].set(jnp.nan),
            "y": jnp.ones(60).at[30].set(-1e6),
        }
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=2.0, orient_by_velocity=True, nan_policy="omit"
        ).order(pos, vel)
        steps = np.diff(t[np.asarray(res.ordering)])
        assert np.all(steps > 0)

    def test_default_pipeline_omits_missing_velocities(self):
        """``pcf.order`` with no orderer must not reject a catalogue with gaps."""
        _, pos, vel = _line()
        vel = {"x": vel["x"].at[10:15].set(jnp.nan), "y": vel["y"]}
        assert int(pcf.order(pos, vel).n_visited) == 60

    @pytest.mark.parametrize("mechanism", sorted(_MECHANISMS))
    def test_integer_velocities(self, mechanism):
        """Integer velocities, zeros included, are valid input."""
        _, pos, _ = _line()
        vel = {
            "x": jnp.ones(60, dtype=jnp.int32).at[30].set(0),
            "y": jnp.zeros(60, dtype=jnp.int32),
        }
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=2.0, **_MECHANISMS[mechanism]
        ).order(pos, vel)
        assert int(res.n_visited) == 60

    def test_small_unit_velocities_keep_their_direction(self):
        """|v| ~ 1e-7 still separates the arms (an absolute floor ignored them)."""
        pos, vel = _hairpin(n_a=50, n_b=40)
        vel = {c: v * 1e-7 for c, v in vel.items()}
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=1.0, sever_cos_threshold=0.0, on_disconnected="largest"
        ).order(pos, vel)
        assert int((res.indices >= 0).sum()) == 50
