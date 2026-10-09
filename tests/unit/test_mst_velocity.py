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


class TestMissingVelocities:
    """Mechanisms 1 and 2 refuse tracers without a velocity direction."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0], ids=["nan", "inf", "zero"])
    @pytest.mark.parametrize(
        "kw",
        [{"velocity_weight": 5.0}, {"sever_cos_threshold": 0.0}],
        ids=["velocity_weight", "sever"],
    )
    def test_raises(self, kw, bad):
        """One directionless velocity raises instead of bridging the arms.

        Every silent stand-in (cos = 0, cos = 1, neighbour imputation) lets
        that one tracer reconnect the anti-parallel arms of the hairpin.
        """
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[20].set(bad), "y": vel["y"]}
        if bad == 0.0:
            vel["y"] = vel["y"].at[20].set(0.0)
        orderer = pcf.orderers.MSTOrderer(
            k=6, jump_cap=1.0, on_disconnected="largest", **kw
        )
        with pytest.raises(ValueError, match="velocity direction"):
            orderer.order(pos, vel)

    def test_raises_under_jit(self):
        """Traced, the same error surfaces from the host stage at run time."""
        pos, vel = _hairpin()
        vel = {"x": vel["x"].at[20].set(jnp.nan), "y": vel["y"]}
        orderer = pcf.orderers.MSTOrderer(k=6, jump_cap=1.0, velocity_weight=5.0)
        with pytest.raises(Exception, match="velocity direction"):
            jax.block_until_ready(
                jax.jit(lambda p, v: orderer.order(p, v).indices)(pos, vel)
            )

    @pytest.mark.parametrize(
        "kw", [{}, {"orient_by_velocity": True}], ids=["spatial", "orient"]
    )
    def test_unused_or_tolerant_mechanisms_accept_nan(self, kw):
        """Velocities the graph never compares may be missing."""
        x = np.linspace(0.0, 10.0, 60)
        pos = {"x": jnp.asarray(x), "y": jnp.zeros(60)}
        vel = {"x": jnp.ones(60).at[10:15].set(jnp.nan), "y": jnp.zeros(60)}
        res = pcf.orderers.MSTOrderer(k=6, jump_cap=2.0, **kw).order(pos, vel)
        steps = np.diff(np.asarray(res.ordering))
        assert len(steps) == 59
        assert np.all(np.abs(steps) == 1)  # neighbours along the line...
        assert np.all(steps == steps[0])  # ...all in one direction

    def test_small_unit_velocities_keep_their_direction(self):
        """|v| ~ 1e-7 still separates the arms (an absolute floor ignored them)."""
        pos, vel = _hairpin(n_a=50, n_b=40)
        vel = {c: v * 1e-7 for c, v in vel.items()}
        res = pcf.orderers.MSTOrderer(
            k=6, jump_cap=1.0, sever_cos_threshold=0.0, on_disconnected="largest"
        ).order(pos, vel)
        assert int((res.indices >= 0).sum()) == 50
