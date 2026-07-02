"""Tests for the opt-in velocity mechanisms of ``MSTOrderer``.

Fixture: a *hairpin* — two spatially-adjacent arms (vertical gap far smaller than
in-arm spacing) with **opposite** velocities. Pure-spatial ordering zigzags
between the arms; velocity information should keep them apart or orient them.
"""

import jax.numpy as jnp
import numpy as np

import phasecurvefit as pcf


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
