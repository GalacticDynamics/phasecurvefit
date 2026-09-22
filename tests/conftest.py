"""Shared curve fixtures for the test suite.

The orderers are all judged the same way: build a curve whose true parameter is
known, shuffle it, order it, and compare. The curves themselves were being
redefined per module -- ``_helix`` existed three times with three different
radii -- so "the helix test" meant a different curve depending on the file.

Each fixture is a *factory*: it returns a callable, so a test asks for the
curve it needs rather than accepting one module's defaults.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from scipy.stats import spearmanr


class Curve(NamedTuple):
    """A curve with its ground-truth parameter.

    ``t`` is the true ordering parameter, so ``spearman(result, curve)``
    measures how well an orderer recovered it.
    """

    positions: dict[str, Array]
    velocities: dict[str, Array]
    t: np.ndarray


def _shuffle(pos, vel, t, rng):
    """Destroy the input ordering, so nothing can pass by accident."""
    perm = rng.permutation(len(t))
    return Curve(
        {k: v[perm] for k, v in pos.items()},
        {k: v[perm] for k, v in vel.items()},
        np.asarray(t)[perm],
    )


def _flat_velocity(n):
    """Return unit speed along +x, for curves whose velocity is never read."""
    return {"x": jnp.ones(n), "y": jnp.zeros(n)}


@pytest.fixture
def flat_vel():
    """Return `_flat_velocity`, for a bespoke ``pos`` with a trivial ``vel``."""
    return _flat_velocity


@pytest.fixture
def line():
    """Build ``n`` points evenly spaced along +x over ``[0, length]``, in order.

    Deliberately not :func:`straight`, which shuffles and carries the true
    derivative so a velocity-aware metric weighs it correctly. These are the
    closed-form cases for the SOM core -- the chord along a straight backbone
    is just the x coordinate -- where the input order is load-bearing (the
    prototypes are asserted monotone) and the velocity is never read, every
    caller passing ``SpatialDistanceMetric``.

    Returns ``(pos, vel)``, so it serves as data and as a backbone alike.
    """

    def make(n, length):
        pos = {"x": jnp.linspace(0.0, length, n), "y": jnp.zeros(n)}
        return pos, _flat_velocity(n)

    return make


@pytest.fixture
def straight():
    """Build a straight line: the trivial case every orderer must get right."""

    def make(n=200, length=10.0, scatter=0.0, sign=1.0, seed=0, *, shuffle=True):
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, n)
        x = sign * length * t + rng.normal(0, scatter, n)
        pos = {"x": jnp.asarray(x), "y": jnp.asarray(rng.normal(0, scatter, n))}
        # dx/dt of `sign * length * t`, not just the direction: a velocity-aware
        # metric weighs velocity against position separations, so a unit
        # velocity here would quietly under-weight it.
        vel = {"x": jnp.full(n, sign * length), "y": jnp.zeros(n)}
        return _shuffle(pos, vel, t, rng) if shuffle else Curve(pos, vel, t)

    return make


@pytest.fixture
def arc():
    """Build a half-circle: open, monotone along its own principal axis."""

    def make(n=200, radius=5.0, scatter=0.0, seed=0, *, shuffle=True):
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, n)
        ang = np.pi * t
        pos = {
            "x": jnp.asarray(radius * np.cos(ang) + rng.normal(0, scatter, n)),
            "y": jnp.asarray(radius * np.sin(ang) + rng.normal(0, scatter, n)),
        }
        dtheta = np.pi  # d(ang)/dt
        vel = {
            "x": jnp.asarray(-radius * dtheta * np.sin(ang)),
            "y": jnp.asarray(radius * dtheta * np.cos(ang)),
        }
        return _shuffle(pos, vel, t, rng) if shuffle else Curve(pos, vel, t)

    return make


@pytest.fixture
def helix():
    """Build a helix, parametrized by turns.

    Past about one turn it doubles back along its own first principal axis, so
    ``turns`` selects whether ``init_prototypes``'s PCA fallback is inside its
    precondition or outside it.
    """

    def make(
        n=200, turns=1.0, radius=1.0, height=2.0, scatter=0.0, seed=0, *, shuffle=True
    ):
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, 1.0, n)
        ang = 2 * np.pi * turns * t
        dtheta = 2 * np.pi * turns
        pos = {
            "x": jnp.asarray(radius * np.cos(ang) + rng.normal(0, scatter, n)),
            "y": jnp.asarray(radius * np.sin(ang) + rng.normal(0, scatter, n)),
            "z": jnp.asarray(height * t),
        }
        vel = {
            "x": jnp.asarray(-radius * dtheta * np.sin(ang)),
            "y": jnp.asarray(radius * dtheta * np.cos(ang)),
            "z": jnp.full(n, height),
        }
        return _shuffle(pos, vel, t, rng) if shuffle else Curve(pos, vel, t)

    return make


@pytest.fixture
def epitrochoid():
    """Build a self-intersecting epitrochoid: many lobes, branches that cross.

    At a crossing the two branches are spatially coincident and differ only in
    velocity, so a position-only metric cannot separate them.
    """

    def make(
        n=700,
        noise=6.0,
        scale=120.0,
        R=5.0,
        r=1.0,
        d=4.5,
        seed=3,
        *,
        shuffle=True,
    ):
        rng = np.random.default_rng(seed)
        t = np.linspace(np.deg2rad(5), np.deg2rad(355), n)
        ratio = (R + r) / r
        x = scale * ((R + r) * np.cos(t) - d * np.cos(ratio * t)) / 5.0
        y = scale * ((R + r) * np.sin(t) - d * np.sin(ratio * t)) / 5.0
        dx = scale * (-(R + r) * np.sin(t) + d * ratio * np.sin(ratio * t)) / 5.0
        dy = scale * ((R + r) * np.cos(t) - d * ratio * np.cos(ratio * t)) / 5.0
        pos = {
            "x": jnp.asarray(x + rng.normal(0, noise, n)),
            "y": jnp.asarray(y + rng.normal(0, noise, n)),
        }
        vel = {"x": jnp.asarray(dx), "y": jnp.asarray(dy)}
        return _shuffle(pos, vel, t, rng) if shuffle else Curve(pos, vel, t)

    return make


@pytest.fixture
def spearman():
    """|rho| between an ordering result and the curve's true parameter."""

    def measure(result, truth):
        """``truth`` is a ``Curve`` or the bare array of true parameters."""
        t = truth.t if isinstance(truth, Curve) else np.asarray(truth)
        idx = np.asarray(result.ordering)
        return abs(spearmanr(t[idx], np.arange(idx.size)).statistic)

    return measure
