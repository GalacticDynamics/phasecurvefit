"""Quick test of bounded while_loop implementation."""

import jax.numpy as jnp

import unxt as u

import localflowwalk as lfw
from localflowwalk import KDTreeStrategy

# Quick test with small epitrochoid data
n = 50
t = jnp.linspace(5.0 * jnp.pi / 180.0, 355.0 * jnp.pi / 180.0, n)
R, r, d = 4.0, 1.0, 3.5
ratio = (R + r) / r
x = ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
y = ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0
dx = (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
dy = ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

pos = {"x": u.Q(x, "m"), "y": u.Q(y, "m")}
vel = {"x": u.Q(dx, "m/s"), "y": u.Q(dy, "m/s")}

print("Testing walk_local_flow with bounded while_loop...")
res = lfw.walk_local_flow(
    pos, vel, start_idx=0, lam=u.Q(0.0, "m"), strategy=KDTreeStrategy(k=10)
)
ordered = [i for i in res.ordered_indices if i >= 0]
print(f"✓ Success! Walk completed, ordered {len(ordered)} out of {n} points")
print(f"Result type: {type(res)}")
