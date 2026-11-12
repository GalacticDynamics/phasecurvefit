"""The shipped ``pcf.bootstrap_track`` on the clean arc, scored two ways.

Usage: python run_shipped.py <seed> <M>

Saves, per dataset: each member's signed offset from the truth along the true normal
(for the per-point interval, as in run_dataset.py), the reference's offset, and the
68%/95% ``TrackEnsemble.radius`` at the reference point nearest each true point (for
the tube about the reference).
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import S_GRID, make_data, normal_offsets, true_curve

import phasecurvefit as pcf

seed, M = int(sys.argv[1]), int(sys.argv[2])
pos, vel = make_data(seed)
cfg = pcf.nn.TrainingConfig(show_pbar=False)

t0 = time.perf_counter()
ens = pcf.bootstrap_track(
    pos, vel, key=jax.random.key(seed), n_samples=M, training_config=cfg
)
t = (time.perf_counter() - t0) / (M + 1)


def xy(r, g):
    q = r(g)
    return np.stack([np.asarray(q["x"]), np.asarray(q["y"])], -1)


A = np.array(
    [normal_offsets(xy(m, jnp.linspace(*m.track_range, 600))) for m in ens.members]
)
ref = ens.reference
g_ref = jnp.linspace(*ref.track_range, 2000)
ref_xy = xy(ref, g_ref)
ref_off = normal_offsets(ref_xy)

# the reference point nearest each true point, and the tube's radius there
xt, yt = true_curve(S_GRID)
truth = np.stack([xt, yt], -1)
near = np.argmin(((ref_xy[None] - truth[:, None]) ** 2).sum(-1), axis=1)
g = g_ref[near]
r68 = np.asarray(ens.radius(g, 0.68))
r95 = np.asarray(ens.radius(g, 0.95))

np.savez(f"out_shipped/ds{seed}.npz", A=A, ref=ref_off, r68=r68, r95=r95, t=t)
print(seed, f"{t:.1f} s per fit")
