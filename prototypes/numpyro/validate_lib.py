"""Known-truth coverage of the library's sample_track with its defaults.

Usage: python validate_lib.py <case> <seed>. Writes out_lib/<case>_ds<seed>.npz.
"""

import sys
import time

import cases
import jax
import jax.numpy as jnp
import numpy as np

import phasecurvefit as pcf

case, seed = sys.argv[1], int(sys.argv[2])
q, p, w, truth = cases.make(case, seed)
keys = "xyz"[: q.shape[1]]
pos = {k: jnp.asarray(q[:, i]) for i, k in enumerate(keys)}
vel = {k: jnp.asarray(p[:, i]) for i, k in enumerate(keys)}
kw = {"n_basis": 32, "n_grid": 512} if case == "loop" else {}
t0 = time.perf_counter()
post = pcf.sample_track(pos, vel, key=jax.random.key(seed), **kw)
dt = time.perf_counter() - t0


def curve(m):
    x = m(jnp.linspace(*m.track_range, 600))
    return np.stack([np.asarray(x[k]) for k in keys], -1)


offs = np.array([cases.offsets(curve(m), truth) for m in post.members])
out = {"svi": offs, "t": dt, "f_bg": float(np.mean(post.samples["f_bg"]))}
if case == "vary":  # width along the true curve, against the truth
    st = cases._arc_frac(truth)
    wid = []
    for m in post.members:
        g = jnp.linspace(*m.track_range, 600)
        c = curve(m)
        s_ = st[np.argmin(np.linalg.norm(c[:, None] - truth[None, ::4], axis=2), 1) * 4]
        o = np.argsort(s_)
        wid.append(
            np.interp(
                cases.S_GRID,
                s_[o],
                np.asarray(m.width(g))[o],
                left=np.nan,
                right=np.nan,
            )
        )
    out["wid"] = np.array(wid)
np.savez(f"out_lib/{case}_ds{seed}.npz", **out)
c = cases.coverage(offs[None])
print(
    case,
    seed,
    f"{c['68'].mean():.2f}/{c['95'].mean():.2f} t {dt:.0f}s f_bg {out['f_bg']:.3f}",
    flush=True,
)
