"""Bootstrap by index resampling (keeps fit_track's outlier clipping on), on data
with 4% interlopers spread over the field. Usage: python run_interlopers.py <seed> <M>
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import make_data, normal_offsets

import phasecurvefit as pcf

seed, M = int(sys.argv[1]), int(sys.argv[2])
pos, vel = make_data(seed)
rng = np.random.default_rng(10_000 + seed)
n_bad = 16
bad_q = np.stack([rng.uniform(-1, 11, n_bad), rng.uniform(-3, 3, n_bad)], -1)
bad_p = rng.normal(0, 1, (n_bad, 2))
pos = {
    k: jnp.concatenate([pos[k], jnp.asarray(bad_q[:, i])]) for i, k in enumerate("xy")
}
vel = {
    k: jnp.concatenate([vel[k], jnp.asarray(bad_p[:, i])]) for i, k in enumerate("xy")
}
N = pos["x"].shape[0]
cfg = pcf.nn.TrainingConfig(show_pbar=False)

offs, t0 = [], time.perf_counter()
for m in range(M):
    k = jax.random.key(1000 * seed + m)
    idx = jax.random.randint(jax.random.fold_in(k, 7), (N,), 0, N)
    p = {c: v[idx] for c, v in pos.items()}
    v_ = {c: v[idx] for c, v in vel.items()}
    _, r, _ = pcf.fit_track(p, v_, key=k, training_config=cfg)
    xy = r(jnp.linspace(*r.track_range, 600))
    offs.append(
        normal_offsets(np.stack([np.asarray(xy["x"]), np.asarray(xy["y"])], -1))
    )
np.savez(f"out_bad/ds{seed}.npz", A=np.array(offs), tA=(time.perf_counter() - t0) / M)
print(seed, (time.perf_counter() - t0) / M)
