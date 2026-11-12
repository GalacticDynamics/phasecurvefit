import time

import jax
import numpy as np
from common import make_data, normal_offsets

import phasecurvefit as pcf

pos, vel = make_data(0)
cfg = pcf.nn.TrainingConfig(show_pbar=False)
for k in range(2):
    t0 = time.perf_counter()
    o, r, _ = pcf.fit_track(pos, vel, key=jax.random.key(k), training_config=cfg)
    g = np.linspace(*r.track_range, 400)
    xy = r(jax.numpy.asarray(g))
    xy = np.stack([np.asarray(xy["x"]), np.asarray(xy["y"])], -1)
    print(time.perf_counter() - t0, np.nanmax(np.abs(normal_offsets(xy))))
