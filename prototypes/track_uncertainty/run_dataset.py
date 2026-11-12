"""Fit one mock dataset with three uncertainty methods; save offsets from the true track.

Usage: python run_dataset.py <dataset_seed> <n_members>

A  bootstrap: whole pipeline (MST -> SOM -> autoencoder) on multinomial resample
   counts passed as `weights`, with a fresh training key per member.
B  seed ensemble: same data, fresh training key per member (deep ensemble).
C  last-layer Laplace on member 0 of B: Gaussian posterior over the decoder's final
   linear layer, noise and prior precision set by evidence, gamma held fixed.
"""

import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from common import make_data, normal_offsets

import phasecurvefit as pcf
from phasecurvefit import som

seed, M = int(sys.argv[1]), int(sys.argv[2])
pos, vel = make_data(seed)
N = pos["x"].shape[0]
cfg = pcf.nn.TrainingConfig(show_pbar=False)


def track_xy(r, n=600):
    xy = r(jnp.linspace(*r.track_range, n))
    return np.stack([np.asarray(xy["x"]), np.asarray(xy["y"])], -1)


out, times = {}, {}
for name in ("A", "B"):
    offs, t0 = [], time.perf_counter()
    for m in range(M):
        k = jax.random.key(1000 * seed + m)
        w = som.bootstrap_weights(jax.random.fold_in(k, 7), N) if name == "A" else None
        _, r, _ = pcf.fit_track(pos, vel, key=k, weights=w, training_config=cfg)
        offs.append(normal_offsets(track_xy(r)))
        if name == "B" and m == 0:
            r0 = r
    out[name] = np.array(offs)
    times[name] = (time.perf_counter() - t0) / M

# ---- C: last-layer Laplace about r0 ----
t0 = time.perf_counter()
model = r0.model
dec = model.decoder
qs, _ = model.normalizer.transform(pos, vel)  # (N, D) normalized positions
member = np.asarray(r0.membership_prob) > 0.5
g = r0.gamma[member]
y = np.asarray(qs)[member]

last = lambda d: d.mlp.layers[-1]
W0, b0 = last(dec).weight, last(dec).bias


def f(theta, gam):
    W, b = theta
    d = eqx.tree_at(lambda d: (last(d).weight, last(d).bias), dec, (W, b))
    return d(gam)


# output is linear in (W, b): its Jacobian row for output 0 is the feature vector
def feats(gam):
    jw, jb = jax.jacfwd(f)((W0, b0), gam)
    return jnp.concatenate([jw[0, 0], jb[0, :1]])


Phi = np.asarray(jax.vmap(feats)(g))  # (n, H+1)
theta0 = np.concatenate([np.asarray(W0), np.asarray(b0)[:, None]], 1)  # (D, H+1)
n, P = Phi.shape
gg = np.linspace(*r0.track_range, 600)
Phig = np.asarray(jax.vmap(feats)(jnp.asarray(gg)))
rng = np.random.default_rng(seed)
Ys = []
for d in range(y.shape[1]):
    e = y[:, d] - Phi @ theta0[d]
    s2 = np.mean(e**2)
    # prior precision by evidence (linear-Gaussian marginal likelihood), on a grid
    best = None
    for alpha in 10.0 ** np.arange(-6, 5):
        A = Phi.T @ Phi / s2 + alpha * np.eye(P)
        mu = np.linalg.solve(A, Phi.T @ y[:, d] / s2)
        ev = (
            P * np.log(alpha)
            - n * np.log(2 * np.pi * s2)
            - np.sum((y[:, d] - Phi @ mu) ** 2) / s2
            - alpha * mu @ mu
            - np.linalg.slogdet(A)[1]
        ) / 2
        if best is None or ev > best[0]:
            best = (ev, A)
    cov = np.linalg.inv(best[1])
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(P))
    draws = (
        theta0[d] + (L @ rng.standard_normal((P, 200))).T
    )  # Laplace: centred on trained weights
    Ys.append(draws @ Phig.T)  # (200, 600)
Ys = np.stack(Ys, -1)  # normalized
offsC = []
for Y in Ys:
    q, _ = model.normalizer.inverse_transform(
        jnp.asarray(Y), jnp.zeros_like(jnp.asarray(Y))
    )
    offsC.append(normal_offsets(np.stack([np.asarray(q["x"]), np.asarray(q["y"])], -1)))
out["C"] = np.array(offsC)
times["C"] = time.perf_counter() - t0

np.savez(f"out/ds{seed}.npz", **out, **{"t" + k: v for k, v in times.items()})
print(seed, times)
