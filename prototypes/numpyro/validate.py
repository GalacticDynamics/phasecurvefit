"""Run one validation case and seed. Saves offsets from the true track.

Usage: python validate.py <case> <seed> [--vel] [--nuts]

case: helix | loop | gap | image | image_norm | image_x5   (see cases.py)
--vel  put velocities in the likelihood (phase-space track, separate width)
--nuts also run NUTS (SVI with a full-rank Gaussian guide always runs)
"""

import sys
import time

import numpyro

numpyro.set_host_device_count(2)

import cases
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tracklik as tl
from numpyro.diagnostics import effective_sample_size, summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, init_to_value

import phasecurvefit as pcf

case, seed = sys.argv[1], int(sys.argv[2])
use_vel = "--vel" in sys.argv or "--tan" in sys.argv
tan = "--tan" in sys.argv  # velocity mean tied to the track tangent
vstart = "--vstart" in sys.argv  # start from a velocity-weighted MST | SOM
v2 = "--v2" in sys.argv  # latent line density and gamma-dependent width
SLACK = 0.05 if "--slack" in sys.argv else 0.0  # loosen the tie (needs --tan)
do_nuts = "--nuts" in sys.argv and seed < 2  # NUTS cut to seeds 0-1 (shared machine)
tag = f"{case}{'_tan' if tan else '_vel' if use_vel else ''}{'_vs' if vstart else ''}{'_v2' if v2 else ''}{'_sl' if SLACK else ''}_ds{seed}"
if case == "loop":
    tl.configure(m_basis=32, k_grid=512)
else:
    tl.configure(m_basis=16, k_grid=256)

q, p, w, truth = cases.make(case, seed)
N, Dq = q.shape
keys = "xyz"[:Dq]
pos = {k: jnp.asarray(q[:, i]) for i, k in enumerate(keys)}
vel = {k: jnp.asarray(p[:, i]) for i, k in enumerate(keys)}
wj = None if w is None else jnp.asarray(w)

scq = tl.Scaler(q)
cols = [scq.fwd(q)]
if use_vel:
    scp = tl.Scaler(p)
    if tan:  # scale only: the tied mean is speed times a unit tangent
        scp.mu = 0.0 * scp.mu
        scp.s = float(np.sqrt(np.mean(np.sum(p**2, 1))))
    cols.append(scp.fwd(p))
z = jnp.asarray(np.concatenate(cols, 1))
D = z.shape[1]
G = jnp.linspace(-1.0, 1.0, 600)
out, info = {}, {}


def save(stage):
    np.savez(f"out_val/{tag}.npz", **out, **info)
    print(
        tag,
        stage,
        {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in info.items()
            if k.endswith(stage)
        },
        flush=True,
    )


N_MEAN = (2 * Dq + 1 if SLACK else Dq + 1) if tan else D  # channels before extras


def _keep(x):
    """With a latent line density, the gamma range holding its central 99%."""
    if not v2:
        return np.ones(len(x), bool)
    cdf = np.cumsum(np.exp(x[:, N_MEAN] - x[:, N_MEAN].max()))
    cdf /= cdf[-1]
    return (cdf > 0.005) & (cdf < 0.995)


def offs_from(xs):
    xs = np.asarray(xs)
    return np.array([cases.offsets(scq.inv(x[_keep(x), :Dq]), truth) for x in xs])


def widths_from(xs, sig):
    """Each draw's width at cases.S_GRID (true-curve arc fraction), data units."""
    st = cases._arc_frac(truth)
    out = np.full((len(xs), len(cases.S_GRID)), np.nan)
    for i, (x, sg) in enumerate(zip(np.asarray(xs), np.asarray(sig))):
        x = x[_keep(x)]
        q_ = scq.inv(x[:, :Dq])
        s_ = st[
            np.argmin(np.linalg.norm(q_[:, None] - truth[None, ::4], axis=2), 1) * 4
        ]
        wd = np.exp(x[:, N_MEAN + 1]) * sg * scq.s
        o = np.argsort(s_)
        out[i] = np.interp(cases.S_GRID, s_[o], wd[o], left=np.nan, right=np.nan)
    return out


# ---- one fit_track, for the point-estimate error ------------------------------------
t0 = time.perf_counter()
try:
    _, r, _ = pcf.fit_track(
        pos,
        vel,
        key=jax.random.key(seed),
        weights=wj,
        training_config=pcf.nn.TrainingConfig(show_pbar=False),
    )
    xy = r(jnp.linspace(*r.track_range, 600))
    out["single"] = cases.offsets(
        np.stack([np.asarray(xy[k]) for k in keys], -1), truth
    )[None]
except Exception as e:  # report, don't stop the Bayesian runs
    info["err_single"] = f"{type(e).__name__}: {e}"[:200]
info["t_single"] = time.perf_counter() - t0
save("single")

# ---- start: rank gamma (by cumulative weight) from an ordering -----------------------
orderer = pcf.orderers.RidgeOrderer() if case.startswith("image") else None
if vstart:
    orderer = (
        pcf.orderers.MSTOrderer(
            velocity_weight=2.0,
            jump_cap=float("inf"),
            orient_by_velocity=True,
            on_disconnected="connect",
            nan_policy="omit",
        )
        | pcf.orderers.SOMOrderer()
    )
ordering = (
    pcf.order(pos, vel, orderer, weights=wj)
    if orderer
    else pcf.orderers.default_pipeline(pos, vel, weights=wj)
)
idx = np.asarray(ordering.ordering)
ww = np.ones(N) if w is None else np.asarray(w)
cw = np.cumsum(ww[idx]) - 0.5 * ww[idx]
g0 = -1 + 2 * (cw - cw[0]) / (cw[-1] - cw[0])
n_extra = 2 if v2 else 0
if tan:  # positions + log-speed channel (speed 1 in these units)
    target = np.zeros((len(idx), N_MEAN))  # log-speed and offsets start at 0
    target[:, :Dq] = np.asarray(z[idx][:, :Dq])
    init = tl.ls_init(target, g0, N_MEAN, n_q=Dq, n_extra=n_extra)
    init["sigma_p"] = jnp.asarray(0.2)
else:
    init = tl.ls_init(z[idx], g0, D, n_q=Dq, n_extra=n_extra)
if v2:  # the density channel starts at the log histogram of the start's gamma
    h, e = np.histogram(g0, bins=32, range=(-1, 1))
    logh = np.log(h + 1.0)
    beta = np.array(init["beta"])
    beta[N_MEAN] = tl.basis_fit(0.5 * (e[1:] + e[:-1]), logh - logh.mean())
    init["beta"] = jnp.asarray(beta)
info["n_visited"] = len(idx)
kw = dict(
    z=z,
    n_q=Dq,
    weights=wj,
    vel_tangent=tan,
    line_density=v2,
    width_varies=v2,
    vel_slack=SLACK,
)

# ---- SVI, full-rank Gaussian ---------------------------------------------------------
t0 = time.perf_counter()
guide = autoguide.AutoMultivariateNormal(
    tl.model, init_loc_fn=init_to_value(values=init)
)
opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(2e-3))
res = SVI(tl.model, guide, opt, Trace_ELBO()).run(
    jax.random.key(seed), 4000, progress_bar=False, **kw
)
post = guide.sample_posterior(jax.random.key(1), res.params, sample_shape=(200,))
jax.block_until_ready(post)
info["t_svi"] = time.perf_counter() - t0
info["nan_svi"] = int(not np.isfinite(res.losses[-1]))
info["sigma_svi"] = float(np.median(post["sigma"])) * scq.s
xs = tl.track_draws(post, G, D)
out["svi"] = offs_from(xs)
if v2:
    out["wid_svi"] = widths_from(xs, post["sigma"])
save("svi")

# ---- NUTS ----------------------------------------------------------------------------
if do_nuts:
    t0 = time.perf_counter()
    mcmc = MCMC(
        NUTS(
            tl.model,
            init_strategy=init_to_value(values=init),
            dense_mass=True,
            target_accept_prob=0.9,
        ),
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(jax.random.key(seed), extra_fields=("diverging",), **kw)
    flat = mcmc.get_samples()
    jax.block_until_ready(flat)
    info["t_nuts"] = time.perf_counter() - t0
    summ = summary(mcmc.get_samples(group_by_chain=True))
    info["rhat_nuts"] = float(max(np.max(v["r_hat"]) for v in summ.values()))
    g12 = jnp.linspace(-1, 1, 12)
    x12 = np.asarray(tl.track_draws(flat, g12, D)).reshape(2, -1, 12 * D)
    info["essx_nuts"] = float(np.min(effective_sample_size(x12)))
    info["div_nuts"] = int(mcmc.get_extra_fields()["diverging"].sum())
    info["sigma_nuts"] = float(np.median(flat["sigma"])) * scq.s
    thin = {k: v[::5] for k, v in flat.items()}
    xs = tl.track_draws(thin, G, D)
    out["nuts"] = offs_from(xs)
    if v2:
        out["wid_nuts"] = widths_from(xs, thin["sigma"])
    save("nuts")
