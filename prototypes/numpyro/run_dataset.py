"""Fit one mock dataset with the NumPyro track models; save offsets from the true track.

Usage: python run_dataset.py <seed> [--bad]   (--bad adds 16 interlopers, 4%)

single   fit_track once (the library's point estimate), for reference
fixed    NUTS, HSGP track, gamma_n fixed at the fitted encoder's values
marginal NUTS, HSGP track, gamma_n marginalised on a grid (no per-star gamma)
svi_*    SVI on the marginal model: AutoNormal, AutoMultivariateNormal, AutoLaplace
"""

import sys
import time

import numpyro

numpyro.set_host_device_count(2)  # run the two chains in parallel

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
import tracklik as tl
from common import make_data, normal_offsets
from numpyro.diagnostics import effective_sample_size, summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, init_to_value

import phasecurvefit as pcf

seed = int(sys.argv[1])
bad = "--bad" in sys.argv
pos, vel = make_data(seed)
if bad:  # same interlopers as the sibling prototype's run_interlopers.py
    rng = np.random.default_rng(10_000 + seed)
    bq = np.stack([rng.uniform(-1, 11, 16), rng.uniform(-3, 3, 16)], -1)
    bp = rng.normal(0, 1, (16, 2))
    pos = {
        k: jnp.concatenate([pos[k], jnp.asarray(bq[:, i])]) for i, k in enumerate("xy")
    }
    vel = {
        k: jnp.concatenate([vel[k], jnp.asarray(bp[:, i])]) for i, k in enumerate("xy")
    }
q = np.stack([np.asarray(pos["x"]), np.asarray(pos["y"])], -1)
sc = tl.Scaler(q)
z = jnp.asarray(sc.fwd(q))
D = 2
G = jnp.linspace(-1.0, 1.0, 600)
log_rho = None
if bad:
    log_rho = float(jnp.log(pcf.nn.uniform_background_density(z)))

out, info = {}, {}
fname = f"out/ds{seed}{'_bad' if bad else ''}.npz"


def save(stage):
    np.savez(fname, **out, **info)
    print(
        seed,
        stage,
        {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in info.items()
            if k.endswith(stage)
        },
        flush=True,
    )


def offsets_from(xs):  # xs: (S, 600, D) scaled
    return np.array([normal_offsets(sc.inv(np.asarray(x))) for x in xs])


# ---- the library's single fit --------------------------------------------------------
t0 = time.perf_counter()
ordering, r, _ = pcf.fit_track(
    pos,
    vel,
    key=jax.random.key(seed),
    training_config=pcf.nn.TrainingConfig(show_pbar=False),
)
xy = r(jnp.linspace(*r.track_range, 600))
out["single"] = normal_offsets(
    np.stack([np.asarray(xy["x"]), np.asarray(xy["y"])], -1)
)[None]
info["t_single"] = time.perf_counter() - t0
save("single")
member = np.asarray(r.membership_prob) > 0.5
g_enc = np.asarray(r.gamma)

# rank gamma from the ordering, as init for the marginal model
order_idx = np.asarray(ordering.ordering)
g_rank = np.full(len(q), np.nan)
g_rank[order_idx] = np.linspace(-1, 1, len(order_idx))
vis = np.isfinite(g_rank)


def run_nuts(name, init, grid, **kw):
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
    mcmc.run(jax.random.key(seed), **kw)
    jax.block_until_ready(mcmc.get_samples())  # dispatch is async: wait before timing
    info["t_" + name] = time.perf_counter() - t0
    s = mcmc.get_samples(group_by_chain=True)
    summ = summary(s)
    ess = np.concatenate([np.ravel(v["n_eff"]) for v in summ.values()])
    rhat = np.concatenate([np.ravel(v["r_hat"]) for v in summ.values()])
    info["ess_" + name] = float(np.min(ess))
    info["essmed_" + name] = float(np.median(ess))
    info["rhat_" + name] = float(np.max(rhat))
    info["div_" + name] = int(mcmc.get_extra_fields()["diverging"].sum())
    info["sigma_" + name] = float(np.median(s["sigma"])) * sc.s
    flat = mcmc.get_samples()
    # ESS of the track itself, at 12 points along it (what the bands are made of)
    g12 = jnp.linspace(float(grid[0]), float(grid[-1]), 12)
    x12 = np.asarray(tl.track_draws(flat, g12, D)).reshape(2, -1, 12 * D)
    ess_x = np.asarray(effective_sample_size(x12))
    info["essx_" + name] = float(np.min(ess_x))
    info["essxmed_" + name] = float(np.median(ess_x))
    thin = {k: v[::5] for k, v in flat.items()}  # 200 draws
    out[name] = offsets_from(tl.track_draws(thin, grid, D))
    save(name)


# ---- fixed gamma (encoder) -----------------------------------------------------------
gm = g_enc[member]
zm = z[member]
G_fixed = jnp.linspace(gm.min(), gm.max(), 600)
init_f = tl.ls_init(zm, gm, D)
run_nuts("fixed", init_f, G_fixed, z=zm, gamma=jnp.asarray(gm))

# ---- marginal gamma ------------------------------------------------------------------
init_m = tl.ls_init(z[vis], g_rank[vis], D)
if bad:
    init_m["f_bg"] = jnp.asarray(0.05)
run_nuts("marginal", init_m, G, z=z, log_rho_bg=log_rho)

# ---- SVI on the marginal model -------------------------------------------------------
for gname, G_ in [
    ("svi_normal", autoguide.AutoNormal),
    ("svi_mvn", autoguide.AutoMultivariateNormal),
    ("svi_laplace", autoguide.AutoLaplaceApproximation),
]:
    t0 = time.perf_counter()
    guide = G_(tl.model, init_loc_fn=init_to_value(values=init_m))
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(2e-3))
    svi = SVI(tl.model, guide, opt, Trace_ELBO())
    res = svi.run(
        jax.random.key(seed), 4000, z=z, log_rho_bg=log_rho, progress_bar=False
    )
    jax.block_until_ready(res)
    info["t_" + gname] = time.perf_counter() - t0
    info["nan_" + gname] = int(not np.isfinite(res.losses[-1]))
    if info["nan_" + gname]:
        save(gname)
        continue
    post = guide.sample_posterior(jax.random.key(1), res.params, sample_shape=(200,))
    out[gname] = offsets_from(tl.track_draws(post, G, D))
    save(gname)
