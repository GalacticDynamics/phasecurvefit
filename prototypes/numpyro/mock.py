"""The cached mock stream (8000 stars, 3-D): self-consistency under subsampling.

Usage: python mock.py ref              fit all 8000 stars (SVI), save the reference track
       python mock.py <seed> [--nuts]  fit a random 1000-star subset, offsets from the ref

There is no known true track, so the reference is the same model fitted to all the
data. Coverage against it measures whether the subset posterior is consistent with the
full-data answer. A bias the model has at any N is invisible to this test.
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

tl.configure(m_basis=32, k_grid=512)
d = np.load("../../docs/tutorials/mockstream.npz")
q_all, p_all = d["q"].astype(float), d["p"].astype(float)
arg = sys.argv[1]
v2 = "--v2" in sys.argv  # latent line density and gamma-dependent width
tan = "--tan" in sys.argv  # velocities in the likelihood, mean tied to the tangent
SLACK = 0.05 if "--slack" in sys.argv else 0.0  # loosen the tie (needs --tan)
sfx = ("_v2" if v2 else "") + ("_tan" if tan else "") + ("_sl" if SLACK else "")
if arg == "ref":
    q, p, tag = q_all, p_all, "mock_ref" + sfx
else:
    seed = int(arg)
    sel = np.random.default_rng(seed).choice(len(q_all), 1000, replace=False)
    q, p, tag = q_all[sel], p_all[sel], f"mock{sfx}_ds{seed}"
pos = {k: jnp.asarray(q[:, i]) for i, k in enumerate("xyz")}
vel = {k: jnp.asarray(p[:, i]) for i, k in enumerate("xyz")}
scq = tl.Scaler(q_all)  # the same scaling for every fit
p_rms = float(np.sqrt(np.mean(np.sum(p_all**2, 1))))  # scale only, no centring
z = jnp.asarray(np.concatenate([scq.fwd(q), p / p_rms], 1) if tan else scq.fwd(q))
N_MEAN = (7 if SLACK else 4) if tan else 3  # track channels before the extras
log_rho = float(jnp.log(pcf.nn.uniform_background_density(jnp.asarray(scq.fwd(q_all)))))
kw = dict(
    z=z,
    n_q=3,
    log_rho_bg=log_rho,
    line_density=v2,
    width_varies=v2,
    vel_tangent=tan,
    vel_slack=SLACK,
)
G = jnp.linspace(-1.0, 1.0, 1200)
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


ordering = pcf.orderers.default_pipeline(pos, vel)
idx = np.asarray(ordering.ordering)
if v2:  # gamma from arc length; the density channel starts at its log histogram
    ch = np.asarray(ordering.chord)[idx]
    g0 = -1 + 2 * (ch - ch.min()) / (ch.max() - ch.min())
    if tan:  # positions + a log-speed channel started at speed 1 (p / p_rms)
        target = np.zeros((len(idx), N_MEAN))  # log-speed and offsets start at 0
        target[:, :3] = np.asarray(z[idx][:, :3])
        init = tl.ls_init(target, g0, N_MEAN, n_q=3, n_extra=2)
        init["sigma_p"] = jnp.asarray(0.3)
    else:
        init = tl.ls_init(z[idx], g0, 3, n_q=3, n_extra=2)
    h, e = np.histogram(g0, bins=64, range=(-1, 1))
    logh = np.log(h + 1.0)
    beta = np.array(init["beta"])
    beta[N_MEAN] = tl.basis_fit(0.5 * (e[1:] + e[:-1]), logh - logh.mean())
    init["beta"] = jnp.asarray(beta)
else:
    init = tl.ls_init(z[idx], np.linspace(-1, 1, len(idx)), 3, n_q=3)
init["f_bg"] = jnp.asarray(0.02)
info["n_visited"] = len(idx)

t0 = time.perf_counter()
guide = autoguide.AutoMultivariateNormal(
    tl.model, init_loc_fn=init_to_value(values=init)
)
opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(2e-3))
res = SVI(tl.model, guide, opt, Trace_ELBO()).run(
    jax.random.key(0), 6000, progress_bar=False, **kw
)
post = guide.sample_posterior(jax.random.key(1), res.params, sample_shape=(200,))
jax.block_until_ready(post)
info["t_svi"] = time.perf_counter() - t0
info["sigma_svi"] = float(np.median(post["sigma"])) * scq.s
info["fbg_svi"] = float(np.median(post["f_bg"]))


def curves(draws):
    """Position polylines for each draw, in data units. With a latent line density,
    cut each to the gamma range holding the central 99% of it: beyond the last stars
    the track is unconstrained and curls off."""
    xs = np.asarray(tl.track_draws(draws, G))
    out = []
    for x in xs:
        keep = slice(None)
        if v2:
            cdf = np.cumsum(np.exp(x[:, N_MEAN] - x[:, N_MEAN].max()))
            cdf /= cdf[-1]
            keep = (cdf > 0.005) & (cdf < 0.995)
        out.append(scq.inv(x[keep, :3]))
    return out


cs = curves(post)
if arg == "ref":
    # median over draws at each gamma, cut to the median draw's range
    xs = np.asarray(tl.track_draws(post, G))
    med = np.median(xs, 0)
    keep = slice(None)
    if v2:
        cdf = np.cumsum(np.exp(med[:, N_MEAN] - med[:, N_MEAN].max()))
        cdf /= cdf[-1]
        keep = (cdf > 0.005) & (cdf < 0.995)
        np.save(
            f"out_val/{tag}_width.npy",
            np.stack(
                [np.asarray(G)[keep], np.exp(med[keep, N_MEAN + 1]) * info["sigma_svi"]]
            ),
        )
    np.save(f"out_val/{tag}_track.npy", scq.inv(med[keep, :3]))
    save("svi")
    raise SystemExit
truth = np.load(f"out_val/mock_ref{sfx}_track.npy")
out["svi"] = np.array([cases.offsets(c, truth) for c in cs])
save("svi")

if "--nuts" in sys.argv:
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
    x12 = np.asarray(tl.track_draws(flat, jnp.linspace(-1, 1, 12)))[..., :3]
    x12 = x12.reshape(2, -1, 36)
    info["essx_nuts"] = float(np.min(effective_sample_size(x12)))
    info["div_nuts"] = int(mcmc.get_extra_fields()["diverging"].sum())
    info["sigma_nuts"] = float(np.median(flat["sigma"])) * scq.s
    cs = curves({k: v[::5] for k, v in flat.items()})
    out["nuts"] = np.array([cases.offsets(c, truth) for c in cs])
    save("nuts")
