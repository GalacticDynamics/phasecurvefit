"""Full Bayesian NN decoder: NUTS over every weight of a small phasecurvefit TrackNet.

Usage: python run_bnn.py <seed> <width> <depth>

The bridge is numpyro.contrib.module.random_eqx_module (NumPyro >= 0.19), which puts
a prior on each inexact-array leaf of an Equinox module. gamma_n is fixed at the
fitted encoder's values, as in the ``fixed`` HSGP model. Weights ~ Normal(0, 1).
"""

import sys
import time

import numpyro

numpyro.set_host_device_count(2)  # run the two chains in parallel

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import tracklik as tl
from common import make_data, normal_offsets
from numpyro.contrib.module import random_eqx_module
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive

import phasecurvefit as pcf

seed, width, depth = map(int, sys.argv[1:4])
pos, vel = make_data(seed)
q = np.stack([np.asarray(pos["x"]), np.asarray(pos["y"])], -1)
sc = tl.Scaler(q)
z = jnp.asarray(sc.fwd(q))
_, r, _ = pcf.fit_track(
    pos,
    vel,
    key=jax.random.key(seed),
    training_config=pcf.nn.TrainingConfig(show_pbar=False),
)
m = np.asarray(r.membership_prob) > 0.5
g, zm = jnp.asarray(r.gamma)[m], z[m]
net0 = pcf.nn.TrackNet(out_size=2, width_size=width, depth=depth, key=jax.random.key(0))
n_par = sum(x.size for x in jax.tree.leaves(net0) if hasattr(x, "size"))


def model(g, zm=None):
    net = random_eqx_module("dec", net0, prior=dist.Normal(0.0, 1.0))
    x = jax.vmap(net)(g)
    x = x[0] if isinstance(x, tuple) else x
    numpyro.deterministic("x", x)
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
    with numpyro.plate("stars", g.shape[0]):
        numpyro.sample("q", dist.Normal(x, sigma).to_event(1), obs=zm)


t0 = time.perf_counter()
mcmc = MCMC(
    NUTS(model),
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    chain_method="parallel",
    progress_bar=False,
)
mcmc.run(jax.random.key(seed), g, zm)
jax.block_until_ready(mcmc.get_samples())
t = time.perf_counter() - t0
s = mcmc.get_samples(group_by_chain=True)
summ = summary({k: v for k, v in s.items() if k != "x"})
ess = np.concatenate([np.ravel(v["n_eff"]) for v in summ.values()])
rhat = np.concatenate([np.ravel(v["r_hat"]) for v in summ.values()])
flat = {k: v[::5] for k, v in mcmc.get_samples().items() if k != "x"}
Gg = jnp.linspace(float(g.min()), float(g.max()), 600)
xs = Predictive(model, posterior_samples=flat)(jax.random.key(1), Gg)["x"]
offs = np.array([normal_offsets(sc.inv(np.asarray(x))) for x in xs])
div = int(mcmc.get_extra_fields()["diverging"].sum())

# mean-field VI on the same model, the usual NumPyro BNN recipe
import optax
from numpyro.infer import SVI, Trace_ELBO, autoguide

t1 = time.perf_counter()
guide = autoguide.AutoNormal(model)
svi = SVI(model, guide, optax.adam(3e-3), Trace_ELBO())
res = svi.run(jax.random.key(seed), 20000, g, zm, progress_bar=False)
post = guide.sample_posterior(jax.random.key(2), res.params, sample_shape=(200,))
jax.block_until_ready(post)
t_vi = time.perf_counter() - t1
post = {k: v for k, v in post.items() if k != "x"}
xs = Predictive(model, posterior_samples=post)(jax.random.key(1), Gg)["x"]
offs_vi = np.array([normal_offsets(sc.inv(np.asarray(x))) for x in xs])
np.savez(
    f"out/bnn{width}x{depth}_ds{seed}.npz",
    bnn=offs,
    t=t,
    ess=np.min(ess),
    essmed=np.median(ess),
    rhat=np.max(rhat),
    div=div,
    n_par=n_par,
    bnn_vi=offs_vi,
    t_vi=t_vi,
)
print("vi", f"t={t_vi:.0f}s final loss {float(res.losses[-1]):.1f}")
print(
    seed,
    width,
    depth,
    n_par,
    f"t={t:.0f}s ess_min={np.min(ess):.0f} "
    f"ess_med={np.median(ess):.0f} rhat_max={np.max(rhat):.2f} div={div}",
)
