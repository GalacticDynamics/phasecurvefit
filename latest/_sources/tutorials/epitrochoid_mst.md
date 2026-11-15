---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Ordering a Self-Intersecting Epitrochoid with the MST Backbone

+++

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GalacticDynamics/phasecurvefit/blob/main/docs/tutorials/epitrochoid_mst.ipynb)

+++

The `MSTOrderer` links each tracer to its $k$ nearest neighbours, keeps the minimum spanning tree (MST) of those links, and orders the tracers along the tree's longest path (its *diameter*). It needs **no start point**: the ends of the diameter are the two tips of the curve. This notebook uses the self-intersecting epitrochoid from the other tutorials to show a subtlety:

1. a **spatial** MST short-circuits across the self-intersections, and
2. a **velocity-aware** MST (severing anti-parallel edges) is needed to keep the ordering on a coherent branch.

The takeaway: velocity information is *necessary* on self-overlapping curves. (For curves with many loops like this one, the `LocalFlowOrderer` is often the better tool, as in the [epitrochoid autoencoder tutorial](epitrochoid_autoencoder.md); the MST's sweet spot is the near-closed single loop.)

It helps to have read the [stream MST tutorial](stream_mst.md) first.

```{code-cell} ipython3
# On Colab, clone the repository, install the locked tutorial dependencies, and run
# from docs/tutorials. Elsewhere it does nothing: run from docs/tutorials with the
# `tutorials` extra.
import pathlib
import sys

repo = pathlib.Path("/content/phasecurvefit")
if "google.colab" in sys.modules:
    %pip install -q uv
    if not repo.exists():
        !git clone -q --depth 1 https://github.com/GalacticDynamics/phasecurvefit {repo}
    %cd {repo}
    !uv export -q --frozen --extra tutorials --no-dev --no-editable --no-hashes -o r.txt
    !uv pip install -q --system --python {sys.executable} -r r.txt
    %cd docs/tutorials
```

```{code-cell} ipython3
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import quaxed.numpy as jnp
import unxt as u

import phasecurvefit as pcf
```

```{code-cell} ipython3
key = jr.key(201030)
usys = u.unitsystems.si
```

## Data: Self-Intersecting Epitrochoid Curve

```{code-cell} ipython3
def make_self_intersecting_stream(
    key,
    *,
    n: int,
    noise_sigma=0.5,
    scale=120.0,
    R: float = 5.0,
    r: float = 1.0,
    d: float = 4.5,
):
    """Return (t, pos, vel) for an OPEN curve with self-intersections."""
    # Single outer rotation: 5° to 355° with a 10° gap
    t_start = 5.0 * jnp.pi / 180.0
    t_end = 355.0 * jnp.pi / 180.0
    t = jnp.linspace(t_start, t_end, n)

    ratio = (R + r) / r  # internal rotations per outer rotation
    x0 = scale * ((R + r) * jnp.cos(t) - d * jnp.cos(ratio * t)) / 5.0
    y0 = scale * ((R + r) * jnp.sin(t) - d * jnp.sin(ratio * t)) / 5.0

    # Derivatives for velocity
    dx0 = scale * (-(R + r) * jnp.sin(t) + d * ratio * jnp.sin(ratio * t)) / 5.0
    dy0 = scale * ((R + r) * jnp.cos(t) - d * ratio * jnp.cos(ratio * t)) / 5.0

    kx, ky = jr.split(key)
    x = x0 + noise_sigma * jr.normal(kx, (n,))
    y = y0 + noise_sigma * jr.normal(ky, (n,))

    pos = {"x": u.Q(x, usys["length"]), "y": u.Q(y, usys["length"])}
    vel = {"x": u.Q(dx0, usys["speed"]), "y": u.Q(dy0, usys["speed"])}
    return t, pos, vel
```

```{code-cell} ipython3
key, subkey = jr.split(key)
t, pos, vel = make_self_intersecting_stream(subkey, n=2048, noise_sigma=6)

# shuffle so the input order carries no ordering information
key, subkey = jr.split(key)
order = jr.permutation(subkey, jnp.arange(len(t)))
qs = jax.tree.map(lambda a: a[order], pos)
ps = jax.tree.map(lambda a: a[order], vel)
t_shuffled = t[order]
```

```{code-cell} ipython3
_fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(qs["x"], qs["y"], color="gray", alpha=0.7, s=8)
every = 12
ax.quiver(
    u.ustrip(usys, qs["x"][::every]),
    u.ustrip(usys, qs["y"][::every]),
    u.ustrip(usys, ps["x"][::every]),
    u.ustrip(usys, ps["y"][::every]),
    scale=10_000,
    width=0.005,
    alpha=0.6,
    color="black",
)
ax.set(aspect="equal", title="Epitrochoid with Self-Intersections")
plt.show()
```

## A helper: how well is the ordering monotone in the true parameter?

We know the true curve parameter `t`, so we can score any ordering by the correlation between each tracer's position in the ordering and its true `t` (|rho| ≈ 1 is perfect).

```{code-cell} ipython3
def rho(result):
    to = np.asarray(t_shuffled)[np.asarray(result.ordering)]
    r = np.corrcoef(np.arange(to.size), to)[0, 1]
    cov = int((result.indices >= 0).sum())
    return abs(r), cov


def plot_order(result, title):
    _fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(qs["x"], qs["y"], s=3, c="lightgray")
    oq = {k: v[result.ordering] for k, v in result.positions.items()}
    timeline = np.linspace(-1, 1, len(result.ordering))
    im = ax.scatter(oq["x"], oq["y"], s=25, c=timeline, cmap="RdYlBu")
    ax.plot(oq["x"], oq["y"], c="k", lw=0.7, alpha=0.6)
    plt.colorbar(im, ax=ax, label="ordering gamma")
    r, cov = rho(result)
    ax.set(aspect="equal", title=f"{title}\n|rho|={r:.2f}, coverage={cov}/{len(t)}")
    plt.show()
```

## 1. Spatial MST — and why it short-circuits

We set `jump_cap` from the data (a few times the median nearest-neighbour distance). With positions only, the MST happily bridges the two branches wherever they cross, so the arc-length ordering jumps between branches.

```{code-cell} ipython3
P = np.stack([np.asarray(u.ustrip(usys, qs[c])) for c in ("x", "y")], axis=1)
med = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))

spatial = pcf.orderers.MSTOrderer(k=16, jump_cap=6.0 * med, on_disconnected="largest")
res_spatial = spatial.order(qs, ps, metadata=pcf.StateMetadata(usys=usys))
plot_order(res_spatial, "Spatial MST (positions only)")
```

## 2. Velocity-aware MST — cutting anti-parallel edges

At a crossing the two branches travel in different directions, so their velocity alignment `cos(v_i, v_j)` is low. `sever_cos_threshold` drops those edges, so the MST stays on a coherent branch. The anti-parallel edges at the crossings are exactly the ones the spatial MST used to cheat, so severing them recovers a clean ordering at essentially full coverage. (On sparser data severing can fragment the graph; `on_disconnected` then decides what to do — here we keep the largest component to be safe.)

```{code-cell} ipython3
velaware = pcf.orderers.MSTOrderer(
    k=16,
    jump_cap=8.0 * med,
    sever_cos_threshold=0.9,
    orient_by_velocity=True,
    on_disconnected="largest",
)
res_vel = velaware.order(qs, ps, metadata=pcf.StateMetadata(usys=usys))
plot_order(res_vel, "Velocity-aware MST (severing)")
```

The velocity-aware ordering follows a single coherent strand cleanly (|rho| jumps from ~0.26 to ~0.9), while the spatial MST zig-zags across branches. This is the sense in which velocity is *necessary* for self-overlapping curves.

That said, a curve with many self-crossings is not the MST's natural home: the `LocalFlowOrderer` follows such a curve directly. The MST is at its best on a **near-closed single loop**, such as a tidal stream, where the velocity field reverses at the progenitor so a walk must start there and go both ways. The MST orders such a loop tip to tip with no start point and little tuning.

+++

## 3. Refining with a Self-Organizing Map

The MST orders the tracers, but its backbone is the graph path through the
observations themselves, so it inherits their scatter. Chaining a
`SOMOrderer` fits a lattice of prototypes — each
an average over many tracers — and re-orders by arc length along a smooth
backbone through them.

Two things make this work here, and both are automatic:

- **The SOM follows the previous stage's velocity-awareness.** The MST above
  used `sever_cos_threshold`, so its result is marked velocity-aware and the SOM
  picks up a phase-space metric. Ordering on position alone would undo the
  severing: at a crossing the two branches are spatially coincident and differ
  only in velocity.
- **The prior ordering seeds the lattice**, so the SOM refines it locally
  instead of re-deriving a global ordering and discarding the MST's work.

`n_prototypes` is the one knob that matters. The smoothing length is
`sigma_end * L / (n_prototypes - 1)`; any structure finer than that is smoothed
away, and a 5-petal rose has a lot of fine structure. The default 25 is far too
coarse here — it *tangles* the ordering, and `SOMOrderer` warns when it does.

```{code-cell} ipython3
refined = (velaware | pcf.orderers.SOMOrderer(n_prototypes=60)).order(
    qs, ps, metadata=pcf.StateMetadata(usys=usys)
)
plot_order(refined, "Velocity-aware MST | SOM (60 prototypes)")
```

To see what the refinement fits, plot the prototypes themselves.
`SOMOrderer` returns only the backbone, so we refit the prototypes with
`pcf.som.SOM1D` exactly as the orderer does, seeded from the MST ordering.
Each prototype averages over many noisy tracers, and the smooth backbone
through them follows every loop and crossing; colouring by the chord shows
the ordering running cleanly from one tip to the other.

```{code-cell} ipython3
# SOMOrderer returns the backbone but not the prototypes, so refit them with
# `pcf.som.SOM1D` the way the orderer does: seeded from the MST ordering, starting
# at `sigma_end`, on a phase-space metric whose velocity term is scaled to the
# lattice spacing.
n_prototypes, sigma_end = 60, 0.7
visited = np.asarray(res_vel.ordering)
mq = {k: u.ustrip(usys, v[visited]) for k, v in qs.items()}
mp = {k: u.ustrip(usys, v[visited]) for k, v in ps.items()}
length = np.hypot(np.diff(mq["x"]), np.diff(mq["y"])).sum()
speed = np.median(np.hypot(mp["x"], mp["y"]))
som = pcf.som.SOM1D.make(
    mq,
    mp,
    n_prototypes=n_prototypes,
    ordering=jnp.arange(visited.size),
    metric=pcf.metrics.FullPhaseSpaceDistanceMetric(),
    metric_scale=sigma_end * length / (n_prototypes - 1) / (2 * speed),
    sigma_start=sigma_end,
    sigma_end=sigma_end,
).fit(mq, mp)
backbone, _ = som.backbone()
lam = som.chord(mq, mp)
proto_lam = som.chord(som.prototype_positions, som.prototype_velocities)

_fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(P[:, 0], P[:, 1], s=3, c="lightgray")
im = ax.scatter(mq["x"], mq["y"], s=12, c=lam, cmap="RdYlBu", alpha=0.5)
ax.plot(backbone["x"], backbone["y"], c="k", lw=1.5)
pts = ax.scatter(
    som.prototype_positions["x"],
    som.prototype_positions["y"],
    s=90,
    c=proto_lam,
    cmap="RdYlBu",
    norm=im.norm,
    edgecolors="k",
    zorder=3,
    label=f"{n_prototypes} prototypes",
)
plt.colorbar(pts, ax=ax, label="chord (arc length along the backbone) [m]")
r = abs(np.corrcoef(np.argsort(np.argsort(lam)), np.asarray(t_shuffled)[visited])[0, 1])
ax.set(
    aspect="equal",
    xlabel="x [m]",
    ylabel="y [m]",
    title=f"SOM prototypes and backbone\n|rho|={r:.2f}",
)
ax.legend(loc="lower left")
plt.show()
```

|rho| goes from ~0.90 to ~1.00: the SOM recovers the exact ordering. The
backbone it produces is also a genuine smooth curve rather than a polyline
through noisy observations, and `refined.chord` carries the arc length of every
tracer along it — the natural affine parameter for the fit that follows.

Try `n_prototypes=25` (the default) to see the failure the warning is there to
catch: the lattice is coarser than the petals, and the ordering collapses to
|rho| ~ 0.23 while still looking plausible on the plot.

+++

## 4. Feeding the ordering to the autoencoder

The autoencoder has two halves: an **encoder** mapping each tracer to its
ordering parameter $\gamma$, and a **decoder** mapping $\gamma$ back to a
position (the smooth "mean path"). Training runs in three phases - encoder,
decoder, then a joint phase.

Two choices matter for a self-intersecting curve like this one:

- **Decoder architecture.** The epitrochoid makes five tight inner loops, so
  the decoder $\gamma \to (x, y)$ must bend sharply five times. A plain MLP (`TrackNet`) over-smooths these loops away.
  We instead use `FourierTrackNet`, which feeds Fourier features
  $[\gamma, \sin(\pi k\gamma), \cos(\pi k\gamma)]$ ($k = 1 \dots K$) to the MLP,
  giving it the high-frequency capacity to follow the loops. The decoder is
  fully swappable via `PathAutoencoder.make(decoder=...)`.
- **The joint phase.** The joint ("both") phase optimises a *different*
  composite objective (membership + velocity alignment), which on this hard
  topology inflates the reported loss without improving the geometry. We set
  `n_epochs_both=0` and stop after the decoder phase, whose spatial
  reconstruction is best here.

```{code-cell} ipython3
key, model_key, train_key = jr.split(key, 3)
enc_key, dec_key = jr.split(model_key)
normalizer = pcf.nn.StandardScalerNormalizer(qs, ps)

# Fourier-feature decoder: high-frequency capacity for the sharp petal excursions
# (a plain TrackNet over-smooths a 5-fold self-intersection).
decoder = pcf.nn.FourierTrackNet(
    out_size=2, n_frequencies=12, width_size=256, depth=4, key=dec_key
)
ae_model = pcf.nn.PathAutoencoder.make(
    normalizer, gamma_range=res_vel.gamma_range, decoder=decoder, key=enc_key
)

# Stop after the decoder phase (see above): no joint phase on this hard topology.
config = pcf.nn.TrainingConfig(
    n_epochs_encoder=400, n_epochs_decoder=600, n_epochs_both=0
)
result, _, losses = pcf.nn.train_autoencoder(
    ae_model, res_vel, key=train_key, config=config
)
```

```{code-cell} ipython3
_fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.asarray(losses))
ax.set(xlabel="Epoch", ylabel="Loss", yscale="log", title="Training loss")
plt.show()
```

```{code-cell} ipython3
_fig, ax = plt.subplots(figsize=(8, 8))
all_gamma, all_probs = result.model.encode(res_vel.positions, res_vel.velocities)
qs_pred = result(jnp.linspace(*result.track_range, 1_000))
im = ax.scatter(
    u.ustrip(usys, qs["x"]),
    u.ustrip(usys, qs["y"]),
    s=20,
    c=np.asarray(all_gamma),
    cmap="RdYlBu",
    alpha=0.8,
)
ax.plot(
    u.ustrip(usys, qs_pred["x"]),
    u.ustrip(usys, qs_pred["y"]),
    c="k",
    lw=3,
    label="mean path",
)
plt.colorbar(im, ax=ax, label="gamma")
ax.set(aspect="equal", title="MST ordering + autoencoder mean path")
ax.legend()
plt.show()
```

**What to look for:** the mean path should trace every inner loop and cross itself where the data do. Compare it with the [running-mean path](epitrochoid_runningmean.md) on the same curve, which cuts inside the loops.

## Takeaways

- On a self-crossing curve, a positions-only MST short-circuits at the crossings. `sever_cos_threshold` cuts links between tracers moving in different directions and recovers the ordering.
- The decoder is swappable: `FourierTrackNet` gives it the capacity for sharp bends.
- Judge the final track by its geometry, not by the loss. Here stopping after the decoder phase (`n_epochs_both=0`) gives the better track.
