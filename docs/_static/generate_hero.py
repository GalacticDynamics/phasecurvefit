#!/usr/bin/env -S uv run python
"""Generate the landing-page figure, ``hero.png``.

It runs the same code as the example on the landing page (``docs/index.md``):
a shuffled, noisy spiral is ordered and fitted with ``pcf.fit_track``. The left
panel shows the tracers in their shuffled input order, the right panel colours
them by the fitted ordering coordinate gamma and draws the fitted mean track.

Run from the repository root::

    uv run --group docs python docs/_static/generate_hero.py
"""

__all__: tuple[str, ...] = ()

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import phasecurvefit as pcf

# --- the landing-page example ------------------------------------------------
t = jnp.linspace(0.5, 0.5 + 3 * jnp.pi, 400)  # 1.5 turns of a spiral
r = 1 + 0.6 * t
kx, ky, kp = jax.random.split(jax.random.key(0), 3)
pos = {
    "x": r * jnp.cos(t) + 0.2 * jax.random.normal(kx, t.shape),
    "y": r * jnp.sin(t) + 0.2 * jax.random.normal(ky, t.shape),
}
vel = {"x": 0.6 * jnp.cos(t) - r * jnp.sin(t), "y": 0.6 * jnp.sin(t) + r * jnp.cos(t)}
shuffle = jax.random.permutation(kp, t.size)  # hide the true order
pos, vel = jax.tree.map(lambda a: a[shuffle], (pos, vel))

config = pcf.nn.TrainingConfig(show_pbar=False)
ordering, fitted, losses = pcf.fit_track(
    pos, vel, key=jax.random.key(1), training_config=config
)
track = fitted(jnp.linspace(*fitted.gamma_range, 300))
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(
    1, 2, figsize=(9, 4.2), sharex=True, sharey=True, layout="constrained"
)
for ax in axs:
    ax.set(aspect="equal", xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set_color("0.6")

axs[0].plot(pos["x"], pos["y"], "-", c="0.6", lw=0.4)
axs[0].scatter(pos["x"], pos["y"], c="0.45", s=6, zorder=2)
axs[0].set_title("Input: shuffled tracers", color="0.4")

im = axs[1].scatter(pos["x"], pos["y"], c=fitted.gamma, cmap="viridis", s=6)
axs[1].plot(track["x"], track["y"], c="#e8590c", lw=2.2, label="fitted track")
axs[1].set_title("pcf.fit_track: ordering and track", color="0.4")
axs[1].legend(loc="upper right", frameon=False, labelcolor="0.4", fontsize=9)
cbar = fig.colorbar(im, ax=axs, location="bottom", shrink=0.4, aspect=30, pad=0.03)
cbar.set_label(r"ordering coordinate $\gamma$", color="0.4")
cbar.ax.tick_params(colors="0.4")
cbar.outline.set_edgecolor("0.6")

out = Path(__file__).parent / "hero.png"
fig.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
print(f"wrote {out}")
