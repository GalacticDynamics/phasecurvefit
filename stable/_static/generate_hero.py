#!/usr/bin/env -S uv run python
"""Generate the landing-page figures, ``hero.png`` and ``hero-field.png``.

Each runs the same code as its tab of the example on the landing page
(``docs/index.md``). ``hero.png``: a shuffled, noisy spiral is ordered and
fitted with ``pcf.fit_track``; the left panel shows the tracers in their
shuffled input order, the right panel colours them by the fitted ordering
coordinate gamma and draws the fitted mean track. ``hero-field.png``: the same
spiral as an image; its pixels are fitted as samples weighted by the density,
and the right panel colours them by gamma.

Run from the repository root::

    uv run --group docs python docs/_static/generate_hero.py
"""

__all__: tuple[str, ...] = ()

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

import phasecurvefit as pcf

OUT = Path(__file__).parent

# --- the landing-page example: point cloud -----------------------------------
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
track = fitted(jnp.linspace(*fitted.track_range, 300))
# ------------------------------------------------------------------------------

# --- the landing-page example: density field ---------------------------------
t = jnp.linspace(0.5, 0.5 + 3 * jnp.pi, 400)  # the same spiral, as an image
r = 1 + 0.6 * t
grid = jnp.linspace(-7.5, 7.5, 120)
X, Y = jnp.meshgrid(grid, grid)
cx, cy = r * jnp.cos(t), r * jnp.sin(t)
d2 = jnp.min((X[..., None] - cx) ** 2 + (Y[..., None] - cy) ** 2, axis=-1)
image = jnp.exp(-d2 / (2 * 0.3**2))  # density, blurred across the curve

keep = image.ravel() > 0.05  # each bright pixel becomes a sample
fpos = {"x": X.ravel()[keep], "y": Y.ravel()[keep]}
fvel = jax.tree.map(jnp.zeros_like, fpos)  # an image has no velocities
weights = image.ravel()[keep]  # weighted by the density there

fordering, ffitted, flosses = pcf.fit_track(
    fpos, fvel, key=jax.random.key(1), weights=weights, training_config=config
)
ftrack = ffitted(jnp.linspace(*ffitted.track_range, 300))
# ------------------------------------------------------------------------------


def _figure(left_title: str, right_title: str, /) -> tuple[Figure, np.ndarray]:
    fig, axs = plt.subplots(
        1, 2, figsize=(9, 4.2), sharex=True, sharey=True, layout="constrained"
    )
    for ax in axs:
        ax.set(aspect="equal", xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_color("0.6")
    axs[0].set_title(left_title, color="0.4")
    axs[1].set_title(right_title, color="0.4")
    return fig, axs


def _finish(
    fig: Figure,
    axs: np.ndarray,
    im: ScalarMappable,
    trk: dict[str, jax.Array],
    name: str,
    /,
) -> None:
    axs[1].plot(trk["x"], trk["y"], c="#e8590c", lw=2.2, label="fitted track")
    axs[1].legend(loc="upper right", frameon=False, labelcolor="0.4", fontsize=9)
    cbar = fig.colorbar(im, ax=axs, location="bottom", shrink=0.4, aspect=30, pad=0.03)
    cbar.set_label(r"ordering coordinate $\gamma$", color="0.4")
    cbar.ax.tick_params(colors="0.4")
    cbar.outline.set_edgecolor("0.6")
    out = OUT / name
    fig.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
    print(f"wrote {out}")


fig, axs = _figure("Input: shuffled tracers", "pcf.fit_track: ordering and track")
axs[0].plot(pos["x"], pos["y"], "-", c="0.6", lw=0.4)
axs[0].scatter(pos["x"], pos["y"], c="0.45", s=6, zorder=2)
im = axs[1].scatter(pos["x"], pos["y"], c=fitted.gamma, cmap="viridis", s=6)
_finish(fig, axs, im, track, "hero.png")

fig, axs = _figure("Input: a density image", "pcf.fit_track with weights")
extent = (grid[0], grid[-1], grid[0], grid[-1])
axs[0].imshow(image, origin="lower", extent=extent, cmap="Greys")
im = axs[1].scatter(
    fpos["x"], fpos["y"], c=ffitted.gamma, cmap="viridis", s=3, alpha=weights
)
_finish(fig, axs, im, ftrack, "hero-field.png")
