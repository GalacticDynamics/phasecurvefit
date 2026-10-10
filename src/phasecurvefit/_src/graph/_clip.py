"""Robust, iterated rejection of outlier nodes by MST edge length."""

__all__: tuple[str, ...] = ("sigma_clip",)

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ._boruvka import boruvka

MIN_RATIO = 2.0  # never cut an edge shorter than MIN_RATIO * median length
SMALL_FRAC = 0.01  # reject split-off pieces smaller than this share of alive
MAX_REJECT_FRAC = 0.5  # never reject more than this share of alive at once
MAD_SCALE = 1.4826  # MAD -> Gaussian sigma


def _median(x: Array, mask: Array, /) -> Array:
    """``np.median`` of ``x[mask]`` (even counts average the middle two)."""
    s = jnp.sort(jnp.where(mask, x, jnp.inf))
    c = jnp.sum(mask)
    lo, hi = jnp.maximum((c - 1) // 2, 0), c // 2
    return 0.5 * (s[lo] + s[jnp.minimum(hi, s.shape[0] - 1)])


def sigma_clip(
    n: int,
    lo: Int[Array, " e"],
    hi: Int[Array, " e"],
    d: Float[Array, " e"],
    tree_mask: Bool[Array, " e"],
    alive: Bool[Array, " n"],
    /,
    *,
    sigma: float,
    max_iters: int,
) -> Bool[Array, " n"]:
    """Surviving nodes after robust MST edge-length clipping.

    Mirrors ``_sigma_clip_edges``: in log space, cut tree edges longer than
    ``median + max(sigma * 1.4826 * MAD, log MIN_RATIO)``, then reject only the
    split-off pieces smaller than ``max(2, ceil(SMALL_FRAC * alive))``; repeat
    until nothing positive, nothing cut, nothing small, or the small pieces
    would hold more than ``MAX_REJECT_FRAC`` of alive (fragmentation), or
    ``max_iters``.
    """
    if lo.shape[0] == 0 or n < 2:
        return alive
    # Work on the forest's (at most n - 1) edges only: medians sort n, not e.
    m = n - 1
    slot = jnp.where(tree_mask, jnp.cumsum(tree_mask) - 1, m)
    lo = jnp.full(m, n, jnp.int32).at[slot].set(lo, mode="drop")
    hi = jnp.full(m, n, jnp.int32).at[slot].set(hi, mode="drop")
    d = jnp.zeros(m, d.dtype).at[slot].set(d, mode="drop")
    tree_mask = jnp.zeros(m, bool).at[slot].set(tree_mask, mode="drop")
    lo, hi = jnp.minimum(lo, n - 1), jnp.minimum(hi, n - 1)  # unused slots: masked
    log_floor = math.log(MIN_RATIO)
    logd = jnp.log(jnp.where(d > 0, d, 1.0))
    zeros = jnp.zeros_like(d)

    def cond(state: tuple) -> Array:
        _, done, it = state
        return ~done & (it < max_iters)

    def body(state: tuple) -> tuple:
        alive, _, it = state
        live = tree_mask & alive[lo] & alive[hi]
        pos = live & (d > 0)
        med = _median(logd, pos)
        scale = MAD_SCALE * _median(jnp.abs(logd - med), pos)
        cut = pos & (logd > med + jnp.maximum(sigma * scale, log_floor))
        _, label = boruvka(n, lo, hi, zeros, live & ~cut)
        size = jnp.zeros(n, jnp.int32).at[label].add(1)
        size_min = jnp.maximum(2, (jnp.sum(alive) + 99) // 100)  # ceil(0.01 a)
        small = alive & (size[label] < size_min)
        # stop if nothing to cut, nothing small, or the small pieces would hold
        # more than MAX_REJECT_FRAC of alive: that is fragmentation of the
        # stream, not outlier rejection. 2 * small > alive is exact (frac 1/2).
        n_small, n_alive = jnp.sum(small), jnp.sum(alive)
        fragmented = 2 * n_small > n_alive
        stop = ~jnp.any(pos) | ~jnp.any(cut) | (n_small == 0) | fragmented
        return jnp.where(stop, alive, alive & ~small), stop, it + 1

    return jax.lax.while_loop(cond, body, (alive, jnp.zeros((), bool), 0))[0]
