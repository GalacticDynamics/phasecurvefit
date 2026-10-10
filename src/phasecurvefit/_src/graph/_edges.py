"""The kNN edge list: lengths, velocity cosines, weights, keep mask; orientation."""

__all__: tuple[str, ...] = ("knn_edges", "orient_flip")

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def knn_edges(
    p: Float[Array, "n d"],
    v: Float[Array, "n d"],
    nbr: Int[Array, "n k"],
    real: Bool[Array, " n"],
    /,
    *,
    jump_cap: float,
    velocity_weight: float,
    sever_cos_threshold: float | None,
    scale: float | Array = 1.0,
) -> tuple[Array, Array, Array, Array, Array]:
    """Undirected edge list ``(lo, hi, d, w, valid)`` from kNN indices.

    ``nbr[i]`` lists ``i``'s neighbours; an index ``>= n`` is a missing
    neighbour. ``d`` is the spatial length, ``w`` the MST weight
    ``max(d + velocity_weight * (1 - cos), tiny)``; ``valid`` drops missing and
    padded neighbours, edges longer than ``jump_cap`` and (if set) edges with
    velocity cosine below ``sever_cos_threshold``.

    ``p`` may be pre-divided by ``scale`` (e.g. a power of two) so squared
    differences cannot overflow; ``d`` is returned times ``scale``, in the
    caller's units. The cosine is scale-free: each velocity is divided by its
    largest component first, so a tiny one keeps its direction; an end with no
    direction (zero, or NaN) gives ``cos = 1`` (no penalty, never severed).
    """
    n, k = nbr.shape
    rows = jnp.repeat(jnp.arange(n), k)
    raw = nbr.reshape(-1)
    cols = jnp.minimum(raw, n - 1)
    diff = p[rows] - p[cols]
    d = jnp.sqrt(jnp.sum(diff * diff, axis=-1)) * scale
    valid = (raw < n) & real[rows] & real[cols] & (d <= jump_cap)
    w = d
    if velocity_weight > 0.0 or sever_cos_threshold is not None:
        big = jnp.max(jnp.abs(v), axis=-1, keepdims=True)  # NaN for NaN rows
        u = jnp.where(big > 0, v / jnp.where(big > 0, big, 1.0), 0.0)
        ui, uj = u[rows], u[cols]
        num = jnp.sum(ui * uj, axis=-1)
        den = jnp.linalg.norm(ui, axis=-1) * jnp.linalg.norm(uj, axis=-1)
        cos = jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), 1.0)
        if velocity_weight > 0.0:
            w = d + velocity_weight * (1.0 - cos)
        if sever_cos_threshold is not None:
            valid = valid & (cos >= sever_cos_threshold)
    w = jnp.maximum(w, jnp.finfo(w.dtype).tiny)
    return jnp.minimum(rows, cols), jnp.maximum(rows, cols), d, w, valid


def orient_flip(
    p: Float[Array, "n d"],
    v: Float[Array, "n d"],
    full: Int[Array, " n"],
    blen: Int[Array, ""],
    /,
) -> Bool[Array, ""]:
    """Whether the path runs against the mean velocity.

    Segments whose velocity term is not wholly finite are skipped (NaN, under
    ``nan_policy="omit"``): one NaN must not make ``nan < 0`` silently never
    flip, and a partly NaN velocity must not count half a dot.
    """
    n = full.shape[0]
    seg = jnp.arange(n - 1) < blen - 1
    tang = p[full[1:]] - p[full[:-1]]
    vmid = 0.5 * (v[full[:-1]] + v[full[1:]])
    dot = tang * vmid
    ok = seg & jnp.all(jnp.isfinite(dot), axis=-1)
    return jnp.sum(jnp.where(ok[:, None], dot, 0.0)) < 0.0
