"""Bridge a graph's components along their shortest spatial links."""

__all__: tuple[str, ...] = ("connect",)

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from ._boruvka import boruvka
from ._pointer import n_rounds

Nearest = Callable[[Array], tuple[Array, Array]]


def connect(
    labels: Int[Array, " n"],
    real: Bool[Array, " n"],
    nearest: Nearest,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Bridge edges ``(lo, hi, d, valid)`` (``2n`` slots) joining all real components.

    Mirrors ``_connect_components``: each round, every component adds its
    shortest link to a point outside it, and linked components merge, so the
    count at least halves. ``labels`` are component labels in ``[0, n)`` (e.g.
    from ``boruvka``). ``nearest(labels)`` must return, for every point, the
    index of its nearest point with a *different* label (``>= n`` if none) and
    the squared distance to it -- exactly (e.g. a kd-tree query with
    ``exclude``). Bridge lengths are spatial; they ignore any cap or severing.
    Padded (non-real) rows never link: they lie beyond the real points'
    diameter, so no real point picks one while another real component exists.
    """
    n = labels.shape[0]
    slots = 2 * n
    dist_dtype = jax.eval_shape(nearest, labels)[1].dtype
    i32 = jnp.int32
    nodes = jnp.arange(n, dtype=i32)
    labels = labels.astype(i32)

    def cond(state: tuple) -> Array:
        *_, go, it = state
        return go & (it < n_rounds(n))

    def body(state: tuple) -> tuple:
        labels, blo, bhi, bd, bv, used, _, it = state
        j, dd = nearest(labels)
        jc = jnp.minimum(j, n - 1)
        ok = real & (j < n) & real[jc]
        # each component's shortest link: lexicographic min of (dd, i)
        do = jnp.where(ok, dd, jnp.inf)
        md = jnp.full(n, jnp.inf, dd.dtype).at[labels].min(do)
        tie = jnp.where(ok & (do == md[labels]), nodes, n)
        best = jnp.full(n, n, i32).at[labels].min(tie)
        has = (best < n) & (labels == nodes)  # component representatives
        i_star = jnp.minimum(best, n - 1)
        j_star = jc[i_star]
        other = labels[j_star]
        # record every component's bridge (as scipy does), in fresh slots
        slot = jnp.where(has, used + jnp.cumsum(has, dtype=i32) - 1, slots)
        blo = blo.at[slot].set(jnp.minimum(i_star, j_star).astype(i32), mode="drop")
        bhi = bhi.at[slot].set(jnp.maximum(i_star, j_star).astype(i32), mode="drop")
        bd = bd.at[slot].set(jnp.sqrt(dd[i_star]), mode="drop")
        bv = bv.at[slot].set(has, mode="drop")
        # merge: use connected components of this round's bridges
        _, rep_labels = boruvka(n, nodes, other, jnp.zeros(n, dist_dtype), has)
        labels = rep_labels[labels]
        used = used + jnp.sum(has, dtype=i32)
        return labels, blo, bhi, bd, bv, used, jnp.any(has), it + 1

    init = (
        labels,
        jnp.full(slots, n, i32),
        jnp.full(slots, n, i32),
        jnp.zeros(slots, dist_dtype),
        jnp.zeros(slots, bool),
        i32(0),
        jnp.bool_(n > 1),
        i32(0),
    )
    _, blo, bhi, bd, bv, *_ = jax.lax.while_loop(cond, body, init)
    bd = jnp.maximum(bd, jnp.finfo(bd.dtype).tiny)
    return blo, bhi, bd, bv
