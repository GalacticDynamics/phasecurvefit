"""Borůvka minimum spanning forest and connected components, fixed shape."""

__all__: tuple[str, ...] = ("boruvka", "largest_component")

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ._pointer import find_roots, n_rounds


def boruvka(
    n: int,
    lo: Int[Array, " e"],
    hi: Int[Array, " e"],
    w: Float[Array, " e"],
    valid: Bool[Array, " e"],
    /,
) -> tuple[Bool[Array, " e"], Int[Array, " n"]]:
    """Minimum spanning forest of an undirected edge list over nodes ``0..n-1``.

    Edges are compared by ``(w, edge index)``, a strict total order, so each
    component's cheapest outgoing edge is two segment-minima (no sorting), and
    an undirected pair listed twice is harmless: both sides pick the
    lower-indexed copy, after which the other is internal. Returns
    ``(tree_mask, labels)``: forest edges in the caller's order, and each node's
    component label (the smallest node index in it).
    """
    e = lo.shape[0]
    i32 = jnp.int32
    nodes = jnp.arange(n, dtype=i32)
    if e == 0:  # static: no edges, every node is its own component
        return jnp.zeros(0, bool), nodes
    idx = jnp.arange(e, dtype=i32)
    inf = jnp.asarray(jnp.inf, w.dtype)

    def cond(state: tuple) -> Array:
        _, _, go, it = state
        return go & (it < n_rounds(n))

    def body(state: tuple) -> tuple:
        labels, tree, _, it = state
        cl, ch = labels[lo], labels[hi]
        wo = jnp.where(valid & (cl != ch), w, inf)
        mw = jnp.full(n, inf).at[cl].min(wo).at[ch].min(wo)
        tie_l = jnp.where((wo < inf) & (wo == mw[cl]), idx, e)
        tie_h = jnp.where((wo < inf) & (wo == mw[ch]), idx, e)
        best = jnp.full(n, e, i32).at[cl].min(tie_l).at[ch].min(tie_h)
        has = best < e
        b = jnp.minimum(best, e - 1)
        other = jnp.where(cl[b] == nodes, ch[b], cl[b])
        mutual = best.at[other].get(mode="fill", fill_value=e) == best
        hook = has & ~(mutual & (nodes < other))
        tree = tree.at[jnp.where(has, best, e)].set(has, mode="drop")
        labels = find_roots(jnp.where(hook, other, nodes))[labels]
        return labels, tree, jnp.any(has), it + 1

    init = (nodes, jnp.zeros(e, bool), jnp.bool_(n > 1), i32(0))
    labels, tree, _, _ = jax.lax.while_loop(cond, body, init)
    smallest = jnp.full(n, n, i32).at[labels].min(nodes)
    return tree, smallest[labels]


def largest_component(
    labels: Int[Array, " n"], real: Bool[Array, " n"], /
) -> tuple[Bool[Array, " n"], Int[Array, ""]]:
    """Mask of the largest real component, and the number of real components.

    Ties in size go to the lowest label, as ``argmax(bincount(labels))`` does.
    """
    n = labels.shape[0]
    size = jnp.zeros(n, jnp.int32).at[labels].add(real.astype(jnp.int32))
    n_comp = jnp.sum((size > 0) & (labels == jnp.arange(n)))
    return labels == jnp.argmax(size), n_comp
