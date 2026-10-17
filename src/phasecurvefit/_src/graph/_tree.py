"""Tree stage: Euler tour, root distances, re-rooting and the diameter path."""

__all__: tuple[str, ...] = ("diameter_path",)

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ._pointer import accumulate, list_rank


def _parents(
    n: int, src: Array, dst: Array, aw: Array, pos: Array, ok: Array, twin: Array, /
) -> tuple[Array, Array, Array, Array]:
    """Parents, root distances and tour entry/exit positions for one rooting.

    An arc is a down-arc iff it precedes its twin in the tour. Returns
    ``(dist, first, last, down)`` per node / arc; nodes never entered keep
    ``first = -1`` and ``last = big`` (the root, or unreachable).
    """
    m = src.shape[0]
    down = ok & (pos < pos[twin])
    tgt = jnp.where(down, dst, n)
    parent = jnp.arange(n).at[tgt].set(src, mode="drop")
    pw = jnp.zeros(n, aw.dtype).at[tgt].set(aw, mode="drop")
    first = jnp.full(n, -1).at[tgt].set(pos, mode="drop")
    last = jnp.full(n, m + 1).at[tgt].set(pos[twin], mode="drop")
    _, dist = accumulate(parent, pw)
    return dist, first, last, down


def diameter_path(
    n: int,
    lo: Int[Array, " e"],
    hi: Int[Array, " e"],
    w: Float[Array, " e"],
    tree_mask: Bool[Array, " e"],
    alive: Bool[Array, " n"],
    /,
) -> tuple[Int[Array, " n"], Int[Array, ""]]:
    """Tip-to-tip path of the tree (forest) restricted to ``alive`` nodes.

    Reproduces ``_diameter_path``: farthest node ``a`` from the smallest alive
    node ``r`` (by tree distance, ties to the smallest index), then farthest
    ``b`` from ``a``; nodes not connected to ``r`` are ignored. Returns
    ``(full, blen)``: the path ``a -> b`` padded to ``n`` by repeating its last
    node, and its length.
    """
    if lo.shape[0] == 0:  # static: no edges, the path is the root alone
        r = jnp.argmax(alive).astype(jnp.int32)
        return jnp.full(n, r), jnp.int32(1)
    m = max(n - 1, 1)  # a forest on n nodes has at most n - 1 edges
    use = tree_mask & alive[lo] & alive[hi]
    slot = jnp.where(use, jnp.cumsum(use) - 1, m)
    t_lo = jnp.full(m, n).at[slot].set(lo, mode="drop")
    t_hi = jnp.full(m, n).at[slot].set(hi, mode="drop")
    tw = jnp.zeros(m, w.dtype).at[slot].set(w, mode="drop")
    tv = jnp.zeros(m, bool).at[slot].set(use, mode="drop")

    # 2m arcs; arc a's twin is a +- m. Sort by (valid first, src, dst).
    src0, dst0 = jnp.concat([t_lo, t_hi]), jnp.concat([t_hi, t_lo])
    aw0, av0 = jnp.concat([tw, tw]), jnp.concat([tv, tv])
    a2 = 2 * m
    order = jnp.lexsort((dst0, src0, ~av0))
    inv = jnp.zeros(a2, jnp.int32).at[order].set(jnp.arange(a2, dtype=jnp.int32))
    src, dst, aw, av = src0[order], dst0[order], aw0[order], av0[order]
    twin = inv[(order + m) % a2]

    # Per node: first sorted arc out of it and how many.
    ar = jnp.arange(a2)
    nsrc = jnp.where(av, src, n)
    start = jnp.full(n, a2).at[nsrc].min(ar, mode="drop")
    count = jnp.zeros(n, jnp.int32).at[nsrc].add(1, mode="drop")
    v = dst  # arc u->v continues with the arc after twin (v->u) in v's list
    after = twin + 1
    vstart = start.at[v].get(mode="fill", fill_value=0)
    vend = vstart + count.at[v].get(mode="fill", fill_value=0)
    succ = jnp.where(av, jnp.where(after < vend, after, vstart), a2)

    # Root r: smallest alive node. Break r's circuit before its first arc.
    r = jnp.argmax(alive)
    f = start[r]
    has_arcs = f < a2
    succ = jnp.where(succ == f, a2, succ)
    steps, reached = list_rank(succ)
    ok = av & reached
    tour = jnp.where(has_arcs, steps.at[f].get(mode="fill", fill_value=0) + 1, 1)
    pos = jnp.where(ok, tour - 1 - steps, 0)

    nodes = jnp.arange(n)
    dist, _, _, down = _parents(n, src, dst, aw, pos, ok, twin)
    entered = jnp.zeros(n, bool).at[jnp.where(down, dst, n)].set(down, mode="drop")
    in_piece = entered | (nodes == r)
    a = jnp.argmax(jnp.where(in_piece, dist, -1))

    # Re-root at a: rotate the circuit to start at a's earliest arc.
    pa = jnp.min(jnp.where(ok & (src == a), pos, a2))
    pos2 = jnp.where(ok, (pos - pa) % tour, 0)
    dist2, first2, last2, _ = _parents(n, src, dst, aw, pos2, ok, twin)
    b = jnp.argmax(jnp.where(in_piece, dist2, -1))

    on = in_piece & (first2 <= first2[b]) & (last2 >= last2[b])
    key = jnp.where(on, first2, a2 + 1)
    path = jnp.argsort(key, stable=True)
    blen = jnp.sum(on)
    full = jnp.where(nodes < blen, path, path[blen - 1])
    return full.astype(jnp.int32), blen.astype(jnp.int32)
