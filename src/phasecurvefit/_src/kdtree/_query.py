"""Exact kNN on a :class:`Tree`: per-point leaf-cell pruning.

For each query:

1. **Bound** ``r2``: the k-th squared distance to the valid points of the
   aligned block of ``BLOCK_LEAVES`` leaves around the query's leaf.
2. **Candidates**: every leaf whose cell is within ``r2`` of the query, by a
   level-synchronous descent with a static frontier cap.
3. **Merge**: direct-difference squared distances (not ``|a|^2+|b|^2-2ab``) to
   the candidates' valid points, then exact k-smallest selection.
4. **Overflow**: queries whose frontier overflowed are compacted (sentinel
   index ``Q``, ``mode="drop"/"fill"``) and finished in ``lax.while_loop`` tiers
   with wider caps, then brute force. Each tier tightens ``r2`` to the k-th
   distance of its own candidate set, which is still a valid bound. No
   ``lax.cond`` is used: under ``vmap``/batched ``lax.map`` it becomes ``select``
   and runs both branches.

Sentinels: tree positions use ``tree.n_pad``; original indices use ``tree.n``.
"""

__all__: tuple[str, ...] = ("all_knn", "knn", "locate_leaves")

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._build import Tree, build_tree
from ._select import ksmallest, kth_smallest

BLOCK_LEAVES = 16
TIERS = (128, 512)  # frontier caps for the overflow tiers, then brute force
TIER_CHUNK = (256, 32)
BRUTE_CHUNK = 128
QUERY_CHUNK = 16384


def _box_d2(q: Array, lo: Array, hi: Array) -> Array:
    gap = jnp.maximum(0.0, jnp.maximum(lo - q, q - hi))
    return jnp.sum(gap * gap, -1)


def _descend(tree: Tree, xq: Array, r2: Array, cap: int) -> tuple[Array, Array]:
    """Leaves whose cell is within ``r2`` of each query, at most ``cap`` of them.

    Returns ``(front (Q, cap) leaf ids with sentinel n_leaves, overflow (Q,))``.
    """
    nq = xq.shape[0]
    rows = jnp.arange(nq)[:, None]
    l0 = min(tree.depth, math.floor(math.log2(cap)))
    n0 = 2**l0
    seed = jnp.concatenate([jnp.arange(n0), jnp.full(cap - n0, n0)]).astype(jnp.int32)
    front = jnp.broadcast_to(seed, (nq, cap))
    over = jnp.zeros(nq, bool)

    def keep(cand: Array, nn: int, lvl: int) -> tuple[Array, Array]:
        lo = tree.cell_lo[lvl].at[cand].get(mode="fill", fill_value=0)
        hi = tree.cell_hi[lvl].at[cand].get(mode="fill", fill_value=0)
        ok = (cand < nn) & (_box_d2(xq[:, None], lo, hi) <= r2[:, None])
        rank = jnp.where(ok, jnp.cumsum(ok, 1) - 1, cap)
        out = jnp.full((nq, cap), nn, jnp.int32).at[rows, rank].set(cand, mode="drop")
        return out, ok.sum(1)

    front, _ = keep(front, n0, l0)
    for lvl in range(l0 + 1, tree.depth + 1):
        nn = 2**lvl
        kids = jnp.stack([2 * front, 2 * front + 1], -1).reshape(nq, 2 * cap)
        front, cnt = keep(jnp.where(kids >= nn, nn, kids), nn, lvl)
        over = over | (cnt > cap)
    return front, over


def _merge(
    tree: Tree, xq: Array, qself: Array, cand: Array, k: int, *, top_k: bool
) -> tuple[Array, Array]:
    """K nearest valid points among the candidate leaves -> (sq_dist, tree position)."""
    nq, m = cand.shape
    b = tree.leaf_size
    pts = tree.points.reshape(tree.n_leaves, b, -1)
    val = tree.valid.reshape(tree.n_leaves, b)
    xc = pts.at[cand].get(mode="fill", fill_value=0).reshape(nq, m * b, -1)
    ok = val.at[cand].get(mode="fill", fill_value=False).reshape(nq, m * b)
    pos = (cand[..., None] * b + jnp.arange(b)).reshape(nq, m * b)
    diff = xq[:, None] - xc
    d2 = jnp.where(ok & (pos != qself[:, None]), jnp.sum(diff * diff, -1), jnp.inf)
    if m * b < k:
        d2 = jnp.concatenate([d2, jnp.full((nq, k - m * b), jnp.inf, d2.dtype)], 1)
        pos = jnp.concatenate(
            [pos, jnp.full((nq, k - m * b), tree.n_pad, jnp.int32)], 1
        )
    if top_k:
        neg, col = jax.lax.top_k(-d2, k)
        dd = -neg
    else:
        dd, col = ksmallest(d2, k)
    ii = jnp.where(jnp.isinf(dd), tree.n_pad, jnp.take_along_axis(pos, col, 1))
    return dd, ii


def _bound(tree: Tree, xq: Array, qself: Array, qleaf: Array, k: int) -> Array:
    """k-th squared distance to the valid points of each query's leaf block."""
    bb = min(BLOCK_LEAVES, tree.n_leaves)
    while bb < tree.n_leaves and bb * max(tree.leaf_size - 1, 0) < k + 1:
        bb *= 2
    s = bb * tree.leaf_size
    blk = qleaf // bb
    xb = tree.points.reshape(-1, s, tree.points.shape[1])[blk]
    vb = tree.valid.reshape(-1, s)[blk]
    pb = blk[:, None] * s + jnp.arange(s)
    diff = xq[:, None] - xb
    d2 = jnp.where(vb & (pb != qself[:, None]), jnp.sum(diff * diff, -1), jnp.inf)
    return kth_smallest(d2, k)


Handler = Callable[[Array, Array, Array], tuple[Array, Array, Array]]


def _finish(
    xq: Array,
    qself: Array,
    over: Array,
    dd: Array,
    ii: Array,
    r2: Array,
    chunk: int,
    handler: Handler,
) -> tuple[Array, Array, Array, Array]:
    """Rerun ``handler`` on the queries flagged in ``over`` (compacted, chunked)."""
    nq = xq.shape[0]
    n_over = over.sum()
    tgt = jnp.where(over, jnp.cumsum(over) - 1, nq + chunk)
    buf = jnp.full(nq + chunk, nq, jnp.int32).at[tgt].set(jnp.arange(nq), mode="drop")
    still = jnp.zeros(nq, bool)

    def body(state: tuple) -> tuple:
        j, dd, ii, still, r2 = state
        q = jax.lax.dynamic_slice(buf, (j * chunk,), (chunk,))
        xs = xq.at[q].get(mode="fill", fill_value=0)
        ss = qself.at[q].get(mode="fill", fill_value=-1)
        rs = r2.at[q].get(mode="fill", fill_value=jnp.inf)
        d_new, i_new, o_new = handler(xs, ss, rs)
        r2 = r2.at[q].set(jnp.minimum(rs, d_new[:, -1]), mode="drop")
        return (
            j + 1,
            dd.at[q].set(d_new, mode="drop"),
            ii.at[q].set(i_new, mode="drop"),
            still.at[q].set(o_new, mode="drop"),
            r2,
        )

    n_chunks = (n_over + chunk - 1) // chunk
    _, dd, ii, still, r2 = jax.lax.while_loop(
        lambda s: s[0] < n_chunks, body, (jnp.int32(0), dd, ii, still, r2)
    )
    return dd, ii, still, r2


def _query(
    tree: Tree,
    xq: Array,
    qself: Array,
    qvalid: Array,
    qleaf: Array,
    k: int,
    frontier: int,
) -> tuple[Array, Array]:
    """Exact kNN for queries ``xq`` -> ``(sq_dist (Q, k), tree position (Q, k))``."""
    nq, d = xq.shape
    qc = min(QUERY_CHUNK, nq)
    n_chunks = -(-nq // qc)
    padn = n_chunks * qc - nq

    def pad(a: Array, fill: float) -> Array:
        return jnp.concatenate([a, jnp.full((padn, *a.shape[1:]), fill, a.dtype)])

    def chunk(args: tuple[Array, Array, Array]) -> tuple[Array, ...]:
        x, s, lf = args
        r2 = _bound(tree, x, s, lf, k)
        front, over = _descend(tree, x, r2, frontier)
        dd, ii = _merge(tree, x, s, front, k, top_k=False)
        return dd, ii, over, r2

    dd, ii, over, r2 = jax.lax.map(
        chunk,
        (
            pad(xq, 0).reshape(n_chunks, qc, d),
            pad(qself, -1).reshape(n_chunks, qc),
            pad(qleaf, 0).reshape(n_chunks, qc),
        ),
    )
    dd, ii = dd.reshape(-1, k)[:nq], ii.reshape(-1, k)[:nq]
    over, r2 = over.reshape(-1)[:nq] & qvalid, r2.reshape(-1)[:nq]
    r2 = jnp.minimum(r2, dd[:, -1])  # even a truncated candidate set bounds the k-th

    for cap, tchunk in zip(TIERS, TIER_CHUNK, strict=True):
        if cap <= frontier:
            continue

        def tier(xs: Array, ss: Array, rs: Array, cap: int = cap) -> tuple[Array, ...]:
            front, o = _descend(tree, xs, rs, cap)
            d_new, i_new = _merge(tree, xs, ss, front, k, top_k=True)
            return d_new, i_new, o

        dd, ii, over, r2 = _finish(xq, qself, over, dd, ii, r2, tchunk, tier)
        over = over & qvalid

    pos_all = jnp.arange(tree.n_pad)

    def brute(xs: Array, ss: Array, rs: Array) -> tuple[Array, ...]:
        del rs
        diff = xs[:, None] - tree.points[None]
        d2 = jnp.sum(diff * diff, -1)
        d2 = jnp.where(tree.valid[None] & (pos_all[None] != ss[:, None]), d2, jnp.inf)
        if tree.n_pad < k:
            extra = jnp.full((d2.shape[0], k - tree.n_pad), jnp.inf, d2.dtype)
            d2 = jnp.concatenate([d2, extra], 1)
        neg, col = jax.lax.top_k(-d2, k)
        d_new = -neg
        i_new = jnp.where(jnp.isinf(d_new), tree.n_pad, col.astype(jnp.int32))
        return d_new, i_new, jnp.zeros(xs.shape[0], bool)

    dd, ii, _, _ = _finish(xq, qself, over, dd, ii, r2, BRUTE_CHUNK, brute)
    return dd, ii


def locate_leaves(tree: Tree, queries: Float[Array, "m d"]) -> Int[Array, " m"]:
    """Return the leaf each query falls in, by descending the split planes."""
    node = jnp.zeros(queries.shape[0], jnp.int32)
    rows = jnp.arange(queries.shape[0])
    for lvl in range(tree.depth):
        dim = tree.split_dim[lvl][node]
        right = queries[rows, dim] >= tree.split_val[lvl][node]
        node = 2 * node + right.astype(jnp.int32)
    return node


def _to_original(tree: Tree, ii: Array) -> Array:
    return tree.perm.at[ii].get(mode="fill", fill_value=tree.n)


def all_knn(
    points: Float[Array, "n d"], k: int, *, leaf_size: int = 16, frontier: int = 16
) -> tuple[Int[Array, "n k"], Float[Array, "n k"]]:
    """Exact k nearest neighbours of every point among the others.

    Self is excluded by index. Returns ``(indices, sq_dist)`` in input order,
    each row sorted by distance; missing neighbours (fewer than ``k`` other
    points) are index ``n`` with distance ``inf``.
    """
    n = points.shape[0]
    if n == 0:
        return jnp.zeros((0, k), jnp.int32), jnp.zeros((0, k), points.dtype)
    tree = build_tree(points, leaf_size=leaf_size)
    pos = jnp.arange(tree.n_pad, dtype=jnp.int32)
    dd, ii = _query(
        tree, tree.points, pos, tree.valid, pos // tree.leaf_size, k, frontier
    )
    out_i = (
        jnp.full((n, k), n, jnp.int32)
        .at[tree.perm]
        .set(_to_original(tree, ii), mode="drop")
    )
    out_d = jnp.full((n, k), jnp.inf, points.dtype).at[tree.perm].set(dd, mode="drop")
    return out_i, out_d


def knn(
    tree: Tree, queries: Float[Array, "m d"], k: int, *, frontier: int = 16
) -> tuple[Int[Array, "m k"], Float[Array, "m k"]]:
    """Exact k nearest tree points to each query (no self-exclusion).

    Returns ``(indices into the tree's original points, sq_dist)``, rows sorted
    by distance; missing neighbours are index ``tree.n`` with distance ``inf``.
    """
    m = queries.shape[0]
    if m == 0 or tree.n == 0:
        return (
            jnp.full((m, k), tree.n, jnp.int32),
            jnp.full((m, k), jnp.inf, queries.dtype),
        )
    leaf = locate_leaves(tree, queries)
    order = jnp.argsort(leaf)  # process in tree order for locality
    dd, ii = _query(
        tree,
        queries[order],
        jnp.full(m, -1, jnp.int32),
        jnp.ones(m, bool),
        leaf[order],
        k,
        frontier,
    )
    out_i = jnp.zeros((m, k), jnp.int32).at[order].set(_to_original(tree, ii))
    out_d = jnp.zeros((m, k), queries.dtype).at[order].set(dd)
    return out_i, out_d
