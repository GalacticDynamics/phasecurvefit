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

Results carry original indices (``tree.perm``), and ties go to the lower one,
so the answer depends only on the input, not on the tree layout or padding.
Sentinels: tree positions use ``tree.n_pad``; original indices use ``tree.n``.
"""

__all__: tuple[str, ...] = ("all_knn", "knn", "locate_leaves", "node_labels")

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._build import Tree, build_tree
from ._select import ksmallest, kth_smallest

BLOCK_LEAVES = 16
TIERS = (128, 512)  # frontier caps for the overflow tiers, then brute force
# Max queries per tier chunk; scaled down as nq // 128 (floor 32) so a sparse
# tier at small n does not pad a few overflowing queries to a wide chunk.
TIER_CHUNK = (1024, 256)
BRUTE_CHUNK = 128
QUERY_CHUNK = 16384
_ONEHOT_MAX_CAP = 32  # _descend compaction: one-hot sum up to here, scatter above


def _box_d2(q: Array, lo: Array, hi: Array, /) -> Array:
    gap = jnp.maximum(0.0, jnp.maximum(lo - q, q - hi))
    return jnp.sum(gap * gap, -1)


def node_labels(
    tree: Tree, labels: Int[Array, " n"], /
) -> tuple[Array, tuple[Array, ...]]:
    """Per-row labels and every node's homogeneous label, for ``exclude``.

    Rows: ``labels`` in tree order, ``tree.n + 1`` (EMPTY) on padding. Nodes
    (per level 0..depth): the label all their points share, ``tree.n`` (MIXED)
    if they differ, EMPTY if they hold no points. Labels must be in ``[0, n)``.
    """
    mixed, empty = tree.n, tree.n + 1
    rl = labels.astype(jnp.int32).at[tree.perm].get(mode="fill", fill_value=empty)
    rl = jnp.where(tree.valid, rl, empty)
    leaf = rl.reshape(tree.n_leaves, tree.leaf_size)
    has = leaf != empty
    mn = jnp.min(jnp.where(has, leaf, empty), 1)
    mx = jnp.max(jnp.where(has, leaf, 0), 1)
    h = jnp.where(~has.any(1), empty, jnp.where(mn == mx, mn, mixed))
    levels = [h]
    for _ in range(tree.depth):
        c0, c1 = h[0::2], h[1::2]
        h = jnp.where(c0 == empty, c1, jnp.where((c1 == empty) | (c0 == c1), c0, mixed))
        levels.append(h)
    return rl, tuple(levels[::-1])


def _skip(tree: Tree, hl: tuple, lvl: int, cand: Array, ql: Array, /) -> Array:
    """Nodes holding only the query's own label (or nothing): never candidates."""
    h = hl[lvl].at[cand].get(mode="fill", fill_value=tree.n + 1)
    return (h == ql) | (h == tree.n + 1)


def _bound_exclude(
    tree: Tree, xq: Array, ql: Array, rl: Array, hl: tuple, k: int, /
) -> Array:
    """Bound r2 for excluded queries by a greedy descent to an other-label leaf.

    At each level step into the nearer child that holds some point with a
    different label; the leaf reached has one (if any exists anywhere), so the
    k-th distance to its other-label points bounds the true k-th (``inf`` if it
    has fewer than k). Avoids an ``inf`` bound when the query's whole leaf block
    shares its label (a point deep inside a large component).
    """
    node = jnp.zeros(xq.shape[0], jnp.int32)
    for lvl in range(1, tree.depth + 1):
        c0, c1 = 2 * node, 2 * node + 1
        s0, s1 = _skip(tree, hl, lvl, c0, ql), _skip(tree, hl, lvl, c1, ql)
        d0 = _box_d2(xq, tree.cell_lo[lvl][c0], tree.cell_hi[lvl][c0])
        d1 = _box_d2(xq, tree.cell_lo[lvl][c1], tree.cell_hi[lvl][c1])
        node = jnp.where(s0 | (~s1 & (d1 < d0)), c1, c0)
    b = tree.leaf_size
    pts = tree.points.reshape(tree.n_leaves, b, -1)[node]
    lab = rl.reshape(tree.n_leaves, b)[node]
    diff = xq[:, None] - pts
    ok = (lab != ql[:, None]) & (lab != tree.n + 1)
    return kth_smallest(jnp.where(ok, jnp.sum(diff * diff, -1), jnp.inf), k)


def _descend(
    tree: Tree,
    xq: Array,
    r2: Array,
    cap: int,
    /,
    ex: tuple | None = None,
) -> tuple[Array, Array]:
    """Leaves whose cell is within ``r2`` of each query, at most ``cap`` of them.

    Returns ``(front (Q, cap) leaf ids with sentinel n_leaves, overflow (Q,))``.
    """
    nq = xq.shape[0]
    rows = jnp.arange(nq)[:, None]
    l0 = min(tree.depth, math.floor(math.log2(cap)))
    n0 = 2**l0
    seed = jnp.concat([jnp.arange(n0), jnp.full(cap - n0, n0)]).astype(jnp.int32)
    front = jnp.broadcast_to(seed, (nq, cap))
    over = jnp.zeros(nq, bool)
    slots = jnp.arange(cap, dtype=jnp.int32)
    # Slack on the cut-off: r2 is often exactly a point's squared distance (the
    # k-th so far), and that point's leaf box distance equals it only up to
    # rounding (computed in another fusion). An exact ``<=`` pruned the k-th
    # neighbour's leaf by one ulp. Extra candidates cannot change the result.
    r2 = r2 * (1.0 + 4.0 * xq.shape[1] * jnp.finfo(xq.dtype).eps)

    def keep(cand: Array, nn: int, lvl: int) -> tuple[Array, Array]:
        lo = tree.cell_lo[lvl].at[cand].get(mode="fill", fill_value=0)
        hi = tree.cell_hi[lvl].at[cand].get(mode="fill", fill_value=0)
        ok = (cand < nn) & (_box_d2(xq[:, None], lo, hi) <= r2[:, None])
        if ex is not None:
            ok = ok & ~_skip(tree, ex[1], lvl, cand, ex[2][:, None])
        rank = jnp.where(ok, jnp.cumsum(ok, 1) - 1, cap)
        if cap <= _ONEHOT_MAX_CAP:  # (Q, cap, 2cap) one-hot beats a CPU scatter
            hit = rank[:, None, :] == slots[None, :, None]
            out = jnp.sum(jnp.where(hit, cand[:, None, :] - nn, 0), -1) + nn
            return out.astype(jnp.int32), ok.sum(1)
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
    tree: Tree,
    xq: Array,
    qself: Array,
    cand: Array,
    k: int,
    /,
    *,
    top_k: bool,
    ex: tuple | None = None,
) -> tuple[Array, Array]:
    """K nearest valid points among the candidate leaves -> (sq_dist, original idx)."""
    nq, m = cand.shape
    b = tree.leaf_size
    pts = tree.points.reshape(tree.n_leaves, b, -1)
    val = tree.valid.reshape(tree.n_leaves, b)
    xc = pts.at[cand].get(mode="fill", fill_value=0).reshape(nq, m * b, -1)
    ok = val.at[cand].get(mode="fill", fill_value=False).reshape(nq, m * b)
    pos = (cand[..., None] * b + jnp.arange(b)).reshape(nq, m * b)
    if top_k:
        orig = tree.perm.at[pos].get(mode="fill", fill_value=tree.n)
    else:  # gathered in (m*b, nq) layout: the networks want ids transposed
        pos_t = (cand.T[:, None] * b + jnp.arange(b)[:, None]).reshape(m * b, nq)
        orig = tree.perm.at[pos_t].get(mode="fill", fill_value=tree.n).T
    diff = xq[:, None] - xc
    if ex is not None:
        ok = ok & (
            ex[0].at[pos].get(mode="fill", fill_value=tree.n + 1) != ex[2][:, None]
        )
    d2 = jnp.where(ok & (pos != qself[:, None]), jnp.sum(diff * diff, -1), jnp.inf)
    dd, ii = ksmallest(d2, orig, k, top_k=top_k)  # pads width < k itself
    return dd, jnp.where(jnp.isinf(dd), tree.n, ii)


def _bound_leaves(tree: Tree, k: int, /) -> int:
    """Leaves per aligned block guaranteed to hold ``k + 1`` valid points.

    Padding is spread evenly (``layout``), so ``bb`` aligned leaves hold at
    most ``ceil(bb * pad / n_leaves)`` padding rows. Assuming ``leaf_size - 1``
    valid per leaf instead is 0 at leaf_size 1, which grew the block to the
    whole tree: O(n^2) time and memory (8.7 GB at n = 1e5).
    """
    bb = min(BLOCK_LEAVES, tree.n_leaves)
    pad = tree.n_pad - tree.n
    while (
        bb < tree.n_leaves
        and bb * tree.leaf_size - (-(-bb * pad // tree.n_leaves)) < k + 1
    ):
        bb *= 2
    return bb


def _bound(tree: Tree, xq: Array, qself: Array, qleaf: Array, k: int, /) -> Array:
    """k-th squared distance to the valid points of each query's leaf block."""
    bb = _bound_leaves(tree, k)
    s = bb * tree.leaf_size
    blk = qleaf // bb
    xb = tree.points.reshape(-1, s, tree.points.shape[1])[blk]
    vb = tree.valid.reshape(-1, s)[blk]
    pb = blk[:, None] * s + jnp.arange(s)
    diff = xq[:, None] - xb
    d2 = jnp.where(vb & (pb != qself[:, None]), jnp.sum(diff * diff, -1), jnp.inf)
    return kth_smallest(d2, k)


Handler = Callable[..., tuple[Array, Array, Array]]


def _finish(
    xq: Array,
    qself: Array,
    over: Array,
    dd: Array,
    ii: Array,
    r2: Array,
    chunk: int,
    handler: Handler,
    qlab: Array | None = None,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Rerun ``handler`` on the queries flagged in ``over`` (compacted, chunked)."""
    nq = xq.shape[0]
    n_over = over.sum()
    tgt = jnp.where(over, jnp.cumsum(over) - 1, nq + chunk)
    buf = (
        jnp.full(nq + chunk, nq, jnp.int32)
        .at[tgt]
        .set(jnp.arange(nq, dtype=jnp.int32), mode="drop")
    )
    still = jnp.zeros(nq, bool)

    def body(state: tuple) -> tuple:
        j, dd, ii, still, r2 = state
        q = jax.lax.dynamic_slice(buf, (j * chunk,), (chunk,))
        xs = xq.at[q].get(mode="fill", fill_value=0)
        ss = qself.at[q].get(mode="fill", fill_value=-1)
        rs = r2.at[q].get(mode="fill", fill_value=jnp.inf)
        if qlab is None:
            d_new, i_new, o_new = handler(xs, ss, rs)
        else:
            qs = qlab.at[q].get(mode="fill", fill_value=0)
            d_new, i_new, o_new = handler(xs, ss, rs, qs)
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
    /,
    exclude: tuple[Array, tuple, Array] | None = None,
) -> tuple[Array, Array]:
    """Exact kNN for queries ``xq`` -> ``(sq_dist (Q, k), original idx (Q, k))``.

    ``exclude = (row_labels, node_labels, query_labels)`` (see
    :func:`node_labels`) skips every point sharing the query's label.
    """
    qlab = () if exclude is None else (exclude[2],)  # per-query labels, if any
    rl, hl = (None, None) if exclude is None else exclude[:2]

    def ex_of(ql: Array | None) -> tuple | None:
        return None if exclude is None else (rl, hl, ql)

    nq, d = xq.shape
    qc = min(QUERY_CHUNK, nq)
    n_chunks = -(-nq // qc)
    padn = n_chunks * qc - nq

    def pad(a: Array, fill: float) -> Array:
        return jnp.concat([a, jnp.full((padn, *a.shape[1:]), fill, a.dtype)])

    def chunk(args: tuple[Array, ...]) -> tuple[Array, ...]:
        x, s, lf, *ql = args
        ql = next(iter(ql), None)
        if exclude is None:
            r2 = _bound(tree, x, s, lf, k)
        else:
            r2 = _bound_exclude(tree, x, ql, rl, hl, k)
        front, over = _descend(tree, x, r2, frontier, ex_of(ql))
        dd, ii = _merge(tree, x, s, front, k, top_k=False, ex=ex_of(ql))
        return dd, ii, over, r2

    inputs = (
        pad(xq, 0).reshape(n_chunks, qc, d),
        pad(qself, -1).reshape(n_chunks, qc),
        pad(qleaf, 0).reshape(n_chunks, qc),
        *(pad(a, 0).reshape(n_chunks, qc) for a in qlab),
    )
    dd, ii, over, r2 = jax.lax.map(chunk, inputs)
    dd, ii = dd.reshape(-1, k)[:nq], ii.reshape(-1, k)[:nq]
    over, r2 = over.reshape(-1)[:nq] & qvalid, r2.reshape(-1)[:nq]
    r2 = jnp.minimum(r2, dd[:, -1])  # even a truncated candidate set bounds the k-th

    for cap, tchunk in zip(TIERS, TIER_CHUNK, strict=True):
        if cap <= frontier:
            continue

        def tier(
            xs: Array, ss: Array, rs: Array, qs: Array | None = None, cap: int = cap
        ) -> tuple[Array, ...]:
            front, o = _descend(tree, xs, rs, cap, ex_of(qs))
            d_new, i_new = _merge(tree, xs, ss, front, k, top_k=True, ex=ex_of(qs))
            return d_new, i_new, o

        tchunk = min(tchunk, max(32, nq // 128))  # noqa: PLW2901
        dd, ii, over, r2 = _finish(xq, qself, over, dd, ii, r2, tchunk, tier, *qlab)
        over = over & qvalid

    pos_all = jnp.arange(tree.n_pad)
    # ponytail: per-chunk (BRUTE_CHUNK, n_pad, d) allocation, ~0.5-1.5 GB at
    # n~1e6 when any query reaches brute force (rare, ~0.03% on interloper
    # data); upgrade = smaller BRUTE_CHUNK or loop over dimensions.

    def brute(
        xs: Array, ss: Array, rs: Array, qs: Array | None = None
    ) -> tuple[Array, ...]:
        del rs
        diff = xs[:, None] - tree.points[None]
        d2 = jnp.sum(diff * diff, -1)
        ok = tree.valid[None] & (pos_all[None] != ss[:, None])
        if exclude is not None:
            ok = ok & (rl[None] != qs[:, None])
        d2 = jnp.where(ok, d2, jnp.inf)
        perm = jnp.broadcast_to(tree.perm, d2.shape)
        d_new, i_new = ksmallest(d2, perm, k, top_k=True)
        i_new = jnp.where(jnp.isinf(d_new), tree.n, i_new)
        return d_new, i_new, jnp.zeros(xs.shape[0], bool)

    dd, ii, _, _ = _finish(xq, qself, over, dd, ii, r2, BRUTE_CHUNK, brute, *qlab)
    return dd, ii


def locate_leaves(tree: Tree, queries: Float[Array, "m d"], /) -> Int[Array, " m"]:
    """Return the leaf each query falls in, by descending the split planes."""
    node = jnp.zeros(queries.shape[0], jnp.int32)
    rows = jnp.arange(queries.shape[0])
    for lvl in range(tree.depth):
        dim = tree.split_dim[lvl][node]
        right = queries[rows, dim] >= tree.split_val[lvl][node]
        node = 2 * node + right.astype(jnp.int32)
    return node


def all_knn(
    points: Float[Array, "n d"], /, k: int, *, leaf_size: int = 16, frontier: int = 16
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
    out_i = jnp.full((n, k), n, jnp.int32).at[tree.perm].set(ii, mode="drop")
    out_d = jnp.full((n, k), jnp.inf, points.dtype).at[tree.perm].set(dd, mode="drop")
    return out_i, out_d


def knn(
    tree: Tree,
    queries: Float[Array, "m d"],
    /,
    k: int,
    *,
    frontier: int = 16,
    exclude: tuple[Int[Array, " n"], Int[Array, " m"]] | None = None,
) -> tuple[Int[Array, "m k"], Float[Array, "m k"]]:
    """Exact k nearest tree points to each query (no self-exclusion).

    ``exclude=(point_labels, query_labels)`` (integers in ``[0, n)``) skips
    every point whose label equals the query's, exactly: subtrees holding only
    that label are pruned whole. Returns ``(indices into the tree's original
    points, sq_dist)``, rows sorted by distance; missing neighbours are index
    ``tree.n`` with distance ``inf``.
    """
    m = queries.shape[0]
    if m == 0 or tree.n == 0:
        return (
            jnp.full((m, k), tree.n, jnp.int32),
            jnp.full((m, k), jnp.inf, queries.dtype),
        )
    leaf = locate_leaves(tree, queries)
    order = jnp.argsort(leaf)  # process in tree order for locality
    ex = None
    if exclude is not None:
        rl, hl = node_labels(tree, exclude[0])
        ex = (rl, hl, exclude[1].astype(jnp.int32)[order])
    dd, ii = _query(
        tree,
        queries[order],
        jnp.full(m, -1, jnp.int32),
        jnp.ones(m, bool),
        leaf[order],
        k,
        frontier,
        ex,
    )
    out_i = jnp.zeros((m, k), jnp.int32).at[order].set(ii)
    out_d = jnp.zeros((m, k), queries.dtype).at[order].set(dd)
    return out_i, out_d
