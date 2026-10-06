"""The MST-backbone orderer.

Orders tracers along a 1-D manifold via the longest path (graph diameter) of the
minimum spanning tree of a kNN graph. Unlike the velocity-following walk, it has
no progenitor and no forward/backward split: the graph diameter finds the two
tips itself, giving a clean tip-to-tip ordering. This is the algorithm of choice
for near-closed-loop / self-overlapping streams where the velocity field reverses
and a single walk cannot traverse the arc.

Velocity information is **opt-in** (pure-spatial is the default) via three
mechanisms, all reusing the phase-space notion of velocity alignment
``cos(v_i, v_j)`` (cf. ``AlignedMomentumDistanceMetric``):

1. *phase-space edge weights* (``velocity_weight``): edge weight
   ``||dq|| + velocity_weight * (1 - cos(v_i, v_j))`` — anti-parallel arms cost
   more, so the MST avoids bridging spatially-close arms that move oppositely;
2. *velocity-aware severing* (``sever_cos_threshold``): drop edges with
   ``cos(v_i, v_j) < threshold`` — cuts the reversal seam of a near-closed loop;
3. *tip orientation* (``orient_by_velocity``): flip the ordering so ``gamma``
   increases along the mean velocity.

The graph algorithms themselves (kNN, MST, shortest-path) remain **host-side**
(NumPy/SciPy) and deterministic -- the *selection* they make (which edges,
which nodes, in what order) is combinatorial and has no meaningful gradient,
since it changes in discrete jumps rather than smoothly as points move.
``order()`` runs them through ``jax.pure_callback`` with their inputs
stop-gradiented, so it is jit/vmap-traceable (``vmap_method="sequential"``: one
host call per batch element) and can sit inside a larger autodiffed pipeline.
The callback itself returns only indices (``indices``, the backbone's node
indices, ``backbone_size``) -- correctly gradient-free. The backbone
*coordinates* are then gathered from ``positions``/``velocities`` in ordinary
JAX (``P[backbone_idx]``), so -- away from the measure-zero set of points where
the selection itself changes -- gradient flows through them exactly as it would
through any other data-dependent gather (e.g. ``x[jnp.argmax(x)]``): real
w.r.t. the gathered values, zero w.r.t. the (integer, non-differentiable)
index that picked them.
"""

__all__: tuple[str, ...] = ("MSTOrderer",)

import queue
import threading
import warnings
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import plum
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    minimum_spanning_tree,
    shortest_path,
)
from scipy.spatial import cKDTree

from .base import AbstractOrderer, _check_component_keys, chord_along_ordering
from .result import OrderingResult
from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents

OnDisconnected = Literal["raise", "warn", "largest", "connect"]
_TINY = 1e-12
# An edge must be at least this multiple of the median length to be a clip
# candidate. Floors the (multiplicative) threshold so a uniformly-sampled
# backbone -- where the robust spread collapses to ~0 -- is not shredded by
# microscopic edge-length variation.
_EDGE_CLIP_MIN_RATIO = 2.0
# After cutting, a component is an outlier clump (rejected) only if it holds
# fewer than this fraction of the working points. Larger pieces are kept and
# reconnected, so cutting a genuine sparse-region edge never discards stream.
_EDGE_CLIP_SMALL_FRAC = 0.01
# cKDTree queries use every core; results are identical to a single worker.
_KDTREE_WORKERS = -1


# A single, long-lived worker thread that runs every _run_in_thread() job.
#
# The original workaround (spawn a fresh threading.Thread per call, from
# wherever _run_in_thread happened to be called) fixed a segfault reliably on
# macOS, but the identical input still segfaulted identically on Linux CI --
# even with an explicit, generous stack size on that fresh thread. That points
# away from "the new thread's stack was too small" and toward "creating a
# *new* thread from inside jax.pure_callback's own native dispatch thread is
# itself unsafe on some platforms" -- plausible, since that dispatch thread is
# a foreign thread CPython has attached a PyThreadState to (not one CPython
# created itself), and further threading operations from such a thread are
# less well-trodden than from an ordinary one.
#
# A single worker thread sidesteps that concern entirely: it is created once,
# here, at import time -- on whatever thread imports this module, which is
# always an ordinary Python thread, never jax's callback-dispatch thread.
# _run_in_thread() then only ever *hands work to* that already-running thread
# via a queue; it never creates a thread from within pure_callback's dispatch
# thread. Confirmed on Linux CI to fix the segfault.
_job_queue: "queue.Queue[tuple[Callable[[], object], queue.Queue]]" = queue.Queue()


def _worker() -> None:
    while True:
        fn, out = _job_queue.get()
        try:
            out.put(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 -- forwarded to the caller
            out.put(("err", exc))


threading.Thread(target=_worker, daemon=True, name="phasecurvefit-mst-host").start()


def _run_in_thread[T](fn: Callable[[], T]) -> T:
    """Run ``fn`` on this module's persistent worker thread; re-raise there.

    Works around a segfault observed when scipy's ``cKDTree``/sparse-graph C
    extensions run directly on the native thread ``jax.pure_callback``
    dispatches the host call onto. See the module-level worker thread's
    comment for why this hands work to an already-running thread rather than
    spawning a new one on demand.
    """
    out: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    _job_queue.put((fn, out))
    kind, payload = out.get()
    if kind == "err":
        raise payload  # type: ignore[misc]
    return payload  # type: ignore[return-value]


def _edge_cosine(V: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Cosine similarity of velocities across each candidate edge (i, j)."""
    vi, vj = V[rows], V[cols]
    num = np.sum(vi * vj, axis=1)
    den = np.linalg.norm(vi, axis=1) * np.linalg.norm(vj, axis=1)
    return np.where(den > _TINY, num / np.maximum(den, _TINY), 0.0)


def _backbone_on_component(
    P: np.ndarray, tree: object, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backbone/order within one connected component.

    Returns ``(order_idx, backbone_nodes, Cb)``: original point indices in arc
    order, the original indices of the backbone vertices, and the tip-to-tip
    backbone polyline coordinates.
    """
    sub = tree[nodes][:, nodes]
    # graph diameter via double shortest-path: farthest node a, then farthest b
    d0 = shortest_path(sub, method="D", indices=0)
    a = int(np.nanargmax(np.where(np.isinf(d0), -1.0, d0)))
    da, pred = shortest_path(sub, method="D", indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isinf(da), -1.0, da)))
    # walk predecessors b -> a to recover the backbone path
    bb: list[int] = []
    j = b
    while j != a and j >= 0:
        bb.append(j)
        j = int(pred[j])
    bb.append(a)
    bb_local = np.asarray(bb[::-1])  # tip a -> tip b, local indices into ``nodes``

    backbone_nodes = nodes[bb_local]
    Cb = P[backbone_nodes]  # backbone polyline coordinates
    seg = np.linalg.norm(np.diff(Cb, axis=0), axis=1)
    s_bb = np.concatenate([[0.0], np.cumsum(seg)])
    # project every component point onto the backbone -> along-track arc length
    _, near = cKDTree(Cb).query(P[nodes], workers=_KDTREE_WORKERS)
    order_local = np.argsort(s_bb[near], kind="stable")
    return nodes[order_local], backbone_nodes, Cb


def _sigma_clip_edges(
    tree: csr_matrix,
    P: np.ndarray,
    nodes: np.ndarray,
    *,
    sigma: float,
    max_iters: int,
) -> np.ndarray:
    """Reject outlier nodes by robust, iterated MST edge-length clipping.

    Works on the *spatial* edge lengths (not the possibly velocity-augmented
    graph weights) in *log* space -- a multiplicative rule, since lengths are
    positive and heavy-tailed. Each iteration:

    1. cut every edge longer than
       ``median(L) * exp(sigma * 1.4826 * MAD(log L))``, but never one shorter
       than ``_EDGE_CLIP_MIN_RATIO * median(L)`` (the floor keeps a
       uniformly-sampled backbone, where the robust spread is ~0, from being
       shredded, and still catches a large jump when ``MAD`` is degenerate);
    2. split the tree at the cut edges and **reject only the small components**
       (< ``_EDGE_CLIP_SMALL_FRAC`` of the working points) -- the isolated
       interlopers. Larger pieces are retained and reconnect through the intact
       ``tree``, so cutting a genuine sparse-region edge cannot discard half the
       stream;
    3. recompute the statistic on the survivors and repeat, until nothing small
       is split off (or ``max_iters``).

    Returns the surviving node set (a subset of ``nodes``, in ascending order).
    """
    log_floor = np.log(_EDGE_CLIP_MIN_RATIO)
    n = tree.shape[0]
    # The tree's edges and lengths, once; survivors are tracked by a mask
    # rather than by re-slicing the tree each iteration.
    tree_edges = tree.tocoo()
    upper = tree_edges.row < tree_edges.col  # undirected edges, once each
    ei, ej = tree_edges.row[upper], tree_edges.col[upper]
    length = np.linalg.norm(P[ei] - P[ej], axis=1)
    alive = np.zeros(n, dtype=bool)
    alive[nodes] = True
    for _ in range(max_iters):
        live = alive[ei] & alive[ej]
        # Zero-length edges join coincident points: they have no log length,
        # carry no spacing information, and can never be too long to keep.
        pos = live & (length > 0.0)
        if not pos.any():
            break
        loglen = np.log(length[pos])
        med = float(np.median(loglen))
        scale = 1.4826 * float(np.median(np.abs(loglen - med)))
        cut = np.zeros_like(pos)
        cut[pos] = loglen > med + max(sigma * scale, log_floor)
        if not cut.any():
            break
        keep = live & ~cut
        g = csr_matrix((np.ones(int(keep.sum())), (ei[keep], ej[keep])), shape=(n, n))
        # Dead nodes have no kept edges, so each is its own component and the
        # live components' sizes count live nodes only.
        _, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels)
        size_min = max(2, int(np.ceil(_EDGE_CLIP_SMALL_FRAC * int(alive.sum()))))
        small = alive & (sizes[labels] < size_min)
        if not small.any():  # cuts split off nothing small (e.g. a sparse gap)
            break
        alive &= ~small
    return np.flatnonzero(alive)


def _connect_components(P: np.ndarray, graph: csr_matrix) -> csr_matrix:
    """Join a graph's connected components along their shortest links.

    Each round, every component adds one edge: the shortest spatial link from
    one of its points to any point outside it. Components that pick each other
    merge, so the count at least halves and the loop ends in ``O(log m)``
    rounds. The bridge edges are spatial lengths only -- they deliberately
    ignore ``jump_cap`` and velocity severing, which are what split the graph.

    Performance: one k-d tree per component per round, ``O(m n log n)``. Fine for
    the few pieces a gap or a clump produces; a ``jump_cap`` far below the
    spacing shatters the graph into ~``n`` pieces and makes this slow. Switch to
    a single k-d tree with a growing ``k`` if that case ever matters.
    """
    n = P.shape[0]
    tiny = np.finfo(graph.dtype).tiny  # scipy treats zero weights as missing
    n_comp, labels = connected_components(graph, directed=False)
    while n_comp > 1:
        rows, cols, lengths = [], [], []
        for c in range(n_comp):
            inside = np.flatnonzero(labels == c)
            outside = np.flatnonzero(labels != c)
            dist, nearest = cKDTree(P[inside]).query(
                P[outside], workers=_KDTREE_WORKERS
            )
            best = int(np.argmin(dist))
            rows.append(inside[nearest[best]])
            cols.append(outside[best])
            lengths.append(dist[best])
        bridges = csr_matrix((np.maximum(lengths, tiny), (rows, cols)), shape=(n, n))
        graph = graph.maximum(bridges.maximum(bridges.T))
        n_comp, labels = connected_components(graph, directed=False)
    return graph


def _disconnected_message(
    n_comp: int, k: int, jump_cap: float, sever_cos_threshold: float | None
) -> str:
    """Explain a disconnected kNN graph, blaming only what could be the cause."""
    causes = [f"k={k} too low"]
    if np.isfinite(jump_cap):  # an infinite cap cannot be "too small"
        causes.insert(0, f"jump_cap={jump_cap} too small")
    if sever_cos_threshold is not None:
        causes.append("severing too aggressive")
    return (
        f"kNN graph is disconnected into {n_comp} components "
        f"({', '.join(causes)}). Set on_disconnected='connect' to bridge the "
        "pieces along their shortest links, or increase k (and jump_cap, relax "
        "sever_cos_threshold) to connect them; 'warn'/'largest' order only the "
        "largest piece and leave the rest unvisited."
    )


def _orient_along_velocity(
    order_idx: np.ndarray, backbone_nodes: np.ndarray, Cb: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse the ordering if it runs against the mean velocity."""
    tang = np.diff(Cb, axis=0)
    vseg = V[backbone_nodes]
    vmid = 0.5 * (vseg[:-1] + vseg[1:])
    # nansum: one NaN velocity must not turn this test into a coin flip
    # (``nan < 0`` is False, so the flip would silently never happen).
    if np.nansum(tang * vmid) < 0.0:
        return order_idx[::-1], backbone_nodes[::-1]
    return order_idx, backbone_nodes


def _mst_backbone(
    P: np.ndarray,
    V: np.ndarray,
    *,
    k: int,
    jump_cap: float,
    velocity_weight: float,
    sever_cos_threshold: float | None,
    orient_by_velocity: bool,
    on_disconnected: OnDisconnected,
    edge_clip_sigma: float | None,
    edge_clip_max_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Order points along the MST longest-path backbone.

    Returns ``(order_idx, backbone_nodes)``: the arc-length ordering (original
    point indices) and the original indices of the tip-to-tip backbone
    vertices, in order. Indices rather than coordinates -- the caller gathers
    ``P[backbone_nodes]`` in JAX so gradient flows through the gather (see the
    module docstring).
    """
    n = len(P)
    if n < 2:
        return np.arange(n), np.arange(n)

    k_eff = int(min(k, n - 1))
    nn_d, nn_i = cKDTree(P).query(P, k=k_eff + 1, workers=_KDTREE_WORKERS)
    nn_d = np.atleast_2d(nn_d)
    nn_i = np.atleast_2d(nn_i)

    # Exclude self by index, not by dropping column 0: with coincident points
    # cKDTree may list a duplicate before the point itself.
    not_self = nn_i != np.arange(n)[:, None]
    rows = np.nonzero(not_self)[0]
    cols = nn_i[not_self]
    d_edges = nn_d[not_self]  # spatial edge length

    # velocity alignment (only computed when a mechanism needs it)
    need_cos = velocity_weight > 0.0 or sever_cos_threshold is not None
    cos = _edge_cosine(V, rows, cols) if need_cos else None

    weights = d_edges.copy()
    if velocity_weight > 0.0:  # Mechanism 1: phase-space edge weights
        weights = d_edges + velocity_weight * (1.0 - cos)
    # scipy's csgraph treats zero weights as missing edges, which would cut
    # coincident points (repeat observations) out of the graph. Floor to the
    # smallest positive float: still "free", but a real edge.
    weights = np.maximum(weights, np.finfo(weights.dtype).tiny)

    keep = d_edges <= jump_cap  # sever long cross-loop edges (spatial)
    if sever_cos_threshold is not None:  # Mechanism 2: velocity-aware severing
        keep = keep & (cos >= sever_cos_threshold)

    # Left directed: csgraph treats it as undirected, taking the smaller nonzero
    # of graph[i, j] and graph[j, i]. Distances and cosines are symmetric, so
    # that equals the explicit symmetrised graph without building it.
    graph = csr_matrix((weights[keep], (rows[keep], cols[keep])), shape=(n, n))
    if on_disconnected == "connect":  # bridge the pieces instead of dropping any
        graph = _connect_components(P, graph)
    tree = minimum_spanning_tree(graph)
    tree = tree + tree.T

    n_comp, labels = connected_components(tree, directed=False)
    if n_comp != 1:
        msg = _disconnected_message(n_comp, k, jump_cap, sever_cos_threshold)
        if on_disconnected == "raise":
            raise ValueError(msg)
        if on_disconnected == "warn":
            warnings.warn(msg, stacklevel=3)
        largest = int(np.argmax(np.bincount(labels)))
        nodes = np.flatnonzero(labels == largest)
    else:
        nodes = np.arange(n)

    if edge_clip_sigma is not None:  # optional: reject outliers by MST edge length
        nodes = _sigma_clip_edges(
            tree, P, nodes, sigma=edge_clip_sigma, max_iters=edge_clip_max_iters
        )

    order_idx, backbone_nodes, Cb = _backbone_on_component(P, tree, nodes)

    if orient_by_velocity:  # Mechanism 3: orient gamma along mean velocity
        order_idx, backbone_nodes = _orient_along_velocity(
            order_idx, backbone_nodes, Cb, V
        )

    return order_idx, backbone_nodes


def _mst_backbone_padded(
    P: np.ndarray,
    V: np.ndarray,
    *,
    k: int,
    jump_cap: float,
    velocity_weight: float,
    sever_cos_threshold: float | None,
    orient_by_velocity: bool,
    on_disconnected: OnDisconnected,
    edge_clip_sigma: float | None,
    edge_clip_max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_mst_backbone``, padded to a shape fixed by ``P`` (for ``pure_callback``).

    The backbone's true vertex count is data-dependent (a shortest-path length),
    so it can't be a ``jax.pure_callback`` output shape on its own. Returns
    ``(idx_full, backbone_idx_full, backbone_len)``: ``idx_full`` is
    ``order_idx`` padded to ``len(P)`` with ``-1`` (as in
    ``OrderingResult.indices``); ``backbone_idx_full`` is ``backbone_nodes``
    padded to ``len(P)`` by repeating its last index; ``backbone_len`` is the
    true vertex count. Indices, not coordinates -- the caller gathers
    ``P[backbone_idx_full]`` in JAX (see the module docstring).
    """
    n = P.shape[0]
    order_idx, backbone_nodes = _mst_backbone(
        P,
        V,
        k=k,
        jump_cap=jump_cap,
        velocity_weight=velocity_weight,
        sever_cos_threshold=sever_cos_threshold,
        orient_by_velocity=orient_by_velocity,
        on_disconnected=on_disconnected,
        edge_clip_sigma=edge_clip_sigma,
        edge_clip_max_iters=edge_clip_max_iters,
    )

    idx_full = np.full(n, -1, dtype=np.int32)
    idx_full[: order_idx.size] = order_idx

    b = backbone_nodes.shape[0]
    backbone_idx_full = np.empty(n, dtype=np.int32)
    backbone_idx_full[:b] = backbone_nodes
    if b > 0:
        backbone_idx_full[b:] = backbone_nodes[-1]  # pad by repeating last index
    return idx_full, backbone_idx_full, np.asarray(b, dtype=np.int32)


class MSTOrderer(AbstractOrderer):
    """Order tracers along the MST longest-path backbone.

    Parameters
    ----------
    k
        Number of nearest neighbours for the kNN graph.
    jump_cap
        Edges longer than this (spatially) are severed before building the MST.
        Should exceed the typical inter-tracer spacing but stay below the
        loop-opening / arm-separation scale.
    velocity_weight
        Mechanism 1. If ``> 0``, edge weights become
        ``||dq|| + velocity_weight * (1 - cos(v_i, v_j))``. ``0`` (default) is
        pure spatial.
    sever_cos_threshold
        Mechanism 2. If not ``None``, edges with ``cos(v_i, v_j)`` below this are
        severed (e.g. ``0.0`` cuts anti-parallel arms).
    orient_by_velocity
        Mechanism 3. If ``True``, flip the ordering so ``gamma`` increases along
        the mean velocity.
    on_disconnected
        Policy when the graph splits into multiple components (a gap in the
        stream, or a ``jump_cap``/``sever_cos_threshold`` that is too tight):
        ``"raise"`` (default), ``"warn"`` (order the largest component, warn,
        leave the rest unvisited), ``"largest"`` (same, silently), or
        ``"connect"`` (join the pieces along their shortest links and order
        everything; the bridge links ignore ``jump_cap`` and velocity severing,
        which is what split the graph).
    edge_clip_sigma
        Optional outlier rejection by MST edge length. If not ``None``, robustly
        sigma-clip the backbone's *spatial* edge lengths in log space: cut edges
        longer than ``median * exp(edge_clip_sigma * 1.4826 * MAD(log L))``
        (never shorter than twice the median), split off the small components
        this isolates, and repeat (see ``edge_clip_max_iters``). Interlopers the
        MST threads in along one long edge fall away and are left unvisited
        (``indices == -1``); the rest of the stream is kept and reconnected, so a
        sparse but continuous tail is not rejected. ``None`` (default) disables
        clipping. Lower ``edge_clip_sigma`` clips more aggressively.
    edge_clip_max_iters
        Maximum sigma-clip iterations (default 5). Ignored when
        ``edge_clip_sigma`` is ``None``.

    Examples
    --------
    Order a simple 2D stream using the MST backbone:

    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf

    >>> positions = {
    ...     "x": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    ...     "y": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    ... }
    >>> velocities = {"x": jnp.ones(5), "y": jnp.full(5, 0.5)}

    >>> orderer = pcf.orderers.MSTOrderer(k=10, jump_cap=3.0)
    >>> result = pcf.order(positions, velocities, orderer)
    >>> result.indices
    Array([4, 3, 2, 1, 0], dtype=int32)

    For near-closed loops where the velocity field reverses, use
    ``velocity_weight`` to penalize edges between opposite-moving arms:

    >>> # Synthetic near-closed loop (two arms moving in opposite directions)
    >>> theta = jnp.linspace(0, 2 * jnp.pi, 100)
    >>> positions = {"x": jnp.cos(theta), "y": jnp.sin(theta)}
    >>> # Velocity tangent to the circle, but reversing at the crossing
    >>> velocities = {"x": -jnp.sin(theta), "y": jnp.cos(theta)}

    Use ``velocity_weight`` to down-weight edges between opposite-moving regions:

    >>> orderer = pcf.orderers.MSTOrderer(k=10, jump_cap=2.0, velocity_weight=1.0)
    >>> result = pcf.order(positions, velocities, orderer)
    >>> result.n_visited > 0  # Most points ordered
    Array(True, dtype=bool)

    Alternatively, use ``sever_cos_threshold`` to explicitly cut edges where
    velocities are anti-parallel:

    >>> orderer = pcf.orderers.MSTOrderer(k=10, jump_cap=2.0, sever_cos_threshold=0.0)
    >>> result = pcf.order(positions, velocities, orderer)

    The result includes a ``backbone`` polyline (the MST longest path) that
    ``__call__`` uses for smooth interpolation:

    >>> result.backbone is not None
    True
    >>> result.backbone["x"].shape
    (100,)

    Orient the ordering to increase along the mean velocity using
    ``orient_by_velocity``:

    >>> orderer = pcf.orderers.MSTOrderer(k=10, jump_cap=2.0, orient_by_velocity=True)
    >>> result = pcf.order(positions, velocities, orderer)

    Reject an interloper by MST edge length with ``edge_clip_sigma``. Here a lone
    point sits far off an otherwise clean line; clipping leaves it unvisited:

    >>> xs = jnp.concatenate([jnp.linspace(0.0, 9.0, 40), jnp.array([30.0])])
    >>> ys = jnp.concatenate([jnp.zeros(40), jnp.array([30.0])])
    >>> pos = {"x": xs, "y": ys}
    >>> vel = {"x": jnp.ones(41), "y": jnp.zeros(41)}
    >>> clipper = pcf.orderers.MSTOrderer(k=10, jump_cap=50.0, edge_clip_sigma=3.0)
    >>> result = pcf.order(pos, vel, clipper)
    >>> int(result.n_skipped)  # the lone interloper is rejected
    1

    """

    k: int = eqx.field(static=True, default=10)
    jump_cap: float = eqx.field(static=True, default=3.0)
    velocity_weight: float = eqx.field(static=True, default=0.0)
    sever_cos_threshold: float | None = eqx.field(static=True, default=None)
    orient_by_velocity: bool = eqx.field(static=True, default=False)
    on_disconnected: OnDisconnected = eqx.field(static=True, default="raise")
    edge_clip_sigma: float | None = eqx.field(static=True, default=None)
    edge_clip_max_iters: int = eqx.field(static=True, default=5)

    def __check_init__(self) -> None:
        """Reject invalid configuration early, at construction."""
        allowed = ("raise", "warn", "largest", "connect")
        if self.on_disconnected not in allowed:
            msg = (
                f"on_disconnected must be one of {allowed}; "
                f"got {self.on_disconnected!r}."
            )
            raise ValueError(msg)
        if self.velocity_weight < 0.0:
            # Only ``> 0.0`` engages the phase-space edge weights, so a negative
            # value would do nothing at all -- and would make ``velocity_aware``
            # read False on an orderer the caller thought used velocity.
            msg = f"velocity_weight must be >= 0, got {self.velocity_weight}."
            raise ValueError(msg)
        if self.edge_clip_sigma is not None and self.edge_clip_sigma <= 0:
            msg = f"edge_clip_sigma must be positive, got {self.edge_clip_sigma}."
            raise ValueError(msg)
        if self.edge_clip_max_iters < 1:
            msg = f"edge_clip_max_iters must be >= 1, got {self.edge_clip_max_iters}."
            raise ValueError(msg)

    @plum.dispatch
    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,  # noqa: ARG002
        init: AbstractResult | None = None,  # noqa: ARG002
    ) -> OrderingResult:
        """Order tracers along the MST backbone.

        Runs directly (no JAX overhead) when called eagerly -- the common case,
        per the ``AbstractOrderer`` contract. When called under ``jax.jit``,
        ``jax.vmap``, or ``jax.grad`` (i.e. ``positions``/``velocities`` are
        traced), the host graph algorithms instead run through
        ``jax.pure_callback`` (inputs stop-gradiented) so tracing doesn't
        break, returning only indices; the backbone coordinates are then
        gathered from ``positions``/``velocities`` in ordinary JAX, so
        gradient flows through them like any other data-dependent gather (see
        the module docstring).

        A caveat of the traced path: ``on_disconnected="raise"`` raises
        ``ValueError`` eagerly, but surfaces as ``jax.errors.JaxRuntimeError``
        (wrapping the same message) under jit/vmap/grad, since the host call
        actually runs at execution time, after ``order()`` has already
        returned traced outputs.
        """
        _check_component_keys(positions, velocities)

        comps = sorted(positions)
        P = jnp.stack([jnp.asarray(positions[c]) for c in comps], axis=1)
        V = jnp.stack([jnp.asarray(velocities[c]) for c in comps], axis=1)
        n, _d = P.shape

        def _host(p: np.ndarray, v: np.ndarray) -> tuple:
            return _mst_backbone_padded(
                np.asarray(p),
                np.asarray(v),
                k=self.k,
                jump_cap=self.jump_cap,
                velocity_weight=self.velocity_weight,
                sever_cos_threshold=self.sever_cos_threshold,
                orient_by_velocity=self.orient_by_velocity,
                on_disconnected=self.on_disconnected,
                edge_clip_sigma=self.edge_clip_sigma,
                edge_clip_max_iters=self.edge_clip_max_iters,
            )

        # The host call only ever needs to see values, never gradients -- the
        # selection it makes is discrete either way -- so its inputs are
        # stop-gradiented up front. That leaves pure_callback with nothing to
        # differentiate (no custom_jvp needed): the real gradient path is the
        # P[backbone_idx_full] gather below, using the original (not
        # stop-gradiented) P.
        P_static = jax.lax.stop_gradient(P)
        V_static = jax.lax.stop_gradient(V)

        if not (isinstance(P, jax.core.Tracer) or isinstance(V, jax.core.Tracer)):
            # Eager call (the common case): run directly on this thread, no
            # JAX overhead -- confirmed safe without _run_in_thread's fix (see
            # below), since it's only pure_callback's own dispatch thread that
            # triggers the crash.
            idx_full, backbone_idx_full, backbone_len = _host(P_static, V_static)
            idx_full = jnp.asarray(idx_full)
            backbone_idx_full = jnp.asarray(backbone_idx_full)
            backbone_len = jnp.asarray(backbone_len)
        else:
            result_shapes = (
                jax.ShapeDtypeStruct((n,), jnp.int32),
                jax.ShapeDtypeStruct((n,), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
            )

            def _host_threaded(p: np.ndarray, v: np.ndarray) -> tuple:
                # _run_in_thread works around a segfault observed when the
                # host computation runs directly on the native thread
                # jax.pure_callback dispatches onto. See its docstring.
                return _run_in_thread(lambda: _host(p, v))

            # Note: under an outer jax.jit/vmap, pure_callback only *records*
            # this call during tracing -- ``_host`` (and any
            # ``on_disconnected="raise"`` ValueError it raises) actually runs
            # later, at execution, after this function has already returned.
            # So a disconnected-graph failure here surfaces to the caller as
            # ``jax.errors.JaxRuntimeError`` (wrapping the original message),
            # not ``ValueError`` as it does eagerly.
            idx_full, backbone_idx_full, backbone_len = jax.pure_callback(
                _host_threaded,
                result_shapes,
                P_static,
                V_static,
                vmap_method="sequential",
            )

        backbone_full = P[backbone_idx_full]  # JAX gather: gradient flows via P
        backbone = {c: backbone_full[:, i] for i, c in enumerate(comps)}
        qs = {key: jnp.asarray(val) for key, val in positions.items()}
        return OrderingResult(
            positions=qs,
            velocities={key: jnp.asarray(val) for key, val in velocities.items()},
            indices=idx_full,
            gamma_range=(-1.0, 1.0),
            backbone=backbone,
            backbone_size=backbone_len,
            chord=chord_along_ordering(qs, idx_full),
            # ``orient_by_velocity`` only picks a direction; it does not make
            # the ordering itself velocity-aware.
            velocity_aware=self.velocity_weight > 0.0
            or self.sever_cos_threshold is not None,
        )
