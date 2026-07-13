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

The computation is **host-side** (NumPy/SciPy) and deterministic; ``order()`` is
not jit/vmap-traceable (per the ``AbstractOrderer`` contract).
"""

__all__: tuple[str, ...] = ("MSTOrderer",)

import warnings
from typing import Literal

import equinox as eqx
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

from .base import AbstractOrderer
from .result import OrderingResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents

OnDisconnected = Literal["raise", "warn", "largest"]
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
    _, near = cKDTree(Cb).query(P[nodes])
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
    current = nodes
    for _ in range(max_iters):
        sub = tree[current][:, current].tocoo()
        upper = sub.row < sub.col  # undirected edges, once each
        ei, ej = sub.row[upper], sub.col[upper]
        if ei.size == 0:
            break
        loglen = np.log(np.linalg.norm(P[current[ei]] - P[current[ej]], axis=1))
        med = float(np.median(loglen))
        scale = 1.4826 * float(np.median(np.abs(loglen - med)))
        cut = loglen > med + max(sigma * scale, log_floor)
        if not cut.any():
            break
        m = current.size
        keep = ~cut
        g = csr_matrix((np.ones(int(keep.sum())), (ei[keep], ej[keep])), shape=(m, m))
        g = g.maximum(g.T)
        _, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels)
        size_min = max(2, int(np.ceil(_EDGE_CLIP_SMALL_FRAC * m)))
        small = np.isin(labels, np.flatnonzero(sizes < size_min))
        if not small.any():  # cuts split off nothing small (e.g. a sparse gap)
            break
        current = current[~small]
    return current


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

    Returns ``(order_idx, backbone)``: the arc-length ordering (original point
    indices) and the tip-to-tip backbone polyline coordinates.
    """
    n = len(P)
    if n < 2:
        return np.arange(n), P.copy()

    k_eff = int(min(k, n - 1))
    nn_d, nn_i = cKDTree(P).query(P, k=k_eff + 1)
    nn_d = np.atleast_2d(nn_d)
    nn_i = np.atleast_2d(nn_i)

    rows = np.repeat(np.arange(n), k_eff)
    cols = nn_i[:, 1:].ravel()
    d_edges = nn_d[:, 1:].ravel()  # spatial edge length

    # velocity alignment (only computed when a mechanism needs it)
    need_cos = velocity_weight > 0.0 or sever_cos_threshold is not None
    cos = _edge_cosine(V, rows, cols) if need_cos else None

    weights = d_edges.copy()
    if velocity_weight > 0.0:  # Mechanism 1: phase-space edge weights
        weights = d_edges + velocity_weight * (1.0 - cos)

    keep = d_edges <= jump_cap  # sever long cross-loop edges (spatial)
    if sever_cos_threshold is not None:  # Mechanism 2: velocity-aware severing
        keep = keep & (cos >= sever_cos_threshold)

    graph = csr_matrix((weights[keep], (rows[keep], cols[keep])), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrise
    tree = minimum_spanning_tree(graph)
    tree = tree + tree.T

    n_comp, labels = connected_components(tree, directed=False)
    if n_comp != 1:
        msg = (
            f"kNN graph is disconnected into {n_comp} components "
            f"(jump_cap={jump_cap} too small, k={k} too low, or severing too "
            "aggressive). Increase jump_cap/k, relax sever_cos_threshold, or set "
            "on_disconnected='warn'/'largest'."
        )
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
        tang = np.diff(Cb, axis=0)
        vseg = V[backbone_nodes]
        vmid = 0.5 * (vseg[:-1] + vseg[1:])
        if np.sum(tang * vmid) < 0.0:
            order_idx = order_idx[::-1]
            Cb = Cb[::-1]

    return order_idx, Cb


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
        Policy when the graph splits into multiple components: ``"raise"``
        (default), ``"warn"`` (order the largest component, warn, leave the rest
        unvisited), or ``"largest"`` (same, silently).
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
        allowed = ("raise", "warn", "largest")
        if self.on_disconnected not in allowed:
            msg = (
                f"on_disconnected must be one of {allowed}; "
                f"got {self.on_disconnected!r}."
            )
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
    ) -> OrderingResult:
        """Order tracers along the MST backbone (host-side)."""
        if set(positions) != set(velocities):
            missing = sorted(set(positions) - set(velocities))
            extra = sorted(set(velocities) - set(positions))
            msg = (
                "positions and velocities must have the same component keys; "
                f"missing={missing}, extra={extra}."
            )
            raise ValueError(msg)

        comps = sorted(positions)
        P = np.stack([np.asarray(positions[c]) for c in comps], axis=1)
        V = np.stack([np.asarray(velocities[c]) for c in comps], axis=1)
        order_idx, backbone_P = _mst_backbone(
            P,
            V,
            k=self.k,
            jump_cap=self.jump_cap,
            velocity_weight=self.velocity_weight,
            sever_cos_threshold=self.sever_cos_threshold,
            orient_by_velocity=self.orient_by_velocity,
            on_disconnected=self.on_disconnected,
            edge_clip_sigma=self.edge_clip_sigma,
            edge_clip_max_iters=self.edge_clip_max_iters,
        )

        n = P.shape[0]
        idx_full = np.full(n, -1, dtype=np.int32)
        idx_full[: order_idx.size] = order_idx

        backbone = {c: jnp.asarray(backbone_P[:, i]) for i, c in enumerate(comps)}
        return OrderingResult(
            positions={key: jnp.asarray(val) for key, val in positions.items()},
            velocities={key: jnp.asarray(val) for key, val in velocities.items()},
            indices=jnp.asarray(idx_full),
            gamma_range=(-1.0, 1.0),
            backbone=backbone,
        )
