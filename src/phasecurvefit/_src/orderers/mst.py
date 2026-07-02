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

from .base import AbstractOrderer
from .result import OrderingResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents

OnDisconnected = Literal["raise", "warn", "largest"]
_TINY = 1e-12


def _load_scipy() -> tuple:
    """Import scipy lazily so it is only needed when MSTOrderer actually runs.

    scipy is an optional dependency (``phasecurvefit[mst]``); importing it here
    rather than at module load keeps ``import phasecurvefit`` working without it.
    """
    try:
        from scipy.sparse import csr_matrix  # noqa: PLC0415
        from scipy.sparse.csgraph import (  # noqa: PLC0415
            connected_components,
            minimum_spanning_tree,
            shortest_path,
        )
        from scipy.spatial import cKDTree  # noqa: PLC0415
    except ImportError:
        msg = (
            "MSTOrderer requires the optional 'scipy' dependency. "
            "Install with: pip install 'phasecurvefit[mst]'."
        )
        raise ImportError(msg) from None
    return (
        cKDTree,
        csr_matrix,
        connected_components,
        minimum_spanning_tree,
        shortest_path,
    )


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
    cKDTree, _, _, _, shortest_path = _load_scipy()
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
) -> tuple[np.ndarray, np.ndarray]:
    """Order points along the MST longest-path backbone.

    Returns ``(order_idx, backbone)``: the arc-length ordering (original point
    indices) and the tip-to-tip backbone polyline coordinates.
    """
    n = len(P)
    if n < 2:
        return np.arange(n), P.copy()

    cKDTree, csr_matrix, connected_components, minimum_spanning_tree, _ = _load_scipy()
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

    """

    k: int = eqx.field(static=True, default=10)
    jump_cap: float = eqx.field(static=True, default=3.0)
    velocity_weight: float = eqx.field(static=True, default=0.0)
    sever_cos_threshold: float | None = eqx.field(static=True, default=None)
    orient_by_velocity: bool = eqx.field(static=True, default=False)
    on_disconnected: OnDisconnected = eqx.field(static=True, default="raise")

    @plum.dispatch
    def order(
        self,
        positions: VectorComponents,
        velocities: VectorComponents,
        *,
        metadata: StateMetadata | None = None,  # noqa: ARG002
    ) -> OrderingResult:
        """Order tracers along the MST backbone (host-side)."""
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
