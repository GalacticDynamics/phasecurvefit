"""The MST-backbone orderer.

Orders tracers along a 1-D manifold via the longest path (graph diameter) of the
minimum spanning tree of a kNN graph. Unlike the velocity-following walk, it has
no progenitor and no forward/backward split: the graph diameter finds the two
tips itself, giving a clean tip-to-tip ordering. This is the algorithm of choice
for near-closed-loop / self-overlapping streams where the velocity field reverses
and a single walk cannot traverse the arc.

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


def _backbone_on_component(
    P: np.ndarray, tree: csr_matrix, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backbone/order within one connected component.

    Returns ``(order_idx, Cb)``: original point indices in arc order and the
    tip-to-tip backbone polyline coordinates.
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
    return nodes[order_local], Cb


def _mst_backbone(
    P: np.ndarray,
    *,
    k: int,
    jump_cap: float,
    on_disconnected: OnDisconnected,
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

    keep = d_edges <= jump_cap  # sever long cross-loop edges (spatial)
    graph = csr_matrix((d_edges[keep], (rows[keep], cols[keep])), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrise
    tree = minimum_spanning_tree(graph)
    tree = tree + tree.T

    n_comp, labels = connected_components(tree, directed=False)
    if n_comp != 1:
        msg = (
            f"kNN graph is disconnected into {n_comp} components "
            f"(jump_cap={jump_cap} too small or k={k} too low). "
            "Increase jump_cap/k, or set on_disconnected='warn'/'largest'."
        )
        if on_disconnected == "raise":
            raise ValueError(msg)
        if on_disconnected == "warn":
            warnings.warn(msg, stacklevel=3)
        largest = int(np.argmax(np.bincount(labels)))
        nodes = np.flatnonzero(labels == largest)
    else:
        nodes = np.arange(n)

    return _backbone_on_component(P, tree, nodes)


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
    on_disconnected
        Policy when the graph splits into multiple components: ``"raise"``
        (default), ``"warn"`` (order the largest component, warn, leave the rest
        unvisited), or ``"largest"`` (same, silently).

    """

    k: int = eqx.field(static=True, default=10)
    jump_cap: float = eqx.field(static=True, default=3.0)
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

        order_idx, backbone_P = _mst_backbone(
            P, k=self.k, jump_cap=self.jump_cap, on_disconnected=self.on_disconnected
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
