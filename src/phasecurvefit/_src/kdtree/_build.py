"""Balanced, pointer-free kd-tree build with static shapes.

Node ``i`` at level ``l`` is the contiguous slice ``[i*m, (i+1)*m)`` of the
permuted points, ``m = n_pad >> l``. Each node splits at the exact median of its
widest-extent dimension. Padding rows (fewer than one per leaf) are +inf in every
dimension, so they sort last within a node and never set a split; per-node valid
counts are static, from :func:`layout`.

Build: one stable argsort per dimension up front, then per level a stable
partition of every dimension's order by cumsum + scatter (2.6x faster than a
batched argsort per level).
"""

__all__: tuple[str, ...] = ("Tree", "build_tree")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from ._layout import layout


class Tree(eqx.Module):
    """A built tree. Arrays are in tree (permuted) order unless noted."""

    points: Float[Array, "n_pad d"]  # padding rows are 0
    perm: Int[Array, " n_pad"]  # original index per row; ``n`` on padding
    valid: Bool[Array, " n_pad"]
    cell_lo: tuple[Float[Array, "nodes d"], ...]  # per level 0..depth
    cell_hi: tuple[Float[Array, "nodes d"], ...]
    split_dim: tuple[Int[Array, " nodes"], ...]  # per level 0..depth-1
    split_val: tuple[Float[Array, " nodes"], ...]
    n: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)  # effective, B_eff
    n_pad: int = eqx.field(static=True)

    @property
    def n_leaves(self) -> int:
        """Number of leaves, ``2**depth``."""
        return 2**self.depth


def build_tree(points: Float[Array, "n d"], *, leaf_size: int = 16) -> Tree:
    """Build a tree over ``points`` (``n >= 0`` rows, any dimension ``d >= 1``)."""
    n, d = points.shape
    lay = layout(n, leaf_size)
    n_pad, inf = lay.n_pad, jnp.inf
    pp = jnp.concatenate([points, jnp.full((n_pad - n, d), inf, points.dtype)])
    ords = [jnp.argsort(pp[:, j], stable=True).astype(jnp.int32) for j in range(d)]
    lo = jnp.full((1, d), -inf, points.dtype)
    hi = jnp.full((1, d), inf, points.dtype)
    cell_lo, cell_hi, split_dim, split_val = [lo], [hi], [], []
    dims = jnp.arange(d)
    for lvl in range(lay.depth):
        nn, m = 2**lvl, n_pad >> lvl
        h = m // 2
        v = lay.leaf_valid.reshape(nn, -1).sum(1)  # static valid per node
        vl = lay.leaf_valid.reshape(2 * nn, -1).sum(1)[0::2]  # static, left child
        vj = jnp.asarray(v, jnp.int32)
        o = jnp.stack(ords).reshape(d, nn, m)  # per dim, node-grouped, sorted
        first = jnp.take_along_axis(pp[o[:, :, 0]], dims[:, None, None], 2)[..., 0]
        last_id = jnp.take_along_axis(
            o, jnp.maximum(vj - 1, 0)[None, :, None].repeat(d, 0), 2
        )[..., 0]
        last = jnp.take_along_axis(pp[last_id], dims[:, None, None], 2)[..., 0]
        dim = jnp.argmax(jnp.where(vj[None] > 0, last - first, -inf), 0)  # (nn,)
        os_ = jnp.take_along_axis(o, dim[None, :, None], 0)[0]  # order on split dim
        i = np.arange(m)[None]
        left = (i < vl[:, None]) | (
            (i >= v[:, None]) & (i < v[:, None] + (h - vl[:, None]))
        )
        side = (
            jnp.zeros(n_pad, bool)
            .at[os_.reshape(-1)]
            .set(jnp.asarray(left).reshape(-1))
        )
        right_first = jnp.take_along_axis(os_, jnp.asarray(vl, jnp.int32)[:, None], 1)[
            :, 0
        ]
        split = jnp.take_along_axis(pp[right_first], dim[:, None], 1)[:, 0]
        base = (np.arange(nn) * m)[:, None]
        new = []
        for j in range(d):
            oj = o[j]
            s = side[oj]
            tgt = base + jnp.where(s, jnp.cumsum(s, 1) - 1, h + jnp.cumsum(~s, 1) - 1)
            new.append(
                jnp.zeros(n_pad, jnp.int32)
                .at[tgt.reshape(-1)]
                .set(oj.reshape(-1), unique_indices=True)
            )
        ords = new
        onehot = jax.nn.one_hot(dim, d, dtype=bool)
        lo = jnp.stack([lo, jnp.where(onehot, split[:, None], lo)], 1).reshape(
            2 * nn, d
        )
        hi = jnp.stack([jnp.where(onehot, split[:, None], hi), hi], 1).reshape(
            2 * nn, d
        )
        cell_lo.append(lo)
        cell_hi.append(hi)
        split_dim.append(dim.astype(jnp.int32))
        split_val.append(split)
    o = ords[0]
    real = o < n
    return Tree(
        points=jnp.where(real[:, None], pp[o], 0.0).astype(points.dtype),
        perm=jnp.where(real, o, n).astype(jnp.int32),
        valid=real,
        cell_lo=tuple(cell_lo),
        cell_hi=tuple(cell_hi),
        split_dim=tuple(split_dim),
        split_val=tuple(split_val),
        n=n,
        depth=lay.depth,
        leaf_size=lay.leaf_size,
        n_pad=n_pad,
    )
