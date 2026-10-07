"""Exact k-smallest selection without ``lax.top_k``.

XLA's CPU ``top_k`` costs ~0.4 us per row, which dominated the query. Two
value-only insertion networks are cheaper: the first finds the k-th smallest
value ``tau``; the second picks columns by the key ``j`` (``d < tau``) or
``W + j`` (``d == tau``), so ties go to the lower column (stable, exact).
"""

__all__: tuple[str, ...] = ("kth_smallest", "ksmallest")

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _network(at: Float[Array, "W Q"], k: int, /) -> Float[Array, "k Q"]:
    """Sorted k smallest of each column of ``at`` by insertion.

    ``t_i' = min(t_i, max(t_{i-1}, c))`` on k separate ``(Q,)`` carries, scanned
    over column blocks of ``u`` (unrolled inside). A ``(k, Q)`` array carry was
    ~20x slower; full unrolling compiles very slowly for ``W >~ 100``.
    """
    w = at.shape[0]
    u = max(d for d in range(1, 33) if w % d == 0)

    def body(t: tuple, block: Array) -> tuple:
        for jj in range(u):
            c = block[jj]
            t = (
                jnp.minimum(t[0], c),
                *[jnp.minimum(t[i], jnp.maximum(t[i - 1], c)) for i in range(1, k)],
            )
        return t, None

    init = (jnp.full(at.shape[1:], jnp.inf, at.dtype),) * k
    t, _ = jax.lax.scan(body, init, at.reshape(w // u, u, *at.shape[1:]))
    return jnp.stack(t, 0)


def _pad_width(d2: Array, k: int, /) -> Array:
    w = d2.shape[-1]
    if w >= k:
        return d2
    pad = jnp.full((*d2.shape[:-1], k - w), jnp.inf, d2.dtype)
    return jnp.concat([d2, pad], axis=-1)


def kth_smallest(d2: Float[Array, "Q W"], /, k: int) -> Float[Array, " Q"]:
    """Return the k-th smallest value of each row (``inf`` if a row has fewer)."""
    return _network(_pad_width(d2, k).T, k)[-1]


def ksmallest(
    d2: Float[Array, "Q W"], /, k: int
) -> tuple[Float[Array, "Q k"], Int[Array, "Q k"]]:
    """Sorted k smallest values per row and their column indices.

    Exact; among equal values the lower column wins. Rows with fewer than k
    finite entries return ``inf`` values (their columns are then meaningless).
    """
    a = _pad_width(d2, k)
    w = a.shape[-1]
    at = a.T
    tau = _network(at, k)[-1]
    j = jnp.arange(w, dtype=at.dtype)[:, None]
    key = jnp.where(at < tau, j, jnp.where(at == tau, w + j, jnp.inf))
    kk = _network(key, k)  # (k, Q) column keys, ascending
    idx = jnp.where(kk >= w, kk - w, kk).astype(jnp.int32).T
    vals = jnp.take_along_axis(a, idx, 1)
    # odd-even transposition sort by (value, column), width k
    v = [vals[:, i] for i in range(k)]
    ix = [idx[:, i] for i in range(k)]
    for r in range(k):
        for i in range(r % 2, k - 1, 2):
            swap = (v[i + 1] < v[i]) | ((v[i + 1] == v[i]) & (ix[i + 1] < ix[i]))
            v[i], v[i + 1] = (
                jnp.where(swap, v[i + 1], v[i]),
                jnp.where(swap, v[i], v[i + 1]),
            )
            ix[i], ix[i + 1] = (
                jnp.where(swap, ix[i + 1], ix[i]),
                jnp.where(swap, ix[i], ix[i + 1]),
            )
    return jnp.stack(v, 1), jnp.stack(ix, 1)
