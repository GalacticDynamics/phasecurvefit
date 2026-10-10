"""Exact k-smallest selection without ``lax.top_k``.

XLA's CPU ``top_k`` costs ~0.4 us per row, which dominated the query. Two
value-only insertion networks are cheaper: the first finds the k-th smallest
value ``tau``; the second picks by the integer key ``j`` (column, ``d < tau``)
or ``W + id`` (``d == tau``), so ties at the boundary go to the lower *id*.
Ids are the caller's (original point indices), so the selection does not
depend on the tree layout or on padding.
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

    init = (jnp.full(at.shape[1:], _top(at.dtype), at.dtype),) * k
    t, _ = jax.lax.scan(body, init, at.reshape(w // u, u, *at.shape[1:]))
    return jnp.stack(t, 0)


def _top(dtype: jnp.dtype, /) -> float | int:
    return jnp.inf if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).max


def _pad_width(d2: Array, k: int, /, fill: float | None = None) -> Array:
    w = d2.shape[-1]
    if w >= k:
        return d2
    fill = _top(d2.dtype) if fill is None else fill
    pad = jnp.full((*d2.shape[:-1], k - w), fill, d2.dtype)
    return jnp.concat([d2, pad], axis=-1)


def kth_smallest(d2: Float[Array, "Q W"], /, k: int) -> Float[Array, " Q"]:
    """Return the k-th smallest value of each row (``inf`` if a row has fewer)."""
    return _network(_pad_width(d2, k).T, k)[-1]


# Above this k the final sort uses lax.sort: the unrolled transposition sort is
# ~2.5x faster at k=10 but its trace grows as k**2 (k=50 compiled for minutes).
_UNROLL_MAX_K = 16
# At k = 7, 8 XLA CPU fuses the whole unrolled sort into the output
# concatenate and recomputes it per column (~15-30x slower than k = 9);
# lax.sort is ~7x faster there. k <= 6 and 9..16 are not affected.
_FUSION_TRAP_K = (7, 8)


def _sort_pairs(v: Array, ix: Array, k: int, /) -> tuple[Array, Array]:
    """Sort each row by (value, id)."""
    if k > _UNROLL_MAX_K or k in _FUSION_TRAP_K:
        return jax.lax.sort((v, ix), dimension=1, num_keys=2)
    v = [v[:, i] for i in range(k)]
    ix = [ix[:, i] for i in range(k)]
    for r in range(k):  # odd-even transposition sort
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


def ksmallest(
    d2: Float[Array, "Q W"], ids: Int[Array, "Q W"], /, k: int, *, top_k: bool = False
) -> tuple[Float[Array, "Q k"], Int[Array, "Q k"]]:
    """Sorted k smallest values per row and their ids, ties to the lower id.

    Exact and independent of column order. ``ids`` must lie in
    ``[0, 2**31 - W)``. Rows with fewer than k finite entries return ``inf``
    values (their ids are then meaningless). ``top_k`` selects with
    ``lax.top_k`` instead of the networks (better for wide rows).
    """
    a, ids = _pad_width(d2, k), _pad_width(ids.astype(jnp.int32), k, 0)
    w = a.shape[-1]
    if top_k:
        # Strictly-nearer entries come straight from the first top_k (its
        # values are exact). Only a row whose group tied at tau does not fit
        # in the remaining slots needs a second top_k to pick the lowest ids;
        # it runs only if some row needs it (continuous data: never). Under
        # vmap the cond becomes a select and both branches run. Float keys:
        # XLA's CPU top_k is ~70x slower on int32. ponytail: ids >= 2**24
        # round in float32, so ties among them may not follow id order (the
        # distances stay exact).
        v1, c1 = jax.lax.top_k(-a, k)
        v1 = -v1
        # max, not [:, -1:]: slicing top_k's output made XLA CPU ~60x slower.
        tau = jnp.max(v1, axis=1, keepdims=True)
        n_lt = jnp.sum(v1 < tau, axis=1, keepdims=True)  # all of them are in v1
        n_eq = jnp.sum(a == tau, axis=1, keepdims=True)
        cut = jnp.any(jnp.isfinite(tau) & (n_lt + n_eq > k))

        def lowest_tied_ids() -> Array:
            kdt = jnp.promote_types(a.dtype, jnp.float32)
            tie_key = jnp.where(a == tau, ids.astype(kdt), jnp.inf)
            c2 = jax.lax.top_k(-tie_key, k)[1]
            slot = jnp.arange(k)[None]
            c2 = jnp.take_along_axis(c2, jnp.clip(slot - n_lt, 0, k - 1), 1)
            return jnp.where(slot < n_lt, c1, c2)

        col = jax.lax.cond(cut, lowest_tied_ids, lambda: c1)
        vals = jnp.where(jnp.arange(k)[None] < n_lt, v1, tau)
        out = jnp.take_along_axis(ids, col, 1)
        return _sort_pairs(vals, out, k)
    # (W, Q) layout for the networks; pass ``ids`` as ``x.T`` to skip a copy.
    at = a.T
    tau = _network(at, k)[-1]
    big = jnp.iinfo(jnp.int32).max
    j = jnp.arange(w, dtype=jnp.int32)[:, None]
    key = jnp.where(at < tau, j, jnp.where(at == tau, w + ids.T, big))
    kk = _network(key, k).T  # at least k entries are <= tau, so every kk < big
    col = jnp.where(kk < w, kk, 0)
    vals = jnp.where(kk < w, jnp.take_along_axis(a, col, 1), tau[:, None])
    out = jnp.where(kk < w, jnp.take_along_axis(ids, col, 1), kk - w)
    return _sort_pairs(vals, out, k)
