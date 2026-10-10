"""Blocked brute-force kNN: the exactness oracle and the small-n path."""

__all__: tuple[str, ...] = ("brute_knn",)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def brute_knn(
    points: Float[Array, "n d"],
    /,
    k: int,
    *,
    queries: Float[Array, "m d"] | None = None,
    chunk: int = 1024,
) -> tuple[Int[Array, "m k"], Float[Array, "m k"]]:
    """Exact kNN comparing every query with every point, ``chunk`` queries at a time.

    Same contract as ``all_knn`` (``queries=None``: self excluded by index) and
    ``knn`` (``queries`` given: no exclusion): ``(indices, sq_dist)``, rows
    sorted, missing neighbours index ``n`` with distance ``inf``.
    """
    n = points.shape[0]
    self_mode = queries is None
    q = points if self_mode else queries
    m = q.shape[0]
    if m == 0:
        return jnp.zeros((0, k), jnp.int32), jnp.zeros((0, k), points.dtype)
    qc = min(chunk, m)
    nc = -(-m // qc)
    qp = jnp.concat([q, jnp.zeros((nc * qc - m, q.shape[1]), q.dtype)])
    ids = jnp.arange(nc * qc, dtype=jnp.int32)

    def one(args: tuple[Array, Array]) -> tuple[Array, Array]:
        xq, iq = args
        diff = xq[:, None] - points[None]
        d2 = jnp.sum(diff * diff, -1)
        if self_mode:
            d2 = jnp.where(jnp.arange(n)[None] == iq[:, None], jnp.inf, d2)
        if n < k:
            d2 = jnp.concat([d2, jnp.full((qc, k - n), jnp.inf, d2.dtype)], axis=1)
        neg, col = jax.lax.top_k(-d2, k)
        dd = -neg
        return jnp.where(jnp.isinf(dd), n, col).astype(jnp.int32), dd

    ii, dd = jax.lax.map(one, (qp.reshape(nc, qc, -1), ids.reshape(nc, qc)))
    return ii.reshape(-1, k)[:m], dd.reshape(-1, k)[:m]
