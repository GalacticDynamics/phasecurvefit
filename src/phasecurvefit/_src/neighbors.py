"""Selectable exact k-nearest-neighbour backends.

``BucketKDTree`` (default) and ``BruteForce`` are JAX-native and trace under
jit/vmap/grad; ``Jaxkd`` wraps the optional jaxkd package; ``Scipy`` is scipy's
``cKDTree`` -- fastest on CPU but eager-only.
"""

__all__: tuple[str, ...] = (
    "AbstractNeighborSearch",
    "BruteForce",
    "BucketKDTree",
    "Jaxkd",
    "Scipy",
    "far_rows",
)

import abc
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from phasecurvefit._src import kdtree as _kd

KnnOut = tuple[Int[Array, "m k"], Float[Array, "m k"]]


def _traced(*xs: object) -> bool:
    """Whether any input is a JAX tracer (inside jit/vmap/grad)."""
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def _safe_sqrt(d2: Array) -> Array:
    """Euclidean distance with gradient 0 (not NaN) at coincident points."""
    pos = d2 > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, d2, 1.0)), 0.0)


def _gathered_distance(points: Array, queries: Array | None, ii: Array) -> Array:
    """Euclidean distance to each selected neighbour, recomputed from coordinates.

    The tree selects indices on stop-gradiented inputs (its overflow tiers use
    ``lax.while_loop``, which reverse-mode autodiff cannot cross); recomputing the
    distances by a plain gather makes them differentiable w.r.t. the points
    with the neighbour selection held fixed. Sentinel indices give ``inf``.
    """
    n = points.shape[0]
    q = points if queries is None else queries
    nb = points.at[ii].get(mode="fill", fill_value=0)
    diff = q[:, None] - nb
    return jnp.where(ii >= n, jnp.inf, _safe_sqrt(jnp.sum(diff * diff, -1)))


def _check_finite(points: Array, queries: Array | None) -> Array:
    bad = ~jnp.all(jnp.isfinite(points))
    if queries is not None:
        bad = bad | ~jnp.all(jnp.isfinite(queries))
    return eqx.error_if(points, bad, "kNN inputs must be finite (no NaN or inf).")


def _bucket(n: int) -> int:
    """Smallest value in {2**j, 3 * 2**(j-1)} that is >= n (and >= 1)."""
    if n <= 1:
        return 1
    p = 1 << (n - 1).bit_length()  # next power of two >= n
    return p * 3 // 4 if p >= 4 and p * 3 // 4 >= n else p


def far_rows(points: Float[Array, "n d"], count: int) -> Float[Array, "count d"]:
    """``count`` distinct rows farther from every point than the points' diameter.

    Padding with these never changes a real query's k nearest real neighbours
    while at least k real points exist: every real point lies within the data's
    diameter, every far row beyond it.
    """
    d = points.shape[1]
    lo, hi = points.min(0), points.max(0)
    span = jnp.max(hi - lo) + 1.0
    offset = (2.0 * math.sqrt(d) + 1.0 + jnp.arange(count, dtype=points.dtype)) * span
    return jnp.broadcast_to(lo, (count, d)).at[:, 0].set(hi[0] + offset)


class AbstractNeighborSearch(eqx.Module):
    """An exact k-nearest-neighbour backend.

    ``knn(points, k)`` returns, for every point, its k nearest *other* points
    (self excluded by index). ``knn(points, k, queries=q)`` returns the k
    nearest points to each query. Both give ``(indices, distances)``: Euclidean
    distances, rows sorted ascending; a missing neighbour (fewer than k
    candidates) is index ``len(points)`` with distance ``inf``.
    """

    @abc.abstractmethod
    def knn(
        self,
        points: Float[Array, "n d"],
        k: int,
        *,
        queries: Float[Array, "m d"] | None = None,
    ) -> KnnOut:
        """Exact k nearest neighbours."""


def _knn_core(
    points: Array, queries: Array | None, k: int, leaf_size: int, frontier: int
) -> KnnOut:
    if queries is None:
        return _kd.all_knn(points, k, leaf_size=leaf_size, frontier=frontier)
    tree = _kd.build_tree(points, leaf_size=leaf_size)
    return _kd.knn(tree, queries, k, frontier=frontier)


_knn_core_jit = jax.jit(_knn_core, static_argnums=(2, 3, 4))


class BucketKDTree(AbstractNeighborSearch):
    """JAX-native exact kd-tree, the default. Traceable under jit/vmap/grad.

    Eager calls pad ``n`` to a size bucket ({2**j, 1.5 * 2**j}) with far rows,
    so differing stream lengths share compiled code (a new bucket compiles in
    ~4-5 s); under ``jit`` the caller's shapes are used as-is.
    """

    leaf_size: int = eqx.field(static=True, default=16)
    frontier: int = eqx.field(static=True, default=16)

    def __check_init__(self) -> None:
        """Reject invalid sizes at construction."""
        if self.leaf_size < 1 or self.frontier < 1:
            msg = (
                "leaf_size and frontier must be >= 1, "
                f"got {self.leaf_size} and {self.frontier}."
            )
            raise ValueError(msg)

    def knn(self, points: Array, k: int, *, queries: Array | None = None) -> KnnOut:
        points = jnp.asarray(points)
        queries = None if queries is None else jnp.asarray(queries)
        n = points.shape[0]
        m = n if queries is None else queries.shape[0]
        if n == 0:
            return jnp.full((m, k), 0, jnp.int32), jnp.full(
                (m, k), jnp.inf, points.dtype
            )
        points = _check_finite(points, queries)
        if _traced(points, queries):
            sq = None if queries is None else jax.lax.stop_gradient(queries)
            ii, _ = _knn_core(
                jax.lax.stop_gradient(points), sq, k, self.leaf_size, self.frontier
            )
            return ii, _gathered_distance(points, queries, ii)
        padded = jnp.concatenate([points, far_rows(points, _bucket(n) - n)])
        qpad = None
        if queries is not None:
            filler = jnp.broadcast_to(queries[:1], (_bucket(m) - m, points.shape[1]))
            qpad = jnp.concatenate([queries, filler])
        ii, d2 = _knn_core_jit(padded, qpad, k, self.leaf_size, self.frontier)
        ii, d2 = ii[:m], d2[:m]
        fake = ii >= n  # only when fewer than k real candidates exist
        return jnp.where(fake, n, ii), _safe_sqrt(jnp.where(fake, jnp.inf, d2))


class BruteForce(AbstractNeighborSearch):
    """Exact brute force, ``chunk`` queries at a time. Traceable."""

    chunk: int = eqx.field(static=True, default=1024)

    def knn(self, points: Array, k: int, *, queries: Array | None = None) -> KnnOut:
        points = _check_finite(jnp.asarray(points), queries)
        q = None if queries is None else jnp.asarray(queries)
        ii, d2 = _kd.brute_knn(points, k, queries=q, chunk=self.chunk)
        return ii, _safe_sqrt(d2)


def _pad_k(ii: Array, dd: Array, k: int, n: int) -> KnnOut:
    if ii.shape[1] >= k:
        return ii[:, :k], dd[:, :k]
    extra = k - ii.shape[1]
    return (
        jnp.concatenate([ii, jnp.full((ii.shape[0], extra), n, jnp.int32)], 1),
        jnp.concatenate([dd, jnp.full((dd.shape[0], extra), jnp.inf, dd.dtype)], 1),
    )


class Jaxkd(AbstractNeighborSearch):
    """The optional ``jaxkd`` package.

    Traceable, but pathological on data with scattered interlopers (minutes at
    n=100k).
    """

    def __check_init__(self) -> None:
        """Fail at construction if jaxkd is missing."""
        try:
            import jaxkd  # noqa: F401, PLC0415
        except ImportError:
            msg = (
                "Jaxkd requires the jaxkd optional dependency. "
                "Install with: uv add phasecurvefit[kdtree]"
            )
            raise ImportError(msg) from None

    def knn(self, points: Array, k: int, *, queries: Array | None = None) -> KnnOut:
        import jaxkd  # noqa: PLC0415

        points = _check_finite(jnp.asarray(points), queries)
        n = points.shape[0]
        q = None if queries is None else jnp.asarray(queries)
        ps = jax.lax.stop_gradient(points)  # jaxkd's traversal is a while_loop
        tree = jaxkd.build_tree(ps)
        if q is not None:
            ii, dd = jaxkd.query_neighbors(tree, jax.lax.stop_gradient(q), k=min(k, n))
            ii, _ = _pad_k(ii.astype(jnp.int32), dd, k, n)
            return ii, _gathered_distance(points, q, ii)
        kk = min(k + 1, n)
        ii, dd = jaxkd.query_neighbors(tree, ps, k=kk)
        is_self = ii == jnp.arange(n)[:, None]
        order = jnp.argsort(is_self, axis=1, stable=True)  # self (if listed) last
        ii = jnp.take_along_axis(ii, order, 1)[:, : kk - 1].astype(jnp.int32)
        ii, _ = _pad_k(ii, dd[:, : kk - 1], k, n)
        return ii, _gathered_distance(points, None, ii)


class Scipy(AbstractNeighborSearch):
    """scipy's ``cKDTree``: fastest on CPU, but eager-only (raises when traced).

    ``workers`` is scipy's thread count: -1 (default) uses every core.
    """

    workers: int = eqx.field(static=True, default=-1)

    def knn(self, points: Array, k: int, *, queries: Array | None = None) -> KnnOut:
        if _traced(points, queries):
            msg = (
                "neighbors.Scipy cannot run under jax.jit/vmap/grad (it is host "
                "code). Use neighbors.BucketKDTree() to trace."
            )
            raise TypeError(msg)
        from scipy.spatial import cKDTree  # noqa: PLC0415

        p = np.asarray(points)
        q = None if queries is None else np.asarray(queries)
        if not np.all(np.isfinite(p)) or (q is not None and not np.all(np.isfinite(q))):
            msg = "kNN inputs must be finite (no NaN or inf)."
            raise ValueError(msg)
        n = p.shape[0]
        tree = cKDTree(p)
        if q is not None:
            dd, ii = tree.query(q, k=k, workers=self.workers)
            ii, dd = ii.reshape(-1, k), dd.reshape(-1, k)
        else:
            dd, ii = tree.query(p, k=k + 1, workers=self.workers)
            ii, dd = ii.reshape(n, k + 1), dd.reshape(n, k + 1)
            order = np.argsort(ii == np.arange(n)[:, None], axis=1, kind="stable")
            ii = np.take_along_axis(ii, order, 1)[:, :k]
            dd = np.take_along_axis(dd, order, 1)[:, :k]
        return jnp.asarray(ii, jnp.int32), jnp.asarray(dd, p.dtype)
