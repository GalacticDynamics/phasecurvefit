"""Selectable exact k-nearest-neighbour backends.

``BucketKDTree`` and ``BruteForce`` are JAX-native and trace under
jit/vmap/grad; ``JaxKD`` wraps the optional jaxkd package; ``SciPy`` is scipy's
``cKDTree`` -- fastest on CPU but eager-only.
"""

__all__: tuple[str, ...] = (
    "AbstractNeighborSearch",
    "BruteForce",
    "BucketKDTree",
    "JaxKD",
    "SciPy",
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
from phasecurvefit._src.optional_deps import OptDeps

if OptDeps.JAXKD.installed:
    import jaxkd

KnnOut = tuple[Int[Array, "m k"], Float[Array, "m k"]]


def _traced(*xs: object) -> bool:
    """Whether any input is a JAX tracer (inside jit/vmap/grad)."""
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def _as_float(x: object, /) -> Array:
    """Promote integer / low-precision inputs to at least float32."""
    x = jnp.asarray(x)
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def _safe_sqrt(d2: Array, /) -> Array:
    """Euclidean distance with gradient 0 (not NaN) at coincident points."""
    pos = d2 > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, d2, 1.0)), 0.0)


def _gathered_distance(points: Array, queries: Array | None, ii: Array, /) -> Array:
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


_NOT_FINITE = "kNN inputs must be finite (no NaN or inf)."


def _check_args(points: Array, k: int, queries: Array | None, /) -> None:
    """Check ``k >= 1`` and (n, d) shapes; shared by every backend."""
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
        msg = f"k must be an integer >= 1, got {k!r}."
        raise ValueError(msg)
    if jnp.ndim(points) != 2:
        msg = f"points must have shape (n, d), got {jnp.shape(points)}."
        raise ValueError(msg)
    if queries is not None and (
        jnp.ndim(queries) != 2 or jnp.shape(queries)[1] != jnp.shape(points)[1]
    ):
        msg = (
            f"queries must have shape (m, {jnp.shape(points)[1]}) to match "
            f"points, got {jnp.shape(queries)}."
        )
        raise ValueError(msg)


# Coordinates whose largest magnitude has binary exponent in this window are
# left unscaled; outside it they are moved to its nearer edge. 2**60 keeps
# float32 squared distances finite in up to 64 dimensions (d * 2**122 < 2**128);
# 2**-20 keeps them well above float32's smallest normal (2**-126).
_SCALE_WINDOW = (-20, 60)


def _pow2_scale(points: Array, queries: Array | None, /) -> Array:
    """Power of two to divide coordinates by so squared distances stay finite.

    1 (no scaling) unless the largest coordinate magnitude falls outside
    ``2**_SCALE_WINDOW``. Dividing by a power of two is exact, so neighbour
    selection, ties included, is unchanged; it only keeps squared distances
    from overflowing float32 (coordinate gaps above ~1.8e19, e.g. metres at
    kpc scale) or underflowing to zero for tiny coordinates. Scaling only out
    of range, rather than always normalising, keeps a tight cluster next to a
    distant outlier from underflowing. Gradient-free.
    """
    m = jnp.max(jnp.abs(points), initial=0.0)
    if queries is not None:
        m = jnp.maximum(m, jnp.max(jnp.abs(queries), initial=0.0))
    _, e = jnp.frexp(m)  # m = f * 2**e, f in [0.5, 1); e = 0 for m = 0
    lo, hi = _SCALE_WINDOW
    shift = e - jnp.clip(e, lo, hi)
    return jax.lax.stop_gradient(jnp.ldexp(jnp.ones((), points.dtype), shift))


def _check_finite(points: Array, queries: Array | None, /) -> Array:
    bad = ~jnp.all(jnp.isfinite(points))
    if queries is not None:
        bad = bad | ~jnp.all(jnp.isfinite(queries))
    if not _traced(points, queries, bad):  # same error type as the SciPy backend
        if bad:
            raise ValueError(_NOT_FINITE)
        return points
    return eqx.error_if(points, bad, _NOT_FINITE)


def _bucket(n: int, /) -> int:
    """Smallest value in {2**j, 3 * 2**(j-1)} that is >= n (and >= 1)."""
    if n <= 1:
        return 1
    p = 1 << (n - 1).bit_length()  # next power of two >= n
    return p * 3 // 4 if p >= 4 and p * 3 // 4 >= n else p


def far_rows(points: Float[Array, "n d"], count: int, /) -> Float[Array, "count d"]:
    """``count`` copies of a row farther from every row of ``points`` than its diameter.

    Padding a search with these never changes the k nearest real neighbours of
    any query that was included in the array passed here (so pass the union of
    points and queries), while at least k real points exist. The row is placed
    relative to the data, within (4 sqrt(d) + 3) times its largest magnitude, so
    it never inflates the coordinate range (and with it the scaling) by more
    than that constant.
    """
    d = points.shape[1]
    lo, hi = points.min(0), points.max(0)
    # Cover the magnitude too: in float32, hi + c * spread rounds back to hi
    # once |hi| / spread >~ 1e7, which would put "far" rows on real points.
    span = jnp.maximum(jnp.max(hi - lo), jnp.max(jnp.abs(jnp.stack([lo, hi]))))
    span = jnp.where(span > 0, span, 1.0)  # all-zero data
    row = lo.at[0].set(hi[0] + (2.0 * math.sqrt(d) + 1.0) * span)
    return jnp.broadcast_to(row, (count, d))


class AbstractNeighborSearch(eqx.Module):
    """An exact k-nearest-neighbour backend.

    ``knn(points, k)`` returns, for every point, its k nearest *other* points
    (self excluded by index). ``knn(points, k, queries=q)`` returns the k
    nearest points to each query. Both give ``(indices, distances)``: Euclidean
    distances, rows sorted ascending; a missing neighbour (fewer than k
    candidates) is index ``len(points)`` with distance ``inf``. Equidistant
    neighbours: ``BucketKDTree`` and ``BruteForce`` take the lower index
    (identically eager and under jit; for ``BucketKDTree`` with indices below
    2**24, beyond which its overflow tiers may order such ties differently,
    distances unaffected); ``JaxKD`` and ``SciPy`` follow their library's
    order.
    """

    @abc.abstractmethod
    def knn(
        self,
        points: Float[Array, "n d"],
        /,
        k: int,
        *,
        queries: Float[Array, "m d"] | None = None,
    ) -> KnnOut:
        """Exact k nearest neighbours."""


def _knn_core(
    points: Array, queries: Array | None, k: int, leaf_size: int, frontier: int, /
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

    def knn(self, points: Array, /, k: int, *, queries: Array | None = None) -> KnnOut:
        _check_args(points, k, queries)
        points = _as_float(points)
        queries = None if queries is None else _as_float(queries)
        points = _check_finite(points, queries)
        n = points.shape[0]
        m = n if queries is None else queries.shape[0]
        if m == 0:
            return jnp.zeros((0, k), jnp.int32), jnp.zeros((0, k), points.dtype)
        if n == 0:
            return jnp.full((m, k), 0, jnp.int32), jnp.full(
                (m, k), jnp.inf, points.dtype
            )
        scale = _pow2_scale(points, queries)
        points = points / scale
        queries = None if queries is None else queries / scale
        if _traced(points, queries):
            sq = None if queries is None else jax.lax.stop_gradient(queries)
            ii, _ = _knn_core_jit(
                jax.lax.stop_gradient(points), sq, k, self.leaf_size, self.frontier
            )
            return ii, _gathered_distance(points, queries, ii) * scale
        extent = points if queries is None else jnp.concat([points, queries])
        padded = jnp.concat([points, far_rows(extent, _bucket(n) - n)])
        qpad = None
        if queries is not None:
            filler = jnp.broadcast_to(queries[:1], (_bucket(m) - m, points.shape[1]))
            qpad = jnp.concat([queries, filler])
        ii, d2 = _knn_core_jit(padded, qpad, k, self.leaf_size, self.frontier)
        ii, d2 = ii[:m], d2[:m]
        fake = ii >= n  # only when fewer than k real candidates exist
        dist = _safe_sqrt(jnp.where(fake, jnp.inf, d2)) * scale
        return jnp.where(fake, n, ii), dist


class BruteForce(AbstractNeighborSearch):
    """Exact brute force, ``chunk`` queries at a time. Traceable."""

    chunk: int = eqx.field(static=True, default=1024)

    def __check_init__(self) -> None:
        """Reject an invalid chunk size at construction."""
        if self.chunk < 1:
            msg = f"chunk must be >= 1, got {self.chunk}."
            raise ValueError(msg)

    def knn(self, points: Array, /, k: int, *, queries: Array | None = None) -> KnnOut:
        _check_args(points, k, queries)
        q = None if queries is None else _as_float(queries)
        points = _check_finite(_as_float(points), q)
        scale = _pow2_scale(points, q)
        q = None if q is None else q / scale
        ii, d2 = _kd.brute_knn(points / scale, k, queries=q, chunk=self.chunk)
        return ii, _safe_sqrt(d2) * scale


def _pad_k(ii: Array, dd: Array, k: int, n: int, /) -> KnnOut:
    if ii.shape[1] >= k:
        return ii[:, :k], dd[:, :k]
    extra = k - ii.shape[1]
    return (
        jnp.concat([ii, jnp.full((ii.shape[0], extra), n, jnp.int32)], axis=1),
        jnp.concat([dd, jnp.full((dd.shape[0], extra), jnp.inf, dd.dtype)], axis=1),
    )


class JaxKD(AbstractNeighborSearch):
    """The optional ``jaxkd`` package.

    Traceable, but pathological on data with scattered interlopers (minutes at
    n=100k).
    """

    def __check_init__(self) -> None:
        """Fail at construction if jaxkd is missing."""
        if not OptDeps.JAXKD.installed:
            msg = (
                "JaxKD requires the jaxkd optional dependency. "
                "Install with: uv add phasecurvefit[kdtree]"
            )
            raise ImportError(msg)

    def knn(self, points: Array, /, k: int, *, queries: Array | None = None) -> KnnOut:
        _check_args(points, k, queries)
        # jaxkd fails on float32 under x64 ("cond branches must have equal
        # output types"), so use the default float dtype.
        fdt = jnp.promote_types(jnp.float32, jnp.result_type(float))
        q = None if queries is None else jnp.asarray(queries).astype(fdt)
        points = _check_finite(jnp.asarray(points).astype(fdt), q)
        n = points.shape[0]
        if n == 0:  # jaxkd cannot build an empty tree
            m = 0 if q is None else q.shape[0]
            return jnp.zeros((m, k), jnp.int32), jnp.full((m, k), jnp.inf, points.dtype)
        scale = _pow2_scale(points, q)
        points = points / scale
        q = None if q is None else q / scale
        ps = jax.lax.stop_gradient(points)  # jaxkd's traversal is a while_loop
        tree = jaxkd.build_tree(ps)
        if q is not None:
            ii, dd = jaxkd.query_neighbors(tree, jax.lax.stop_gradient(q), k=min(k, n))
            ii, _ = _pad_k(ii.astype(jnp.int32), dd, k, n)
            return ii, _gathered_distance(points, q, ii) * scale
        kk = min(k + 1, n)
        ii, dd = jaxkd.query_neighbors(tree, ps, k=kk)
        is_self = ii == jnp.arange(n)[:, None]
        order = jnp.argsort(is_self, axis=1, stable=True)  # self (if listed) last
        ii = jnp.take_along_axis(ii, order, 1)[:, : kk - 1].astype(jnp.int32)
        ii, _ = _pad_k(ii, dd[:, : kk - 1], k, n)
        return ii, _gathered_distance(points, None, ii) * scale


SCIPY_TRACED = (
    "neighbors.SciPy cannot take traced inputs (jax.jit/vmap/grad arguments; "
    "it is host code). Use neighbors.BucketKDTree() to trace."
)


class SciPy(AbstractNeighborSearch):
    """scipy's ``cKDTree``: fastest on CPU, but host-only (raises on traced inputs).

    ``workers`` is scipy's thread count: -1 (default) uses every core.
    """

    workers: int = eqx.field(static=True, default=-1)

    def knn(self, points: Array, /, k: int, *, queries: Array | None = None) -> KnnOut:
        _check_args(points, k, queries)
        if _traced(points, queries):
            raise TypeError(SCIPY_TRACED)
        from scipy.spatial import cKDTree  # noqa: PLC0415

        p = np.asarray(_as_float(points))
        q = None if queries is None else np.asarray(_as_float(queries))
        if not np.all(np.isfinite(p)) or (q is not None and not np.all(np.isfinite(q))):
            raise ValueError(_NOT_FINITE)
        n = p.shape[0]
        if n == 0:  # don't rely on cKDTree's behaviour for an empty tree
            m = 0 if q is None else q.shape[0]
            return jnp.zeros((m, k), jnp.int32), jnp.full((m, k), jnp.inf, p.dtype)
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
