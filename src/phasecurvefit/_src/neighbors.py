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
)

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxknn
import numpy as np
from jaxtyping import Array, Float, Int

from phasecurvefit._src.checks import value_error_if
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


def _check_finite(points: Array, queries: Array | None, /) -> Array:
    bad = ~jnp.all(jnp.isfinite(points))
    if queries is not None:
        bad = bad | ~jnp.all(jnp.isfinite(queries))
    return value_error_if(points, bad, _NOT_FINITE)  # same type as SciPy's check


def _prep(
    points: Array, k: int, queries: Array | None, /, *, dtype: object = None
) -> tuple[Array, Array | None, Array]:
    """Validate, cast and rescale inputs: ``(points, queries, scale)``.

    Used by ``JaxKD`` (``jaxknn`` validates its own inputs). ``dtype=None``
    casts with `_as_float`. Returns coordinates already divided by
    `jaxknn.pow2_scale`'s ``scale``.
    """
    _check_args(points, k, queries)

    def cast(x: Array) -> Array:
        return _as_float(x) if dtype is None else jnp.asarray(x).astype(dtype)

    q = None if queries is None else cast(queries)
    points = _check_finite(cast(points), q)
    scale = jaxknn.pow2_scale(points, q)
    return points / scale, None if q is None else q / scale, scale


def _no_neighbors(m: int, k: int, dtype: object, /) -> KnnOut:
    """Return the empty-search result: index 0, distance inf, for every slot."""
    return jnp.zeros((m, k), jnp.int32), jnp.full((m, k), jnp.inf, dtype)


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


class BucketKDTree(AbstractNeighborSearch):
    """JAX-native exact kd-tree (``jaxknn.knn``). Traceable under jit/vmap/grad.

    ``MSTOrderer``'s default when its inputs are traced. ``SciPy`` is
    eager-only, so under jit/vmap/grad this (or ``BruteForce``/``JaxKD``) is
    the choice; eagerly, ``SciPy`` is faster and compiles nothing, which is
    why the default uses it there.

    Eager calls pad ``n`` to a size bucket ({2**j, 1.5 * 2**j}) with far rows,
    so differing stream lengths share compiled code (each new bucket compiles
    once: seconds, ~9 s at n = 100k, k = 8); under ``jit`` the caller's shapes
    are used as-is.

    Tuning, which matters only where ``SciPy`` cannot run (traced calls):

    - Large ``k`` (``>= 32``): set ``frontier`` near ``k``; at ``k = 32``,
      ``frontier=32`` measured 3-15% faster than the default 16.
    - High dimension (``d >= 6``): every query overflows the default frontier,
      and the tree is ~30-50x slower than ``SciPy``; ``leaf_size=32`` measured
      ~1.45x faster. Eagerly, prefer ``SciPy``.
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
        return jaxknn.knn(
            points,
            k,
            queries=queries,
            leaf_size=self.leaf_size,
            frontier=self.frontier,
        )


class BruteForce(AbstractNeighborSearch):
    """Exact brute force (``jaxknn.brute_knn``), ``chunk`` queries at a time.

    Traceable.
    """

    chunk: int = eqx.field(static=True, default=1024)

    def __check_init__(self) -> None:
        """Reject an invalid chunk size at construction."""
        if self.chunk < 1:
            msg = f"chunk must be >= 1, got {self.chunk}."
            raise ValueError(msg)

    def knn(self, points: Array, /, k: int, *, queries: Array | None = None) -> KnnOut:
        return jaxknn.brute_knn(points, k, queries=queries, chunk=self.chunk)


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
        # jaxkd fails on float32 under x64 ("cond branches must have equal
        # output types"), so use the default float dtype.
        fdt = jnp.promote_types(jnp.float32, jnp.result_type(float))
        points, q, scale = _prep(points, k, queries, dtype=fdt)
        n = points.shape[0]
        if n == 0:  # jaxkd cannot build an empty tree
            return _no_neighbors(0 if q is None else q.shape[0], k, points.dtype)
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
    "neighbors.SciPy cannot run inside jax.jit/vmap/grad (it is host code): "
    "its inputs are traced there, including arrays captured by a jitted "
    "function. Use neighbors.BucketKDTree() to trace."
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
            return _no_neighbors(0 if q is None else q.shape[0], k, p.dtype)
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
