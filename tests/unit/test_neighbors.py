"""Tests for the kNN backends (``phasecurvefit.neighbors``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src import neighbors as nb_src

BACKENDS = [
    pcf.neighbors.BucketKDTree(),
    pcf.neighbors.BruteForce(),
    pcf.neighbors.Jaxkd(),
    pcf.neighbors.Scipy(),
]
IDS = ["bucket", "brute", "jaxkd", "scipy"]


def _ref(points, k, queries=None):
    q = points if queries is None else queries
    d2 = ((q[:, None].astype(np.float64) - points[None]) ** 2).sum(-1)
    if queries is None:
        np.fill_diagonal(d2, np.inf)
    if points.shape[0] < k:
        d2 = np.concatenate([d2, np.full((len(q), k - points.shape[0]), np.inf)], 1)
    return np.sqrt(np.sort(d2, 1)[:, :k])


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
class TestContract:
    """Every backend honours the same contract."""

    @pytest.mark.parametrize("n", [1, 2, 11, 300])
    def test_all_points(self, backend, n):
        """Euclidean distances, sorted, self excluded, sentinel n when short."""
        p = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
        idx, dist = map(np.asarray, backend.knn(jnp.asarray(p), 10))
        ref = _ref(p, 10)
        np.testing.assert_array_equal(np.isfinite(dist), np.isfinite(ref))
        fin = np.isfinite(ref)
        np.testing.assert_allclose(dist[fin], ref[fin], rtol=1e-5, atol=1e-6)
        assert np.all(idx[~fin] == n)
        assert not np.any(idx == np.arange(n)[:, None])

    def test_queries(self, backend):
        """Bichromatic queries, k=1 and k=10."""
        rng = np.random.default_rng(7)
        p = rng.normal(size=(300, 3)).astype(np.float32)
        q = rng.normal(size=(40, 3)).astype(np.float32)
        for k in (1, 10):
            _, dist = backend.knn(jnp.asarray(p), k, queries=jnp.asarray(q))
            np.testing.assert_allclose(
                np.asarray(dist), _ref(p, k, q), rtol=1e-5, atol=1e-6
            )

    def test_non_finite_raises(self, backend):
        """Review Focus 2: NaN input is an error, not a silently wrong answer."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        p[3, 1] = np.nan
        with pytest.raises(Exception, match="finite"):
            jax.block_until_ready(backend.knn(jnp.asarray(p), 5))


class TestTracing:
    """Which backends trace."""

    @pytest.mark.parametrize("backend", BACKENDS[:3], ids=IDS[:3])
    def test_jax_backends_trace(self, backend):
        """BucketKDTree, BruteForce and Jaxkd run under jit."""
        p = jnp.asarray(np.random.default_rng(1).normal(size=(200, 3)), jnp.float32)
        idx, _ = jax.jit(lambda x: backend.knn(x, 5))(p)
        np.testing.assert_array_equal(np.asarray(idx), np.asarray(backend.knn(p, 5)[0]))

    def test_scipy_raises_when_traced(self):
        """The scipy backend is eager-only."""
        p = jnp.ones((10, 3))
        with pytest.raises(TypeError, match="BucketKDTree"):
            jax.jit(lambda x: pcf.neighbors.Scipy().knn(x, 3))(p)

    def test_gradient_finite_at_coincident_points(self):
        """Distances differentiate with neighbour selection fixed; no NaN at d=0."""
        p = np.random.default_rng(2).normal(size=(64, 3)).astype(np.float32)
        p[1] = p[0]

        def loss(x):
            return jnp.sum(pcf.neighbors.BucketKDTree().knn(x, 4)[1])

        g = np.asarray(jax.grad(loss)(jnp.asarray(p)))
        assert np.all(np.isfinite(g))
        assert np.any(g != 0)


class TestBucketing:
    """Eager size buckets for BucketKDTree."""

    def test_bucket_sizes(self):
        """Buckets are 2**j or 1.5 * 2**j, never smaller than n."""
        sizes = [nb_src._bucket(n) for n in [1, 2, 3, 5, 7, 100, 1000]]
        assert sizes == [1, 2, 3, 6, 8, 128, 1024]

    def test_same_bucket_reuses_compilation(self):
        """Review Focus 1: n in an already-compiled bucket does not recompile."""
        backend = pcf.neighbors.BucketKDTree()
        rng = np.random.default_rng(3)
        backend.knn(jnp.asarray(rng.normal(size=(1000, 3)), jnp.float32), 10)
        before = nb_src._knn_core_jit._cache_size()
        backend.knn(jnp.asarray(rng.normal(size=(990, 3)), jnp.float32), 10)
        assert nb_src._knn_core_jit._cache_size() == before

    def test_far_rows_never_win(self):
        """Padding rows are farther than the data's diameter from every point."""
        p = jnp.asarray(np.random.default_rng(4).normal(size=(100, 3)), jnp.float32)
        far = np.asarray(nb_src.far_rows(p, 7))
        diam = np.max(
            np.linalg.norm(np.asarray(p)[:, None] - np.asarray(p)[None], axis=-1)
        )
        dmin = np.min(np.linalg.norm(np.asarray(p)[:, None] - far[None], axis=-1))
        assert dmin > diam
        assert len({tuple(r) for r in far.tolist()}) == 7
