"""Tests for the kNN backends (``phasecurvefit.neighbors``)."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src import neighbors as nb_src
from phasecurvefit._src.optional_deps import OptDeps

_NO_JAXKD = pytest.mark.skipif(
    not OptDeps.JAXKD.installed, reason="jaxkd not installed"
)
# Lazy factories: JaxKD() raises at construction when jaxkd is missing.
_FACTORIES = {
    "bucket": pcf.neighbors.BucketKDTree,
    "brute": pcf.neighbors.BruteForce,
    "jaxkd": pcf.neighbors.JaxKD,
    "scipy": pcf.neighbors.SciPy,
}
PARAMS = [
    pytest.param(f, id=i, marks=[_NO_JAXKD] if i == "jaxkd" else [])
    for i, f in _FACTORIES.items()
]


@pytest.fixture(params=PARAMS)
def backend(request):
    """Each backend, constructed lazily."""
    return request.param()


@pytest.fixture(params=PARAMS[:3])
def jax_backend(request):
    """Yield each backend that traces under jit."""
    return request.param()


def _ref(points, k, queries=None):
    q = points if queries is None else queries
    d2 = ((q[:, None].astype(np.float64) - points[None]) ** 2).sum(-1)
    if queries is None:
        np.fill_diagonal(d2, np.inf)
    if points.shape[0] < k:
        d2 = np.concat([d2, np.full((len(q), k - points.shape[0]), np.inf)], axis=1)
    return np.sqrt(np.sort(d2, 1)[:, :k])


def _assert_indices_give_distances(points, queries, idx, dist):
    """Each finite neighbour index reproduces its reported distance."""
    q = points if queries is None else queries
    fin = np.isfinite(dist)
    rows = np.nonzero(fin)[0]
    got = np.linalg.norm(q[rows].astype(np.float64) - points[idx[fin]], axis=1)
    np.testing.assert_allclose(got, dist[fin], rtol=1e-5, atol=1e-6)


class TestContract:
    """Every backend honours the same contract."""

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 11, 300])  # 5: padded, n < k
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
        _assert_indices_give_distances(p, None, idx, dist)

    @pytest.mark.parametrize("k", [5, 40])
    def test_duplicates_excluded_by_index(self, backend, k):
        """Coincident points are neighbours at distance 0; self never is."""
        p = np.random.default_rng(3).normal(size=(80, 3)).astype(np.float32)
        p[10:40] = p[0]
        idx, dist = map(np.asarray, backend.knn(jnp.asarray(p), k))
        assert not np.any(idx == np.arange(80)[:, None])
        assert np.sum(dist[0] == 0) == min(k, 30)
        _assert_indices_give_distances(p, None, idx, dist)

    def test_queries(self, backend):
        """Bichromatic queries, k=1 and k=10."""
        rng = np.random.default_rng(7)
        p = rng.normal(size=(300, 3)).astype(np.float32)
        q = rng.normal(size=(40, 3)).astype(np.float32)
        for k in (1, 10):
            idx, dist = backend.knn(jnp.asarray(p), k, queries=jnp.asarray(q))
            np.testing.assert_allclose(
                np.asarray(dist), _ref(p, k, q), rtol=1e-5, atol=1e-6
            )
            _assert_indices_give_distances(p, q, np.asarray(idx), np.asarray(dist))

    def test_far_queries(self, backend):
        """Queries farther from the data than its diameter still find real points."""
        p = np.random.default_rng(11).normal(size=(100, 3)).astype(np.float32)
        q = np.array([[50, 0, 0], [0, -40, 0]], np.float32)
        idx, dist = backend.knn(jnp.asarray(p), 3, queries=jnp.asarray(q))
        np.testing.assert_allclose(
            np.asarray(dist), _ref(p, 3, q), rtol=1e-5, atol=1e-6
        )
        assert not np.any(np.asarray(idx) == 100)

    @pytest.mark.parametrize("dtype", [jnp.int32, jnp.bfloat16], ids=["int32", "bf16"])
    def test_integer_and_low_precision_inputs(self, backend, dtype):
        """Integers / bfloat16 behave as their float32 values (no inf rows)."""
        rng = np.random.default_rng(5)
        p = jnp.asarray(rng.integers(-50, 50, size=(120, 3))).astype(dtype)
        q = jnp.asarray(rng.integers(-60, 60, size=(15, 3))).astype(dtype)
        p32, q32 = p.astype(jnp.float32), q.astype(jnp.float32)
        for qq, qq32 in ((None, None), (q, q32)):
            got_i, got_d = backend.knn(p, 4, queries=qq)
            _, want_d = backend.knn(p32, 4, queries=qq32)
            assert jnp.issubdtype(got_d.dtype, jnp.floating)
            assert np.all(np.isfinite(np.asarray(got_d)))
            np.testing.assert_allclose(np.asarray(got_d), np.asarray(want_d), rtol=1e-5)
            assert np.all(np.asarray(got_i) < 120)

    def test_signature(self, backend):
        """``points`` is positional-only; ``k`` may be bound by keyword (for jit)."""
        p = jnp.asarray(np.random.default_rng(1).normal(size=(20, 2)), jnp.float32)
        idx, _ = backend.knn(p, k=3)
        assert idx.shape == (20, 3)
        with pytest.raises(TypeError, match="positional"):
            backend.knn(points=p, k=3)

    def test_float64(self, backend):
        """Under x64, float64 stays float64 and nothing scatters int64 into int32."""
        p = np.random.default_rng(9).normal(size=(200, 3))
        q = p[:20] * 3
        with jax.enable_x64(new_val=True):
            for qq in (None, jnp.asarray(q)):
                _, dist = backend.knn(jnp.asarray(p), 4, queries=qq)
                assert dist.dtype == jnp.float64
                ref = _ref(p, 4, None if qq is None else q)
                np.testing.assert_allclose(np.asarray(dist), ref, rtol=1e-12)

    def test_empty_queries(self, backend):
        """Zero queries give (0, k) outputs."""
        p = jnp.asarray(np.random.default_rng(0).normal(size=(30, 3)), jnp.float32)
        idx, dist = backend.knn(p, 4, queries=jnp.zeros((0, 3), jnp.float32))
        assert idx.shape == dist.shape == (0, 4)

    def test_empty_points(self, backend):
        """Zero points: every query gets sentinel index 0 (= n) at distance inf."""
        q = jnp.asarray(np.random.default_rng(0).normal(size=(5, 3)), jnp.float32)
        idx, dist = backend.knn(jnp.zeros((0, 3), jnp.float32), 4, queries=q)
        assert idx.shape == dist.shape == (5, 4)
        assert np.all(np.asarray(idx) == 0)
        assert np.all(np.isinf(np.asarray(dist)))

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            ((np.ones((5, 2)), 0), {}),
            ((np.ones((5, 2)), -1), {}),
            ((np.ones((5, 2)), 2.0), {}),
            ((np.ones((5, 2)), True), {}),
            ((np.ones(5), 2), {}),
            ((np.ones((5, 2)), 2), {"queries": np.ones((3, 3))}),
        ],
        ids=["k0", "k-1", "kfloat", "kbool", "1d", "query-dim"],
    )
    def test_invalid_args_raise(self, backend, args, kwargs):
        """Bad k or shapes raise the same ValueError from every backend."""
        with pytest.raises(ValueError, match="must"):
            backend.knn(*args, **kwargs)

    def test_non_finite_queries_raise(self, backend):
        """NaN in queries is caught too, as ValueError when eager."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        q = p[:4].copy()
        q[2, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            backend.knn(jnp.asarray(p), 3, queries=jnp.asarray(q))

    @pytest.mark.parametrize("scale", [1e19, 3e19, 1e30])
    def test_huge_coordinates_float32(self, backend, scale):
        """Squared distances that overflow float32 still give real neighbours.

        1 kpc is ~3.1e19 m: unscaled, these d2 overflow to inf and came back as
        sentinels. Coordinates are scaled by a power of two (exact) first.
        """
        p = (np.random.default_rng(4).normal(size=(30, 3)) * scale).astype(np.float32)
        idx, dist = map(np.asarray, backend.knn(jnp.asarray(p), 2))
        assert np.all(idx < 30)
        np.testing.assert_allclose(dist, _ref(p, 2), rtol=1e-5)

    @pytest.mark.parametrize("scale", [1e-14, 1e-30])
    def test_tiny_coordinates_float32(self, backend, scale):
        """Tiny coordinates keep their neighbours (no squared-distance underflow)."""
        p = (np.random.default_rng(6).normal(size=(30, 3)) * scale).astype(np.float32)
        idx, dist = map(np.asarray, backend.knn(jnp.asarray(p), 2))
        np.testing.assert_allclose(dist, _ref(p, 2), rtol=1e-5)
        _assert_indices_give_distances(p, None, idx, dist)

    def test_near_float32_max_coordinates(self, backend):
        """Coordinates near float32's maximum do not overflow the scale factor."""
        p = np.array([[3e38], [1e38], [2e38]], np.float32)
        idx, dist = map(np.asarray, backend.knn(jnp.asarray(p), 1))
        np.testing.assert_array_equal(idx[:, 0], [2, 2, 1])
        assert np.all(np.isfinite(dist))

    def test_tight_cluster_beside_distant_outlier(self, backend):
        """A 1e-15 cluster beside a 1e15 outlier keeps its own neighbours.

        Normalising every coordinate below 1 underflowed the cluster's squared
        distances to 0; in range, coordinates are left unscaled.
        """
        rng = np.random.default_rng(7)
        p = np.concat([rng.normal(size=(50, 2)) * 1e-15, [[1e15, 0.0]]])
        p = p.astype(np.float32)
        idx, dist = map(np.asarray, backend.knn(jnp.asarray(p), 3))
        np.testing.assert_allclose(dist, _ref(p, 3), rtol=1e-5)
        _assert_indices_give_distances(p, None, idx, dist)

    def test_huge_queries_beside_small_points(self, backend):
        """Scaling covers the queries too: huge queries against unit points."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        q = np.array([[3e19, 0, 0], [0, -1e25, 0]], np.float32)
        idx, dist = map(
            np.asarray, backend.knn(jnp.asarray(p), 2, queries=jnp.asarray(q))
        )
        assert np.all(idx < 50)
        np.testing.assert_allclose(dist, _ref(p, 2, q), rtol=1e-5)

    def test_non_finite_with_empty_input_raises(self, backend):
        """NaN is caught even when the other input is empty."""
        nan = jnp.full((2, 3), jnp.nan, jnp.float32)
        empty = jnp.zeros((0, 3), jnp.float32)
        for pts, qs in ((empty, nan), (nan, empty)):
            with pytest.raises(ValueError, match="finite"):
                backend.knn(pts, 2, queries=qs)

    def test_non_finite_raises(self, backend):
        """Review Focus 2: NaN input is an error, not a silently wrong answer."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        p[3, 1] = np.nan
        with pytest.raises(Exception, match="finite"):
            jax.block_until_ready(backend.knn(jnp.asarray(p), 5))


class TestTracing:
    """Which backends trace."""

    def test_jax_backends_trace(self, jax_backend):
        """BucketKDTree, BruteForce and JaxKD run under jit."""
        p = jnp.asarray(np.random.default_rng(1).normal(size=(200, 3)), jnp.float32)
        idx, _ = jax.jit(lambda x: jax_backend.knn(x, 5))(p)
        np.testing.assert_array_equal(
            np.asarray(idx), np.asarray(jax_backend.knn(p, 5)[0])
        )

    @pytest.mark.parametrize("k", [3, 8])
    def test_ties_independent_of_jit_and_padding(self, k):
        """On a grid (many equidistant neighbours) eager, jit and brute agree."""
        g = np.stack(np.meshgrid(np.arange(7.0), np.arange(5.0)), -1).reshape(-1, 2)
        g = jnp.asarray(g, jnp.float32)
        eager = pcf.neighbors.BucketKDTree().knn(g, k)[0]
        jitted = jax.jit(lambda x: pcf.neighbors.BucketKDTree().knn(x, k))(g)[0]
        brute = pcf.neighbors.BruteForce().knn(g, k)[0]
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(brute))

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(pcf.neighbors.BucketKDTree, id="bucket"),
            pytest.param(pcf.neighbors.JaxKD, id="jaxkd", marks=[_NO_JAXKD]),
        ],
    )
    def test_gradient_matches_brute_force(self, make):
        """Distance gradients equal brute force's (tie-free data)."""
        p = jnp.asarray(np.random.default_rng(6).normal(size=(64, 3)), jnp.float32)

        def loss(backend):
            return jax.grad(lambda x: jnp.sum(backend.knn(x, 4)[1] ** 1.5))(p)

        np.testing.assert_allclose(
            np.asarray(loss(make())),
            np.asarray(loss(pcf.neighbors.BruteForce())),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_non_finite_raises_under_jit(self, jax_backend):
        """Traced NaN input is a runtime error too, not a silently wrong answer."""
        p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        p[3, 1] = np.nan
        with pytest.raises(Exception, match="finite"):
            jax.block_until_ready(jax.jit(lambda x: jax_backend.knn(x, 5))(p))

    @pytest.mark.parametrize("queries", [False, True])
    def test_vmap_matches_loop(self, jax_backend, queries):
        """Vmap over a batch of clouds equals separate eager calls."""
        rng = np.random.default_rng(8)
        pb = jnp.asarray(rng.normal(size=(3, 120, 3)), jnp.float32)
        qb = jnp.asarray(rng.normal(size=(3, 9, 3)), jnp.float32)
        if queries:
            got = jax.vmap(lambda a, b: jax_backend.knn(a, 4, queries=b))(pb, qb)
        else:
            got = jax.vmap(lambda a: jax_backend.knn(a, 4))(pb)
        for i in range(3):
            want = jax_backend.knn(pb[i], 4, queries=qb[i] if queries else None)
            np.testing.assert_array_equal(np.asarray(got[0][i]), np.asarray(want[0]))
            np.testing.assert_allclose(
                np.asarray(got[1][i]), np.asarray(want[1]), rtol=1e-6
            )

    def test_scipy_raises_when_traced(self):
        """The scipy backend is eager-only."""
        p = jnp.ones((10, 3))
        with pytest.raises(TypeError, match="BucketKDTree"):
            jax.jit(lambda x: pcf.neighbors.SciPy().knn(x, 3))(p)

    def test_gradient_finite_at_coincident_points(self, jax_backend):
        """Distances differentiate with neighbour selection fixed; no NaN at d=0."""
        p = np.random.default_rng(2).normal(size=(64, 3)).astype(np.float32)
        p[1] = p[0]

        def loss(x):
            return jnp.sum(jax_backend.knn(x, 4)[1])

        g = np.asarray(jax.grad(loss)(jnp.asarray(p)))
        assert np.all(np.isfinite(g))
        assert np.any(g != 0)


class TestBucketing:
    """Eager size buckets for BucketKDTree."""

    def test_bucket_sizes(self):
        """Buckets are 2**j or 1.5 * 2**j, never smaller than n."""
        sizes = [nb_src._bucket(n) for n in [1, 2, 3, 5, 7, 100, 1000]]
        assert sizes == [1, 2, 3, 6, 8, 128, 1024]

    def test_same_bucket_reuses_compilation(self, monkeypatch):
        """Review Focus 1: n in an already-compiled bucket does not recompile."""
        traces = []
        all_knn = nb_src._kd.all_knn

        def counting(*args, **kwargs):  # runs only while jit traces
            traces.append(args[0].shape)
            return all_knn(*args, **kwargs)

        monkeypatch.setattr(nb_src._kd, "all_knn", counting)
        # An unusual leaf_size, and a cleared cache, so the first call traces.
        nb_src._knn_core_jit.clear_cache()
        backend = pcf.neighbors.BucketKDTree(leaf_size=13)
        rng = np.random.default_rng(3)
        backend.knn(jnp.asarray(rng.normal(size=(1000, 3)), jnp.float32), 10)
        backend.knn(jnp.asarray(rng.normal(size=(990, 3)), jnp.float32), 10)
        assert traces == [(1024, 3)]

    def test_far_rows_never_win(self):
        """Padding rows are farther than the data's diameter from every point."""
        p = jnp.asarray(np.random.default_rng(4).normal(size=(100, 3)), jnp.float32)
        far = np.asarray(nb_src.far_rows(p, 7))
        diam = np.max(
            np.linalg.norm(np.asarray(p)[:, None] - np.asarray(p)[None], axis=-1)
        )
        dmin = np.min(np.linalg.norm(np.asarray(p)[:, None] - far[None], axis=-1))
        assert dmin > diam

    @pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
    def test_far_rows_stay_near_the_data_scale(self, scale):
        """The padding row's magnitude is a constant multiple of the data's.

        An absolute ``+1`` offset (or one growing with the row count) inflated
        tiny data's range, and the coordinate scaling then underflowed it.
        """
        p = np.random.default_rng(5).normal(size=(100, 3)) * scale
        far = np.asarray(nb_src.far_rows(jnp.asarray(p, jnp.float32), 500))
        assert np.abs(far).max() <= (4 * np.sqrt(3) + 3) * np.abs(p).max() * 1.01

    def test_far_rows_beyond_the_data_at_large_magnitude(self):
        """|x| >> spread in float32: the far row still lies beyond the data."""
        p = np.c_[np.full(41, 1e9), np.linspace(0, 0.5, 41)].astype(np.float32)
        far = np.asarray(nb_src.far_rows(jnp.asarray(p), 7), np.float64)
        gap = np.linalg.norm(p.astype(np.float64)[:, None] - far[None], axis=-1)
        assert gap.min() > 0.5  # farther than the data's diameter

    def test_far_rows_large_magnitude_float32(self):
        """|x| >> spread in float32: far rows must not round onto real points."""
        p = np.c_[np.full(41, 1e9), np.linspace(0, 0.5, 41)].astype(np.float32)
        idx, dist = pcf.neighbors.BucketKDTree().knn(jnp.asarray(p), 3)
        assert np.all(np.asarray(idx) < 41)
        np.testing.assert_allclose(np.asarray(dist), _ref(p, 3), rtol=1e-5)


def test_scipy_empty_points_does_not_build_a_tree(monkeypatch):
    """SciPy's n == 0 result comes from the backend, not from cKDTree."""
    import scipy.spatial  # noqa: PLC0415

    def boom(*_args, **_kwargs):
        msg = "cKDTree built for empty points"
        raise AssertionError(msg)

    monkeypatch.setattr(scipy.spatial, "cKDTree", boom)
    q = jnp.ones((3, 2), jnp.float32)
    idx, dist = pcf.neighbors.SciPy().knn(jnp.zeros((0, 2)), 4, queries=q)
    assert np.all(np.asarray(idx) == 0)
    assert np.all(np.isinf(np.asarray(dist)))


class TestConstruction:
    """Invalid backend settings fail at construction."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: pcf.neighbors.BucketKDTree(leaf_size=0),
            lambda: pcf.neighbors.BucketKDTree(frontier=0),
            lambda: pcf.neighbors.BruteForce(chunk=0),
        ],
        ids=["leaf_size", "frontier", "chunk"],
    )
    def test_bad_sizes(self, make):
        """Sizes below 1 raise ValueError."""
        with pytest.raises(ValueError, match=">= 1"):
            make()

    def test_jaxkd_missing(self, monkeypatch):
        """JaxKD() without jaxkd installed raises ImportError with a hint."""
        missing = SimpleNamespace(JAXKD=SimpleNamespace(installed=False))
        monkeypatch.setattr(nb_src, "OptDeps", missing)
        with pytest.raises(ImportError, match="kdtree"):
            pcf.neighbors.JaxKD()

    @_NO_JAXKD
    def test_jaxkd_float32_under_x64(self):
        """Jaxkd itself fails on float32 under x64; the wrapper upcasts."""
        p = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)
        with jax.enable_x64(new_val=True):
            _, dist = pcf.neighbors.JaxKD().knn(jnp.asarray(p), 3)
        np.testing.assert_allclose(np.asarray(dist), _ref(p, 3), rtol=1e-5)
