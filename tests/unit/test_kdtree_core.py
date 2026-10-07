"""Tests for the self-contained JAX kd-tree (``phasecurvefit._src.kdtree``)."""

import functools
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasecurvefit._src import kdtree as kd
from phasecurvefit._src.kdtree._build import build_tree
from phasecurvefit._src.kdtree._layout import layout
from phasecurvefit._src.kdtree._select import ksmallest, kth_smallest


class TestLayout:
    """Static tree layout."""

    @pytest.mark.parametrize("n", [0, 1, 2, 15, 16, 17, 100, 1000, 100_000])
    @pytest.mark.parametrize("leaf_size", [8, 16, 32])
    def test_padding_under_one_row_per_leaf(self, n, leaf_size):
        """n_pad covers n, padding is < one row per leaf, spread as 0/1 per leaf."""
        lay = layout(n, leaf_size)
        assert lay.n_leaves == 2**lay.depth
        assert lay.n_pad == lay.n_leaves * lay.leaf_size >= n
        assert lay.n_pad - n < lay.n_leaves or n == 0
        assert int(lay.leaf_valid.sum()) == max(n, 0) or n == 0
        assert set(np.unique(lay.leaf_size - lay.leaf_valid)) <= {0, 1}
        assert lay.leaf_size <= max(leaf_size, 1) or lay.depth == 0


class TestSelect:
    """Exact k-smallest selection."""

    @pytest.mark.parametrize("width", [3, 13, 52, 208])
    def test_matches_sort_with_ties_and_infs(self, width):
        """Values equal np.sort; columns reproduce them, distinct, stable on ties."""
        k = 10
        rng = np.random.default_rng(width)
        a = rng.random((500, width)).astype(np.float32)
        a[:, 1 % width] = 0.5
        a[:, 2 % width] = 0.5  # ties
        a[:7] = np.inf
        a[:7, 0] = 0.1  # rows with fewer than k finite entries
        vals, cols = jax.jit(functools.partial(ksmallest, k=k))(jnp.asarray(a))
        vals, cols = np.asarray(vals), np.asarray(cols)
        padded = np.concatenate(
            [a, np.full((500, max(0, k - width)), np.inf, np.float32)], 1
        )
        ref = np.sort(padded, 1)[:, :k]
        np.testing.assert_array_equal(vals, ref)
        finite = np.isfinite(vals)
        np.testing.assert_array_equal(
            np.take_along_axis(padded, cols, 1)[finite], ref[finite]
        )
        for r in range(7, 500):
            assert len(set(cols[r].tolist())) == k
        kth = np.asarray(kth_smallest(jnp.asarray(a), k))
        np.testing.assert_array_equal(kth, ref[:, -1])


def _stream(n, seed=0, interlopers=0.0):
    """Noisy, shuffled 3-D stream; optionally 1% (etc.) scattered interlopers."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    m = int(interlopers * n)
    if m:
        idx = rng.choice(n, m, replace=False)
        pad = np.array([0.0, 3.0, 3.0])
        p[idx] = rng.uniform(p.min(0) - pad, p.max(0) + pad, (m, 3))
    return p[rng.permutation(n)].astype(np.float32)


class TestBuild:
    """Balanced, pointer-free build."""

    @pytest.mark.parametrize("n", [1, 2, 15, 17, 100, 1001])
    def test_perm_is_a_permutation_and_points_match(self, n):
        """Valid rows hold every input point once; padding is masked with sentinel n."""
        p = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
        tree = jax.jit(build_tree)(jnp.asarray(p))
        perm, valid = np.asarray(tree.perm), np.asarray(tree.valid)
        assert sorted(perm[valid].tolist()) == list(range(n))
        assert np.all(perm[~valid] == n)
        np.testing.assert_array_equal(np.asarray(tree.points)[valid], p[perm[valid]])

    @pytest.mark.parametrize("n", [17, 100, 1001])
    def test_leaf_cells_contain_their_points(self, n):
        """Every valid point lies inside (or on the boundary of) its leaf's cell."""
        p = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
        tree = jax.jit(build_tree)(jnp.asarray(p))
        b = tree.leaf_size
        pts = np.asarray(tree.points).reshape(tree.n_leaves, b, 3)
        ok = np.asarray(tree.valid).reshape(tree.n_leaves, b)
        lo = np.asarray(tree.cell_lo[tree.depth])[:, None]
        hi = np.asarray(tree.cell_hi[tree.depth])[:, None]
        inside = np.all((pts >= lo) & (pts <= hi), -1)
        assert np.all(inside[ok])

    def test_duplicates_and_degenerate_extent(self):
        """Hundreds of identical points still build a valid tree."""
        p = np.zeros((300, 3), np.float32)
        tree = jax.jit(build_tree)(jnp.asarray(p))
        assert sorted(np.asarray(tree.perm)[np.asarray(tree.valid)].tolist()) == list(
            range(300)
        )


def _ref_sq(points, k, queries=None):
    """Float64 brute-force sorted squared distances (self excluded if no queries)."""
    q = points if queries is None else queries
    d2 = ((q[:, None].astype(np.float64) - points[None]) ** 2).sum(-1)
    if queries is None:
        np.fill_diagonal(d2, np.inf)
    if points.shape[0] < k:
        d2 = np.concatenate([d2, np.full((len(q), k - points.shape[0]), np.inf)], 1)
    return np.sort(d2, 1)[:, :k]


def _assert_exact(points, idx, d2, k, queries=None):
    idx, d2 = np.asarray(idx), np.asarray(d2)
    n = len(points)
    ref = _ref_sq(points, k, queries)
    fin = np.isfinite(ref)
    np.testing.assert_array_equal(np.isfinite(d2), fin)
    if fin.any():
        rel = np.abs(d2[fin] - ref[fin]) / np.maximum(ref[fin], 1e-30)
        assert rel.max() < 1e-5
        q = points if queries is None else queries
        rows = np.nonzero(fin)
        realised = ((q[rows[0]] - points[idx[rows]]) ** 2).sum(-1)
        np.testing.assert_allclose(realised, d2[rows], rtol=1e-5, atol=1e-12)
    assert np.all(idx[~fin] == n)
    if queries is None:
        assert not np.any(idx == np.arange(n)[:, None])
    for r in range(min(len(idx), 500)):
        row = idx[r][fin[r]]
        assert len(set(row.tolist())) == len(row)


def _all(points, k=10, **kw):
    return jax.jit(functools.partial(kd.all_knn, k=k, **kw))(jnp.asarray(points))


def _bi(points, queries, k=10, leaf_size=16, **kw):
    f = jax.jit(lambda p, q: kd.knn(kd.build_tree(p, leaf_size=leaf_size), q, k, **kw))
    return f(jnp.asarray(points), jnp.asarray(queries))


SIZES = [0, 1, 2, 9, 10, 11, 15, 16, 17, 31, 33, 100, 257, 1000]
KNOBS = [{}, {"leaf_size": 8}, {"frontier": 2}]  # frontier=2 forces every overflow tier


class TestAllKnn:
    """Exact all-points kNN."""

    @pytest.mark.parametrize("kw", KNOBS, ids=["default", "leaf8", "frontier2"])
    @pytest.mark.parametrize("n", SIZES)
    def test_edge_sizes(self, n, kw):
        """Every n, including n <= k (sentinel rows), matches brute force."""
        p = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
        _assert_exact(p, *_all(p, **kw), 10)

    @pytest.mark.parametrize("frontier", [16, 2])
    def test_duplicates(self, frontier):
        """More than a leaf of identical points, including the last row."""
        p = np.random.default_rng(1).normal(size=(500, 3)).astype(np.float32)
        p[100:200] = p[0]
        p[-1] = p[0]
        _assert_exact(p, *_all(p, frontier=frontier), 10)

    def test_all_identical(self):
        """A single coincident clump."""
        p = np.zeros((300, 3), np.float32)
        _assert_exact(p, *_all(p), 10)

    def test_far_last_row(self):
        """Row n-1 is never corrupted by a -1 sentinel (regression)."""
        p = np.random.default_rng(2).normal(size=(1001, 3)).astype(np.float32)
        p[-1] = 100.0
        _assert_exact(p, *_all(p), 10)

    @pytest.mark.parametrize("d", [1, 2, 4])
    def test_dimensions(self, d):
        """Dimensions other than 3."""
        p = np.random.default_rng(d).normal(size=(700, d)).astype(np.float32)
        _assert_exact(p, *_all(p), 10)

    @pytest.mark.parametrize("frac", [0.0, 0.01], ids=["stream", "interlopers"])
    def test_stream_10k(self, frac):
        """A realistic stream, with and without interlopers."""
        p = _stream(10_000, interlopers=frac)
        _assert_exact(p, *_all(p), 10)

    def test_vmap_mixed_batch_exact(self):
        """Review Focus 4: vmap over clouds that differ stays exact for each."""
        batch = np.stack([_stream(2000, 0), _stream(2000, 1, interlopers=0.05)])
        idx, d2 = jax.jit(jax.vmap(functools.partial(kd.all_knn, k=10)))(
            jnp.asarray(batch)
        )
        for b in range(2):
            _assert_exact(batch[b], idx[b], d2[b], 10)


class TestKnn:
    """Exact bichromatic kNN (queries are not tree points)."""

    @pytest.mark.parametrize("kw", KNOBS, ids=["default", "leaf8", "frontier2"])
    @pytest.mark.parametrize("n", [1, 2, 9, 11, 17, 100, 1000])
    @pytest.mark.parametrize("k", [1, 10])
    def test_edge_sizes(self, n, k, kw):
        """Random queries against random trees, k=1 (projection) and k=10."""
        rng = np.random.default_rng(n + k)
        p = rng.normal(size=(n, 3)).astype(np.float32)
        q = rng.normal(size=(37, 3)).astype(np.float32)
        _assert_exact(p, *_bi(p, q, k=k, **kw), k, queries=q)

    def test_stream_projection_10k(self):
        """k=1 queries near a stream, as the MST projection uses it."""
        p = _stream(10_000)
        q = p[::7] + 0.01
        _assert_exact(p, *_bi(p, q, k=1), 1, queries=q)

    def test_locate_leaves_finds_the_containing_cell(self):
        """Each query is inside the cell of the leaf it is located in."""
        p = _stream(3000)
        tree = jax.jit(kd.build_tree)(jnp.asarray(p))
        q = np.random.default_rng(3).normal(size=(200, 3)).astype(np.float32) * 5
        leaf = np.asarray(jax.jit(kd.locate_leaves)(tree, jnp.asarray(q)))
        lo = np.asarray(tree.cell_lo[tree.depth])[leaf]
        hi = np.asarray(tree.cell_hi[tree.depth])[leaf]
        assert np.all((q >= lo) & (q <= hi))


class TestBrute:
    """The brute-force oracle itself."""

    def test_matches_reference(self):
        """All-points and bichromatic brute force match float64 NumPy."""
        rng = np.random.default_rng(4)
        p = rng.normal(size=(700, 3)).astype(np.float32)
        q = rng.normal(size=(50, 3)).astype(np.float32)
        _assert_exact(
            p, *jax.jit(functools.partial(kd.brute_knn, k=10))(jnp.asarray(p)), 10
        )
        _assert_exact(
            p, *kd.brute_knn(jnp.asarray(p), 10, queries=jnp.asarray(q)), 10, queries=q
        )


class TestBruteTier:
    """The final brute-force overflow tier, forced by tiny tier constants."""

    @pytest.mark.parametrize("frontier", [1, 2])
    def test_forced_tiers_exact(self, monkeypatch, frontier):
        """All-points and bichromatic (incl. far) queries stay exact through brute."""
        q_mod = importlib.import_module("phasecurvefit._src.kdtree._query")
        monkeypatch.setattr(q_mod, "TIERS", (2, 4))
        monkeypatch.setattr(q_mod, "TIER_CHUNK", (3, 5))
        monkeypatch.setattr(q_mod, "BRUTE_CHUNK", 7)
        chunks = []
        real = q_mod._finish

        def spy(*args):
            chunks.append(args[6])  # the chunk size of each overflow stage
            return real(*args)

        monkeypatch.setattr(q_mod, "_finish", spy)
        jax.clear_caches()  # constants are read at trace time
        p = _stream(300, seed=3, interlopers=0.1)
        rng = np.random.default_rng(1)
        q = np.concatenate(
            [p[:20] + 0.01, 100 * rng.normal(size=(10, 3)).astype(np.float32)]
        ).astype(np.float32)
        _assert_exact(p, *_all(p, k=8, leaf_size=4, frontier=frontier), 8)
        _assert_exact(p, *_bi(p, q, k=8, leaf_size=4, frontier=frontier), 8, queries=q)
        assert 7 in chunks  # brute tier was traced
        assert 5 in chunks  # ... after an overflow tier
        jax.clear_caches()
