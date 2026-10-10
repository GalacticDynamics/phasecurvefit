"""Tests for the pure-JAX graph package (``phasecurvefit._src.graph``).

Each module is checked against its reference: scipy's csgraph, or the host
helpers in ``orderers/mst.py`` that the ``SciPy()`` path still uses.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from phasecurvefit._src import graph as gr
from phasecurvefit._src.graph._pointer import accumulate, find_roots, list_rank
from phasecurvefit._src.orderers.mst import (
    _connect_components,
    _diameter_path,
    _edge_cosine,
    _sigma_clip_edges,
)

_boruvka = jax.jit(gr.boruvka, static_argnums=0)
_diameter = jax.jit(gr.diameter_path, static_argnums=0)


class TestPointer:
    """Pointer jumping: roots, path sums and list ranking."""

    def test_find_roots_and_accumulate(self):
        """Roots and root-path sums of a random forest equal a direct walk."""
        rng = np.random.default_rng(0)
        n = 300
        parent = np.array(
            [rng.integers(0, i) if i and rng.random() < 0.9 else i for i in range(n)]
        )
        w = np.where(parent == np.arange(n), 0.0, rng.random(n)).astype(np.float32)
        roots = np.asarray(jax.jit(find_roots)(jnp.asarray(parent)))
        r2, dist = map(
            np.asarray, jax.jit(accumulate)(jnp.asarray(parent), jnp.asarray(w))
        )
        for i in range(n):
            j, total = i, 0.0
            while parent[j] != j:
                total += w[j]
                j = parent[j]
            assert roots[i] == r2[i] == j
            assert dist[i] == pytest.approx(total, rel=1e-5)

    def test_list_rank(self):
        """Steps to the end of a list; elements on a cycle are flagged."""
        order = np.random.default_rng(1).permutation(40)
        succ = np.full(50, 50)
        succ[order[:-1]] = order[1:]  # a 40-element list
        succ[40:50] = np.r_[41:50, 40]  # a 10-element cycle
        steps, reached = map(np.asarray, jax.jit(list_rank)(jnp.asarray(succ)))
        np.testing.assert_array_equal(steps[order], np.arange(39, -1, -1))
        np.testing.assert_array_equal(reached, np.arange(50) < 40)


def _random_graph(rng, n, *, ties):
    """Draw an undirected edge list (lo <= hi) with invalid and self edges."""
    e = int(rng.integers(0, 4 * n + 1))
    a, b = rng.integers(0, n, e), rng.integers(0, n, e)
    w = rng.integers(1, 4, e) if ties else rng.random(e) + 0.01
    valid = (a != b) & (rng.random(e) < 0.9)
    return np.minimum(a, b), np.maximum(a, b), w.astype(np.float32), valid


def _scipy_graph(n, lo, hi, w, valid):
    """Sparse graph for scipy; a pair listed twice keeps its smaller weight.

    (``csr_matrix`` would *sum* duplicate entries.)
    """
    best = {}
    for i, j, ww in zip(lo[valid], hi[valid], w[valid], strict=True):
        best[(int(i), int(j))] = min(best.get((int(i), int(j)), np.inf), float(ww))
    rows = [i for i, _ in best]
    cols = [j for _, j in best]
    vals = np.maximum(list(best.values()), 1e-30)
    return csr_matrix((vals, (rows, cols)), shape=(n, n)), best


def _scipy_forest(n, lo, hi, w, valid):
    """Return scipy's MST (symmetric) and its component labels."""
    g, _ = _scipy_graph(n, lo, hi, w, valid)
    t = minimum_spanning_tree(g)
    t = t + t.T
    return t, connected_components(t, directed=False)[1]


def _edge_set(lo, hi, mask):
    return {(int(i), int(j)) for i, j in zip(lo[mask], hi[mask], strict=True)}


class TestBoruvka:
    """Minimum spanning forest and components equal scipy's."""

    @pytest.mark.parametrize("seed", range(30))
    @pytest.mark.parametrize("ties", [False, True])
    def test_matches_scipy(self, seed, ties):
        """Same partition, weight and edge count; the same edges if no ties."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 60))
        lo, hi, w, valid = _random_graph(rng, n, ties=ties)
        tree, labels = map(
            np.asarray, _boruvka(n, *map(jnp.asarray, (lo, hi, w, valid)))
        )
        t, ref = _scipy_forest(n, lo, hi, w, valid)
        for i in range(n):  # label = smallest node of the component
            assert labels[i] == np.flatnonzero(ref == ref[i]).min()
        assert tree.sum() == n - len(np.unique(ref))
        assert float(w[tree].sum()) == pytest.approx(float(t.sum()) / 2, rel=1e-5)
        if not ties:
            tc = t.tocoo()
            want = {
                (int(i), int(j)) for i, j in zip(tc.row, tc.col, strict=True) if i < j
            }
            assert _edge_set(lo, hi, tree) == want

    def test_no_edges(self):
        """No edges: every node is its own component."""
        z = jnp.zeros(0, jnp.int32)
        tree, labels = _boruvka(4, z, z, jnp.zeros(0), jnp.zeros(0, bool))
        assert tree.shape == (0,)
        np.testing.assert_array_equal(np.asarray(labels), np.arange(4))

    def test_duplicate_pair_is_used_once(self):
        """(i, j) listed twice (as kNN lists do) enters the tree once."""
        lo, hi = jnp.array([0, 0, 1]), jnp.array([1, 1, 2])
        w, valid = jnp.array([1.0, 1.0, 2.0]), jnp.ones(3, bool)
        tree, _ = _boruvka(3, lo, hi, w, valid)
        assert int(tree.sum()) == 2

    def test_largest_component(self):
        """Largest real component; ties to the lowest label; padding not counted."""
        labels = jnp.array([0, 0, 2, 2, 4, 5])
        real = jnp.array([True, True, True, True, True, False])
        mask, n_comp = gr.largest_component(labels, real)
        np.testing.assert_array_equal(np.asarray(mask), [1, 1, 0, 0, 0, 0])
        assert int(n_comp) == 3


class TestDiameterPath:
    """The tip-to-tip path equals ``_diameter_path`` on the same forest."""

    @pytest.mark.parametrize("seed", range(30))
    def test_matches_host(self, seed):
        """Including alive sets that split the restricted tree into pieces."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 60))
        lo, hi, w, valid = _random_graph(rng, n, ties=False)
        args = tuple(map(jnp.asarray, (lo, hi, w, valid)))
        tree, _ = _boruvka(n, *args)
        t, ref = _scipy_forest(n, lo, hi, w, valid)
        comp = ref == ref[int(rng.integers(n))]
        alive = comp & (rng.random(n) < (1.0 if seed % 2 else 0.8))
        alive = alive if alive.any() else comp
        full, blen = _diameter(n, *args[:3], tree, jnp.asarray(alive))
        want = _diameter_path(t.tocsr(), np.flatnonzero(alive))
        np.testing.assert_array_equal(np.asarray(full)[: int(blen)], want)
        assert np.all(np.asarray(full)[int(blen) :] == want[-1])  # padded

    def test_single_node(self):
        """One alive node with no edges: the path is that node."""
        z = jnp.zeros(1, jnp.int32)
        full, blen = _diameter(
            3, z, z, jnp.ones(1), jnp.zeros(1, bool), jnp.array([False, True, False])
        )
        assert int(blen) == 1
        np.testing.assert_array_equal(np.asarray(full), [1, 1, 1])


class TestEdges:
    """The kNN edge list follows the host formulas."""

    def test_matches_host_formulas(self):
        """Lengths, weights and keep mask, incl. missing/padded neighbours."""
        rng = np.random.default_rng(3)
        n, k = 50, 4
        p = rng.normal(size=(n, 3)).astype(np.float32)
        v = rng.normal(size=(n, 3)).astype(np.float32)
        v[:5] = 0.0  # zero velocity: cosine defined as 0
        nbr = rng.integers(0, n + 1, (n, k))  # n = missing
        real = np.arange(n) < 45
        f = jax.jit(
            functools.partial(
                gr.knn_edges,
                jump_cap=2.0,
                velocity_weight=0.7,
                sever_cos_threshold=-0.2,
            )
        )
        lo, hi, d, w, valid = map(np.asarray, f(*map(jnp.asarray, (p, v, nbr, real))))
        rows, cols = np.repeat(np.arange(n), k), nbr.ravel()
        ok = cols < n
        c = np.minimum(cols, n - 1)
        d_ref = np.linalg.norm(p[rows] - p[c], axis=1)
        cos = _edge_cosine(v.astype(np.float64), rows, c)
        np.testing.assert_allclose(d, d_ref, rtol=1e-6)
        # atol: a self-neighbour has d = 0, cos = 1 -> w ~ 0 (float32 rounding)
        np.testing.assert_allclose(
            w, np.maximum(d_ref + 0.7 * (1 - cos), 1e-30), rtol=1e-5, atol=1e-6
        )
        keep = ok & real[rows] & real[c] & (d_ref <= 2.0) & (cos >= -0.2)
        np.testing.assert_array_equal(valid, keep)
        np.testing.assert_array_equal(lo, np.minimum(rows, c))
        np.testing.assert_array_equal(hi, np.maximum(rows, c))

    def test_orient_flip(self):
        """Flip iff the path runs against the velocity; NaN velocity is ignored."""
        p = jnp.stack([jnp.arange(5.0), jnp.zeros(5)], 1)
        v = jnp.stack([-jnp.ones(5), jnp.zeros(5)], 1).at[2, 0].set(jnp.nan)
        full, blen = jnp.arange(5), jnp.asarray(5)
        assert bool(gr.orient_flip(p, v, full, blen))
        assert not bool(gr.orient_flip(p, -v, full, blen))


def _stream(n, seed, interlopers):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    p = np.c_[10 * t, np.sin(3 * t), 0.3 * np.cos(2 * t)] + rng.normal(0, 0.02, (n, 3))
    m = int(interlopers * n)
    if m:
        i = rng.choice(n, m, replace=False)
        p[i] = rng.uniform(p.min(0) - 3, p.max(0) + 3, (m, 3))
    return p[rng.permutation(n)].astype(np.float32)


def _knn_graph(p, k):
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argsort(d2, 1)[:, :k]
    rows = np.repeat(np.arange(len(p)), k)
    cols = nbr.ravel()
    d = np.linalg.norm(p[rows] - p[cols], axis=1).astype(np.float32)
    return np.minimum(rows, cols), np.maximum(rows, cols), d


class TestSigmaClip:
    """Clipping keeps the same nodes as ``_sigma_clip_edges``."""

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("sigma", [1.5, 3.0])
    def test_matches_host(self, seed, sigma):
        """Interlopers rejected, the stream kept -- node for node."""
        p = _stream(400, seed, interlopers=0.03)
        n = len(p)
        lo, hi, d = _knn_graph(p, 8)
        valid = np.ones(len(lo), bool)
        args = tuple(map(jnp.asarray, (lo, hi, d, valid)))
        tree, labels = _boruvka(n, *args)
        alive = np.asarray(labels) == np.asarray(labels)[0]
        clip = jax.jit(
            functools.partial(gr.sigma_clip, sigma=sigma, max_iters=5), static_argnums=0
        )
        got = np.asarray(clip(n, args[0], args[1], args[2], tree, jnp.asarray(alive)))
        tm = np.asarray(tree)
        t = csr_matrix((d[tm], (lo[tm], hi[tm])), shape=(n, n))
        want = _sigma_clip_edges(
            t + t.T, p, np.flatnonzero(alive), sigma=sigma, max_iters=5
        )
        np.testing.assert_array_equal(np.flatnonzero(got), want)

    @pytest.mark.parametrize("extra", [0, 7], ids=["pairs", "pairs+7"])
    def test_fragmentation_matches_host(self, extra):
        """Cuts that shatter the stream stop instead of rejecting most of it."""
        gaps = np.tile([1.0, 10.0], 300)
        if extra:
            gaps = np.r_[gaps, np.ones(extra - 1), 10.0]
        x = np.concatenate([[0.0], np.cumsum(gaps)])
        p = np.c_[x, np.zeros_like(x)].astype(np.float32)
        n = len(p)
        lo, hi, d = _knn_graph(p, 4)
        args = tuple(map(jnp.asarray, (lo, hi, d, np.ones(len(lo), bool))))
        tree, labels = _boruvka(n, *args)
        alive = np.asarray(labels) == np.asarray(labels)[0]
        clip = jax.jit(
            functools.partial(gr.sigma_clip, sigma=0.1, max_iters=5),
            static_argnums=0,
        )
        got = np.asarray(clip(n, args[0], args[1], args[2], tree, jnp.asarray(alive)))
        tm = np.asarray(tree)
        t = csr_matrix((d[tm], (lo[tm], hi[tm])), shape=(n, n))
        want = _sigma_clip_edges(
            t + t.T, p, np.flatnonzero(alive), sigma=0.1, max_iters=5
        )
        np.testing.assert_array_equal(np.flatnonzero(got), want)
        assert got.sum() == n  # nothing rejected


def _brute_nearest(p):
    def nearest(labels):
        d2 = jnp.sum((p[:, None] - p[None]) ** 2, -1)
        d2 = jnp.where(labels[:, None] == labels[None], jnp.inf, d2)
        j = jnp.argmin(d2, 1)
        dd = jnp.min(d2, 1)
        return jnp.where(jnp.isinf(dd), p.shape[0], j), dd

    return nearest


class TestConnect:
    """Bridging gives the same final MST as ``_connect_components``."""

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_host(self, seed):
        """Pieces of a cut stream are bridged into scipy's spanning tree."""
        rng = np.random.default_rng(seed)
        p = _stream(int(rng.integers(30, 300)), seed, interlopers=0.0)
        n = len(p)
        lo, hi, d = _knn_graph(p, 4)
        valid = d <= float(rng.choice([0.02, 0.05, 0.2]))
        _, labels = _boruvka(n, *map(jnp.asarray, (lo, hi, d, valid)))
        real = jnp.ones(n, bool)
        blo, bhi, bd, bv = jax.jit(gr.connect, static_argnums=2)(
            labels, real, _brute_nearest(jnp.asarray(p))
        )
        all_lo = np.r_[lo, np.asarray(blo)]
        all_hi = np.r_[hi, np.asarray(bhi)]
        all_w = np.r_[d, np.asarray(bd)]
        all_valid = np.r_[valid, np.asarray(bv)]
        tree, labels = map(
            np.asarray,
            _boruvka(n, *map(jnp.asarray, (all_lo, all_hi, all_w, all_valid))),
        )
        assert np.all(labels == 0)  # one component
        g, _ = _scipy_graph(n, lo, hi, d, valid)
        t = minimum_spanning_tree(_connect_components(p, g, workers=1)).tocoo()
        want = {(min(i, j), max(i, j)) for i, j in zip(t.row, t.col, strict=True)}
        assert _edge_set(all_lo, all_hi, tree) == want

    def test_tied_distances_cycle(self):
        """Tied distances that form cycles are broken correctly.

        Repro: 1D points [0,10,20,21,1,11] with labels [0,1,2,0,1,2].
        Components A={0,3}, B={1,4}, C={2,5} form a pick cycle at distance 1.
        """
        p = np.array([[0], [10], [20], [21], [1], [11]], dtype=np.float32)
        labels = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
        real = jnp.ones(6, bool)

        blo, bhi, bd, bv = jax.jit(gr.connect, static_argnums=2)(
            labels, real, _brute_nearest(jnp.asarray(p))
        )

        # Bridges, taken between component labels, must join all components
        _, comp = _boruvka(3, labels[blo], labels[bhi], bd, bv)
        assert np.all(np.asarray(comp) == 0), "bridges must form one component"

        # Every valid bridge should have length 1.0 (the tied distance)
        valid_distances = np.asarray(bd)[np.asarray(bv)]
        assert np.allclose(valid_distances, 1.0), (
            "all bridges should be at distance 1.0"
        )
