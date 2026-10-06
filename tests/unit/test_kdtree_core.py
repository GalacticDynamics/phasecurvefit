"""Tests for the self-contained JAX kd-tree (``phasecurvefit._src.kdtree``)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
