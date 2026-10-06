"""Tests for the self-contained JAX kd-tree (``phasecurvefit._src.kdtree``)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
