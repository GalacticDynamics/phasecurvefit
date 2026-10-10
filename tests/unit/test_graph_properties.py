"""Property tests for the pure-JAX graph package on arbitrary small graphs.

Sizes are drawn from a fixed menu so that JAX compiles a bounded number of
shapes; deadlines are off for the same reason.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    minimum_spanning_tree,
    shortest_path,
)

from phasecurvefit._src import graph as gr

_boruvka = jax.jit(gr.boruvka, static_argnums=0)
_diameter = jax.jit(gr.diameter_path, static_argnums=0)
SIZES = st.sampled_from([1, 2, 3, 8, 33, 80])


def _graph(seed, n, ties):
    rng = np.random.default_rng(seed)
    e = 3 * n
    a, b = rng.integers(0, n, e), rng.integers(0, n, e)
    w = (rng.integers(1, 4, e) if ties else rng.random(e) + 0.01).astype(np.float32)
    valid = (a != b) & (rng.random(e) < 0.8)
    return np.minimum(a, b), np.maximum(a, b), w, valid


def _sym(n, lo, hi, w, mask):
    best = {}
    for i, j, ww in zip(lo[mask], hi[mask], w[mask], strict=True):
        best[(int(i), int(j))] = min(best.get((int(i), int(j)), np.inf), float(ww))
    g = csr_matrix(
        (list(best.values()), ([i for i, _ in best], [j for _, j in best])),
        shape=(n, n),
    )
    return g + g.T


@settings(max_examples=40, deadline=None)
@given(n=SIZES, seed=st.integers(0, 2**31 - 1), ties=st.booleans())
def test_boruvka_is_a_minimum_spanning_forest(n, seed, ties):
    """Same total weight and components as scipy, and a forest (n - c edges)."""
    lo, hi, w, valid = _graph(seed, n, ties)
    tree, labels = map(np.asarray, _boruvka(n, *map(jnp.asarray, (lo, hi, w, valid))))
    ref = minimum_spanning_tree(_sym(n, lo, hi, w, valid))
    n_comp, ref_labels = connected_components(ref, directed=False)
    assert tree.sum() == n - n_comp
    assert np.isclose(w[tree].sum(), ref.sum(), rtol=1e-5)
    for i in range(n):
        assert (labels == labels[i]).tolist() == (ref_labels == ref_labels[i]).tolist()


@settings(max_examples=40, deadline=None)
@given(n=SIZES, seed=st.integers(0, 2**31 - 1))
def test_diameter_path_is_a_longest_tree_path(n, seed):
    """A path along tree edges, as long as the root piece's diameter."""
    lo, hi, w, valid = _graph(seed, n, ties=False)
    args = tuple(map(jnp.asarray, (lo, hi, w, valid)))
    tree, labels = _boruvka(n, *args)
    alive = np.asarray(labels) == np.asarray(labels)[0]
    full, blen = _diameter(n, *args[:3], tree, jnp.asarray(alive))
    path = np.asarray(full)[: int(blen)]
    t = _sym(n, lo, hi, w, np.asarray(tree)).toarray()
    assert len(set(path.tolist())) == len(path)
    assert all(t[a, b] > 0 for a, b in itertools.pairwise(path))
    nodes = np.flatnonzero(alive)
    dist = shortest_path(t[np.ix_(nodes, nodes)], directed=False)
    length = sum(t[a, b] for a, b in itertools.pairwise(path))
    assert np.isclose(length, dist[np.isfinite(dist)].max(), rtol=1e-5)


@settings(max_examples=25, deadline=None)
@given(n=SIZES, seed=st.integers(0, 2**31 - 1))
def test_connect_joins_everything(n, seed):
    """Bridges plus the original edges always form one component."""
    rng = np.random.default_rng(seed)
    p = jnp.asarray(rng.normal(size=(n, 2)), jnp.float32)
    lo, hi, w, valid = _graph(seed, n, ties=False)
    _, labels = _boruvka(n, *map(jnp.asarray, (lo, hi, w, valid)))

    def nearest(lab):
        d2 = jnp.sum((p[:, None] - p[None]) ** 2, -1)
        d2 = jnp.where(lab[:, None] == lab[None], jnp.inf, d2)
        dd = jnp.min(d2, 1)
        return jnp.where(jnp.isinf(dd), n, jnp.argmin(d2, 1)), dd

    blo, bhi, bd, bv = jax.jit(gr.connect, static_argnums=2)(
        labels, jnp.ones(n, bool), nearest
    )
    all_ = [np.r_[a, np.asarray(b)] for a, b in ((lo, blo), (hi, bhi), (w, bd))]
    av = np.r_[valid, np.asarray(bv)]
    _, joined = _boruvka(n, *map(jnp.asarray, (*all_, av)))
    assert np.all(np.asarray(joined) == 0)
