"""Property test: the kd-tree equals brute force on arbitrary small clouds.

Sizes, dimensions and k are drawn small so that JAX compiles a bounded number of
shapes; deadlines are off for the same reason.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st

from phasecurvefit._src import kdtree as kd


def _lower_index_oracle(p, k):
    """Float64 brute force, self excluded, ties to the lower index."""
    d2 = ((p[:, None].astype(np.float64) - p[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    if len(p) < k:
        d2 = np.concat([d2, np.full((len(p), k - len(p)), np.inf)], axis=1)
    return np.argsort(d2, axis=1, kind="stable")[:, :k]


@settings(max_examples=25, deadline=None)
@given(
    n=st.sampled_from([0, 1, 2, 5, 17, 64, 200]),
    d=st.integers(1, 6),
    k=st.sampled_from([1, 3, 10]),
    frontier=st.sampled_from([2, 16]),
    leaf_size=st.sampled_from([1, 2, 4, 16]),
    seed=st.integers(0, 2**31 - 1),
    data=st.sampled_from(["normal", "clump", "lattice"]),
)
def test_all_knn_equals_brute(n, d, k, frontier, leaf_size, seed, data):
    """Squared distances equal brute force; indices realise them, never self.

    On integer lattices (exact float32 distances, many ties) the indices must
    equal a float64 oracle that breaks ties by the lower index.
    """
    rng = np.random.default_rng(seed)
    if data == "lattice":
        p = rng.integers(0, 4, size=(n, d)).astype(np.float32)
    else:
        p = rng.normal(size=(n, d)).astype(np.float32)
    if data == "clump" and n > 3:
        p[: n // 2] = p[0]
    f = functools.partial(kd.all_knn, k=k, frontier=frontier, leaf_size=leaf_size)
    idx, d2 = jax.jit(f)(jnp.asarray(p))
    ref_i, ref_d = kd.brute_knn(jnp.asarray(p), k)
    np.testing.assert_allclose(np.asarray(d2), np.asarray(ref_d), rtol=1e-5, atol=1e-12)
    idx = np.asarray(idx)
    fin = np.isfinite(np.asarray(d2))
    rows = np.nonzero(fin)
    realised = ((p[rows[0]] - p[idx[rows]]) ** 2).sum(-1)
    np.testing.assert_allclose(realised, np.asarray(d2)[rows], rtol=1e-5, atol=1e-12)
    assert np.all(idx[~fin] == n)
    assert not np.any(idx == np.arange(n)[:, None])
    if data == "lattice":
        want = _lower_index_oracle(p, k)
        np.testing.assert_array_equal(idx[fin], want[fin])


@settings(max_examples=25, deadline=None)
@given(
    n=st.sampled_from([1, 2, 5, 17, 64, 200]),
    m=st.sampled_from([1, 7, 30]),
    d=st.integers(1, 3),
    k=st.sampled_from([1, 3, 10]),
    frontier=st.sampled_from([2, 16]),
    seed=st.integers(0, 2**31 - 1),
    far=st.booleans(),
)
def test_knn_equals_brute(n, m, d, k, frontier, seed, far):
    """Bichromatic squared distances equal float64 brute force, far queries too."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, d)).astype(np.float32)
    q = (rng.normal(size=(m, d)) * (100 if far else 1)).astype(np.float32)
    f = jax.jit(
        lambda a, b: kd.knn(kd.build_tree(a), b, k, frontier=frontier),
    )
    idx, d2 = map(np.asarray, f(jnp.asarray(p), jnp.asarray(q)))
    ref = ((q[:, None].astype(np.float64) - p[None]) ** 2).sum(-1)
    ref = np.sort(ref, 1)
    ref = np.concat([ref, np.full((m, max(0, k - n)), np.inf)], axis=1)[:, :k]
    fin = np.isfinite(ref)
    np.testing.assert_array_equal(np.isfinite(d2), fin)
    np.testing.assert_allclose(d2[fin], ref[fin], rtol=1e-5, atol=1e-12)
    rows = np.nonzero(fin)
    realised = ((q[rows[0]] - p[idx[rows]]) ** 2).sum(-1)
    np.testing.assert_allclose(realised, d2[rows], rtol=1e-5, atol=1e-12)
    assert np.all(idx[~fin] == n)
