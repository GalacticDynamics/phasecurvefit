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


@settings(max_examples=25, deadline=None)
@given(
    n=st.sampled_from([0, 1, 2, 5, 17, 64, 200]),
    d=st.integers(1, 3),
    k=st.sampled_from([1, 3, 10]),
    frontier=st.sampled_from([2, 16]),
    seed=st.integers(0, 2**31 - 1),
    clump=st.booleans(),
)
def test_all_knn_equals_brute(n, d, k, frontier, seed, clump):
    """Squared distances equal brute force; indices realise them."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, d)).astype(np.float32)
    if clump and n > 3:
        p[: n // 2] = p[0]
    idx, d2 = jax.jit(functools.partial(kd.all_knn, k=k, frontier=frontier))(
        jnp.asarray(p)
    )
    ref_i, ref_d = kd.brute_knn(jnp.asarray(p), k)
    np.testing.assert_allclose(np.asarray(d2), np.asarray(ref_d), rtol=1e-5, atol=1e-12)
    idx = np.asarray(idx)
    fin = np.isfinite(np.asarray(d2))
    rows = np.nonzero(fin)
    realised = ((p[rows[0]] - p[idx[rows]]) ** 2).sum(-1)
    np.testing.assert_allclose(realised, np.asarray(d2)[rows], rtol=1e-5, atol=1e-12)
    assert np.all(idx[~fin] == n)
