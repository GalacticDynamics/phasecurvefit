"""Pointer jumping: forest roots, list ranking, path sums. Log-depth, fixed shape."""

__all__: tuple[str, ...] = ("accumulate", "find_roots", "list_rank", "n_rounds")

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def n_rounds(n: int, /) -> int:
    """Pointer-jumping rounds that resolve any chain of length ``n``."""
    return max(1, math.ceil(math.log2(max(n, 2)))) + 1


def accumulate(
    parent: Int[Array, " n"], w: Float[Array, " n"], /
) -> tuple[Int[Array, " n"], Float[Array, " n"]]:
    """Root of every node and the sum of ``w`` along its path to the root.

    ``w[i]`` is the weight of the edge from ``i`` to ``parent[i]``; roots must
    carry 0. Sums of non-negative weights, so no cancellation.
    """

    def cond(state: tuple) -> Array:
        p, _, it = state
        return (it < n_rounds(parent.shape[0])) & jnp.any(p != p[p])

    def body(state: tuple) -> tuple:
        p, d, it = state
        return p[p], d + d[p], it + 1

    p, d, _ = jax.lax.while_loop(cond, body, (parent, w, 0))
    return p, d


def find_roots(parent: Int[Array, " n"], /) -> Int[Array, " n"]:
    """Find the root of every node of a forest (roots point to themselves)."""
    return accumulate(parent, jnp.zeros(parent.shape[0], jnp.int32))[0]


def list_rank(succ: Int[Array, " m"], /) -> tuple[Int[Array, " m"], Bool[Array, " m"]]:
    """Count each element's steps to the end of its list (``succ == m`` ends it).

    Returns ``(steps, reached)``; ``reached`` is False for elements on a cycle
    (they never reach an end), whose ``steps`` are then meaningless.
    """
    m = succ.shape[0]
    s = jnp.concat([succ, jnp.array([m], succ.dtype)])  # row m: the end, fixed
    d = jnp.concat([(succ != m).astype(jnp.int32), jnp.zeros(1, jnp.int32)])
    s, d = accumulate(s, d)
    return d[:m], s[:m] == m
