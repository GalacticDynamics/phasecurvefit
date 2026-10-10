"""Static tree layout: depth, leaf count, effective leaf size and padding."""

__all__: tuple[str, ...] = ("Layout", "layout")

import math
from typing import NamedTuple

import numpy as np


class Layout(NamedTuple):
    """Static shape of a balanced, pointer-free tree over ``n`` points."""

    depth: int
    n_leaves: int
    leaf_size: int  # effective rows per leaf, ``B_eff``
    n_pad: int  # ``n_leaves * leaf_size >= n``
    leaf_valid: np.ndarray  # (n_leaves,) real points per leaf: B_eff or B_eff - 1


def layout(n: int, leaf_size: int, /) -> Layout:
    """Spread fewer than one padding row per leaf evenly across ``2**depth`` leaves.

    >>> layout(100, 16)[:4]
    (3, 8, 13, 104)
    >>> layout(0, 16)[:4]
    (0, 1, 1, 1)

    """
    depth = 0 if n <= leaf_size else math.ceil(math.log2(n / leaf_size))
    n_leaves = 2**depth
    b_eff = max(1, math.ceil(n / n_leaves))
    n_pad = n_leaves * b_eff
    n_extra = n_pad - n  # < n_leaves
    j = np.arange(n_leaves)
    leaf_pad = (j + 1) * n_extra // n_leaves - j * n_extra // n_leaves  # 0 or 1
    return Layout(depth, n_leaves, b_eff, n_pad, b_eff - leaf_pad)
