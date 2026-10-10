"""Self-contained exact kd-tree kNN in JAX.

Imports only the standard library, numpy, jax, jaxtyping and equinox -- never
the parent package -- so it can be lifted into its own library.
"""

__all__: tuple[str, ...] = (
    "Tree",
    "all_knn",
    "brute_knn",
    "build_tree",
    "knn",
    "locate_leaves",
    "node_labels",
)

from ._brute import brute_knn
from ._build import Tree, build_tree
from ._query import all_knn, knn, locate_leaves, node_labels
