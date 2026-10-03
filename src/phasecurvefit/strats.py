r"""Search Strategies for the local-flow walk."""

__all__: tuple[str, ...] = (
    "AbstractQueryStrategy",
    "BruteForce",
    "KDTree",
    "QueryResult",
)

from ._src.strategies import AbstractQueryStrategy, BruteForce, KDTree, QueryResult
