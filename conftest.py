"""Pytest configuration for phasecurvefit tests."""

from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from importlib.util import find_spec

import jax
import pytest
from jaxtyping import PRNGKeyArray
from sybil import Document, Region, Sybil
from sybil.parsers import myst, rest

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

# Shared parsers for both markdown and Python
markdown_parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

# Markdown documentation tests
docs = Sybil(parsers=markdown_parsers, patterns=["*.md"])

# Python source code tests (includes markdown parsers + rest parsers)
python = Sybil(
    parsers=[
        *markdown_parsers,
        rest.PythonCodeBlockParser(),
        rest.DocTestParser(optionflags=optionflags),
        rest.SkipParser(),
    ],
    patterns=["*.py"],
)

# Combine both for pytest collection
pytest_collect_file = (docs + python).pytest()


collect_ignore_glob = [
    path
    for pkg, path in (
        ("unxt", "tests/unit/test_interop_unxt.py"),
        ("matplotlib", "tests/usage/test_epitrochoid.py"),
        ("jaxkd", "tests/unit/test_kdtree.py"),
    )
    if find_spec(pkg) is None
]


def pytest_configure(config):
    """Configure pytest."""
    # Suppress JAX warning about no GPU
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    """Provide a JAX random key for tests."""
    return jax.random.key(37)
