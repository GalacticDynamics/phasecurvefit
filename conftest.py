"""Pytest configuration for phasecurvefit tests."""

import os
from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

# Tests never need a GUI: without this, matplotlib picks TkAgg on Windows
# runners, whose Tcl/Tk install is intermittently broken ("Can't find a usable
# tk.tcl"), failing doc examples that only draw a figure.
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import pytest
from jaxtyping import PRNGKeyArray
from sybil import Document, Region, Sybil
from sybil.parsers import myst, rest

from optional_dependencies import OptionalDependencyEnum, auto

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


class OptDeps(OptionalDependencyEnum):  # pylint: disable=invalid-enum-extension
    """Optional dependencies for phasecurvefit."""

    UNXT = auto()
    JAXKD = auto()
    MATPLOTLIB = auto()


collect_ignore_glob = []
if not OptDeps.UNXT.installed:
    collect_ignore_glob.append("tests/unit/test_interop_unxt.py")
if not OptDeps.MATPLOTLIB.installed:
    collect_ignore_glob.append("tests/usage/test_epitrochoid.py")
if not OptDeps.JAXKD.installed:
    collect_ignore_glob.append("tests/unit/test_kdtree.py")


def pytest_configure(config):
    """Configure pytest."""
    # Suppress JAX warning about no GPU
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    """Provide a JAX random key for tests."""
    return jax.random.key(37)
