"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

__all__: tuple[str, ...] = ()

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pytz
from pygments.lexers.python import PythonLexer

# Add the package to the path
_docs_dir = Path(__file__).parent
_repo_root = _docs_dir.parent
sys.path.insert(0, str(_repo_root / "src"))

# Import version
try:
    from phasecurvefit._version import version as __version__
except ImportError:
    __version__ = "0.1.0"

# -- Project information -----------------------------------------------------

author = "phasecurvefit Developers"
project = "phasecurvefit"
copyright = f"{datetime.now(pytz.timezone('UTC')).year}, {author}"
version = __version__

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # General MyST markdown support
    "nbsphinx",  # Jupyter notebook support
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx-prompt",
    "sphinxext.opengraph",
    "sphinx_togglebutton",
]

python_use_unqualified_type_names = True

# Use a known lexer for notebook code cells to avoid Pygments warnings.
nbsphinx_codecell_lexer = "python"
# On ReadTheDocs, don't execute notebooks (no kernel available)
# Locally, execute them to update outputs
nbsphinx_execute = "never" if os.environ.get("READTHEDOCS") else "always"
pygments_lexers = {"ipython3": PythonLexer()}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Napoleon settings ---------------------------------------------------

napoleon_use_math = True
# Render "Attributes" sections as field lists, so dataclass fields documented by
# ``:members:`` are not described twice.
napoleon_use_ivar = True

# -- Autodoc settings ---------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

always_document_param_types = True
typehints_use_signature = True
typehints_fully_qualified = False
simplify_optional_unions = True

# Map unqualified type names to fully qualified names
autodoc_type_aliases = {
    "ndarray": "numpy.ndarray",
    "ArrayLike": "jaxtyping.ArrayLike",
    "Array": "jax.Array",
}

# Autosummary settings
autosummary_generate = True

# -- Nitpick ignore patterns -------------------------------------------------

# Match single shape tokens: N (batch), D (dimension), etc.
_SHAPE_NAME_RE: Final[str] = r"^(?:N|D|1|2|3|\.\.\.)$"

# Match space-separated shape tuples like "N D", "N 3"
_SHAPE_TUPLE_RE: Final[str] = r"^(?:N|D|\d+|\.\.\.)(?:\s+(?:N|D|\d+|\.\.\.))*$"

nitpick_ignore_regex: Final[list[tuple[str, str]]] = [
    ("py:class", _SHAPE_TUPLE_RE),
    ("py:data", _SHAPE_TUPLE_RE),
    ("py:class", _SHAPE_NAME_RE),
    ("py:data", _SHAPE_NAME_RE),
    # Array annotations. jaxtyping publishes no Sphinx inventory, and the jax /
    # numpy names are rendered unqualified from the jaxtyping annotations.
    ("py:class", r"^jaxtyping\..*"),
    ("py:class", r"^(?:Array|ndarray|TypedNdArray|PRNGKeyArray|Float|Int)$"),
    ("py:class", r"^(?:jax|quax|optax)\._src\..*|^quax\._values\..*"),
    # unxt / Python typing names missing from their projects' inventories
    ("py:class", r"^(?:unxt\.)?(?:AbstractQuantity|Quantity)$"),
    ("py:data", r"^typing\.Union$"),
    # Private implementation classes (e.g. base classes via :show-inheritance:)
    ("py:class", r"^phasecurvefit\._src\..*"),
    # Type aliases documented with ``.. data::`` but referenced as classes
    ("py:class", r"^(?:ScalarComponents|VectorComponents|FLikeSz0|FSzN|ISzN|Bool)$"),
    ("py:class", r"^AbstractResult$"),
    # Free-text fragments numpydoc "type" fields split into references:
    # "optional", "default=0.5", "shape (N, D)", quoted literals, ...
    ("py:class", r"^(?:optional|keyword-only|callable|scalar|shape|M|2\*n_dims|2D)$"),
    # The first name of a combined "a, b : type" parameter line (renders fine)
    ("py:class", r"^(?:width_size|prototype_positions)$"),
    ("py:class", r"^default=.*"),
    ("py:class", r"""^['"{ ].*"""),
]

# optax's recursive ``OptState`` alias contains an unresolvable ``ArrayTree``.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
    ("py:class", "NoneType"),
]

# -- MyST Settings -------------------------------------------------

myst_enable_extensions = [
    "amsmath",  # for direct LaTeX math
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",  # for $, $$
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

myst_substitutions = {
    "ArrayLike": ":obj:`jaxtyping.ArrayLike`",
    "Array": ":obj:`jax.Array`",
}


# -- HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "phasecurvefit"
html_copy_source = True
html_favicon = "_static/favicon.png"
html_logo = "_static/favicon.png"

html_static_path = ["_static"]
html_css_files = []

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/GalacticDynamics/phasecurvefit",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GalacticDynamics/phasecurvefit",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/phasecurvefit/",
            "icon": "https://img.shields.io/pypi/v/phasecurvefit",
            "type": "url",
        },
    ],
}
