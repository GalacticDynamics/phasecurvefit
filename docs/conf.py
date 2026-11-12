"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

__all__: tuple[str, ...] = ()

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import sphinx.ext.autodoc.directive as autodoc_directive
from pygments.lexers.python import PythonLexer
from sphinx.parsers import RSTParser
from sphinx.util.docutils import new_document

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
copyright = f"{datetime.now(UTC).year}, {author}"
version = __version__

master_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # MyST markdown + Jupyter notebook support
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
    "sphinx_reredirects",
]

python_use_unqualified_type_names = True

# Execute every guide and tutorial on every build, here and on ReadTheDocs. No
# outputs are stored in the sources, so each page shows what the current code
# produces.
nb_execution_mode = "force"
# Except pages whose code cells only assert and display nothing. Each executed
# page costs a kernel start and an `import phasecurvefit`, and the build has to
# fit Read the Docs' time limit. Sybil still runs these cells in the doc tests.
nb_execution_excludepatterns = ["contributing.md", "reference/migration/*"]
# A cell that hangs fails the build here, with a log, instead of running into
# Read the Docs' build limit, which kills the build and throws its log away.
nb_execution_timeout = 300
nb_execution_raise_on_error = True
pygments_lexers = {"ipython3": PythonLexer()}

# -- Redirects for pages moved in the Diataxis restructure ------------------

# sphinx-reredirects makes an absolute target relative by prepending ".." per
# directory level of the old page, so a top-level page needs a relative target
# itself. Its redirect page carries the URL fragment across.
redirects: Final[dict[str, str]] = {
    "guides/quickstart": "/start/quickstart.html",
    "guides/algorithm": "/explanation/walk.html",
    "guides/metrics": "/how-to/tune-metric.html",
    "guides/orderers": "/how-to/choose-orderer.html",
    "guides/som": "/how-to/som.html",
    "guides/nn": "/how-to/autoencoder.html",
    "guides/outliers": "/how-to/outliers.html",
    "guides/jax-integration": "/how-to/jax.html",
    "api/index": "/reference/api.html",
    "migration/v0.1-to-v0.2": "/reference/migration/v0.1-to-v0.2.html",
    "migration/v0.2-to-v0.3": "/reference/migration/v0.2-to-v0.3.html",
    "migration/v0.3-to-v0.4": "/reference/migration/v0.3-to-v0.4.html",
    "citation": "reference/citation.html",  # top level: no ".." to prepend
}

exclude_patterns = [
    "superpowers",  # local design notes, git-ignored
    "_build",
    "**.ipynb_checkpoints",
    # Each tutorial's source is its MyST Markdown file. The .ipynb is generated
    # from it, only for the Colab badges.
    "tutorials/*.ipynb",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
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


# -- autodoc inside MyST ---------------------------------------------------


def setup(app: Any) -> None:
    """Let ```{autoclass}``` and friends work in MyST markdown pages.

    autodoc generates reStructuredText, but inside a MyST page it hands that
    text to MyST's markdown state, so it renders as literal text. Parse what
    autodoc generates with the reStructuredText parser instead.
    """

    def parse_generated_content(state: Any, content: Any, documenter: Any) -> list:
        doc = new_document(state.document["source"], state.document.settings)
        parser = RSTParser()
        parser.set_application(app)
        parser.parse("\n".join(content), doc)
        return list(doc.children)

    autodoc_directive.parse_generated_content = parse_generated_content
