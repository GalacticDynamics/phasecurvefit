---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Contributing to phasecurvefit

Thank you for your interest in contributing! This guide covers the essentials.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/phasecurvefit.git
   cd phasecurvefit
   ```

2. **Install dependencies:**

   ```bash
   uv sync --all-extras
   ```

## Code Style

- **Type hints**: Complete type annotations on all functions
- **`__all__` first**: Define before imports (except `__future__`)
- **JAX compatible**: Must work with `jit`, `vmap`, `grad`
- **Dict-based phase-space**: Use dicts for position/velocity data

## Testing and Quality

Run tests:

```bash
uv run pytest tests/ -v
```

Lint your code:

```bash
uv run pre-commit run --all-files
```

## The code graph

`graphify-out/graph.json` is a [graphify](https://github.com/Graphify-Labs/graphify)
map of this repository: its symbols and the relationships between them. AI
assistants query it instead of grepping, and you can too. Extraction is local
tree-sitter parsing — no model call, and nothing leaves your machine — and every
edge is marked `EXTRACTED` (explicit in the source) or `INFERRED` (resolved by
graphify).

The graph is committed, and a pre-commit hook rebuilds it. Install the pinned
release and register the merge driver once per clone, so two branches that both
changed the graph merge cleanly (git will not take a merge-driver command from a
tracked file, so the repository cannot do this for you):

```bash
uv tool install "graphifyy==0.9.119"
PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"
```

Then query it:

```bash
graphify query "how is the ordering parameter gamma assigned"
graphify explain "MSTOrderer"            # one symbol and its neighbours
graphify path "order" "PathAutoencoder"  # how two things connect
open graphify-out/graph.html             # the clickable graph
```

Day to day:

- The rebuild runs with your other pre-commit checks. Like a formatter, it
  rebuilds and stops the commit; `git add graphify-out/graph.json` and commit
  again.
- Commit without rebuilding: `SKIP=graphify-update git commit ...`.
- If a rebuild is refused because the graph would shrink — you deleted code —
  use `GRAPHIFY_FORCE=1 git commit ...`.
- If the graph conflicts in a merge, rebuild rather than edit it:
  `rm graphify-out/graph.json && graphify update . --force`.
- A changed `graph.json` in a pull request is generated output: one line,
  collapsed by GitHub, not reviewed.
- The rebuild may rewrite the graph the first time you commit, and the weekly
  `Refresh code graph` job may rewrite it back. That is expected: graphify's
  output depends on which files are on disk, so a checkout with the generated
  `src/phasecurvefit/_version.py` in it and a fresh clone disagree about
  community numbering even though the symbols are identical.

## Making Changes

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code** following the style guide above

3. **Add tests** in `tests/` for your changes

4. **Run tests and linting:**

   ```bash
   uv run pytest tests/ -v
   uv run pre-commit run --all-files
   ```

5. **Commit and push:**

   ```bash
   git add .
   git commit -m "Brief description"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** with a clear description

## Writing Tests

Tests must:

- **Test something**: Include assertions
- **Be JAX-compatible**: Test with `jit`, `vmap`
- **Have clear names**: `test_ordering_increases_gamma` not `test_1`
- **Be deterministic**: Use fixed seeds

Example:

```{code-cell} python
import jax.numpy as jnp
import phasecurvefit as pcf


def test_basic_ordering():
    """Test basic ordering works."""
    position = {"x": jnp.array([0.0, 1.0, 2.0])}
    velocity = {"x": jnp.array([1.0, 1.0, 1.0])}

    result = pcf.order(position, velocity)

    assert jnp.array_equal(result.indices, jnp.array([0, 1, 2]))
```

## Documentation

- Use NumPy-style docstrings with Parameters, Returns, Examples sections
- All code examples in docs are tested - keep them executable
- Update guides if you change public APIs

## Performance

For hot-path code:

- Use `jax.lax` primitives instead of Python loops
- Use `jax.vmap` for vectorization
- Avoid Python control flow in JAX-compiled functions

## Reporting Issues

**Bugs**: Include minimal reproducible example, expected behavior, and error message

**Features**: Explain the use case and proposed API

## Code Review

We look for:

- Correctness and tests
- JAX compatibility
- Code quality
- Performance

We aim to provide feedback within 1 week.

## License

By contributing, you agree your work will be licensed under the project's MIT license.

Thank you for contributing to phasecurvefit!
