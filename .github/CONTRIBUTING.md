# Contributing

## Quick Development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `uv run nox` to automatically install and run it.

### Development with Nox

To run all the checks with nox:

```bash
uv run nox -s all
```

To run specific sessions:

```bash
uv run nox -s lint      # Run linting checks
uv run nox -s test      # Run test suite
uv run nox -s docs      # Build documentation
```

### Development without Nox

If you'd prefer to set up a local development environment manually:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with development dependencies
uv sync --all-extras --all-groups

# Install pre-commit hooks
uv run pre-commit install
```

Then you can run the checks manually:

```bash
# Run pre-commit checks
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ -v

# Run pylint
uv run pylint src/
```

## The code graph

`graphify-out/graph.json` is a
[graphify](https://github.com/Graphify-Labs/graphify) map of this repository:
its symbols and the relationships between them. AI assistants query it instead
of grepping, and you can too. Extraction is local tree-sitter parsing — no model
call, and nothing leaves your machine — and every edge is marked `EXTRACTED`
(explicit in the source) or `INFERRED` (resolved by graphify), so you can tell
what was read from what was deduced.

The graph is committed, and the pre-commit hooks rebuild it. Install the pinned
release and register the merge driver once per clone, so two branches that both
changed the graph merge cleanly:

```bash
uv tool install "graphifyy==0.9.119"
PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"
```

Git will not take a merge-driver command from a tracked file, which is why this
is a per-clone step rather than something the repository can configure for you.

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
- If a rebuild is ever refused because the graph would shrink — you deleted code
  — use `GRAPHIFY_FORCE=1 git commit ...`. It usually just shrinks and says
  nothing.
- If the graph ever conflicts in a merge, rebuild rather than edit it:
  `rm graphify-out/graph.json && graphify update . --force && git add graphify-out/graph.json`.
- A changed `graph.json` in a pull request is generated output. It is one line,
  collapsed by GitHub, and is not reviewed.
- The rebuild may rewrite the graph the first time you commit in a working
  environment, and the weekly `Refresh code graph` job may rewrite it back. That
  is expected, not a bug: graphify's output depends on which files are on disk,
  so a built checkout — one with the generated `src/phasecurvefit/_version.py`
  in it — and a fresh clone disagree about community numbering even though the
  symbols are identical.

## Preparing for a PR

Before submitting a pull request:

1. Ensure pre-commit passes: `uv run pre-commit run --all-files`
2. Run the full test suite: `uv run pytest tests/ -v`
3. Check test coverage:
   `uv run pytest tests/ --cov=src --cov-report=term-missing`
4. Build the docs to verify they work: `uv run nox -s docs`

The CI will run these checks automatically, but running them locally first saves
time.
