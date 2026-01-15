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

## Preparing for a PR

Before submitting a pull request:

1. Ensure pre-commit passes: `uv run pre-commit run --all-files`
2. Run the full test suite: `uv run pytest tests/ -v`
3. Check test coverage:
   `uv run pytest tests/ --cov=src --cov-report=term-missing`
4. Build the docs to verify they work: `uv run nox -s docs`

The CI will run these checks automatically, but running them locally first saves
time.
