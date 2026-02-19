# Contributing to localflowwalk

Thank you for your interest in contributing! This guide covers the essentials.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nearest-neighbours-with-momentum.git
   cd nearest-neighbours-with-momentum
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

```python
import jax.numpy as jnp
import phasecurvefit as pcf


def test_basic_ordering():
    """Test basic ordering works."""
    position = {"x": jnp.array([0.0, 1.0, 2.0])}
    velocity = {"x": jnp.array([1.0, 1.0, 1.0])}

    result = pcf.walk_local_flow(position, velocity, start_idx=0, metric_scale=1.0)

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

Thank you for contributing to localflowwalk!
