# Project Overview

`localflowwalk` is a JAX-native library for ordering phase-space observations
using a variety of tools.

- **Language**: Python 3.12+
- **JAX integration**: All core operations are JIT-compatible, differentiable,
  and work with `vmap`. Objects are PyTrees via Equinox.
- **Design goals**: Maximum performance, pluggable components (metrics,
  strategies), optional unit support via `unxt`

## Main Components

### Core Algorithm

- `walk_local_flow(positions, velocities, ...)`: Main entry point for ordering
  phase-space data. Returns `LocalFlowWalkResult` with ordered indices.
- Phase-space data: Two dicts with matching keys, e.g.,
  `{"x": array, "y": array}` for positions and velocities.

### Distance Metrics (`localflowwalk.metrics`)

Pluggable metrics determine how the algorithm selects the next point:

- `FullPhaseSpaceDistanceMetric` (default): True 6D Euclidean distance
- `AlignedMomentumDistanceMetric`: NN+p metric with velocity alignment
- `SpatialDistanceMetric`: Position-only (standard nearest-neighbor)
- `AbstractDistanceMetric`: Base class for custom metrics

### Query Strategies

Strategies control how neighbors are found:

- `BruteForce()`: Default, computes distances to all points
- `KDTree(k=...)`: KD-tree prefiltering (requires `jaxkd`)

### Autoencoder for Gap Filling (`localflowwalk.nn`)

Neural network for interpolating skipped tracers (Appendix A.2 of the paper):

- `Autoencoder`: Maps phase-space → ordering parameter γ ∈ [-1, 1]
- `train_autoencoder(autoencoder, result, config)`: Train on output
- `fill_ordering_gaps(result, autoencoder)`: Fill in skipped indices
- `TrainingConfig`: Configure epochs, learning rate, etc.

### Unit Support (`unxt` integration)

When `unxt` is installed, `walk_local_flow` accepts `Quantity` values:

```python
import unxt as u

pos = {"x": u.Q([0, 1, 2], "kpc"), "y": u.Q([0, 0.5, 1], "kpc")}
vel = {"x": u.Q([1, 1, 1], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}
result = pcf.walk_local_flow(pos, vel, start_idx=0, lam=u.Q(1.0, "kpc"))
```

## Folder Structure

- `/src/localflowwalk/`: Public API
  - `__init__.py`: Main exports (`walk_local_flow`, strategies, result types)
  - `metrics.py`: Distance metric classes
  - `nn.py`: Autoencoder neural network module
  - `w.py`: Phase-space accessor utilities
- `/src/localflowwalk/_src/`: Private implementation
  - `algorithm.py`: Core `walk_local_flow` implementation
  - `autoencoder.py`: Neural network implementation (Equinox)
  - `metrics.py`: Metric base classes and implementations
  - `strategies.py`: Query strategy classes
- `/src/localflowwalk/_interop/`: Optional dependency integrations
  - `interop_unxt.py`: `unxt` Quantity support via Quax dispatch
- `/docs/guides/`: User guides (quickstart, metrics, JAX integration, etc.)
- `/tests/`: Test suite organized by component

## Coding Style

### Module Structure: `__all__` Before Imports

**CRITICAL**: In all Python modules, `__all__` must be defined **before** any
imports (except `__future__` imports):

```python
"""Module docstring."""

__all__: tuple[str, ...] = (
    "PublicClass",
    "public_function",
)

from collections.abc import Mapping

# ... rest of imports
```

### Key Conventions

- Always use type hints (standard typing, `jaxtyping.Array`)
- `__all__` should be a tuple (not list) for immutability
- **NEVER use `from __future__ import annotations`** — causes issues with Plum
  dispatch and runtime type introspection
- Use `jax.tree.map` and `jax.tree.leaves` for operations on component dicts
- Phase-space notation: `q` for position, `p` for momentum/velocity, `w` for
  full phase-space point

### Multiple Dispatch with Plum

This package uses `plum-dispatch` for multiple dispatch:

- Check all registered implementations via `function.methods`
- Dispatches exist for plain arrays and `unxt.Quantity` types
- When adding new dispatches, search for existing ones first

### Quax Integration (Unit Support)

The `_interop/interop_unxt.py` module registers Quax dispatches for JAX
primitives when operating on `Quantity` values:

- `scan_p` dispatches handle the bounded while loop with Quantities
- Strategy dispatches strip units before calling `jaxkd`, restore after
- **FIRM REQUIREMENT**: Quantities must flow through the algorithm — no
  stripping at the API level

## Tooling

- **Package manager**: `uv` for dependency and environment management
- **Task runner**: `nox` for all development tasks
- **Linting**: `ruff` (check and format), configured in `pyproject.toml`
- **Type checking**: `mypy` with strict settings

Common commands:

```bash
uv run nox -s lint      # Run linters (pre-commit + ruff)
uv run nox -s test      # Run pytest suite
uv run pytest tests/ -v # Run tests directly
uv run pre-commit run -a # Run all pre-commit hooks
```

## Testing

- Use `pytest` for all test suites
- Add unit tests for every new function or class
- Test JAX compatibility (`jit`, `vmap`, `grad`) where applicable
- Tests for Quantity support in `tests/test_quantity_support.py`
- Assertions should be atomic (no `assert a and b`, use separate asserts)

## Architecture Notes

### Result Structure

`LocalFlowWalkResult` contains:

- `indices`: Array of indices in walk order (-1 for skipped)
- `positions`: Original position dict
- `velocities`: Original velocity dict
- `skipped_indices`: Property returning indices that were skipped

## Final Notes

- Preserve JAX compatibility above all — functions must work with `jit`, `vmap`
- Use dict-based APIs for flexibility with different coordinate systems
- When extending metrics or strategies, follow existing patterns
- Documentation examples must be executable (tested via Sybil)
