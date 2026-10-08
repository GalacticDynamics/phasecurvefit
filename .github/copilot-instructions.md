# Project Overview

`phasecurvefit` is a JAX-native library for ordering phase-space observations
using a variety of tools.

- **Language**: Python 3.12+
- **JAX integration**: All core operations are JIT-compatible, differentiable,
  and work with `vmap`. Objects are PyTrees via Equinox.
- **Design goals**: Maximum performance, pluggable components (metrics,
  strategies), optional unit support via `unxt`

## Main Components

### Ordering (`phasecurvefit.order`, `phasecurvefit.orderers`)

- `order(positions, velocities, orderer=None)`: Main entry point. Returns an
  `OrderingResult`. With no `orderer` it runs `default_pipeline`: an MST
  backbone refined by a SOM (`MSTOrderer() | SOMOrderer()`), falling back to the
  MST alone when there are too few tracers for the SOM.
- Orderers (`phasecurvefit.orderers`), all subclasses of `AbstractOrderer`:
  - `LocalFlowOrderer`: the local-flow walk (Nibauer et al. 2022); takes a
    `WalkConfig`, `metric_scale`, `start_idx`, `direction`, `max_dist`, etc.
    - `MSTOrderer`: velocity-aware minimum-spanning-tree backbone, with optional
      edge-length sigma-clipping (`edge_clip_sigma`) for outlier rejection.
  - `SOMOrderer`: 1-D Self-Organizing Map (Starkman et al. 2023); see also the
    `phasecurvefit.som` module.
  - `ChainOrderer`: runs stages in sequence, threading each result into the next
    as `init`. Built with `|`, e.g. `MSTOrderer(k=10) | LocalFlowOrderer()`.
- `fit_track(positions, velocities, key=...)`: end to end — runs the default
  pipeline, then builds and trains a `PathAutoencoder`.
- `walk_local_flow` was **removed** in v0.4; use `order()` with
  `LocalFlowOrderer`.
- `WalkConfig(metric=..., strategy=...)` composes a distance metric and a query
  strategy.
- Phase-space data: Two dicts with matching keys, e.g.,
  `{"x": array, "y": array}` for positions and velocities.

### Distance Metrics (`phasecurvefit.metrics`)

Pluggable metrics determine how the algorithm selects the next point:

- `AlignedMomentumDistanceMetric` (default): NN+p metric with velocity alignment
- `FullPhaseSpaceDistanceMetric`: True 6D Euclidean distance
- `SpatialDistanceMetric`: Position-only (standard nearest-neighbor)
- `AbstractDistanceMetric`: Base class for custom metrics

### Query Strategies (`phasecurvefit.strats`)

Strategies control how neighbors are found:

- `BruteForce()`: Default, computes distances to all points
- `KDTree(k=...)`: KD-tree prefiltering (requires `jaxkd`). `k` counts usable
  neighbors: the query point itself is excluded, and `k` is clamped to the
  number of points.

### Neural Networks (`phasecurvefit.nn`)

Networks for interpolating skipped tracers (Appendix A.2 of the paper) and
modelling the track. Training is built on `jaxmore.nn`.

- `PathAutoencoder` (`AbstractAutoencoder`): Maps phase-space → ordering
  parameter γ; decoders include `EncoderExternalDecoder` and
  `RunningMeanDecoder`
- `OrderingNet`, `TrackNet` / `FourierTrackNet`: ordering and track networks
- `train_autoencoder`, `train_ordering_net`, with `TrainingConfig` /
  `OrderingTrainingConfig`
- `fill_ordering_gaps(result, autoencoder)`: Fill in skipped indices
- Membership / outlier rejection (Hogg, Bovy & Lang 2010, sec. 3):
  `MixtureMembershipConfig`, `posterior_membership`, `mixture_membership_loss`,
  etc.

### Unit Support (`unxt` integration)

When `unxt` is installed, `order` accepts `Quantity` values:

```python
import unxt as u

pos = {"x": u.Q([0, 1, 2], "kpc"), "y": u.Q([0, 0.5, 1], "kpc")}
vel = {"x": u.Q([1, 1, 1], "km/s"), "y": u.Q([0.5, 0.5, 0.5], "km/s")}
orderer = pcf.orderers.LocalFlowOrderer(metric_scale=u.Q(1.0, "kpc"))
meta = pcf.StateMetadata(usys=u.unitsystems.galactic)
result = pcf.order(pos, vel, orderer, metadata=meta)
```

Quantity inputs require a unit system via `metadata=StateMetadata(usys=...)`;
without it the orderers raise.

## Folder Structure

- `/src/phasecurvefit/`: Public API
  - `__init__.py`: Main exports (`order`, `fit_track`, `WalkConfig`, result
    types)
  - `orderers.py`: Orderer classes, `default_pipeline`, and `OrderingResult`
  - `som.py`: Self-Organizing Map
  - `metrics.py`: Distance metric classes
  - `strats.py`: Query strategy classes
  - `nn.py`: Neural network module
  - `w.py`: Phase-space utilities (distances, directions, similarities)
- `/src/phasecurvefit/_src/`: Private implementation
  - `orderers/`: `order()`, `AbstractOrderer`, the orderers, `default_pipeline`,
    `OrderingResult`
  - `algorithm.py`: Local-flow walk implementation, `StateMetadata`
  - `som.py`: SOM implementation
  - `pipeline.py`: `fit_track`
  - `nn/`: Neural networks, training, and membership (Equinox)
  - `metrics.py`: Metric base classes and implementations
  - `strategies.py`: Query strategy classes
  - `query_config.py`: `WalkConfig`
  - `phasespace.py`: Phase-space operations
- `/src/phasecurvefit/_interop/`: Optional dependency integrations
  - `interop_unxt.py`: `unxt` Quantity support via Quax dispatch
- `/docs/guides/`: User guides (quickstart, orderers, SOM, metrics, outliers,
  JAX integration, etc.); `/docs/migration/` records API changes per version
- `/tests/`: `unit/`, `integration/`, `smoke/`, `static/` (source-sweep checks),
  `benchmarks/`, and `usage/`

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
- Tests for Quantity support in `tests/unit/test_quantity_support.py`,
  `tests/unit/test_interop_unxt.py`, and `tests/unit/test_orderers_unxt.py`
- Tests run with beartype runtime type checking and warnings as errors
- Assertions should be atomic (no `assert a and b`, use separate asserts)

## Architecture Notes

### Result Structure

`OrderingResult` (returned by all orderers) contains:

- `indices`: Array of indices in order (`-1` for unvisited slots; never use
  these directly as indices)
- `positions`: Original (not reordered) position dict
- `velocities`: Original velocity dict
- `gamma_range`: Valid range of the ordering parameter for `__call__`
- `backbone`: Optional ordered polyline (e.g. from `MSTOrderer`); `__call__`
  interpolates along it, or along the visited observations if `None`

`WalkLocalFlowResult` (from `LocalFlowOrderer`) is a thin subclass.

## Final Notes

- Preserve JAX compatibility above all — functions must work with `jit`, `vmap`
- Use dict-based APIs for flexibility with different coordinate systems
- When extending metrics or strategies, follow existing patterns
- Documentation examples must be executable (tested via Sybil)
