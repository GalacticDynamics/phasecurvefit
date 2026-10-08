---
name: code-review
description: >-
  Review pull requests to phasecurvefit, a JAX-native library for ordering
  stellar-stream phase-space data. Use when reviewing any change under src/,
  tests/, or docs/: checks JAX tracing safety, numerical robustness (NaN
  gradients, integer overflow, empty/small inputs), unxt Quantity support, plum
  dispatch, the order()/orderer API, and project style rules.
---

# phasecurvefit code review

Review for **correctness first**. Style is enforced by `ruff`, `mypy` and
pre-commit in CI — only comment on style the tools cannot catch (listed under
"Project rules"). Prefer a few high-confidence comments over many speculative
ones. Every comment should name the concrete input that breaks and what happens.

## How to review

1. Read the whole changed function, not just the diff hunk. Trace the change to
   its callers (`order()`, the orderers in `_src/orderers/`, `ChainOrderer`,
   `default_pipeline`, `fit_track`, the `nn` trainers) and to any
   `_interop/interop_unxt.py` dispatch for it.
2. **A defect is a sample of a class.** When you find a bug, search the same
   module (and siblings) for the same shape — the same risky call, the same
   index arithmetic — and flag every instance, not just the first. In this repo
   the unflagged sibling has repeatedly been the real bug.
3. Check that a test exercises the failing case. A regression test should hit
   the edge case directly, not just the happy path the fix was reproduced on.

## High-risk bug classes (check every PR)

### JAX tracing and transforms

- Python control flow on traced values (`if x > 0`, `int(arr)`, `bool(...)`,
  `len()` of a data-dependent shape, `.item()`) inside code that is `jit`/`vmap`
  /`scan`-ed. Must use `jnp.where`, `lax.cond`, `lax.while_loop`, etc.
- Data-dependent output shapes (`jnp.nonzero`, boolean-mask indexing,
  `jnp.unique` without `size=`) inside jitted code.
- Equinox modules: array fields vs static fields. A Python int/str/callable that
  changes behaviour must be `eqx.field(static=True)`; a value that should be
  traced must not be static (causes recompiles or tracer leaks).
- In-place mutation of arrays or of the input `position`/`velocity` dicts.
- Randomness: PRNG keys must be split, never reused across calls.

### Numerical robustness

- **NaN gradients:** `jnp.linalg.norm`, `jnp.sqrt`, division, `arccos`, `log` on
  values that can be exactly zero — e.g. coincident/duplicated points or zero
  velocities, which occur in real data. Forward pass may be fine while `grad` is
  NaN. Look for the safe-norm/`jnp.where` double-where pattern, and check
  _every_ norm in the module, not just one.
- **int32 overflow:** index arithmetic like `arange(m) * n`, `i * n + j`,
  pairwise-count products. JAX defaults to int32; flag products that can exceed
  2^31 for realistic N·K.
- `-1` sentinel indices (skipped tracers) used directly to index an array —
  `x[-1]` silently reads the last element instead of failing.
- Small inputs: N < k for `KDTree(k=...)`, N = 1 or 2, all points skipped, empty
  result. `k` must mean usable neighbours (the query point itself is excluded).
- Dtype handling: float32 vs float64 (`jax_enable_x64`), integer inputs silently
  truncated.

### API and dispatch

- Ordering goes through `pcf.order(pos, vel, orderer)`; `walk_local_flow` was
  removed in v0.4 — flag any reintroduction. With no orderer, `order()` runs
  `default_pipeline` (`MSTOrderer() | SOMOrderer()`); changes to either stage
  change the default for every user.
- Orderers composed with `|` (`ChainOrderer`) receive the previous stage's
  result as `init`. A new orderer must accept `init` to sit after another stage,
  and must not silently discard it. `ChainOrderer.order` stays a plain method,
  not a plum dispatch.
- Never use a `StateMetadata()` (or other mutable object) as a parameter default
  — it is shared across calls. Use `None` and construct in the body;
  `tests/static/` sweeps the source for this.
- New options must default to an exact no-op so existing behaviour is unchanged
  (e.g. `edge_clip_sigma=None`).
- plum dispatch: adding a method — is there already one for those types? Do
  `Array` and `unxt.Quantity` overloads both exist and agree?
- **Quantities must flow through.** Code must not strip units at the API level.
  Unit stripping is only allowed right before an external library that can't
  take Quantities (e.g. `jaxkd`), with units restored afterwards. Mixing a
  Quantity with a bare float (e.g. `metric_scale`) must be dimensionally
  correct. Quantity inputs need `metadata=StateMetadata(usys=...)`; examples and
  tests using Quantities must pass it.
- Public symbols: exported in the right `__all__` and from the right public
  module (e.g. `pcf.strats.BruteForce`, not `pcf.BruteForce`).
- Physics/statistics formulas (metrics, membership probabilities,
  sigma-clipping): check against the cited reference; ask for the citation if
  missing.

## Project rules the linters don't catch

- `__all__` is a **tuple** and is defined **before** imports (only
  `from __future__` may precede it — and that import is banned, see next).
- Never `from __future__ import annotations` (breaks plum and runtime type
  checking).
- Type hints everywhere (`jaxtyping` for arrays). Tests run with beartype
  enabled, so wrong annotations become runtime failures.
- Phase-space naming: `q` position, `p` velocity/momentum, `w` full point.
- Tests: one condition per `assert` (no `assert a and b`); test `jit`/`vmap`
  /`grad` where applicable; Quantity paths covered in `tests/unit/*unxt*` /
  `tests/unit/test_quantity_support.py`. New tests go in the matching
  `tests/unit/`, `integration/`, `smoke/` or `static/` folder. Warnings are
  errors in pytest, so new warnings need handling.
- Docstring and `docs/` examples are executed by Sybil — check they would run
  and that their printed output matches.
- New user-facing behaviour needs a docs update in `docs/guides/`; API breaks
  need an entry in the matching `docs/migration/vX-to-vY.md`.

## Don't comment on

- Formatting, import order, or anything `ruff`/`mypy` will flag.
- Dependency bump PRs from Dependabot, beyond obvious breakage.
- Hypothetical refactors unrelated to the change.
