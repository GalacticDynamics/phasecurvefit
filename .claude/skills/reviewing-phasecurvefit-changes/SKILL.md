---
name: reviewing-phasecurvefit-changes
description:
  Use when reviewing a diff, branch, or pull request in phasecurvefit, or
  self-reviewing a change before committing or opening a PR — orderers, metrics,
  query strategies, walk_local_flow / order(), nn autoencoders, SOMs, or the
  unxt interop.
---

# Reviewing phasecurvefit changes

## Overview

Generic review misses this repo's real failure modes: code that runs eagerly but
breaks under `jit`/`vmap`/`grad`, silent numeric corruption (NaN gradients,
int32 overflow), and `unxt.Quantity` units stripped at the API. A reported
defect is a sample of a class — review the class.

## Process

1. Get the diff: `git diff main...HEAD`, `gh pr diff <N>`, or `git diff A B`.
   Environment: `uv sync --all-extras` (a fresh worktree has no `.venv`); apply
   an unchecked-out diff in a scratch copy, not the working tree. Read every
   touched file in full, plus each caller of every changed function
   (`grep -rn "<name>(" src tests docs`).
2. For deleted lines, run `git log -L <start>,<end>:<file>` — a "simplification"
   that removes a guard and its explanatory comment is often a revert of a fix.
3. Walk the hazard table below against each changed function.
4. **Sweep siblings.** For every finding, grep the module (and `src/`) for the
   same shape — e.g. every `jnp.linalg.norm`, every index `* n` product, every
   `jaxkd` call. Report each hit that shares the defect, not just the first.
5. Run checks and report actual output, not expectations: `uv run nox -s lint`
   and `uv run nox -s pytest` (covers README, `docs/`, and doctests via Sybil).
   Scope with `uv run pytest tests/test_<x>.py` while iterating.
6. Report findings in the format below.

## Hazards

| Area              | Look for                                                                                                                                                                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JAX tracing       | Python `if`/`for`/`int()`/`bool()` on traced values; shapes derived from data; `.item()`; side effects inside `jit`. Needs a `jit` (and `vmap`/`grad` where claimed) test.                                                                                |
| Gradients         | `jnp.linalg.norm`, `sqrt`, division, `arccos` at zero/coincident points → NaN grad. Duplicated observations are ordinary input.                                                                                                                           |
| Integer width     | `arange(m) * n`, flat indices, `astype(jnp.int32)` — overflow once N·K > 2³¹.                                                                                                                                                                             |
| Off-by-one / self | Neighbor queries returning the query point (not guaranteed at slot 0 when points coincide — mask by index, not `[:, 1:]`); `k` larger than the dataset (default `KDTree(k=50)` on small inputs); `-1` sentinel for skipped indices leaking into indexing. |
| Units (`unxt`)    | Quantities must flow through the algorithm — no `.value`/`ustrip` at the public API. New ops need a plum or Quax dispatch in `_interop/`; check `func.methods` for existing ones. Test in `tests/test_*unxt*.py` / `test_quantity_support.py`.            |
| Public API        | `__all__` tuple defined before imports; no `from __future__ import annotations` (breaks plum); new symbols exported in `src/phasecurvefit/*.py`; deprecated paths (`walk_local_flow`) warn on direct call only, not via `order()`.                        |
| Tests             | Regression test hits a _different_ instance than the reported one; atomic asserts (no `assert a and b`); docs examples must show/assert the result — Sybil only catches crashes, not wrong output.                                                        |

## Output format

Findings ranked most severe first. Each one:

```
[severity: bug | risk | nit] path/to/file.py:LINE — one-sentence defect
  Scenario: concrete input/state → wrong output/crash
  Siblings: other sites with the same shape (or "swept: none")
```

End with the lint/test command output summary. No findings → say so plainly.

## Common mistakes

- Verifying only the reproduction in the PR description, then stopping.
- Trusting eager-mode tests for code that is only ever called under `jit`.
- Suggesting a unit strip "to simplify" — it violates a firm project rule.
- Flagging what `ruff`/`mypy` already enforce instead of running them.
