# Agent guidelines

Project overview, layout and the main coding conventions live in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md); read it
first. This file adds rules for coding agents.

## Function signatures

- **Private functions (leading `_`) always take their parameters as
  positional-only** (`/`), never as position-or-keyword. Options may still be
  keyword-only (after `*`).

  ```python
  # Good
  def _unit_rows(x: Float[Array, "N D"], /) -> Float[Array, "N D"]:
      ...


  def _pad_to_multiple(
      mask: Bool[Array, " N"], /, *args: Array, batch_size: int, pad_value: float
  ) -> ...:
      ...


  # Bad: `x` can be passed by keyword, so renaming it breaks callers
  def _unit_rows(x: Float[Array, "N D"]) -> Float[Array, "N D"]:
      ...
  ```

  Positional-only parameters keep the name out of the call contract, so internal
  helpers can rename arguments freely, and they match how JAX and Plum call
  functions (positionally).
