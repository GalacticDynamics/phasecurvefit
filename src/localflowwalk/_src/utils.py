from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

# A (very) general PyTree type: any nested structure of JAX arrays/pytrees.
_T = TypeVar("_T")
_BoolScalar = jax.Array  # convention: shape () boolean array


def bounded_while_loop(
    cond_fn: Callable[[_T], Any],
    body_fn: Callable[[_T], _T],
    init_val: _T,
    *,
    max_steps: int,
) -> _T:
    r"""Reverse-mode-friendly, bounded `while_loop` implemented via `lax.scan`.

    This function emulates:

    ```python
    val = init_val
    while cond_fn(val):
        val = body_fn(val)
    return val
    ```

    but with two crucial differences:

    1. **A hard iteration bound**: the loop is unrolled as a fixed-length `scan`
       of length `max_steps`. This is often much friendlier to reverse-mode AD
       than an unbounded `lax.while_loop`.
    2. **Early stop without wasted work**: once the user condition fails
       (i.e. `cond_fn(val)` becomes `False`), we stop applying `body_fn` and
       run only a no-op for the remaining scan steps. This preserves the fixed
       length required by `scan` *without* performing unnecessary computation.

    If the user condition is still `True` after `max_steps` iterations (i.e. the
    loop would continue), an error is raised using `equinox.error_if`.

    Parameters
    ----------
    cond_fn
        Predicate of the loop, in the same sense as `jax.lax.while_loop`:
        it should return a boolean scalar indicating whether to **continue**
        iterating. The loop halts when this becomes `False`.
    body_fn
        Loop body, mapping the loop carry to a new carry.
    init_val
        Initial loop carry (any PyTree of JAX arrays / scalars / nested containers).
    max_steps
        Maximum number of iterations to attempt. Must be a non-negative Python int.

    Returns
    -------
    _T
        Final carry value, either when `cond_fn` first returns `False`, or (if
        that never happens) after `max_steps` iterations (but in that case an
        error is raised).

    Notes
    -----
    Semantics and implementation details:

    We convert the unbounded while loop into a bounded scan by augmenting the
    carry with a boolean flag `done`:

    - `done == False` means we are still logically inside the while loop.
    - `done == True` means the loop has logically terminated; remaining scan
      steps must be no-ops.

    At each scan step we do:

    - If `done` is already `True`: do nothing (no-op).
    - Else (not done):
        - Evaluate `continue_ = cond_fn(val)`.
        - If `continue_` is `True`: apply `body_fn`.
        - If `continue_` is `False`: mark `done = True` and do *not* apply
          `body_fn`.

    After the scan finishes, if `done` is still `False`, then `cond_fn` never
    became false within the allowed steps, meaning the bounded loop
    “overflowed”.  We then raise via `eqx.error_if`.

    Notes on efficiency:

    - The remaining post-termination scan iterations are routed through a branch
      that returns the carry unchanged. At runtime this avoids executing
      `body_fn` after termination.
    - `body_fn` and `cond_fn` are still traced/compiled as part of the JAX
      program (that is unavoidable), but they are not *executed* once
      `done=True`.

    """
    if not isinstance(max_steps, int) or max_steps < 0:
        msg = "max_steps must be a non-negative Python int."
        raise ValueError(msg)

    # Trivial bound: no iterations allowed.
    if max_steps == 0:
        # Mirror the semantics: we didn't even check cond_fn; we just return init.
        return init_val

    def scan_step(
        carry: tuple[_T, _BoolScalar], _unused: Any
    ) -> tuple[tuple[_T, _BoolScalar], None]:
        """One bounded step.

        carry
            (val, done) where:
            - val: the user loop carry
            - done: whether the loop has already terminated
        """
        val, done = carry

        def already_done(_: Any) -> tuple[_T, _BoolScalar]:
            # No-op: preserve carry; remain done.
            return val, done

        def not_done(_: Any) -> tuple[_T, _BoolScalar]:
            # We are still "in the loop": check whether to continue.
            continue_ = jnp.asarray((cond_fn(val)), dtype=bool)

            def do_body(_: Any) -> tuple[_T, _BoolScalar]:
                # Continue: apply body; still not done.
                return body_fn(val), jnp.asarray(False)  # noqa: FBT003

            def stop_now(_: Any) -> tuple[_T, _BoolScalar]:
                # Stop: mark done, and do not run body.
                return val, jnp.asarray(True)  # noqa: FBT003

            # If continue_ is True, run body. Otherwise terminate (done=True).
            return lax.cond(continue_, do_body, stop_now, operand=None)

        # If we've already terminated, skip everything. Otherwise, proceed as above.
        new_val, new_done = lax.cond(done, already_done, not_done, operand=None)
        return (new_val, new_done), None

    # Carry includes the termination flag. `done` starts False: we have not terminated.
    init_carry: tuple[_T, _BoolScalar] = (init_val, jnp.asarray(False))  # noqa: FBT003

    # Run for exactly max_steps steps. `_` is a dummy scan “sequence” input.
    (final_val, final_done), _ = lax.scan(
        scan_step, init_carry, xs=None, length=max_steps
    )

    # If final_done is False, then cond_fn never became False within max_steps,
    # meaning the corresponding while-loop would still be continuing.
    final_done = eqx.error_if(
        final_done,
        jnp.logical_not(final_done),
        "bounded_while_loop exceeded max_steps without cond_fn becoming False.",
    )

    return final_val
