"""Input checks: ``ValueError`` on concrete inputs, ``eqx.error_if`` when traced."""

__all__: tuple[str, ...] = ("check_nan_policy", "check_velocities", "value_error_if")

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp


def value_error_if[T](x: T, pred: object, msg: str, /) -> T:
    """Raise ``ValueError(msg)`` if ``pred``; return ``x`` unchanged otherwise.

    ``eqx.error_if`` raises ``EquinoxRuntimeError`` (a ``RuntimeError``) even
    on concrete inputs, so the orderers' input checks raised different types
    for the same mistake (``MSTOrderer``: ``ValueError``). On a concrete
    ``pred`` this raises ``ValueError``; under ``jit``/``vmap``/``grad`` it
    defers to ``eqx.error_if``, which fails at run time with a JAX runtime
    error carrying the same ``msg``.
    """
    if isinstance(pred, jax.core.Tracer):
        return eqx.error_if(x, pred, msg)
    if bool(pred):
        raise ValueError(msg)
    return x


def check_nan_policy(nan_policy: object, /) -> None:
    """Raise ``ValueError`` unless ``nan_policy`` is ``"raise"`` or ``"omit"``."""
    if nan_policy not in ("raise", "omit"):
        msg = f"nan_policy must be 'raise' or 'omit', got {nan_policy!r}."
        raise ValueError(msg)


def check_velocities[T: Mapping](
    velocities: T, nan_policy: str, /, *, name: str, nan_msg: str
) -> T:
    """Raise on an infinite velocity, and on NaN unless ``nan_policy="omit"``.

    For orderers that read velocities. Left alone, a non-finite velocity makes
    every distance it touches non-finite. ``name`` is the orderer, for the inf
    message; ``nan_msg`` says how to handle a NaN in that orderer. Uses
    `value_error_if`: a ``ValueError`` on concrete input, and it survives
    ``jit`` and ``vmap``.
    """
    # Per component, then reduced: no (N * D) copy just to validate.
    comps = [jnp.asarray(v) for v in velocities.values()]
    velocities = value_error_if(
        velocities,
        jnp.any(jnp.stack([jnp.any(jnp.isinf(v)) for v in comps])),
        f"{name} found an infinite velocity. inf is not a measurement -- it "
        "comes from an overflow or a bug upstream -- so it raises under either "
        "nan_policy. Fix or drop those tracers.",
    )
    if nan_policy == "omit":
        return velocities
    return value_error_if(
        velocities, jnp.any(jnp.stack([jnp.any(jnp.isnan(v)) for v in comps])), nan_msg
    )
