"""Input checks: ``ValueError`` on concrete inputs, ``eqx.error_if`` when traced."""

__all__: tuple[str, ...] = ("value_error_if",)

import equinox as eqx
import jax


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
