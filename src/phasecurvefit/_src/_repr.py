"""Print behavior for phasecurvefit objects, via `wadler_lindig_parts`.

`ReprMixin` sends ``repr``, ``str`` and ``format`` through one engine. ``repr``
is the constructor form with arrays shown by dtype and shape, broken at 80
columns (the spec ``call-type-width=80``, which matches `equinox.Module`'s
repr). A format spec picks any other form the engine knows:

>>> import jax.numpy as jnp
>>> import phasecurvefit as pcf
>>> cfg = pcf.WalkConfig(strategy=pcf.strats.BruteForce())
>>> cfg
WalkConfig(metric=AlignedMomentumDistanceMetric(), strategy=BruteForce())
>>> print(f"{cfg:call-width=40}")
WalkConfig(
  metric=AlignedMomentumDistanceMetric(),
  strategy=BruteForce()
)

>>> meta = pcf.StateMetadata(scale=jnp.array([1.0, 2.0]))
>>> meta
StateMetadata(scale=f32[2])
>>> f"{meta:call-array}"
'StateMetadata(scale=Array([1., 2.], dtype=float32))'

"""

__all__: tuple[str, ...] = ("ReprMixin", "attrs_pdoc")

from collections.abc import Iterable
from typing import Any

import jax
import wadler_lindig as wl
import wadler_lindig_parts as wlp

#: The spec ``repr`` and ``str`` use.
_REPR_SPEC = wlp.parse_spec("call-type-width=80")


def _short_jax_array(obj: object, /) -> wl.AbstractDoc | None:
    """Show a JAX array as ``f32[3]``, as `equinox.tree_pformat` does."""
    if not isinstance(obj, jax.Array):
        return None
    dtype = obj.dtype.name
    if getattr(obj, "weak_type", False):
        dtype = f"weak_{dtype}"
    return wl.array_summary(obj.shape, dtype, kind=None)


def _render(obj: object, spec: wlp.Spec, /) -> str:
    """Render ``obj``; under ``type``, JAX arrays drop wadler-lindig's ``(jax)``."""
    extra = {"custom": _short_jax_array} if spec["value"] == "type" else {}
    return wlp.render(obj, spec, **extra)


class ReprMixin:
    """Route ``repr``, ``str`` and ``format`` through `wadler_lindig_parts`.

    Put it first among the bases, so it wins over `equinox.Module.__repr__`.
    ``repr`` renders the object's `wadler_lindig` document: the dataclass
    fields for an Equinox module, or the object's own ``__pdoc__``. A class
    that is neither must define ``__pdoc__``, for example with `attrs_pdoc`.

    ``format`` takes a `wadler_lindig_parts` spec: ``call`` for the
    constructor form, ``array`` / ``values`` / ``type`` for how much of each
    array to show, ``width=<n>`` / ``indent=<n>`` for line breaking, and
    ``html`` / ``latex`` for markup. An empty spec is ``str``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return _render(self, _REPR_SPEC)

    def __str__(self) -> str:
        return _render(self, _REPR_SPEC)

    def __format__(self, spec: str, /) -> str:
        if not spec:
            return str(self)
        return _render(self, wlp.parse_spec(spec, obj=self))


@wlp.pparts.dispatch
def _pparts_repr_mixin(
    obj: ReprMixin,
    /,
    *,
    markup: str = "text",  # the engine applies the markup
    short_arrays: Any = "compact",
    value_spec: str | None = None,
    custom: Any = None,
    **kw: Any,  # axes these objects do not act on
) -> tuple[Any, ...]:
    """Show the whole object as one fragment: its constructor form."""
    if value_spec is not None:
        msg = (
            f"{type(obj).__name__} does not take a value format spec "
            f"({value_spec!r}): it holds several values, not one"
        )
        raise TypeError(msg)
    extra = {} if custom is None else {"custom": custom}
    doc = wl.pdoc(obj, indent=2, short_arrays=short_arrays, **extra)
    return (wlp.doc_part("value", doc),)


def attrs_pdoc(
    obj: object, items: Iterable[tuple[str, Any]], /, **kw: Any
) -> wl.AbstractDoc:
    """Return a ``Name(key=value, ...)`` document for a non-dataclass object."""
    return wl.bracketed(
        begin=wl.TextDoc(f"{type(obj).__name__}("),
        docs=wl.named_objs(list(items), **kw),
        sep=wl.comma,
        end=wl.TextDoc(")"),
        indent=kw.get("indent", 2),
    )
