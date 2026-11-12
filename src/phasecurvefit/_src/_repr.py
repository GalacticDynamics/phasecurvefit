"""Print behavior for phasecurvefit objects, via `wadler_lindig_parts`.

`ReprMixin` is `wadler_lindig_parts.ReprMixin` with phasecurvefit's spec,
``call-type-width=80``. ``repr`` is the constructor form with arrays shown by
dtype and shape, broken at 80 columns, which matches `equinox.Module`'s repr.
A format spec picks any other form the engine knows:

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

A notebook shows the same ``repr``: a constructor expression has no HTML form,
so ``_repr_html_`` returns `None`.

"""

__all__: tuple[str, ...] = ("ReprMixin", "attrs_pdoc")

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypedDict

import jax
import wadler_lindig as wl
import wadler_lindig_parts as wlp


class ReprMixin(wlp.ReprMixin):  # type: ignore[misc]  # wadler_lindig_parts is untyped
    """`wadler_lindig_parts.ReprMixin` with phasecurvefit's ``repr`` spec.

    Put it first among the bases, so it wins over `equinox.Module.__repr__`.
    ``repr`` renders the object's `wadler_lindig` document: the dataclass
    fields for an Equinox module, or the object's own ``__pdoc__``. A class
    that is neither must define ``__pdoc__``, for example with `attrs_pdoc`.

    ``format`` takes a `wadler_lindig_parts` spec: ``call`` for the
    constructor form, ``array`` / ``values`` / ``type`` for how much of each
    array to show, and ``width=<n>`` / ``indent=<n>`` for line breaking. The
    ``product`` specs (``type`` on its own, ``html``, ``latex``) give the same
    expression on one line. An empty spec is ``str``.
    """

    __repr_spec__ = "call-type-width=80"


def _short_jax_array(obj: object, /) -> wl.AbstractDoc | None:
    """Show a JAX array as ``f32[3]``, as `equinox.tree_pformat` does."""
    if not isinstance(obj, jax.Array):
        return None
    dtype = obj.dtype.name
    if getattr(obj, "weak_type", False):
        dtype = f"weak_{dtype}"
    return wl.array_summary(obj.shape, dtype, kind=None)


class _PformatExtra(TypedDict, total=False):
    custom: Callable[[object], wl.AbstractDoc | None]


@wlp.render_options.dispatch  # type: ignore[untyped-decorator]  # untyped plum dispatcher
def _render_options_repr_mixin(
    obj: ReprMixin,  # dispatch only
    /,
    *,
    spec: wlp.Spec,
) -> Mapping[str, Any]:
    """Under ``type``, drop the ``(jax)`` that wadler-lindig adds to arrays."""
    return {"custom": _short_jax_array} if spec["value"] == "type" else {}


@wlp.pparts.dispatch  # type: ignore[untyped-decorator]  # untyped plum dispatcher
def _pparts_repr_mixin(
    obj: ReprMixin,
    /,
    *,
    markup: str = "text",  # the engine applies the markup
    short_arrays: Any = "compact",
    value_spec: str | None = None,
    **kw: Any,  # axes these objects do not act on
) -> tuple[Any, ...]:
    """Show the whole object as one fragment: its constructor form, on one line.

    Plain text, so the engine escapes it for the markup. The ``product`` specs
    are for markup, where a line break would only be collapsed again.
    """
    if value_spec is not None:
        msg = (
            f"{type(obj).__name__} does not take a value format spec "
            f"({value_spec!r}): it holds several values, not one"
        )
        raise TypeError(msg)
    extra: _PformatExtra = {"custom": _short_jax_array} if short_arrays is True else {}
    text = wl.pformat(obj, width=wlp.FLAT, short_arrays=short_arrays, **extra)
    return (wlp.PPart("value", text),)


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
