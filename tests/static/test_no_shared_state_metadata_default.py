"""Static source-sweep check: no function defaults to a shared ``StateMetadata()``."""

import ast
import pathlib

import phasecurvefit as pcf


def _names_state_metadata(func: ast.expr) -> bool:
    """Whether ``func`` names ``StateMetadata``, bare or qualified.

    Both spellings carry the identical hazard, so matching only ``ast.Name``
    would let ``metadata=pcf.StateMetadata()`` reintroduce it unnoticed.
    """
    if isinstance(func, ast.Name):
        return func.id == "StateMetadata"
    return isinstance(func, ast.Attribute) and func.attr == "StateMetadata"


def test_no_function_takes_a_state_metadata_default():
    """``StateMetadata()`` as a default is one instance shared by every call.

    Its *attributes* are frozen -- ``m["usys"] = ...`` raises -- but ``_data``
    is an ordinary dict, so a mutation reaching through that private attribute
    persists into every subsequent call that omits the argument. ``None`` plus
    an in-body construction gives each call its own.

    This sweeps the package source rather than the two sites that had the
    defect, because the hazard is the construct, not the call site. It reads
    the source instead of introspecting objects because these functions are
    ``plum`` dispatches: the module attribute is a ``plum.Function``, which
    ``inspect.isfunction`` rejects -- so an object-level sweep silently skips
    exactly the functions at issue.
    """
    root = pathlib.Path(next(iter(pcf.__path__)))
    offenders = []
    for path in sorted(root.rglob("*.py")):
        # Explicit utf-8, not the platform default: the package source
        # carries non-ASCII (theta, lambda, pi, em dashes), which an
        # ASCII default locale refuses outright and cp1252 silently
        # mis-decodes. ``filename`` puts the real path in any SyntaxError.
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = node.args
            defaults = [
                *args.defaults,
                *(d for d in args.kw_defaults if d is not None),
            ]
            offenders += [
                f"{path.relative_to(root)}:{d.lineno} in {node.name}()"
                for d in defaults
                if isinstance(d, ast.Call) and _names_state_metadata(d.func)
            ]

    assert offenders == []
