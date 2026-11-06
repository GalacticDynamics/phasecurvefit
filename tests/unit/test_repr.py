"""Tests for the print behavior set by `phasecurvefit._src._repr.ReprMixin`."""

import jax.numpy as jnp
import pytest

import phasecurvefit as pcf


@pytest.fixture
def result() -> pcf.orderers.OrderingResult:
    t = jnp.linspace(0, 1, 4)
    pos = {"x": t, "y": t}
    vel = {"x": jnp.ones(4), "y": jnp.ones(4)}
    return pcf.order(pos, vel, pcf.orderers.LocalFlowOrderer())


def test_plain_classes_show_constructor() -> None:
    assert repr(pcf.strats.BruteForce()) == "BruteForce()"
    assert repr(pcf.StateMetadata(usys=None)) == "StateMetadata(usys=None)"


def test_str_matches_repr() -> None:
    cfg = pcf.WalkConfig()
    assert str(cfg) == repr(cfg)


def test_format_array_shows_values(result: pcf.orderers.OrderingResult) -> None:
    assert "f32[4]" in repr(result)
    assert "Array(" in f"{result:call-array}"


def test_format_width() -> None:
    assert "\n" not in f"{pcf.WalkConfig():call-width=200}"
    assert "\n" in f"{pcf.WalkConfig():call-width=20}"


def test_value_spec_is_refused(result: pcf.orderers.OrderingResult) -> None:
    with pytest.raises(TypeError, match="value format spec"):
        format(result, ".2f")


def test_repr_html_escapes_and_matches_repr() -> None:
    meta = pcf.StateMetadata(usys="<si>")
    html = meta._repr_html_()
    assert html.startswith("<pre>")
    assert "&lt;si&gt;" in html
    assert "<si>" not in html


def test_markup_honours_width() -> None:
    cfg = pcf.WalkConfig()
    assert "\n" not in f"{cfg:html-width=200}"
    assert "\n" in f"{cfg:html-width=20}"


def test_weak_typed_array_is_marked() -> None:
    meta = pcf.StateMetadata(scale=jnp.asarray(1.0))
    assert repr(meta) == "StateMetadata(scale=weak_f32[])"


def test_empty_spec_is_str() -> None:
    cfg = pcf.WalkConfig()
    assert f"{cfg}" == str(cfg)


def test_normalizer_shows_attributes() -> None:
    qs = {"x": jnp.array([0.0, 1.0])}
    ps = {"vx": jnp.array([1.0, 2.0])}
    text = repr(pcf.nn.StandardScalerNormalizer(qs, ps))
    assert text.startswith("StandardScalerNormalizer(\n")
    assert "q_comps=['x']," in text
    assert "q_mean=f32[1]," in text
