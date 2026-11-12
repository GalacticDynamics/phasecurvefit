"""Tests for per-observation weights through the orderers and the networks.

A density field enters as one observation per pixel, weighted by the density
there. The oracle throughout is that weight ``w`` must act like ``w`` copies of
an observation: two parallel lines weighted 3:1 have a weighted centre line at
a quarter of the way across, not halfway.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phasecurvefit as pcf
from phasecurvefit._src.checks import check_weights
from phasecurvefit._src.nn import autoencoder as ae_mod
from phasecurvefit._src.nn.order_net import _cumulative_weight_gamma

pytest.importorskip("scipy")


def _arc(n: int = 60):
    ang = jnp.linspace(0.0, jnp.pi, n)
    pos = {"x": 5.0 * jnp.cos(ang), "y": 5.0 * jnp.sin(ang)}
    vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    return pos, vel


def _two_lines(n: int = 100, gap: float = 1.0):
    """Two parallel lines, y=0 weighted 3 and y=gap weighted 1.

    With ``gap`` well below the x spacing (0.1) the MST zigzags across the two
    lines and reads them as one stream; with ``gap=1`` it reads them as a
    hairpin, so only orderers that do not start from the MST see one stream.
    """
    x = jnp.tile(jnp.linspace(0.0, 10.0, n), 2)
    y = jnp.concatenate([jnp.zeros(n), jnp.full(n, gap)])
    w = jnp.concatenate([jnp.full(n, 3.0), jnp.ones(n)])
    pos = {"x": x, "y": y}
    vel = {"x": jnp.ones(2 * n), "y": jnp.zeros(2 * n)}
    return pos, vel, w


# ---------------------------------------------------------------------------
# Validation


class TestCheckWeights:
    """Tests for check_weights."""

    def test_returns_float(self):
        """Integer weights come back as floats."""
        w = check_weights(jnp.array([1, 2, 0]), 3)
        assert jnp.issubdtype(w.dtype, jnp.floating)

    @pytest.mark.parametrize(
        ("weights", "match"),
        [
            (jnp.ones(4), "shape"),
            (jnp.array([1.0, -1.0, 1.0]), "non-negative"),
            (jnp.array([1.0, jnp.nan, 1.0]), "finite"),
            (jnp.zeros(3), "positive"),
        ],
    )
    def test_rejects(self, weights, match):
        """Wrong shape, negative, non-finite and all-zero weights raise."""
        with pytest.raises(ValueError, match=match):
            check_weights(weights, 3)


# ---------------------------------------------------------------------------
# Orderers


ORDERERS = [
    pytest.param(None, id="default"),
    pytest.param(pcf.orderers.MSTOrderer(), id="mst"),
    pytest.param(pcf.orderers.LocalFlowOrderer(metric_scale=1.0), id="localflow"),
    pytest.param(pcf.orderers.SOMOrderer(n_prototypes=10), id="som"),
    pytest.param(
        pcf.orderers.MSTOrderer() | pcf.orderers.SOMOrderer(n_prototypes=10),
        id="chain",
    ),
]


class TestOrderersCarryWeights:
    """Every orderer accepts weights and carries them on its result."""

    @pytest.mark.parametrize("orderer", ORDERERS)
    def test_result_carries_weights(self, orderer):
        """``result.weights`` is what was passed in."""
        pos, vel = _arc()
        w = jnp.linspace(1.0, 2.0, 60)
        result = pcf.order(pos, vel, orderer, weights=w)
        np.testing.assert_allclose(result.weights, w)

    @pytest.mark.parametrize("orderer", ORDERERS)
    def test_no_weights_gives_none(self, orderer):
        """Without weights the result's ``weights`` is ``None``."""
        pos, vel = _arc()
        assert pcf.order(pos, vel, orderer).weights is None

    @pytest.mark.parametrize("orderer", ORDERERS)
    def test_invalid_weights_raise(self, orderer):
        """Every orderer validates the weights it is given."""
        pos, vel = _arc()
        with pytest.raises(ValueError, match="shape"):
            pcf.order(pos, vel, orderer, weights=jnp.ones(59))

    def test_mst_ignores_weights(self):
        """The MST tree is geometric: weights do not change the ordering."""
        pos, vel = _arc()
        mst = pcf.orderers.MSTOrderer()
        plain = pcf.order(pos, vel, mst)
        weighted = pcf.order(pos, vel, mst, weights=jnp.linspace(1.0, 5.0, 60))
        np.testing.assert_array_equal(plain.indices, weighted.indices)

    def test_mst_traces_with_weights(self):
        """MSTOrderer still runs under jit when given weights."""
        pos, vel = _arc()
        mst = pcf.orderers.MSTOrderer()
        result = jax.jit(lambda w: pcf.order(pos, vel, mst, weights=w))(jnp.ones(60))
        np.testing.assert_allclose(result.weights, jnp.ones(60))


class TestSOMWeights:
    """SOMOrderer fits the weighted density."""

    def test_weighted_centre_line(self):
        """Lines at y=0 and y=1 weighted 3:1 give a backbone at y = 1/4."""
        pos, vel, w = _two_lines()
        som = pcf.orderers.SOMOrderer(n_prototypes=10, n_epochs=20)
        result = som.order(pos, vel, weights=w)
        assert result.backbone is not None
        y = np.asarray(result.backbone["y"])
        np.testing.assert_allclose(y, 0.25, atol=1e-3)

    def test_unweighted_centre_line(self):
        """Control: without weights the same backbone sits at y = 1/2."""
        pos, vel, _ = _two_lines()
        som = pcf.orderers.SOMOrderer(n_prototypes=10, n_epochs=20)
        backbone = som.order(pos, vel).backbone
        assert backbone is not None
        y = np.asarray(backbone["y"])
        np.testing.assert_allclose(y, 0.5, atol=1e-3)

    def test_constant_weights_change_nothing(self):
        """Scaling every weight by the same factor leaves the fit unchanged."""
        pos, vel = _arc()
        som = pcf.orderers.SOMOrderer(n_prototypes=10)
        plain = som.order(pos, vel)
        scaled = som.order(pos, vel, weights=jnp.full(60, 7.0))
        np.testing.assert_array_equal(plain.indices, scaled.indices)
        assert plain.backbone is not None
        assert scaled.backbone is not None
        np.testing.assert_allclose(
            plain.backbone["x"], scaled.backbone["x"], rtol=1e-5, atol=1e-5
        )

    def test_weights_reach_chained_som(self):
        """A chain hands its weights to the SOM stage, not just the result."""
        pos, vel, w = _two_lines(gap=0.04)
        chain = pcf.orderers.MSTOrderer() | pcf.orderers.SOMOrderer(
            n_prototypes=10, n_epochs=20
        )
        y = np.asarray(pcf.order(pos, vel, chain, weights=w).backbone["y"])
        np.testing.assert_allclose(y, 0.25 * 0.04, atol=2e-3)

    @staticmethod
    def _two_lines_with_background(n_bg: int = 20):
        """Lines at y=0 (weight 3) and y=0.2 (weight 1), and a flat background."""
        pos, vel, w = _two_lines(gap=0.2)
        rng = np.random.default_rng(0)
        pos = {
            "x": jnp.concatenate([pos["x"], jnp.asarray(rng.uniform(0, 10, n_bg))]),
            "y": jnp.concatenate([pos["y"], jnp.asarray(rng.uniform(-2.5, 2.5, n_bg))]),
        }
        vel = {k: jnp.concatenate([v, jnp.ones(n_bg) * v[0]]) for k, v in vel.items()}
        return pos, vel, jnp.concatenate([w, jnp.ones(n_bg)])

    @pytest.mark.parametrize(("weighted", "centre"), [(True, 0.05), (False, 0.1)])
    def test_mixture_weighted_centre_line(self, weighted, centre):
        """With the outlier mixture on, the backbone is the 3:1 weighted centre.

        The background is rejected, the two lines are kept, and the weights
        still set where between them the backbone runs.
        """
        pos, vel, w = self._two_lines_with_background()
        som = pcf.orderers.SOMOrderer(
            n_prototypes=10, n_epochs=20, outlier_mixture=pcf.som.SOMMixture()
        )
        result = som.order(pos, vel, weights=w if weighted else None)
        assert result.backbone is not None
        np.testing.assert_allclose(result.backbone["y"], centre, atol=5e-3)
        visited = np.zeros(220, bool)
        idx = np.asarray(result.indices)
        visited[idx[idx >= 0]] = True
        assert visited[:200].all()  # both lines kept
        assert (~visited[200:]).sum() >= 15  # most of the background rejected


# ---------------------------------------------------------------------------
# Networks


class TestCumulativeWeightGamma:
    """Tests for _cumulative_weight_gamma."""

    def test_equal_weights_is_linspace(self):
        """Equal weights reproduce ``linspace``."""
        g = _cumulative_weight_gamma(jnp.full(7, 2.0), (-1.0, 1.0))
        np.testing.assert_allclose(g, jnp.linspace(-1.0, 1.0, 7), atol=1e-6)

    def test_heavy_tracer_takes_more_room(self):
        """A weight-2 tracer sits where two weight-1 tracers would, on average."""
        g = _cumulative_weight_gamma(jnp.array([1.0, 2.0, 1.0]), (0.0, 1.0))
        # Copies at ranks 0, 1, 2, 3 of 4: the middle tracer is ranks 1-2.
        np.testing.assert_allclose(g, [0.0, 0.5, 1.0], atol=1e-6)

    def test_zero_spread_falls_back_to_uniform(self):
        """All-zero ordered weights give uniform spacing, not a constant."""
        g = _cumulative_weight_gamma(jnp.zeros(5), (-1.0, 1.0))
        np.testing.assert_allclose(g, jnp.linspace(-1.0, 1.0, 5), atol=1e-6)


class TestRunningMeanDecoderWeights:
    """Tests for the weighted running mean."""

    def test_weighted_mean(self):
        """Two members in the window, weighted 3:1, average to 1/4."""
        dec = pcf.nn.RunningMeanDecoder(
            window_size=1.0,
            gamma_train=jnp.array([0.0, 0.1]),
            positions_train=jnp.array([[0.0], [1.0]]),
            member_train=jnp.array([True, True]),
            weights_train=jnp.array([3.0, 1.0]),
        )
        np.testing.assert_allclose(dec(jnp.array(0.05)), [0.25])


class _NoWeightsDecoder(pcf.nn.RunningMeanDecoder):
    """A decoder whose ``update`` predates the ``weights`` keyword."""

    def update(self, model, all_ws):  # type: ignore[override]
        return pcf.nn.RunningMeanDecoder.update(self, model, all_ws)


class TestExternalDecoderWithoutWeights:
    """An external decoder without ``weights`` still trains unweighted."""

    def test_update_without_weights_kwarg(self):
        """``update`` is called without ``weights=`` when none are given."""
        pos, vel = _arc()
        model = pcf.nn.EncoderExternalDecoder(
            pcf.nn.OrderingNet(in_size=4, width_size=16, depth=1, key=jr.key(1)),
            _NoWeightsDecoder(window_size=0.2),
            pcf.nn.StandardScalerNormalizer(pos, vel),
        )
        result = pcf.order(pos, vel, pcf.orderers.MSTOrderer())
        cfg = pcf.nn.OrderingTrainingConfig(n_epochs=2, show_pbar=False)
        ae, _, _ = pcf.nn.train_autoencoder(model, result, config=cfg, key=jr.key(2))
        assert ae.gamma.shape == (60,)


def _fast_config(**kw):
    return pcf.nn.TrainingConfig(
        n_epochs_encoder=400,
        n_epochs_decoder=300,
        n_epochs_both=0,
        batch_size=200,
        show_pbar=False,
        **kw,
    )


class TestTrainAutoencoderWeights:
    """train_autoencoder honours weights from the result or the call."""

    def _model(self, pos, vel, result):
        return pcf.nn.PathAutoencoder.make(
            normalizer=pcf.nn.StandardScalerNormalizer(pos, vel),
            gamma_range=result.gamma_range,
            key=jr.key(0),
        )

    def test_weighted_track(self):
        """The decoded track follows the 3:1 weighted centre line."""
        pos, vel, w = _two_lines()
        result = pcf.orderers.SOMOrderer(n_prototypes=10).order(pos, vel, weights=w)
        ae, _, losses = pcf.nn.train_autoencoder(
            self._model(pos, vel, result), result, config=_fast_config(), key=jr.key(1)
        )
        assert bool(jnp.all(jnp.isfinite(losses)))
        y = ae(jnp.linspace(-0.8, 0.8, 9))["y"]
        np.testing.assert_allclose(y, 0.25, atol=0.1)

    def test_call_weights_override_result(self):
        """``weights=`` on the call takes precedence over ``result.weights``."""
        pos, vel, w = _two_lines()
        result = pcf.orderers.SOMOrderer(n_prototypes=10).order(pos, vel, weights=w)
        ae, _, _ = pcf.nn.train_autoencoder(
            self._model(pos, vel, result),
            result,
            config=_fast_config(),
            weights=jnp.ones(200),
            key=jr.key(1),
        )
        y = ae(jnp.linspace(-0.8, 0.8, 9))["y"]
        np.testing.assert_allclose(y, 0.5, atol=0.1)

    def test_joint_weights_keep_density_scale(self, monkeypatch):
        """Phase-3 weights keep the masked mean the density weights have alone.

        Per-tracer weights correlated with the density weights must not shift
        the overall loss scale, so the product is rescaled after combining.
        """
        pos, vel, w = _two_lines()
        result = pcf.orderers.SOMOrderer(n_prototypes=10).order(pos, vel, weights=w)
        model = self._model(pos, vel, result)
        qs, ps = model.normalizer.transform(pos, vel)
        ws = jnp.concatenate([qs, ps], axis=1)
        mask = jnp.ones(200, dtype=bool)
        density_w = jnp.where(jnp.arange(200) < 100, 4.0, 1.0)  # correlated with w
        monkeypatch.setattr(ae_mod, "compute_weights", lambda *_, **__: density_w)

        seen = {}

        def _capture(self, model, data, mask, /, **kw):
            _padded_ws, seen["w"] = data
            msg = "stop"
            raise RuntimeError(msg)

        monkeypatch.setattr(ae_mod.PathAutoencoderTrainer, "init", _capture)
        cfg = pcf.nn.TrainingConfig(
            n_epochs_both=1, weight_by_density=True, show_pbar=False
        ).autoencoder_config()
        with pytest.raises(RuntimeError, match="stop"):
            ae_mod.train_ordering_and_track_net(
                model, ws, mask=mask, config=cfg, weights=w, key=jr.key(0)
            )
        got = np.asarray(seen["w"])[:200]
        np.testing.assert_allclose(got.mean(), density_w.mean(), rtol=1e-6)
        # Shape of the weighting is the product: 4*3 on line 0, 1*1 on line 1.
        np.testing.assert_allclose(got[0] / got[-1], 12.0, rtol=1e-6)

    def test_mixture_weighted_track(self):
        """With mixture membership the track follows the 3:1 weighted centre line.

        A near-zero background makes every pixel a member, so the mixture
        reduces to a weighted Gaussian fit and its track is the weighted mean.
        """
        pos, vel, w = _two_lines()
        result = pcf.orderers.SOMOrderer(n_prototypes=10).order(pos, vel, weights=w)
        cfg = pcf.nn.TrainingConfig(
            n_epochs_encoder=400,
            n_epochs_decoder=300,
            n_epochs_both=200,
            batch_size=200,
            show_pbar=False,
            freeze_encoder_final_training=True,
            membership=pcf.nn.MixtureMembershipConfig(
                sigma_ceiling=(4.0, 1.0), sigma_init=1.0, background_density=1e-6
            ),
        )
        ae, _, losses = pcf.nn.train_autoencoder(
            self._model(pos, vel, result), result, config=cfg, key=jr.key(1)
        )
        assert bool(jnp.all(jnp.isfinite(losses)))
        y = ae(jnp.linspace(-0.8, 0.8, 9))["y"]
        np.testing.assert_allclose(y, 0.25, atol=0.05)

    def test_mixture_loss_uses_weights(self):
        """The joint mixture loss scales with the per-tracer weights."""
        pos, vel, _ = _two_lines()
        result = pcf.orderers.SOMOrderer(n_prototypes=10).order(pos, vel)
        model = self._model(pos, vel, result)
        membership = pcf.nn.MixtureMembershipConfig(lambda_velocity=0.0)
        model = eqx.tree_at(
            lambda m: m.width,
            model,
            membership.make_width_net(key=jr.key(2)),
            is_leaf=lambda x: x is None,
        )
        qs, ps = model.normalizer.transform(pos, vel)
        ws = jnp.concatenate([qs, ps], axis=1)
        mask = jnp.ones(200, dtype=bool)

        def loss(weights):
            return float(
                ae_mod.compute_decoder_loss(
                    model,
                    ws,
                    weights,
                    mask,
                    lambda_q=1.0,
                    lambda_p=1.0,
                    member_threshold=0.5,
                    key=jr.key(3),
                    membership=membership,
                    log_bg_density=-2.0,
                )
            )

        assert loss(jnp.full(200, 3.0)) == pytest.approx(3 * loss(jnp.ones(200)))

    def test_fit_track_passes_weights(self):
        """fit_track threads weights through to the ordering it trains on."""
        pos, vel = _arc()
        w = jnp.linspace(1.0, 2.0, 60)
        ordering, fitted, _ = pcf.fit_track(
            pos,
            vel,
            key=jr.key(0),
            weights=w,
            n_prototypes=10,
            training_config=_fast_config(),
        )
        np.testing.assert_allclose(ordering.weights, w)
        assert bool(jnp.all(jnp.isfinite(fitted.gamma)))


# ---------------------------------------------------------------------------
# Quantities


class TestUnxtWeights:
    """Weights pass through the Quantity dispatches."""

    @pytest.mark.parametrize(
        "orderer",
        [
            pcf.orderers.MSTOrderer(),
            pcf.orderers.SOMOrderer(n_prototypes=10),
            pcf.orderers.LocalFlowOrderer(),
        ],
        ids=["mst", "som", "localflow"],
    )
    def test_quantity_result_carries_weights(self, orderer):
        """Quantity inputs keep plain-array weights on the result."""
        u = pytest.importorskip("unxt")
        pos, vel = _arc()
        q = {k: u.Q(v, "kpc") for k, v in pos.items()}
        p = {k: u.Q(v, "km/s") for k, v in vel.items()}
        meta = pcf.StateMetadata(usys=u.unitsystems.galactic)
        w = jnp.linspace(1.0, 2.0, 60)
        result = pcf.order(q, p, orderer, metadata=meta, weights=w)
        np.testing.assert_allclose(result.weights, w)
