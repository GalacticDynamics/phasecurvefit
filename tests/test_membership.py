"""Tests for mixture-model membership (outlier rejection).

The behavioural tests build a synthetic arc-shaped stream with known outliers
and check that the model actually finds them. They are the tests that matter:
the unit tests below them only pin the pieces in place.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phasecurvefit as pcf
from phasecurvefit.nn import (
    MixtureMembershipConfig,
    StreamWidthNet,
    membership_rampup,
    membership_responsibility,
    mixture_membership_loss,
    sigma_ceiling,
    uniform_background_density,
)

# ============================================================
# StreamWidthNet


class TestStreamWidthNet:
    """The gamma-dependent stream width network."""

    def test_initialises_near_sigma_init(self) -> None:
        """The net is constructed to predict ~sigma_init everywhere."""
        net = StreamWidthNet(sigma_init=0.25, key=jr.key(0))
        gammas = jnp.linspace(-1.0, 1.0, 32)
        sigmas = jax.vmap(net)(gammas)

        # Exactly sigma_init + sigma_min, since the final layer is zeroed.
        assert jnp.allclose(sigmas, 0.25 + net.sigma_min, atol=1e-6)

    def test_strictly_positive(self) -> None:
        """Widths must never be <= 0, whatever the weights."""
        net = StreamWidthNet(sigma_init=0.1, sigma_min=1e-3, key=jr.key(1))
        # Perturb the weights hard; positivity is structural (exp + floor).
        net = jax.tree_util.tree_map(lambda x: x - 50.0 if eqx.is_array(x) else x, net)
        sigmas = jax.vmap(net)(jnp.linspace(-5.0, 5.0, 64))

        assert bool(jnp.all(sigmas >= net.sigma_min))
        assert bool(jnp.all(jnp.isfinite(sigmas)))

    def test_can_vary_along_track(self) -> None:
        """After training, sigma(gamma) is genuinely gamma-dependent."""
        net = StreamWidthNet(sigma_init=0.1, key=jr.key(2))
        gammas = jnp.linspace(-1.0, 1.0, 64)
        # Target: width doubles along the track (a fanning stream).
        target = 0.1 + 0.1 * (gammas + 1) / 2

        def loss(net):
            return jnp.mean((jax.vmap(net)(gammas) - target) ** 2)

        opt = optax.adam(1e-2)
        st = opt.init(eqx.filter(net, eqx.is_array))
        for _ in range(500):
            g = eqx.filter_grad(loss)(net)
            u, st = opt.update(g, st, eqx.filter(net, eqx.is_array))
            net = eqx.apply_updates(net, u)

        got = jax.vmap(net)(gammas)
        assert jnp.allclose(got, target, atol=0.02)
        assert float(got[-1]) > 1.5 * float(got[0])  # really does vary

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_sigma_init(self, bad: float) -> None:
        """A non-positive initial width is meaningless."""
        with pytest.raises(ValueError, match="sigma_init must be positive"):
            StreamWidthNet(sigma_init=bad, key=jr.key(0))


# ============================================================
# Schedules


class TestSigmaCeiling:
    """The annealing ceiling that stops the width inflating."""

    def test_endpoints(self) -> None:
        """The ceiling starts at `start` and ends at `stop`."""
        assert float(sigma_ceiling(0, 10, start=1.0, stop=0.1)) == pytest.approx(1.0)
        assert float(sigma_ceiling(9, 10, start=1.0, stop=0.1)) == pytest.approx(0.1)

    def test_geometric_midpoint(self) -> None:
        """Geometric, not linear: the midpoint is the geometric mean."""
        mid = float(sigma_ceiling(2, 5, start=1.0, stop=0.01))
        assert mid == pytest.approx(0.1, rel=1e-5)

    def test_monotonically_decreasing(self) -> None:
        """The ceiling only ever anneals downward."""
        vals = np.array(
            [float(sigma_ceiling(i, 20, start=2.0, stop=0.1)) for i in range(20)]
        )
        assert np.all(np.diff(vals) < 0)

    def test_single_epoch(self) -> None:
        """A one-epoch run just uses `start`."""
        assert float(sigma_ceiling(0, 1, start=1.0, stop=0.1)) == pytest.approx(1.0)

    def test_rejects_nonpositive(self) -> None:
        """Widths must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            sigma_ceiling(0, 10, start=1.0, stop=0.0)


class TestMembershipRampup:
    """The warm-up ramp that stops membership collapsing."""

    def test_zero_during_warmup_one_after(self) -> None:
        """The ramp is 0 during warm-up and 1 afterwards."""
        assert float(membership_rampup(0, 10, warmup_frac=0.5)) == pytest.approx(0.0)
        assert float(membership_rampup(5, 10, warmup_frac=0.5)) == pytest.approx(1.0)
        assert float(membership_rampup(9, 10, warmup_frac=0.5)) == pytest.approx(1.0)

    def test_linear_in_between(self) -> None:
        """The ramp rises linearly across the warm-up window."""
        got = float(membership_rampup(2, 10, warmup_frac=0.5))
        assert got == pytest.approx(0.4444, abs=1e-3)

    def test_no_warmup(self) -> None:
        """warmup_frac=0 means the mixture is on from epoch zero."""
        assert float(membership_rampup(0, 10, warmup_frac=0.0)) == pytest.approx(1.0)

    def test_rejects_out_of_range(self) -> None:
        """warmup_frac must be in [0, 1)."""
        with pytest.raises(ValueError, match=r"warmup_frac must be in \[0, 1\)"):
            membership_rampup(0, 10, warmup_frac=1.0)


# ============================================================
# Background density


class TestUniformBackgroundDensity:
    """The flat background density rho_bg."""

    def test_unit_square(self) -> None:
        """A unit square has unit background density."""
        qs = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        assert uniform_background_density(qs) == pytest.approx(1.0)

    def test_scales_as_inverse_volume(self) -> None:
        """Density is the reciprocal of the field volume."""
        qs = jnp.array([[0.0, 0.0], [2.0, 2.0]])
        assert uniform_background_density(qs) == pytest.approx(0.25)

    def test_degenerate_axis_does_not_blow_up(self) -> None:
        """A flat axis would give zero volume -> infinite density. Guard it."""
        qs = jnp.array([[0.0, 0.0], [2.0, 0.0]])  # y is degenerate
        rho = uniform_background_density(qs)
        assert np.isfinite(rho)
        assert rho == pytest.approx(0.5)  # treated as unit extent in y

    def test_inflate(self) -> None:
        """`inflate` scales each axis before taking the volume."""
        qs = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        assert uniform_background_density(qs, inflate=2.0) == pytest.approx(0.25)


# ============================================================
# The loss itself


class TestMixtureMembershipLoss:
    """The mixture negative log-likelihood."""

    def test_responsibility_tracks_residual(self) -> None:
        """On-track -> member; far off -> background. This is the whole point."""
        qs_meas = jnp.array([[0.0, 0.0], [0.0, 9.0]])
        qs_pred = jnp.zeros((2, 2))
        prob = jnp.array([0.5, 0.5])
        sigma = jnp.array([0.1, 0.1])
        mask = jnp.ones(2, dtype=bool)

        _, resp = mixture_membership_loss(
            qs_meas, qs_pred, prob, sigma, mask, log_bg_density=float(jnp.log(0.01))
        )

        assert float(resp[0]) > 0.99
        assert float(resp[1]) < 0.01

    def test_finite_for_absurd_residual(self) -> None:
        """A wildly discrepant star must not produce -inf/NaN."""
        qs_meas = jnp.array([[0.0, 1e6]])
        qs_pred = jnp.zeros((1, 2))
        loss, resp = mixture_membership_loss(
            qs_meas,
            qs_pred,
            jnp.array([0.999999]),
            jnp.array([1e-3]),
            jnp.ones(1, dtype=bool),
            log_bg_density=float(jnp.log(1e-3)),
        )
        assert bool(jnp.isfinite(loss))
        assert bool(jnp.isfinite(resp[0]))

    def test_gradient_is_finite(self) -> None:
        """Including at saturated pi, where the naive log would blow up."""
        qs_meas = jnp.array([[0.0, 0.0], [5.0, 5.0]])
        qs_pred = jnp.zeros((2, 2))
        sigma = jnp.array([0.1, 0.1])
        mask = jnp.ones(2, dtype=bool)

        def f(prob):
            loss, _ = mixture_membership_loss(
                qs_meas, qs_pred, prob, sigma, mask, log_bg_density=-4.6
            )
            return loss

        for p in (0.0, 1.0, 0.5):
            g = jax.grad(f)(jnp.full(2, p))
            assert bool(jnp.all(jnp.isfinite(g))), f"NaN gradient at pi={p}"

    def test_rampup_zero_ignores_membership(self) -> None:
        """At rampup=0 the loss is a pure reconstruction NLL: pi has no effect."""
        qs_meas = jnp.array([[0.0, 0.3]])
        qs_pred = jnp.zeros((1, 2))
        sigma = jnp.array([0.1])
        mask = jnp.ones(1, dtype=bool)

        losses = [
            float(
                mixture_membership_loss(
                    qs_meas,
                    qs_pred,
                    jnp.array([p]),
                    sigma,
                    mask,
                    log_bg_density=-4.6,
                    rampup=0.0,
                )[0]
            )
            for p in (0.01, 0.5, 0.99)
        ]
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)
        assert losses[1] == pytest.approx(losses[2], rel=1e-6)

    def test_mask_excludes_padding(self) -> None:
        """Masked-out rows must not move the loss."""
        qs_pred = jnp.zeros((3, 2))
        prob = jnp.full(3, 0.9)
        sigma = jnp.full(3, 0.1)

        good = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        withjunk = jnp.array([[0.0, 0.0], [0.0, 0.0], [99.0, 99.0]])
        mask = jnp.array([True, True, False])

        l1, _ = mixture_membership_loss(
            good, qs_pred, prob, sigma, mask, log_bg_density=-4.6
        )
        l2, _ = mixture_membership_loss(
            withjunk, qs_pred, prob, sigma, mask, log_bg_density=-4.6
        )

        assert float(l1) == pytest.approx(float(l2), rel=1e-9)

    def test_empty_mask_is_zero_not_nan(self) -> None:
        """An all-False mask must give a finite loss, not NaN."""
        loss, _ = mixture_membership_loss(
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
            jnp.full(2, 0.5),
            jnp.full(2, 0.1),
            jnp.zeros(2, dtype=bool),
            log_bg_density=-4.6,
        )
        assert bool(jnp.isfinite(loss))


class TestMembershipResponsibility:
    """The posterior membership (E-step)."""

    def test_matches_bayes_by_hand(self) -> None:
        """Check the posterior against an explicit computation."""
        prob, r2, sigma, rho = 0.3, 0.04, 0.2, 0.05
        got = float(
            membership_responsibility(
                jnp.array([prob]),
                jnp.array([r2]),
                jnp.array([sigma]),
                log_bg_density=float(np.log(rho)),
                n_dims=2,
            )[0]
        )
        fg = prob * np.exp(-0.5 * r2 / sigma**2) / (2 * np.pi * sigma**2)
        bg = (1 - prob) * rho
        assert got == pytest.approx(fg / (fg + bg), rel=1e-5)


# ============================================================
# Config


class TestMixtureMembershipConfig:
    """Config validation and defaults."""

    def test_default_is_opt_in(self) -> None:
        """The whole feature must be off unless asked for."""
        assert pcf.nn.TrainingConfig().membership is None

    def test_rejects_upward_ceiling(self) -> None:
        """A ceiling that grows would let the width inflate."""
        with pytest.raises(ValueError, match="must anneal downward"):
            MixtureMembershipConfig(sigma_ceiling=(0.1, 0.5))

    def test_rejects_bad_warmup(self) -> None:
        """warmup_frac must be in [0, 1)."""
        with pytest.raises(ValueError, match=r"warmup_frac must be in \[0, 1\)"):
            MixtureMembershipConfig(warmup_frac=1.5)

    def test_resolve_background_density_prefers_explicit(self) -> None:
        """An explicit density wins over the derived one."""
        cfg = MixtureMembershipConfig(background_density=0.123)
        qs = jnp.array([[0.0, 0.0], [100.0, 100.0]])
        assert cfg.resolve_background_density(qs) == pytest.approx(0.123)

    def test_resolve_background_density_derives(self) -> None:
        """With no explicit value, derive it from the field extent."""
        cfg = MixtureMembershipConfig()
        qs = jnp.array([[0.0, 0.0], [2.0, 2.0]])
        assert cfg.resolve_background_density(qs) == pytest.approx(0.25)


# ============================================================
# Behaviour: does it actually find outliers?


@pytest.fixture
def arc_with_outliers():
    """Build an arc-shaped stream (like a tidal tail) plus known outliers."""
    n_in, n_out, sigma = 300, 20, 0.15
    g = jnp.linspace(-1, 1, n_in)
    th = g * 2.4
    track = jnp.stack([6 * jnp.cos(th) - 4, 6 * jnp.sin(th) - 1], -1)
    tang = jnp.stack([-jnp.sin(th), jnp.cos(th)], -1)
    normal = jnp.stack([jnp.cos(th), jnp.sin(th)], -1)

    q_in = track + sigma * jr.normal(jr.key(1), (n_in, 2))
    v_in = tang + 0.05 * jr.normal(jr.key(2), (n_in, 2))

    # Outliers: displaced 15 stream-widths perpendicular to the track, with
    # velocities uncorrelated with the stream.
    idx = jr.randint(jr.key(3), (n_out,), 0, n_in)
    off = 15.0 * sigma * jnp.sign(jr.normal(jr.key(4), (n_out, 1)))
    q_out = track[idx] + off * normal[idx]
    v_out = jr.normal(jr.key(5), (n_out, 2))
    v_out = v_out / jnp.linalg.norm(v_out, axis=1, keepdims=True)

    ws = jnp.concatenate(
        [jnp.concatenate([q_in, q_out]), jnp.concatenate([v_in, v_out])], axis=-1
    )
    gamma = jnp.concatenate([g, g[idx]])
    is_outlier = jnp.concatenate(
        [jnp.zeros(n_in, dtype=bool), jnp.ones(n_out, dtype=bool)]
    )
    return ws, gamma, is_outlier, sigma


def _fit_mixture(ws, gamma_target, cfg, *, n_epochs=1500, key=None):
    """Fit a minimal phase-3 stand-in, exercising the real membership functions."""
    key = jr.key(0) if key is None else key

    class Enc(eqx.Module):
        mlp: eqx.nn.MLP

        def __init__(self, k):
            self.mlp = eqx.nn.MLP(4, 2, 64, 3, activation=jax.nn.tanh, key=k)

        def __call__(self, w):
            o = self.mlp(w)
            return jnp.tanh(o[0]), jax.nn.sigmoid(o[1])

    class Dec(eqx.Module):
        mlp: eqx.nn.MLP

        def __init__(self, k):
            self.mlp = eqx.nn.MLP("scalar", 2, 64, 3, activation=jax.nn.tanh, key=k)

        def __call__(self, g):
            return self.mlp(g)

    k1, k2, k3 = jr.split(key, 3)
    qs = ws[:, :2]
    mask = jnp.ones(len(ws), dtype=bool)
    log_bg = float(jnp.log(cfg.resolve_background_density(qs)))

    params = (Enc(k1), Dec(k2), cfg.make_width_net(key=k3))
    dyn, static = eqx.partition(params, eqx.is_array)
    opt = optax.adam(3e-3)
    st = opt.init(dyn)

    def loss_fn(dyn, ceil, ramp):
        enc, dec, width = eqx.combine(dyn, static)
        g, p = jax.vmap(enc)(ws)
        qp = jax.vmap(dec)(g)
        sig = jnp.minimum(jax.vmap(width)(g), ceil)
        nll, resp = mixture_membership_loss(
            qs, qp, p, sig, mask, log_bg_density=log_bg, rampup=ramp
        )
        w = jax.lax.stop_gradient(resp)
        anchor = jnp.sum(w * (gamma_target - g) ** 2) / jnp.sum(w)
        return nll + anchor

    def body(carry, i):
        dyn, st = carry
        ceil = sigma_ceiling(
            i, n_epochs, start=cfg.sigma_ceiling[0], stop=cfg.sigma_ceiling[1]
        )
        ramp = membership_rampup(i, n_epochs, warmup_frac=cfg.warmup_frac)
        _, grads = jax.value_and_grad(loss_fn)(dyn, ceil, ramp)
        upd, st = opt.update(grads, st, dyn)
        return (eqx.apply_updates(dyn, upd), st), None

    (dyn, _), _ = jax.lax.scan(body, (dyn, st), jnp.arange(n_epochs))
    enc, dec, width = eqx.combine(dyn, static)

    g, p = jax.vmap(enc)(ws)
    qp = jax.vmap(dec)(g)
    sig = jnp.minimum(jax.vmap(width)(g), jnp.asarray(cfg.sigma_ceiling[1]))
    r2 = jnp.sum((qs - qp) ** 2, axis=-1)
    resp = membership_responsibility(p, r2, sig, log_bg_density=log_bg, n_dims=2)
    return resp, sig


class TestFindsOutliers:
    """End-to-end behaviour: the model must actually reject outliers."""

    def test_flags_outliers_and_keeps_members(self, arc_with_outliers) -> None:
        """The headline claim: outliers get low posterior, members keep high."""
        ws, gamma, is_outlier, _ = arc_with_outliers
        cfg = MixtureMembershipConfig(
            sigma_init=0.3, sigma_ceiling=(1.5, 0.15), warmup_frac=0.3
        )
        resp, _ = _fit_mixture(ws, gamma, cfg)

        caught = int(jnp.sum(resp[is_outlier] < 0.5))
        lost = int(jnp.sum(resp[~is_outlier] < 0.5))
        n_out = int(jnp.sum(is_outlier))
        n_in = int(jnp.sum(~is_outlier))

        assert caught >= 0.8 * n_out, f"only caught {caught}/{n_out} outliers"
        assert lost <= 0.05 * n_in, f"wrongly cut {lost}/{n_in} genuine members"

    def test_recovers_stream_width(self, arc_with_outliers) -> None:
        """Sigma should land near the true injected width."""
        ws, gamma, _, true_sigma = arc_with_outliers
        cfg = MixtureMembershipConfig(
            sigma_init=0.3, sigma_ceiling=(1.5, 0.15), warmup_frac=0.3
        )
        _, sigma = _fit_mixture(ws, gamma, cfg)

        assert float(jnp.median(sigma)) == pytest.approx(true_sigma, abs=0.1)

    def test_warmup_prevents_membership_collapse(self, arc_with_outliers) -> None:
        """Regression: without the ramp, the model can reject *everything*.

        With no warm-up the mixture can find the degenerate optimum where every
        star is background -- at which point the stream component carries no
        gradient, the track never fits, and training is dead. The ramp exists to
        make that unreachable.
        """
        ws, gamma, is_outlier, _ = arc_with_outliers
        n_in = int(jnp.sum(~is_outlier))

        cfg = MixtureMembershipConfig(
            sigma_init=0.3, sigma_ceiling=(1.5, 0.15), warmup_frac=0.3
        )
        resp, _ = _fit_mixture(ws, gamma, cfg)

        # The thing we must never do: reject the entire stream.
        lost = int(jnp.sum(resp[~is_outlier] < 0.5))
        assert lost < 0.5 * n_in, (
            f"membership collapsed: {lost}/{n_in} genuine members rejected"
        )
