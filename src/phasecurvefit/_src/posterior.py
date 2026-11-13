r"""A posterior for the track: a NumPyro model and :func:`sample_track`.

The track is a Hilbert-space Gaussian process in $\gamma \in [-1, 1]$, one
channel per position component. Each star's $\gamma$ is summed out on a grid,
so the model needs no per-star $\gamma$. Stars scatter about the track with a
Gaussian whose width is latent. A flat background with a latent fraction takes
the interlopers (Hogg, Bovy & Lang 2010, sec. 3). When velocities are given,
their mean is tied to the track: the speed times the unit tangent, plus a small
offset.

Needs NumPyro, the ``phasecurvefit[numpyro]`` extra.
"""

__all__: tuple[str, ...] = ("TrackPosterior", "sample_track", "track_model")

import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import plum
from jaxtyping import Array, Float, PRNGKeyArray

from phasecurvefit._src.abstract_result import AbstractResult
from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.nn.membership import uniform_background_density
from phasecurvefit._src.optional_deps import OptDeps
from phasecurvefit._src.orderers.default_pipeline import default_pipeline
from phasecurvefit._src.pipeline import TrackEnsemble
from phasecurvefit._src.track_draw import TrackDraw

# Every use of numpyro below follows _require_numpyro().
# pylint: disable=possibly-used-before-assignment
if OptDeps.NUMPYRO.installed:
    import numpyro
    import numpyro.contrib.hsgp.laplacian
    import numpyro.diagnostics
    import numpyro.distributions
    import numpyro.infer
    import numpyro.infer.autoguide

_L_BOX = 1.3  # HSGP box half-width; gamma lives in [-1, 1]
_A0, _L0 = 1.0, 0.3  # GP amplitude and length scale the start is fitted at
_RHAT_WARN = 1.05


def _require_numpyro() -> None:
    if not OptDeps.NUMPYRO.installed:
        msg = (
            "sample_track and track_model need NumPyro: "
            "pip install 'phasecurvefit[numpyro]'."
        )
        raise ImportError(msg)


def _sqrt_spd(amp: Array, length: Array, n_basis: int, /) -> Array:
    """Sqrt of the squared-exponential spectral density at the basis frequencies.

    In closed form, because the square root of NumPyro's density underflows to 0
    at short length scales, where its gradient is NaN. ``amp`` is the kernel's
    standard deviation.
    """
    omega = jnp.arange(1, n_basis + 1) * jnp.pi / (2 * _L_BOX)
    return (
        amp
        * (2 * jnp.pi) ** 0.25
        * jnp.sqrt(length)
        * jnp.exp(-((length * omega) ** 2) / 4)
    )


def _basis(gamma: Array, n_basis: int, /) -> Array:
    return numpyro.contrib.hsgp.laplacian.eigenfunctions(gamma, _L_BOX, n_basis)


def _channels(gamma: Array, alpha: Array, length: Array, beta: Array, /) -> Array:
    """Every GP channel at ``gamma``: ``(..., G, C)`` from ``(..., C, M)`` latents."""
    spd = _sqrt_spd(alpha, length, beta.shape[-1])
    return _basis(gamma, beta.shape[-1]) @ jnp.swapaxes(beta * spd, -1, -2)


def _log_normal_iso(r2: Array, sigma: Array, d: int, /) -> Array:
    return -0.5 * r2 / sigma**2 - d * jnp.log(sigma) - 0.5 * d * jnp.log(2 * jnp.pi)


def _layout(d: int, n_q: int, velocity_tie: float | None, /) -> tuple[bool, int, int]:
    """Whether the velocities are tied, the slack channels, and the mean channels."""
    tied = velocity_tie is not None and d > n_q
    n_slack = n_q if tied and velocity_tie else 0
    return tied, n_slack, (n_q + 1 + n_slack if tied else d)


def track_model(
    z: Float[Array, "N D"],
    /,
    *,
    n_q: int,
    weights: Float[Array, " N"] | None = None,
    log_rho_bg: float | None = None,
    n_basis: int = 16,
    n_grid: int = 256,
    line_density: bool = True,
    width_varies: bool = True,
    velocity_tie: float | None = 0.05,
) -> None:
    r"""Run the NumPyro model of the track, for use inside a model of your own.

    The latents are a Gaussian process per channel (sites ``alpha``,
    ``length`` and ``beta``, one row per channel), the width ``sigma``, the
    velocity width ``sigma_p`` when there are velocities, and the background
    fraction ``f_bg`` when ``log_rho_bg`` is given. The likelihood is the
    factor site ``loglik``.

    Parameters
    ----------
    z
        Scaled phase-space coordinates, positions in the first ``n_q``
        columns. The priors assume positions with zero mean and unit spread.
        :func:`sample_track` subtracts each position's mean and divides by one
        length; it divides velocities by their rms speed and does not centre
        them, which the velocity tie needs.
    n_q
        Number of position columns. Any later columns are velocities, in the
        same component order.
    weights
        Per-star weights. They act as counts: a weight of 2 is two stars, so
        scaling every weight narrows the band.
    log_rho_bg
        Log of the flat background density, in the units of ``z``. ``None``
        leaves out the background.
    n_basis
        Basis functions per channel. More resolve sharper bends.
    n_grid
        Points of the $\gamma$ grid that each star's $\gamma$ is summed over.
    line_density
        Give $\gamma$ a latent density along the track (a channel normalised
        over the grid) instead of a uniform one, so a sparse tail need not be
        squeezed into a sliver of $\gamma$.
    width_varies
        Let the width change along the track, as ``sigma`` times the
        exponential of a channel.
    velocity_tie
        ``None`` fits the velocities as free channels. A number ties their
        mean to the track: the speed (a log channel) times the unit tangent,
        plus ``velocity_tie`` times an offset channel per component. Stream
        stars move a few degrees off their track, and 0.05 lets them. ``0``
        is a hard tie.

    """
    _require_numpyro()
    n, d = z.shape
    tied, n_slack, n_mean = _layout(d, n_q, velocity_tie)
    n_ch = n_mean + int(line_density) + int(width_varies)

    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(0.5))
    with numpyro.plate("channel", n_ch, dim=-2):
        alpha = numpyro.sample("alpha", numpyro.distributions.HalfNormal(1.0))
        length = numpyro.sample(
            "length", numpyro.distributions.LogNormal(jnp.log(_L0), 0.7)
        )
        with numpyro.plate("basis", n_basis, dim=-1):
            beta = numpyro.sample("beta", numpyro.distributions.Normal(0.0, 1.0))
    x_all = _channels(jnp.linspace(-1.0, 1.0, n_grid), alpha, length, beta)
    x, col = x_all[:, :n_mean], n_mean

    if tied:
        t = jnp.gradient(x[:, :n_q], axis=0)
        t = t / jnp.linalg.norm(t, axis=-1, keepdims=True).clip(1e-12)
        v = jnp.exp(x[:, n_q : n_q + 1]) * t
        if n_slack:
            v = v + (velocity_tie or 0.0) * x[:, n_q + 1 :]
        x = jnp.concatenate([x[:, :n_q], v], -1)

    log_pi: Array | float = -jnp.log(n_grid)
    if line_density:
        log_pi, col = jax.nn.log_softmax(x_all[:, col]), col + 1
    if width_varies:
        sigma = sigma * jnp.exp(x_all[:, col])

    d2 = (z[:, None, :] - x[None]) ** 2  # (N, K, D)
    ll = _log_normal_iso(jnp.sum(d2[..., :n_q], -1), sigma, n_q)
    if d > n_q:
        sigma_p = numpyro.sample("sigma_p", numpyro.distributions.HalfNormal(0.5))
        ll = ll + _log_normal_iso(jnp.sum(d2[..., n_q:], -1), sigma_p, d - n_q)
    log_fg = jax.nn.logsumexp(ll + log_pi, -1)
    if log_rho_bg is not None:
        f_bg = numpyro.sample("f_bg", numpyro.distributions.Beta(1.0, 9.0))
        log_fg = jnp.logaddexp(jnp.log1p(-f_bg) + log_fg, jnp.log(f_bg) + log_rho_bg)
    if weights is not None:  # multiplied in: handlers.scale refuses a weight of 0
        log_fg = weights * log_fg
    with numpyro.plate("stars", n):
        numpyro.factor("loglik", log_fg)


def _basis_fit(gamma: np.ndarray, y: np.ndarray, n_basis: int, /) -> np.ndarray:
    """Ridge fit of ``y(gamma)`` on the basis: non-centred ``beta``, ``(C, M)``."""
    x = np.asarray(_basis(jnp.asarray(gamma), n_basis)) * np.asarray(
        _sqrt_spd(jnp.asarray(_A0), jnp.asarray(_L0), n_basis)
    )
    a = x.T @ x + 0.1 * np.eye(n_basis)
    return np.linalg.solve(a, x.T @ np.asarray(y)).T


def _start(
    z: Array,
    gamma0: np.ndarray,
    idx: np.ndarray,
    /,
    *,
    n_q: int,
    n_basis: int,
    line_density: bool,
    width_varies: bool,
    velocity_tie: float | None,
    **_unused: object,
) -> dict[str, Array]:
    """Fit the track to the ordering by least squares, for the starting latents."""
    d = z.shape[1]
    tied, _, n_mean = _layout(d, n_q, velocity_tie)
    zo = np.asarray(z)[idx]
    target = np.zeros((len(idx), n_mean))  # tied: log-speed and offsets start at 0
    target[:, : (n_q if tied else d)] = zo[:, : (n_q if tied else d)]
    rows = [_basis_fit(gamma0, target, n_basis)]
    if line_density:  # the log histogram of the start's gamma
        h, e = np.histogram(gamma0, bins=32, range=(-1, 1))
        logh = np.log(h + 1.0)
        rows.append(
            _basis_fit(0.5 * (e[1:] + e[:-1]), logh - logh.mean(), n_basis)[None]
        )
    if width_varies:
        rows.append(np.zeros((1, n_basis)))
    beta = np.concatenate(rows)
    x = np.asarray(_basis(jnp.asarray(gamma0), n_basis)) * np.asarray(
        _sqrt_spd(jnp.asarray(_A0), jnp.asarray(_L0), n_basis)
    )
    resid = target - x @ beta[:n_mean].T
    n_ch = beta.shape[0]
    init = {
        "alpha": jnp.full((n_ch, 1), _A0),
        "length": jnp.full((n_ch, 1), _L0),
        "beta": jnp.asarray(beta),
        "sigma": jnp.asarray(max(float(resid[:, :n_q].std()), 0.02)),
    }
    if d > n_q:
        sp = 0.2 if tied else max(float(resid[:, n_q:].std()), 0.02)
        init["sigma_p"] = jnp.asarray(sp)
    return init


@plum.dispatch
def _scale(
    positions: VectorComponents, velocities: VectorComponents, /
) -> tuple[Array, dict[str, Any], Any]:
    """Scale to the model's units: ``z``, each position's mean, and one length.

    Positions lose their means and are divided by one length, so an isotropic
    width stays isotropic. Velocities are divided by their rms speed and keep
    their sign, which the velocity tie needs. Velocity columns follow the
    positions' component order.
    """
    q = jnp.stack([jnp.asarray(positions[k]) for k in positions], axis=1)
    p = jnp.stack([jnp.asarray(velocities[k]) for k in positions], axis=1)
    mu = jnp.mean(q, axis=0)
    s = jnp.sqrt(jnp.mean(jnp.var(q, axis=0)))
    sv = jnp.sqrt(jnp.mean(jnp.sum(p**2, axis=1)))
    z = jnp.concatenate([(q - mu) / s, p / sv], axis=1)
    return z, dict(zip(positions, mu, strict=True)), s


class TrackPosterior(TrackEnsemble):
    r"""Draws from the posterior of the track.

    Made by :func:`sample_track`. ``members`` are draws from the posterior and
    ``reference`` is the track at the mean of the drawn latents. Each is a
    :class:`~phasecurvefit.TrackDraw`. The methods
    :meth:`~phasecurvefit.TrackEnsemble.offsets` and
    :meth:`~phasecurvefit.TrackEnsemble.radius` work as they do for a bootstrap.

    The draws share the model's $\gamma$, so they can also be compared at equal
    $\gamma$.

    Attributes
    ----------
    reference : TrackDraw
        The track at the mean of the drawn latents.
    members : tuple[TrackDraw, ...]
        The posterior draws.
    samples : dict[str, Array]
        The drawn latents of :func:`track_model`, one row per draw.
    losses : Array or None
        The SVI loss at each step, for checking convergence; ``None`` for NUTS.

    """

    samples: dict[str, Array]
    losses: Float[Array, " n_steps"] | None


def _draws(
    samples: dict[str, Array],
    /,
    *,
    n_q: int,
    n_grid: int,
    mean_channels: int,
    line_density: bool,
    width_varies: bool,
    centre: dict[str, Any],
    scale: Any,
) -> list[TrackDraw]:
    grid = jnp.linspace(-1.0, 1.0, n_grid)
    x = _channels(grid, samples["alpha"], samples["length"], samples["beta"])
    sigma = jnp.broadcast_to(samples["sigma"][:, None], x.shape[:2])
    col = mean_channels + int(line_density)
    if width_varies:
        sigma = sigma * jnp.exp(x[..., col])
    out = []
    for i in range(x.shape[0]):
        lo, hi = -1.0, 1.0
        if line_density:  # the central 99% of the line density
            cdf = np.cumsum(np.asarray(jax.nn.softmax(x[i, :, mean_channels])))
            g = np.asarray(grid)
            lo, hi = float(np.interp(0.005, cdf, g)), float(np.interp(0.995, cdf, g))
        out.append(
            TrackDraw(
                gamma=grid,
                x=x[i, :, :n_q],
                sigma=sigma[i],
                centre=centre,
                scale=scale,
                track_range=(lo, hi),
            )
        )
    return out


def sample_track(
    positions: VectorComponents,
    velocities: VectorComponents,
    *,
    key: PRNGKeyArray,
    method: Literal["svi", "nuts"] = "svi",
    metadata: StateMetadata | None = None,
    weights: Float[Array, " N"] | None = None,
    ordering: AbstractResult | None = None,
    n_draws: int = 200,
    n_steps: int = 4000,
    n_warmup: int = 500,
    n_basis: int = 16,
    n_grid: int = 256,
    background: bool = True,
    line_density: bool = True,
    width_varies: bool = True,
    velocity_tie: float | None = 0.05,
) -> TrackPosterior:
    r"""Fit a track and draw from its posterior, with NumPyro.

    The model is :func:`track_model`. It starts from an ordering: the stars'
    rank in it, counted by weight, gives each a starting $\gamma$, and the track
    is fitted to that by least squares. Then SVI with a full-rank Gaussian
    guide, or NUTS, samples the posterior. The ordering matters: a start that
    takes a wrong turn where the stream crosses itself stays wrong.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers, arrays or `unxt` Quantities, with matching
        component names. Values must be finite.
    key
        Split for the inference and for drawing.
    method
        ``"svi"`` (the default) fits a full-rank Gaussian to the posterior.
        ``"nuts"`` runs two chains of NUTS, which is slower and warns when the
        chains disagree (split $\hat R$ above 1.05). The posterior often has
        more than one mode, and then NUTS chains stay in different ones.
    metadata
        Passed to :func:`~phasecurvefit.orderers.default_pipeline`. Quantity
        inputs need its unit system.
    weights
        Per-tracer weights. They act as counts: the band of a weight-2 star is
        that of two stars, so scaling every weight changes the band.
    ordering
        The ordering to start from. ``None`` runs
        :func:`~phasecurvefit.orderers.default_pipeline`.
    n_draws
        Number of posterior draws to return.
    n_steps
        SVI optimisation steps.
    n_warmup
        NUTS warmup steps per chain. Each chain then keeps ``n_draws / 2``.
    n_basis, n_grid, line_density, width_varies, velocity_tie
        Passed to :func:`track_model`. A stream that crosses itself may need
        ``n_basis=32, n_grid=512``.
    background
        Include a flat background over the stars' bounding box in phase space,
        with a latent fraction.

    Returns
    -------
    TrackPosterior
        The draws, as tracks callable on $\gamma$.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import phasecurvefit as pcf
    >>> ang = jnp.linspace(0.0, jnp.pi, 80)
    >>> noise = 0.05 * jax.random.normal(jax.random.key(1), (2, 80))
    >>> pos = {"x": 5 * jnp.cos(ang) + noise[0], "y": 5 * jnp.sin(ang) + noise[1]}
    >>> vel = {"x": -jnp.sin(ang), "y": jnp.cos(ang)}
    >>> post = pcf.sample_track(
    ...     pos, vel, key=jax.random.key(0), n_steps=500, n_draws=20
    ... )
    >>> len(post.members)
    20
    >>> post.radius(jnp.linspace(-0.5, 0.5, 3)).shape
    (3,)

    """
    _require_numpyro()
    if method not in ("svi", "nuts"):
        msg = f"method must be 'svi' or 'nuts', got {method!r}."
        raise ValueError(msg)
    z, centre, scale = _scale(positions, velocities)
    if not bool(jnp.all(jnp.isfinite(z))):
        msg = "sample_track needs finite positions and velocities."
        raise ValueError(msg)
    n_q = len(positions)
    if ordering is None:
        ordering = default_pipeline(
            positions, velocities, metadata=metadata, weights=weights
        )
    idx = np.asarray(ordering.ordering)
    w = np.ones(z.shape[0]) if weights is None else np.asarray(weights)
    cw = np.cumsum(w[idx]) - 0.5 * w[idx]
    gamma0 = -1 + 2 * (cw - cw[0]) / (cw[-1] - cw[0])

    opts: dict[str, Any] = {
        "n_q": n_q,
        "n_basis": n_basis,
        "n_grid": n_grid,
        "line_density": line_density,
        "width_varies": width_varies,
        "velocity_tie": velocity_tie,
    }
    init = _start(z, gamma0, idx, **opts)
    log_rho = float(jnp.log(uniform_background_density(z))) if background else None
    data: dict[str, Any] = {
        "weights": None if weights is None else jnp.asarray(weights),
        "log_rho_bg": log_rho,
    }

    def model() -> None:
        track_model(z, **opts, **data)

    key_fit, key_draw = jr.split(key)
    losses = None
    if method == "svi":
        guide = numpyro.infer.autoguide.AutoMultivariateNormal(
            model, init_loc_fn=numpyro.infer.init_to_value(values=init)
        )
        opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(2e-3))
        res = numpyro.infer.SVI(model, guide, opt, numpyro.infer.Trace_ELBO()).run(
            key_fit, n_steps, progress_bar=False
        )
        losses = res.losses
        samples = guide.sample_posterior(key_draw, res.params, sample_shape=(n_draws,))
    else:
        n_chains = 2
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(
                model,
                init_strategy=numpyro.infer.init_to_value(values=init),
                dense_mass=True,
                target_accept_prob=0.9,
            ),
            num_warmup=n_warmup,
            num_samples=-(-n_draws // n_chains),
            num_chains=n_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(key_fit)
        rhat = max(
            float(np.nanmax(v["r_hat"]))
            for v in numpyro.diagnostics.summary(
                mcmc.get_samples(group_by_chain=True)
            ).values()
        )
        if rhat > _RHAT_WARN:
            warnings.warn(
                f"NUTS chains disagree (max split R-hat {rhat:.2f} > {_RHAT_WARN}); "
                "they may sit in different modes. Compare with method='svi'.",
                RuntimeWarning,
                stacklevel=2,
            )
        samples = {k: v[:n_draws] for k, v in mcmc.get_samples().items()}

    samples = {k: v for k, v in samples.items() if k != "loglik"}
    _, _, n_mean = _layout(z.shape[1], n_q, velocity_tie)
    build: dict[str, Any] = {
        "n_q": n_q,
        "n_grid": n_grid,
        "mean_channels": n_mean,
        "line_density": line_density,
        "width_varies": width_varies,
        "centre": centre,
        "scale": scale,
    }
    members = _draws(samples, **build)
    mean = {k: jnp.mean(v, axis=0, keepdims=True) for k, v in samples.items()}
    reference = _draws(mean, **build)[0]
    return TrackPosterior(
        reference=reference, members=tuple(members), samples=samples, losses=losses
    )
