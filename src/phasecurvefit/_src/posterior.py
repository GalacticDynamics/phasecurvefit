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

import functools
import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import plum
from jaxtyping import Array, Float, PRNGKeyArray

from phasecurvefit._src.algorithm import StateMetadata
from phasecurvefit._src.checks import check_weights
from phasecurvefit._src.custom_types import VectorComponents
from phasecurvefit._src.nn.membership import uniform_background_density
from phasecurvefit._src.optional_deps import OptDeps
from phasecurvefit._src.orderers.default_pipeline import default_pipeline
from phasecurvefit._src.orderers.result import OrderingResult
from phasecurvefit._src.pipeline import TrackEnsemble
from phasecurvefit._src.track_draw import TrackDraw

# Imported on first use by _require_numpyro, so `import phasecurvefit` stays
# fast for users who never sample.
numpyro: Any = None

_L_BOX = 1.3  # HSGP box half-width; gamma lives in [-1, 1]
_A0, _L0 = 1.0, 0.3  # GP amplitude and length scale the start is fitted at
_RHAT_WARN = 1.05


def _require_numpyro() -> None:
    # pylint: disable=global-statement,import-outside-toplevel
    global numpyro
    if not OptDeps.NUMPYRO.installed:
        msg = (
            "sample_track and track_model need NumPyro: "
            "pip install 'phasecurvefit[numpyro]'."
        )
        raise ImportError(msg)
    if numpyro is None:
        import numpyro.contrib.hsgp.laplacian  # noqa: PLC0415
        import numpyro.diagnostics  # noqa: PLC0415
        import numpyro.distributions  # noqa: PLC0415
        import numpyro.infer  # noqa: PLC0415
        import numpyro.infer.autoguide  # noqa: PLC0415


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
    _require_numpyro()
    return numpyro.contrib.hsgp.laplacian.eigenfunctions(gamma, _L_BOX, n_basis)


def _channels(gamma: Array, alpha: Array, length: Array, beta: Array, /) -> Array:
    """Every GP channel at ``gamma``: ``(..., G, C)`` from ``(..., C, M)`` latents."""
    spd = _sqrt_spd(alpha, length, beta.shape[-1])
    return _basis(gamma, beta.shape[-1]) @ jnp.swapaxes(beta * spd, -1, -2)


def _log_normal_iso(r2: Array, sigma: Array, d: int, /) -> Array:
    return -0.5 * r2 / sigma**2 - d * jnp.log(sigma) - 0.5 * d * jnp.log(2 * jnp.pi)


def _sq_dist(a: Array, b: Array, /) -> Array:
    """Squared distances ``(N, K)`` between rows, without an ``(N, K, D)`` array."""
    r2 = jnp.sum(a**2, -1)[:, None] - 2 * a @ b.T + jnp.sum(b**2, -1)[None]
    return jnp.maximum(r2, 0.0)  # rounding can take a near-zero distance below 0


def _layout(d: int, n_q: int, velocity_tie: float | None, /) -> tuple[bool, int, int]:
    """Whether the velocities are tied, the slack channels, and the mean channels."""
    tied = velocity_tie is not None and d > n_q
    n_slack = n_q if tied and velocity_tie else 0
    return tied, n_slack, (n_q + 1 + n_slack if tied else d)


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def _log_likelihood(
    z: Array,
    sites: dict[str, Array],
    weights: Array | None,
    log_rho_bg: Array | float | None,
    n_q: int,
    n_grid: int,
    line_density: bool,  # noqa: FBT001  # positional, as jit's static args
    width_varies: bool,  # noqa: FBT001
    velocity_tie: float | None,
    /,
) -> Array:
    """Each star's log-likelihood, given the latents.

    Jitted on its own because NumPyro runs the model op by op while it sets up
    a guide, and each op would compile separately (seconds in all, per call).
    """
    d = z.shape[1]
    tied, n_slack, n_mean = _layout(d, n_q, velocity_tie)
    x_all = _channels(
        jnp.linspace(-1.0, 1.0, n_grid), sites["alpha"], sites["length"], sites["beta"]
    )
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
    sigma = sites["sigma"]
    if width_varies:
        sigma = sigma * jnp.exp(x_all[:, col])

    ll = _log_normal_iso(_sq_dist(z[:, :n_q], x[:, :n_q]), sigma, n_q)
    if d > n_q:
        r2 = _sq_dist(z[:, n_q:], x[:, n_q:])
        ll = ll + _log_normal_iso(r2, sites["sigma_p"], d - n_q)
    log_fg = jax.nn.logsumexp(ll + log_pi, -1)
    if log_rho_bg is not None:
        f_bg = sites["f_bg"]
        log_fg = jnp.logaddexp(jnp.log1p(-f_bg) + log_fg, jnp.log(f_bg) + log_rho_bg)
    if weights is not None:  # multiplied in: handlers.scale refuses a weight of 0
        log_fg = weights * log_fg
    return log_fg


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
        scaling every weight up narrows the band.
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
        is a hard tie. The tie needs velocities in the frame where the track
        is at rest: a shared velocity several times the speed along the track
        (heliocentric velocities, say) can't be fitted, and the track comes out
        wrong. Use ``None`` for velocities in another frame.

    """
    _require_numpyro()
    n, d = z.shape
    n_q, n_basis, n_grid, line_density, width_varies, velocity_tie = _model_options(
        d, n_q, n_basis, n_grid, line_density, width_varies, velocity_tie
    )
    if weights is not None:
        weights = check_weights(weights, n)
    _, _, n_mean = _layout(d, n_q, velocity_tie)
    n_ch = n_mean + int(line_density) + int(width_varies)

    sites: dict[str, Array] = {
        "sigma": numpyro.sample("sigma", numpyro.distributions.HalfNormal(0.5))
    }
    with numpyro.plate("channel", n_ch, dim=-2):
        sites["alpha"] = numpyro.sample("alpha", numpyro.distributions.HalfNormal(1.0))
        sites["length"] = numpyro.sample(
            "length", numpyro.distributions.LogNormal(jnp.log(_L0), 0.7)
        )
        with numpyro.plate("basis", n_basis, dim=-1):
            sites["beta"] = numpyro.sample(
                "beta", numpyro.distributions.Normal(0.0, 1.0)
            )
    if d > n_q:
        sites["sigma_p"] = numpyro.sample(
            "sigma_p", numpyro.distributions.HalfNormal(0.5)
        )
    if log_rho_bg is not None:
        sites["f_bg"] = numpyro.sample("f_bg", numpyro.distributions.Beta(1.0, 9.0))
    log_fg = _log_likelihood(
        z,
        sites,
        weights,
        log_rho_bg,
        n_q,
        n_grid,
        line_density,
        width_varies,
        velocity_tie,
    )
    with numpyro.plate("stars", n):
        numpyro.factor("loglik", log_fg)


def _basis_fit(
    gamma: np.ndarray, y: np.ndarray, n_basis: int, w: np.ndarray, /
) -> np.ndarray:
    """Weighted ridge fit of ``y(gamma)`` on the basis: non-centred ``beta``."""
    x = np.asarray(_basis(jnp.asarray(gamma), n_basis)) * np.asarray(
        _sqrt_spd(jnp.asarray(_A0), jnp.asarray(_L0), n_basis)
    )
    a = (x * w[:, None]).T @ x + 0.1 * np.eye(n_basis)
    return np.linalg.solve(a, (x * w[:, None]).T @ np.asarray(y)).T


def _wstd(r: np.ndarray, w: np.ndarray, /) -> float:
    """Weighted standard deviation of every entry of ``r``, rows weighted by ``w``."""
    m = np.average(r, axis=0, weights=w)
    return float(np.sqrt(np.average(np.mean((r - m) ** 2, axis=1), weights=w)))


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
    weights: np.ndarray | None = None,
    **_unused: object,
) -> dict[str, Array]:
    """Fit the track to the ordering by least squares, for the starting latents.

    ``weights`` are the ordered tracers' weights. Their start is spaced by
    cumulative weight, so its line density is flat and starts at 0.
    """
    d = z.shape[1]
    tied, _, n_mean = _layout(d, n_q, velocity_tie)
    zo = np.asarray(z)[idx]
    target = np.zeros((len(idx), n_mean))  # tied: log-speed and offsets start at 0
    target[:, : (n_q if tied else d)] = zo[:, : (n_q if tied else d)]
    w = np.ones(len(idx)) if weights is None else np.asarray(weights, dtype=float)
    rows = [_basis_fit(gamma0, target, n_basis, w)]
    rows.append(np.zeros((int(line_density) + int(width_varies), n_basis)))
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
        "sigma": jnp.asarray(max(_wstd(resid[:, :n_q], w), 0.02)),
    }
    if d > n_q:
        sp = 0.2 if tied else max(_wstd(resid[:, n_q:], w), 0.02)
        init["sigma_p"] = jnp.asarray(sp)
    return init


@plum.dispatch
def _scale_positions(
    positions: VectorComponents, /
) -> tuple[Array, dict[str, Any], Any]:
    """Scale positions: ``z``, each component's mean, and one length.

    Positions lose their means and are divided by one length, so an isotropic
    width stays isotropic.
    """
    q = jnp.stack([jnp.asarray(positions[k]) for k in positions], axis=1)
    mu = jnp.mean(q, axis=0)
    s = jnp.sqrt(jnp.mean(jnp.var(q, axis=0)))
    return (q - mu) / s, dict(zip(positions, mu, strict=True)), s


@plum.dispatch
def _scale_velocities(velocities: VectorComponents, keys: tuple[str, ...], /) -> Array:
    """Divide velocities by their rms speed, in the positions' component order.

    They keep their sign and are not centred, which the velocity tie needs.
    """
    p = jnp.stack([jnp.asarray(velocities[k]) for k in keys], axis=1)
    return p / jnp.sqrt(jnp.mean(jnp.sum(p**2, axis=1)))


def _scale(
    positions: VectorComponents,
    velocities: VectorComponents | None,
    weights: Array | np.ndarray | None = None,
    /,
) -> tuple[Array, dict[str, Any], Any]:
    """Scale to the model's units: ``z``, each position's mean, and one length.

    The means and the scales are weighted, so a weight of 2 scales the data as
    a repeated row does, and a tracer of weight 0 changes nothing.
    """
    comps = [*positions.values(), *(velocities or {}).values()]
    if len({hasattr(v, "unit") for v in comps}) > 1:
        msg = "the inputs mix Quantities and plain arrays; give every component a unit."
        raise ValueError(msg)
    zq, centre, s = _scale_positions(positions)
    if weights is not None:  # re-centre and re-scale with the weights
        p = jnp.asarray(weights) / jnp.sum(weights)
        m = p @ zq
        f = jnp.sqrt(jnp.mean(p @ (zq - m) ** 2))
        centre = {k: c + m[i] * s for i, (k, c) in enumerate(centre.items())}
        zq, s = (zq - m) / f, s * f
    if velocities is None:
        return zq, centre, s
    zp = _scale_velocities(velocities, tuple(positions))
    if weights is not None:
        zp = zp / jnp.sqrt(p @ jnp.sum(zp**2, axis=1))
    return jnp.concatenate([zq, zp], 1), centre, s


class TrackPosterior(TrackEnsemble):
    r"""Draws from the posterior of the track.

    Made by :func:`sample_track`. ``members`` are draws from the posterior and
    ``reference`` is the track at the mean of the drawn latents. Each is a
    :class:`~phasecurvefit.TrackDraw`. The methods
    :meth:`~phasecurvefit.TrackEnsemble.offsets` and
    :meth:`~phasecurvefit.TrackEnsemble.radius` work as they do for a bootstrap.

    Compare draws in space, as ``offsets`` does, and not at equal $\gamma$.
    With the latent line density a draw's $\gamma$ can shift along the track,
    and on test arcs the central 68% of the draws' points at one $\gamma$
    spanned 3.6 to 5.1 times the 68% radius along it.

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
                track_range=jnp.array([lo, hi]),
                keys=tuple(centre),
            )
        )
    return out


def _warn_if_chains_disagree(by_chain: dict[str, Array], /) -> None:
    """Warn when the largest split R-hat is above 1.05, or can't be computed.

    A chain stuck at one value gives a NaN R-hat, which counts as disagreeing.
    """
    _require_numpyro()
    with np.errstate(invalid="ignore", divide="ignore"):  # a stuck chain: 0 / 0
        stats = numpyro.diagnostics.summary(by_chain).values()
    rhat = np.concatenate([np.ravel(v["r_hat"]) for v in stats])
    worst = float(np.nan if np.isnan(rhat).any() else rhat.max())
    if np.isnan(worst) or worst > _RHAT_WARN:
        warnings.warn(
            f"NUTS chains disagree (max split R-hat {worst:.2f}, above {_RHAT_WARN} "
            "or not computable); they may sit in different modes or be stuck. "
            "The reference, at the mean of the draws, may then match neither "
            "mode. Compare with method='svi'.",
            RuntimeWarning,
            stacklevel=3,
        )


def _warn_if_divergent(n_divergent: int, /) -> None:
    """Warn when NUTS reported divergent transitions after warmup."""
    if n_divergent:
        warnings.warn(
            f"NUTS had {n_divergent} divergent transitions; the draws near them may "
            "be biased. Try a larger n_warmup, or compare with method='svi'.",
            RuntimeWarning,
            stacklevel=3,
        )


def _runs_backwards(zo: np.ndarray, wo: np.ndarray, n_q: int, /) -> bool:
    """Whether an ordered start runs against its velocities, by a vote of steps.

    Each step between neighbours votes with the sign of its dot product with
    the mean velocity of its two ends. Weights act as counts: copies of a
    tracer add only zero-length steps, so a step between tracers of weight 1
    or more votes 1, and a step to a tracer of weight below 1 votes that
    weight (the smaller of its ends', capped at 1). Only
    steps up to 3 times the median step vote, so long jumps out to
    interlopers, whose velocities are random, don't decide it, and steps of
    zero length don't vote. Reversing the order flips every vote.
    """
    step = np.diff(zo[:, :n_q], axis=0)
    v = 0.5 * (zo[1:, n_q:] + zo[:-1, n_q:])
    length = np.linalg.norm(step, axis=1)
    moved = length > 0  # repeated positions carry no direction
    typical = moved & (length <= 3 * np.median(length[moved] if moved.any() else 0))
    count = np.minimum(np.minimum(wo[1:], wo[:-1]), 1.0)
    vote = np.sign(np.sum(step * v, axis=1)) * count
    return bool(np.sum(vote[typical]) < 0)


@plum.dispatch
def _ratio(x: Array, /) -> Array:
    """Return a dimensionless ratio as a bare array (a Quantity one is converted)."""
    return x


def _check_ordering(
    positions: VectorComponents,
    centre: dict[str, Any],
    scale: Any,
    ordering: OrderingResult,
    keep: np.ndarray,
    /,
) -> None:
    """Raise unless ``ordering`` is of these tracers, in this order.

    Only the tracers in ``keep`` (those of positive weight) are compared, so a
    tracer of weight 0 may hold anything, NaN included.
    """
    zq = jnp.stack(
        [_ratio((positions[k] - c) / scale) for k, c in centre.items()], axis=1
    )
    n = zq.shape[0]
    if ordering.indices.shape[0] != n:
        msg = f"ordering is of {ordering.indices.shape[0]} tracers, not these {n}."
        raise ValueError(msg)
    if set(ordering.positions) != set(centre):
        msg = "ordering is of tracers with other position components."
        raise ValueError(msg)
    # Scaled like these tracers, so the ratio is dimensionless for Quantities.
    try:
        zo = jnp.stack(
            [_ratio((ordering.positions[k] - c) / scale) for k, c in centre.items()],
            axis=1,
        )
    except (TypeError, ValueError) as err:  # astropy's unit errors are ValueErrors
        msg = "ordering is of other tracers: its positions have other units."
        raise ValueError(msg) from err
    # float32 rounds a unit conversion to about 1e-7 of the value, and the
    # scaled positions sit up to |centre| / scale from 0.
    far = max(float(np.abs(_ratio(c / scale))) for c in centre.values())
    zo, zq = np.asarray(zo)[keep], np.asarray(zq)[keep]
    if not np.allclose(zo, zq, rtol=0, atol=1e-4 + 1e-6 * far):
        msg = "ordering is of other tracers: its positions differ from these."
        raise ValueError(msg)


def _start_gamma(
    z: Array, n_q: int, order: np.ndarray, weights: Array | None, /
) -> tuple[np.ndarray, np.ndarray]:
    """Order the start, and give each tracer a gamma by cumulative weight.

    Only tracers of positive weight take part, so a tracer of weight 0 changes
    nothing about the start. With velocities, the start is turned to run along
    them, since a reversed start predicts every velocity backwards and the fit
    does not recover.
    """
    w = np.ones(z.shape[0]) if weights is None else np.asarray(weights)
    idx = np.asarray(order)
    idx = idx[w[idx] > 0]
    if idx.size < 2:
        msg = (
            f"the ordering visits {idx.size} tracers of positive weight; "
            "it needs at least 2."
        )
        raise ValueError(msg)
    if n_q < z.shape[1] and _runs_backwards(np.asarray(z)[idx], w[idx], n_q):
        idx = idx[::-1]
    cw = np.cumsum(w[idx]) - 0.5 * w[idx]
    return idx, -1 + 2 * (cw - cw[0]) / (cw[-1] - cw[0])


def _scalar(x: object, kinds: str, /) -> Any:
    """``x`` as a Python number if it is a scalar of a NumPy dtype kind in ``kinds``."""
    if isinstance(x, int) and not isinstance(x, bool):  # NumPy can't hold huge ones
        return x if "i" in kinds else None
    try:
        a = np.asarray(x)
    except (TypeError, ValueError):
        return None
    return a.item() if a.ndim == 0 and a.dtype.kind in kinds else None


def _model_options(
    d: int,
    n_q: object,
    n_basis: object,
    n_grid: object,
    line_density: object,
    width_varies: object,
    velocity_tie: object,
    /,
) -> tuple[int, int, int, bool, bool, float | None]:
    """Check :func:`track_model`'s options and return them as Python values.

    Integers and reals may be Python, NumPy or JAX scalars, and flags Python,
    NumPy or JAX bools; a bool is not accepted as a number.
    """
    ints = {}
    for name, value, low in (
        ("n_q", n_q, 1),
        ("n_basis", n_basis, 1),
        ("n_grid", n_grid, 2),
    ):
        v = _scalar(value, "iu")
        if v is None or v < low:
            msg = f"{name} must be an integer >= {low}, got {value!r}."
            raise ValueError(msg)
        ints[name] = int(v)
    if d not in (ints["n_q"], 2 * ints["n_q"]):
        msg = (
            f"z needs n_q = {ints['n_q']} position columns, or those and as many "
            f"velocity columns; it has {d}."
        )
        raise ValueError(msg)
    flags = []
    for name, value in (("line_density", line_density), ("width_varies", width_varies)):
        v = _scalar(value, "b")
        if v is None:
            msg = f"{name} must be True or False, got {value!r}."
            raise ValueError(msg)
        flags.append(bool(v))
    tie = None if velocity_tie is None else _scalar(velocity_tie, "iuf")
    if velocity_tie is not None and (tie is None or not 0 <= tie < np.inf):
        msg = (
            f"velocity_tie must be None or a finite number >= 0, got {velocity_tie!r}."
        )
        raise ValueError(msg)
    try:
        tie = None if tie is None else float(tie)
    except OverflowError:
        tie = np.inf
    if tie is not None and not tie < np.inf:
        msg = (
            f"velocity_tie must be None or a finite number >= 0, got {velocity_tie!r}."
        )
        raise ValueError(msg)
    return ints["n_q"], ints["n_basis"], ints["n_grid"], flags[0], flags[1], tie


def _require_finite(samples: dict[str, Array], method: str, /) -> None:
    """Raise when any posterior draw is not finite."""
    if not all(bool(jnp.all(jnp.isfinite(v))) for v in samples.values()):
        msg = (
            f"{method} returned draws that are not finite; try more tracers "
            "or another key."
        )
        raise RuntimeError(msg)


def _keep_positive(
    positions: VectorComponents,
    velocities: VectorComponents | None,
    weights: Array | None,
    /,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None, Array | None]:
    """Drop the tracers of weight 0, so nothing downstream sees them.

    Returns the mask of kept tracers, then the kept positions, velocities and
    weights.
    """
    n = jnp.shape(next(iter(positions.values())))[0]
    keep = np.ones(n, dtype=bool)
    if weights is not None:
        weights = check_weights(weights, n)
        keep = np.asarray(weights) > 0
    if keep.sum() < 2:
        msg = "weights must be positive for at least 2 tracers."
        raise ValueError(msg)

    def take(comps: VectorComponents, /) -> dict[str, Any]:
        return {k: v[keep] for k, v in comps.items()}  # type: ignore[index]  # arrays or Quantities

    return (
        keep,
        take(positions),
        None if velocities is None else take(velocities),
        None if weights is None else weights[keep],
    )


def _check_options(
    positions: VectorComponents,
    velocities: VectorComponents | None,
    method: str,
    counts: dict[str, object],
    /,
) -> dict[str, int]:
    """Raise on the inference options :func:`sample_track` can't use.

    Returns the counts as Python ints. They may be Python, NumPy or JAX
    integers; a bool is refused.
    """
    if not positions:
        msg = "positions needs at least one component."
        raise ValueError(msg)
    arrays = [*positions.values(), *(velocities or {}).values()]
    if not all(hasattr(v, "shape") for v in arrays):
        msg = "every component must be an array or a Quantity."
        raise ValueError(msg)
    comps = {**{f"positions[{k!r}]": v for k, v in positions.items()}}
    if velocities is not None:
        comps |= {f"velocities[{k!r}]": v for k, v in velocities.items()}
    lengths = {k: jnp.shape(v) for k, v in comps.items()}
    if len(set(lengths.values())) > 1 or any(len(s) != 1 for s in lengths.values()):
        msg = (
            "every component needs one value per tracer, the same number; "
            f"got {lengths}."
        )
        raise ValueError(msg)
    if not isinstance(method, str) or method not in ("svi", "nuts"):  # type: ignore[redundant-expr]  # runtime guard on a mistyped argument
        msg = f"method must be 'svi' or 'nuts', got {method!r}."
        raise ValueError(msg)
    # Each NUTS chain keeps half the draws, rounded up, and split R-hat needs 4.
    least = {"n_draws": 7 if method == "nuts" else 1}
    least |= {"n_steps": 1} if method == "svi" else {"n_warmup": 1}
    out = {}
    for name, low in least.items():
        value = _scalar(counts[name], "iu")
        if value is None or value < low:
            where = f" for {method}" if name == "n_draws" else ""
            msg = f"{name} must be an integer >= {low}{where}, got {counts[name]!r}."
            raise ValueError(msg)
        out[name] = int(value)
    if velocities is not None and set(velocities) != set(positions):
        msg = (
            f"velocities need the positions' components {sorted(positions)}, "
            f"got {sorted(velocities)}."
        )
        raise ValueError(msg)
    return out


def sample_track(
    positions: VectorComponents,
    velocities: VectorComponents | None,
    *,
    key: PRNGKeyArray,
    method: Literal["svi", "nuts"] = "svi",
    metadata: StateMetadata | None = None,
    weights: Float[Array, " N"] | None = None,
    ordering: OrderingResult | None = None,
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
    guide, or NUTS, samples the posterior. The ordering matters: without the
    velocity tie, a start that takes a wrong turn where the stream crosses
    itself stays wrong. A hard tie got some of those back in the prototypes;
    the default loose tie wasn't measured from such a start.

    Parameters
    ----------
    positions, velocities
        Phase-space tracers, arrays or `unxt` Quantities, with matching
        component names. Values must be finite. Every position component
        needs the same dimension, since one length scales them all, and the
        velocities are their time derivatives, in the frame where the track
        is at rest (see ``velocity_tie``). A catalogue with some missing
        velocities, which the default ordering accepts, needs those tracers
        left out here, or ``velocities=None``. ``velocities=None`` fits the
        positions alone, for example the pixels of an image; it needs an
        ``ordering``, which you can make by ordering the positions with zero
        velocities (e.g. with :class:`~phasecurvefit.orderers.RidgeOrderer`).
        With one position component the track has no direction across it,
        so ``radius`` is 0 everywhere; use the draws' ``width`` and their
        spread along the track instead.
    key
        Split for the inference and for drawing.
    method
        ``"svi"`` (the default) fits a full-rank Gaussian to the posterior.
        ``"nuts"`` runs two chains of NUTS, which is slower and warns when the
        chains disagree (split $\hat R$ above 1.05). The posterior often has
        more than one mode, and then NUTS chains stay in different ones.
    metadata
        Passed to :func:`~phasecurvefit.orderers.default_pipeline` when there
        is no ``ordering``; Quantity inputs then need its unit system.
    weights
        Per-tracer weights. They act as counts: the band of a weight-2 star is
        that of two stars, so scaling every weight changes the band.
    ordering
        The ordering of these tracers to start from. ``None`` runs
        :func:`~phasecurvefit.orderers.default_pipeline`. When there are
        velocities, the start is turned to run along them.
    n_draws
        Number of posterior draws to return. NUTS needs at least 7, so that
        each chain keeps the 4 draws its split $\hat R$ needs.
    n_steps
        SVI optimisation steps. The learning rate falls from 2e-2 to 2e-4
        over them, so ``losses`` flattens at the end even when the fit could
        still improve. On the helix test case the loss at 4000 steps was 16
        nats short of a 12000-step run, and its band was within 7%.
    n_warmup
        NUTS warmup steps per chain. Each chain then draws half of
        ``n_draws``, rounded up, and the first ``n_draws`` are kept.
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
    counts: dict[str, object] = {
        "n_draws": n_draws,
        "n_steps": n_steps,
        "n_warmup": n_warmup,
    }
    checked = _check_options(positions, velocities, method, counts)
    n_draws = checked["n_draws"]
    n_steps = checked.get("n_steps", n_steps)
    n_warmup = checked.get("n_warmup", n_warmup)
    d = len(positions) * (1 if velocities is None else 2)
    _, n_basis, n_grid, line_density, width_varies, velocity_tie = _model_options(
        d, len(positions), n_basis, n_grid, line_density, width_varies, velocity_tie
    )
    if _scalar(background, "b") is None:
        msg = f"background must be True or False, got {background!r}."
        raise ValueError(msg)
    background = bool(background)
    keep, kept_pos, kept_vel, weights = _keep_positive(positions, velocities, weights)
    z, centre, scale = _scale(kept_pos, kept_vel, weights)
    if not bool(jnp.all(jnp.isfinite(z))):
        msg = (
            "sample_track needs finite positions that are not all equal, and "
            "finite velocities that are not all zero (pass velocities=None to "
            "fit positions alone)."
        )
        raise ValueError(msg)
    n_q = len(positions)
    if ordering is None:
        if kept_vel is None:
            msg = "sample_track(velocities=None) needs an ordering of the positions."
            raise ValueError(msg)
        ordering = default_pipeline(
            kept_pos, kept_vel, metadata=metadata, weights=weights
        )
        order = np.asarray(ordering.ordering)
    else:  # of every tracer: check it, then keep the tracers of positive weight
        _check_ordering(positions, centre, scale, ordering, keep)
        full = np.asarray(ordering.ordering)
        order = (np.cumsum(keep) - 1)[full[keep[full]]]
    idx, gamma0 = _start_gamma(z, n_q, order, weights)

    opts: dict[str, Any] = {
        "n_q": n_q,
        "n_basis": n_basis,
        "n_grid": n_grid,
        "line_density": line_density,
        "width_varies": width_varies,
        "velocity_tie": velocity_tie,
    }
    w_start = None if weights is None else np.asarray(weights)[idx]
    init = _start(z, gamma0, idx, **opts, weights=w_start)
    log_rho = float(jnp.log(uniform_background_density(z))) if background else None
    if background:  # NumPyro would otherwise start it anywhere in (0.12, 0.88)
        init["f_bg"] = jnp.asarray(0.05)
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
        # From 2e-2 down to 2e-4: big early steps, little gradient noise at the
        # end. A constant 2e-3, with f_bg started at random, gave bands 30-80%
        # wider at 4000 steps.
        rate = optax.exponential_decay(2e-2, max(n_steps, 1), 1e-2)
        opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(rate))
        res = numpyro.infer.SVI(model, guide, opt, numpyro.infer.Trace_ELBO()).run(
            key_fit, n_steps, progress_bar=False
        )
        losses = res.losses
        if not bool(jnp.isfinite(losses[-1])):
            msg = (
                "SVI diverged (the loss is not finite). Try more tracers, a "
                "smaller n_basis, or another key."
            )
            raise RuntimeError(msg)
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
        mcmc.run(key_fit, extra_fields=("diverging",))
        _warn_if_chains_disagree(mcmc.get_samples(group_by_chain=True))
        _warn_if_divergent(int(jnp.sum(mcmc.get_extra_fields()["diverging"])))
        samples = {k: v[:n_draws] for k, v in mcmc.get_samples().items()}

    samples = {k: v for k, v in samples.items() if k != "loglik"}
    _require_finite(samples, method)
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
